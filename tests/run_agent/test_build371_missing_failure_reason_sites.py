"""BUILD-371 regression: two more terminal failure returns in
``conversation_loop.py`` omitted ``failure_reason`` entirely, so a kanban
worker dying at either site always exited 1 ("crashed" + circuit breaker)
instead of the graceful 75 ("rate_limited"/requeue) exit that ``cli.py``'s
kanban gate grants ``rate_limit`` / ``billing`` / ``upstream_rate_limit``.

Site 1 — the Nous Portal preemptive rate-limit guard (top of the retry
loop, before any API call is attempted this iteration): a genuine quota
death with no fallback provider configured must report
``failure_reason: "rate_limit"``.

Site 2 — the ``is_client_error`` non-retryable terminal return: billing is
deliberately NOT excluded from ``is_client_error`` (see the inline comment
above it in conversation_loop.py), so a quota death can land here after the
fallback chain exhausts on a non-quota client error. Must apply the same
origin-preference semantics as the other three sites (BUILD-343): prefer
an earlier-recorded quota-class origin over this hop's own non-quota
classification.

Harness mirrors test_build342_pool_exhausted_fallback.py and
test_build343_quota_origin_survives_chain_exhaustion.py: drive the real
``run_conversation()`` loop, mock only the two true I/O boundaries (the
outer API call and the fallback provider's client construction).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(provider="deepseek", model="deepseek-chat",
                 base_url="https://api.deepseek.com", fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key-abcdef12",
            base_url=base_url,
            provider=provider,
            model=model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="fallback/model", usage=None)


def _billing_error():
    err = Exception(
        "Error code: 402 - Insufficient balance, please add funds to your account."
    )
    err.status_code = 402
    return err


def _mock_fallback_client(base_url="https://open.bigmodel.cn/api/coding/paas/v4", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    mock._custom_headers = None
    mock.default_headers = None
    return mock


_FB_CHAIN = [
    {
        "provider": "zai",
        "model": "glm-4.7",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
    }
]


class TestNousPreemptiveGuardReportsRateLimitFailureReason:
    """Site 1: Nous Portal rate limit active, no fallback configured —
    the guard must die with failure_reason='rate_limit' so the kanban
    exit-75 gate fires instead of counting this as a crash."""

    def test_nous_rate_limit_no_fallback_reports_rate_limit_failure_reason(self):
        agent = _make_agent(
            provider="nous",
            model="hermes-4-405b",
            base_url="https://inference.nous.nousresearch.com/v1",
            fallback_model=None,
        )

        with (
            patch(
                "agent.nous_rate_guard.nous_rate_limit_remaining",
                return_value=600,
            ),
            # Guard against a vacuous pass: if the guard failed to return
            # early, this would raise instead of silently succeeding.
            patch.object(
                agent, "_interruptible_api_call",
                side_effect=AssertionError(
                    "should not reach the API call — Nous rate limit guard "
                    "must return before this"
                ),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["failed"] is True
        assert result["completed"] is False
        assert "rate limit" in result["error"].lower()
        assert result["failure_reason"] == "rate_limit"

    def test_nous_rate_limit_with_fallback_available_still_falls_back(self):
        """Control: when a fallback IS configured, the guard escalates to
        it instead of returning — this failure_reason addition must not
        change that existing recovery behavior."""
        agent = _make_agent(
            provider="nous",
            model="hermes-4-405b",
            base_url="https://inference.nous.nousresearch.com/v1",
            fallback_model=_FB_CHAIN,
        )

        def fake_api_call(api_kwargs):
            return _mock_response("Recovered via fallback")

        mock_fb_client = _mock_fallback_client()

        with (
            patch(
                "agent.nous_rate_guard.nous_rate_limit_remaining",
                return_value=600,
            ),
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb_client, "glm-4.7"),
            ) as mock_resolve,
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation("hello")

        mock_resolve.assert_called_once()
        assert agent._fallback_activated is True
        assert result.get("failed") is not True
        assert result["completed"] is True
        assert result["final_response"] == "Recovered via fallback"


class TestClientErrorTerminalReportsFailureReason:
    """Site 2: the is_client_error non-retryable terminal return."""

    def test_plain_nonquota_client_error_reports_own_reason(self):
        """Control: no quota event anywhere in the turn — a plain 422
        (format_error, non-retryable, no fallback configured) must report
        its own classification unchanged."""
        agent = _make_agent(fallback_model=None)

        def fake_api_call(api_kwargs):
            err = Exception("Unprocessable Entity: invalid request shape")
            err.status_code = 422
            raise err

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["failed"] is True
        assert result["completed"] is False
        assert result["failure_reason"] == "format_error"

    def test_billing_then_nonquota_client_error_reports_billing_origin(self):
        """Primary hits billing (402) -> eager fallback activates the
        (only) fallback tier and records quota_origin_reason='billing' ->
        that tier then hits a plain non-retryable 422 -> chain is already
        exhausted (one entry, already consumed) -> is_client_error
        terminal return. The park's own ``classified.reason`` is
        'format_error' (not quota-class), so without the recorded-origin
        override this would report 'format_error' and miss the kanban
        exit-75 gate."""
        agent = _make_agent(fallback_model=_FB_CHAIN)

        calls = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            if len(calls) == 1:
                raise _billing_error()
            err = Exception("Unprocessable Entity: invalid request shape")
            err.status_code = 422
            raise err

        mock_fb_client = _mock_fallback_client()

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb_client, "glm-4.7"),
            ) as mock_resolve,
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation("hello")

        # Guard against a vacuous pass: both hops actually fired and the
        # single-entry chain is genuinely exhausted by the time of the
        # second (client-error) failure — no third tier exists.
        assert calls == [
            ("deepseek", "deepseek-chat"),
            ("zai", "glm-4.7"),
        ]
        mock_resolve.assert_called_once()
        assert agent._fallback_activated is True

        assert result["failed"] is True
        assert result["completed"] is False
        assert result["failure_reason"] == "billing"
