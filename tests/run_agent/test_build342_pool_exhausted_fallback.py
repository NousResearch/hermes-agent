"""BUILD-342 regression: CredentialPoolExhausted must escalate to the
cross-provider fallback chain instead of hard-returning a failed turn.

Incident (2026-07-10 01:27): a provider's credential pool exhausts (e.g.
deepseek ran out of credit). ``agent._recover_with_credential_pool()``
raises ``CredentialPoolExhausted`` by design (BUILD-262) so the caller can
park/abort instead of spinning on a dead pool with zero backoff. But the
``except CredentialPoolExhausted`` handler in
``agent/conversation_loop.py`` hard-``return``ed a failed turn ~550 lines
*before* the eager cross-provider fallback block
(``agent._try_activate_fallback``) — a configured ``fallback_providers``
chain (ending at a healthy local model) was completely unreachable.

Fix: the handler now makes ONE bounded ``agent._try_activate_fallback(
reason=classified.reason)`` attempt — the same call the rate-limit /
billing / auth failover paths use a few hundred lines below — before
parking. This does not reintroduce the BUILD-262 spin: it is bounded by
the fallback chain's length (``_try_activate_fallback`` returns False
once exhausted), never a retry of pool rotation itself.

Mirrors the harness in test_32646_fallback_429_after_timeout.py and
test_auth_provider_failover.py: drive the real ``run_conversation()``
loop, mock only the two true I/O boundaries (the outer API call and the
fallback provider's client construction).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.credential_pool import CredentialPool, CredentialPoolExhausted
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


def _billing_error(msg="Error code: 402 - Insufficient balance, please add funds to your account."):
    err = Exception(msg)
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


class TestCredentialPoolExhaustedActivatesFallback:
    """Acceptance: billing/402 pool exhaustion with a healthy next tier
    completes the turn on the fallback provider instead of parking."""

    def test_run_conversation_falls_back_after_pool_exhausted(self):
        agent = _make_agent(fallback_model=_FB_CHAIN)

        calls = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            if len(calls) == 1:
                raise _billing_error()
            return _mock_response("Recovered via fallback")

        pool_exhausted = CredentialPoolExhausted(provider="deepseek", earliest_available_at=None)
        mock_fb_client = _mock_fallback_client()

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_recover_with_credential_pool", side_effect=pool_exhausted),
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

        # Guard against a vacuous pass: the mocked pool-exhaustion must
        # actually have fired, and a second call must actually have
        # happened on the fallback provider.
        assert calls == [
            ("deepseek", "deepseek-chat"),
            ("zai", "glm-4.7"),
        ]
        mock_resolve.assert_called_once()
        assert agent._fallback_activated is True
        assert result.get("failed") is not True
        assert result["completed"] is True
        assert result["final_response"] == "Recovered via fallback"

    def test_zero_entry_pool_also_falls_back(self):
        """The second CredentialPoolExhausted variant (``mark_exhausted_
        and_rotate_or_raise`` raising immediately because the pool has no
        credentials configured at all) must advance the chain too — same
        exception class, same handler. Uses a REAL, empty CredentialPool
        so the exception is genuinely raised by production pool code
        (agent._recover_with_credential_pool is NOT mocked here)."""
        agent = _make_agent(fallback_model=_FB_CHAIN)
        agent._credential_pool = CredentialPool(provider="deepseek", entries=[])

        calls = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            if len(calls) == 1:
                raise _billing_error()
            return _mock_response("Recovered via fallback")

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

        assert calls == [
            ("deepseek", "deepseek-chat"),
            ("zai", "glm-4.7"),
        ]
        mock_resolve.assert_called_once()
        assert agent._fallback_activated is True
        assert result.get("failed") is not True
        assert result["completed"] is True
        assert result["final_response"] == "Recovered via fallback"


class TestCredentialPoolExhaustedChainExhaustedStillParks:
    """Unchanged behavior: no configured fallback (or a chain that is
    itself exhausted) still returns the terminal failed-turn result —
    the fix must not turn every pool exhaustion into an infinite chain
    walk or silently swallow a genuine dead end."""

    def test_no_fallback_configured_still_parks(self):
        agent = _make_agent(fallback_model=None)

        def fake_api_call(api_kwargs):
            raise _billing_error()

        pool_exhausted = CredentialPoolExhausted(provider="deepseek", earliest_available_at=None)

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_recover_with_credential_pool", side_effect=pool_exhausted),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["failed"] is True
        assert result["completed"] is False
        assert "deepseek" in result["error"]
        # BUILD-343: this park IS a quota-exhaustion terminal (billing-out
        # pool, no fallback configured) — without failure_reason here,
        # cli.py's kanban exit-75 gate never sees it and the deepseek
        # incident's worker would exit 1 ("crashed") instead of 75
        # ("rate_limited"), tripping the circuit breaker on a quota wall.
        assert result["failure_reason"] == "billing"

    def test_fallback_chain_already_exhausted_still_parks(self):
        """The chain exists but every entry has already been consumed
        this turn (_fallback_index has walked past the end) — activation
        must return False and the handler must fall through to parking,
        not loop forever."""
        agent = _make_agent(fallback_model=_FB_CHAIN)
        agent._fallback_index = len(agent._fallback_chain)  # already exhausted

        def fake_api_call(api_kwargs):
            raise _billing_error()

        pool_exhausted = CredentialPoolExhausted(provider="deepseek", earliest_available_at=None)

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_recover_with_credential_pool", side_effect=pool_exhausted),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["failed"] is True
        assert result["completed"] is False
        assert "deepseek" in result["error"]
        # BUILD-343: same as above — a chain that exists but is already
        # exhausted still parks on a quota-class reason and must report it.
        assert result["failure_reason"] == "billing"


class TestParkReportsRecordedQuotaOriginOverOwnNonQuotaReason:
    """BUILD-343: the park path's failure_reason must prefer a quota
    origin recorded EARLIER in the turn even when the pool-exhaustion
    park itself is triggered by a non-quota-class error.

    Sequence: primary hits billing -> eager fallback activates the (only)
    fallback tier and records quota_origin_reason="billing" ->  that
    tier then hits a plain transport error whose OWN credential-pool
    recovery attempt raises CredentialPoolExhausted -> chain is already
    exhausted (one entry, already consumed) -> park. The park's own
    ``classified.reason`` is ``timeout`` (not quota-class), so without
    the recorded-origin override the deepseek-style incident would still
    report "timeout" and miss the kanban exit-75 gate.
    """

    def test_park_after_non_quota_pool_exhaustion_reports_earlier_quota_origin(self):
        agent = _make_agent(fallback_model=_FB_CHAIN)

        calls = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            if len(calls) == 1:
                raise _billing_error()
            raise ConnectionError("connection refused: fallback tier unreachable")

        def fake_recover(*, status_code, has_retried_429, classified_reason=None, error_context=None):
            # First hop (billing): let it fall through to the eager
            # quota-fallback block below instead of pool rotation.
            if len(calls) == 1:
                return (False, has_retried_429)
            # Second hop (transport error on the fallback tier): its own
            # credential pool is exhausted too.
            raise CredentialPoolExhausted(provider="zai", earliest_available_at=None)

        mock_fb_client = _mock_fallback_client()

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_recover_with_credential_pool", side_effect=fake_recover),
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
        # second CredentialPoolExhausted (no third tier exists).
        assert calls == [
            ("deepseek", "deepseek-chat"),
            ("zai", "glm-4.7"),
        ]
        mock_resolve.assert_called_once()
        assert agent._fallback_activated is True

        assert result["failed"] is True
        assert result["completed"] is False
        # The park's own error text is the pool-exhaustion message for the
        # SECOND hop ("zai") — that part is unchanged by this fix. Only
        # the failure_reason classification prefers the earlier quota
        # origin ("billing") over this hop's own transport-triggered
        # pool exhaustion.
        assert result["failure_reason"] == "billing"
        assert "zai" in result["error"]
