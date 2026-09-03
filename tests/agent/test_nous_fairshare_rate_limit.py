"""Nous fair-share (model-scoped) 429 handling.

Some Nous inference models are "fairshare-governed": exceeding a per-identity
adaptive rate returns a 429 that is specific to THAT model.  Other Nous models
remain available and the credential is healthy, so the correct recovery is a
model switch (preferring the gateway's live ``alternates``) or an honest
``retry_after`` wait — never credential rotation and never the cross-session
Nous guard, which would block every Nous model for every session.

Portal-wide per-key 429s (plain ``{status, message}`` body, no ``reason``)
must keep today's behaviour exactly; the detection predicate is unreachable
on them.
"""

import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import (
    FAIRSHARE_REASONS,
    FailoverReason,
    classify_api_error,
    parse_fairshare_refusal,
)


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError (optionally with a response)."""

    def __init__(self, message, status_code=None, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        if headers is not None:
            self.response = SimpleNamespace(headers=headers)


FAIRSHARE_BODY = {
    "status": 429,
    "message": "You've reached this model's current fair-share rate limit.",
    "reason": "rate_limited",
    "retry_after": 51,
    "alternates": ["model-b", "model-c"],
    "upgrade_url": "https://portal.nousresearch.com/upgrade",
}

PLAIN_NOUS_BODY = {"status": 429, "message": "Rate limit exceeded (rpm)."}


def _fairshare_error(body=None, headers=None, message=None):
    body = dict(FAIRSHARE_BODY if body is None else body)
    return MockAPIError(
        message or body.get("message", "rate limited"),
        status_code=429,
        body=body,
        headers=headers,
    )


# ── Classification ──────────────────────────────────────────────────────


class TestFairshareClassification:
    def test_fairshare_429_is_model_scoped_with_context(self):
        result = classify_api_error(_fairshare_error(), provider="nous", model="model-a")
        assert result.reason == FailoverReason.upstream_rate_limit
        assert result.should_rotate_credential is False
        assert result.should_fallback is True
        assert result.retryable is True
        ctx = result.error_context
        assert ctx["fairshare"] is True
        assert ctx["fairshare_reason"] == "rate_limited"
        assert ctx["retry_after"] == 51
        assert ctx["alternates"] == ["model-b", "model-c"]
        assert ctx["upgrade_url"] == "https://portal.nousresearch.com/upgrade"

    @pytest.mark.parametrize("provider", ["nous", "nous-portal", "nousresearch", "Nous"])
    def test_all_nous_provider_aliases_detect(self, provider):
        result = classify_api_error(_fairshare_error(), provider=provider)
        assert result.reason == FailoverReason.upstream_rate_limit
        assert result.error_context.get("fairshare") is True

    def test_plain_nous_429_classifies_exactly_as_before(self):
        """Today's {status, message} Nous 429 → global rate_limit, rotation on,
        no fairshare context.  Pins the pre-fairshare contract."""
        err = MockAPIError("Rate limit exceeded (rpm).", status_code=429, body=PLAIN_NOUS_BODY)
        result = classify_api_error(err, provider="nous", model="model-a")
        assert result.reason == FailoverReason.rate_limit
        assert result.should_rotate_credential is True
        assert result.should_fallback is True
        assert result.retryable is True
        assert "fairshare" not in result.error_context
        assert result.message == "Rate limit exceeded (rpm)."

    def test_plain_nous_429_with_retry_after_header_only_is_unchanged(self):
        """A Retry-After header alone (no body fields) must not trigger."""
        err = MockAPIError(
            "Rate limit exceeded", status_code=429, body=PLAIN_NOUS_BODY,
            headers={"Retry-After": "30"},
        )
        result = classify_api_error(err, provider="nous")
        assert result.reason == FailoverReason.rate_limit
        assert "fairshare" not in result.error_context

    @pytest.mark.parametrize("provider", ["openrouter", "openai", "anthropic", "", "custom"])
    def test_fairshare_shaped_body_from_non_nous_provider_is_unchanged(self, provider):
        result = classify_api_error(_fairshare_error(), provider=provider)
        assert result.reason == FailoverReason.rate_limit
        assert result.should_rotate_credential is True
        assert "fairshare" not in result.error_context

    def test_reason_without_retry_after_does_not_match(self):
        body = dict(FAIRSHARE_BODY)
        del body["retry_after"]
        result = classify_api_error(_fairshare_error(body), provider="nous")
        assert result.reason == FailoverReason.rate_limit

    def test_retry_after_without_reason_does_not_match(self):
        body = dict(FAIRSHARE_BODY)
        del body["reason"]
        result = classify_api_error(_fairshare_error(body), provider="nous")
        assert result.reason == FailoverReason.rate_limit

    def test_unknown_reason_does_not_match(self):
        body = dict(FAIRSHARE_BODY, reason="something_else")
        result = classify_api_error(_fairshare_error(body), provider="nous")
        assert result.reason == FailoverReason.rate_limit

    @pytest.mark.parametrize("reason", sorted(FAIRSHARE_REASONS))
    def test_every_enum_reason_matches(self, reason):
        body = dict(FAIRSHARE_BODY, reason=reason)
        result = classify_api_error(_fairshare_error(body), provider="nous")
        assert result.reason == FailoverReason.upstream_rate_limit
        assert result.error_context["fairshare_reason"] == reason

    def test_non_429_fairshare_body_is_not_fairshare(self):
        err = MockAPIError("boom", status_code=503, body=FAIRSHARE_BODY)
        result = classify_api_error(err, provider="nous")
        assert result.reason != FailoverReason.upstream_rate_limit
        assert "fairshare" not in result.error_context


class TestFairshareDefensiveParsing:
    def test_absent_alternates_and_upgrade_url(self):
        body = {"status": 429, "message": "m", "reason": "rate_limited", "retry_after": 10}
        ctx = classify_api_error(_fairshare_error(body), provider="nous").error_context
        assert ctx["fairshare"] is True
        assert ctx["alternates"] == []
        assert ctx["upgrade_url"] is None

    def test_empty_alternates(self):
        body = dict(FAIRSHARE_BODY, alternates=[])
        ctx = classify_api_error(_fairshare_error(body), provider="nous").error_context
        assert ctx["alternates"] == []

    def test_retry_after_zero_still_matches(self):
        body = dict(FAIRSHARE_BODY, retry_after=0)
        result = classify_api_error(_fairshare_error(body), provider="nous")
        assert result.reason == FailoverReason.upstream_rate_limit
        assert result.error_context["retry_after"] == 0

    @pytest.mark.parametrize("bad", ["soon", None, True, -5, [1]])
    def test_invalid_retry_after_does_not_match(self, bad):
        body = dict(FAIRSHARE_BODY, retry_after=bad)
        result = classify_api_error(_fairshare_error(body), provider="nous")
        assert result.reason == FailoverReason.rate_limit

    def test_retry_after_exceeding_120s_is_kept_exact(self):
        body = dict(FAIRSHARE_BODY, retry_after=301)
        ctx = classify_api_error(_fairshare_error(body), provider="nous").error_context
        assert ctx["retry_after"] == 301

    def test_malformed_alternates_are_sanitised(self):
        body = dict(FAIRSHARE_BODY, alternates=["model-b", 42, "", " model-c ", "model-b", None])
        ctx = classify_api_error(_fairshare_error(body), provider="nous").error_context
        assert ctx["alternates"] == ["model-b", "model-c"]

    def test_non_list_alternates_degrade_to_empty(self):
        body = dict(FAIRSHARE_BODY, alternates="model-b")
        ctx = classify_api_error(_fairshare_error(body), provider="nous").error_context
        assert ctx["alternates"] == []

    @pytest.mark.parametrize("bad_url", ["javascript:alert(1)", "portal", 7, ""])
    def test_non_http_upgrade_url_is_dropped(self, bad_url):
        body = dict(FAIRSHARE_BODY, upgrade_url=bad_url)
        ctx = classify_api_error(_fairshare_error(body), provider="nous").error_context
        assert ctx["upgrade_url"] is None

    def test_nested_error_envelope_is_accepted(self):
        body = {"error": dict(FAIRSHARE_BODY)}
        result = classify_api_error(_fairshare_error(body, message="m"), provider="nous")
        assert result.reason == FailoverReason.upstream_rate_limit

    def test_parse_helper_rejects_non_dict(self):
        assert parse_fairshare_refusal("nope", provider="nous") is None
        assert parse_fairshare_refusal(None, provider="nous") is None


class TestFairshareHeaders:
    def test_retry_after_header_is_authoritative_over_body(self):
        headers = {
            "Retry-After": "77",
            "RateLimit": '"fairshare";r=0;t=77',
            "RateLimit-Policy": '"fairshare";q=1000;qu="tokens";w=60',
        }
        ctx = classify_api_error(
            _fairshare_error(headers=headers), provider="nous"
        ).error_context
        assert ctx["retry_after"] == 77
        assert ctx["fairshare_header_confirmed"] is True

    def test_missing_headers_leave_body_value_and_unconfirmed(self):
        ctx = classify_api_error(_fairshare_error(), provider="nous").error_context
        assert ctx["retry_after"] == 51
        assert ctx["fairshare_header_confirmed"] is False

    def test_unparseable_retry_after_header_falls_back_to_body(self):
        ctx = classify_api_error(
            _fairshare_error(headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}),
            provider="nous",
        ).error_context
        assert ctx["retry_after"] == 51

    def test_fairshare_headers_without_body_fields_do_not_match(self):
        """Headers alone never flip a plain 429 — body fields are required."""
        err = MockAPIError(
            "Rate limit exceeded", status_code=429, body=PLAIN_NOUS_BODY,
            headers={"RateLimit-Policy": '"fairshare";q=1;qu="tokens";w=60', "Retry-After": "5"},
        )
        result = classify_api_error(err, provider="nous")
        assert result.reason == FailoverReason.rate_limit


# ── Cross-session guard hygiene ─────────────────────────────────────────


@pytest.fixture
def rate_guard_env(tmp_path, monkeypatch):
    hermes_home = str(tmp_path / ".hermes")
    os.makedirs(hermes_home, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", hermes_home)
    return hermes_home


class TestGuardNonRecording:
    def test_fairshare_context_is_not_recorded(self, rate_guard_env):
        from agent.nous_rate_guard import (
            _state_path,
            nous_rate_limit_remaining,
            record_nous_rate_limit,
        )

        ctx = classify_api_error(_fairshare_error(), provider="nous").error_context
        record_nous_rate_limit(headers={"retry-after": "51"}, error_context=ctx)
        assert not os.path.exists(_state_path())
        assert nous_rate_limit_remaining() is None

    def test_plain_429_context_still_recorded_with_same_schema(self, rate_guard_env):
        from agent.nous_rate_guard import _state_path, record_nous_rate_limit

        record_nous_rate_limit(
            headers={"x-ratelimit-reset-requests-1h": "900"},
            error_context={"reason": "rate_limit", "message": "rpm"},
        )
        with open(_state_path()) as f:
            state = json.load(f)
        assert set(state) == {"reset_at", "recorded_at", "reset_seconds"}
        assert state["reset_at"] > time.time()

    def test_loop_record_condition_excludes_fairshare(self):
        """Mirror the conversation-loop gate: fairshare never reaches the
        breaker even when is_rate_limited is true."""
        classified = classify_api_error(_fairshare_error(), provider="nous")
        is_rate_limited = classified.reason in {
            FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit,
        }
        assert is_rate_limited
        should_record = (
            is_rate_limited
            and classified.reason == FailoverReason.rate_limit
            and not (classified.error_context or {}).get("fairshare")
        )
        assert should_record is False

        plain = classify_api_error(
            MockAPIError("rpm", status_code=429, body=PLAIN_NOUS_BODY), provider="nous"
        )
        assert (
            plain.reason == FailoverReason.rate_limit
            and not (plain.error_context or {}).get("fairshare")
        )


# ── Recovery: alternates-first fallback ordering ────────────────────────


def _make_agent(fallback_model=None, model="model-a"):
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://inference-api.nousresearch.com/v1",
            model=model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        agent.provider = "nous"
        return agent


def _mock_client(api_key="fb-key", base_url="https://inference-api.nousresearch.com/v1"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


class TestAlternatesPreferredOverStaticChain:
    def test_alternates_are_spliced_ahead_of_configured_chain(self):
        from agent.chat_completion_helpers import (
            FAIRSHARE_ALTERNATE_MARKER,
            inject_fairshare_alternates,
        )

        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        n = inject_fairshare_alternates(agent, ["model-b", "model-c"])
        assert n == 2
        assert [(fb["provider"], fb["model"]) for fb in agent._fallback_chain] == [
            ("nous", "model-b"), ("nous", "model-c"), ("openai", "gpt-4o"),
        ]
        assert all(fb.get(FAIRSHARE_ALTERNATE_MARKER) for fb in agent._fallback_chain[:2])
        assert FAIRSHARE_ALTERNATE_MARKER not in agent._fallback_chain[2]

    def test_empty_alternates_leave_chain_untouched(self):
        from agent.chat_completion_helpers import inject_fairshare_alternates

        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        before = list(agent._fallback_chain)
        assert inject_fairshare_alternates(agent, []) == 0
        assert inject_fairshare_alternates(agent, None) == 0
        assert agent._fallback_chain == before

    def test_current_model_and_duplicates_are_skipped(self):
        from agent.chat_completion_helpers import inject_fairshare_alternates

        agent = _make_agent(fallback_model=[{"provider": "nous", "model": "model-c"}])
        n = inject_fairshare_alternates(agent, ["model-a", "model-b", "model-c", "model-b"])
        assert n == 1
        assert [fb["model"] for fb in agent._fallback_chain] == ["model-b", "model-c"]

    def test_alternates_work_with_no_configured_chain(self):
        from agent.chat_completion_helpers import inject_fairshare_alternates

        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert inject_fairshare_alternates(agent, ["model-b"]) == 1
        assert agent._fallback_index < len(agent._fallback_chain)

    def test_injection_respects_current_index(self):
        """Mid-chain (already on a fallback), alternates go at the cursor so
        they are the NEXT thing tried; entries already walked stay behind."""
        from agent.chat_completion_helpers import inject_fairshare_alternates

        agent = _make_agent(fallback_model=[
            {"provider": "nous", "model": "model-x"},
            {"provider": "openai", "model": "gpt-4o"},
        ])
        agent._fallback_index = 1
        inject_fairshare_alternates(agent, ["model-b"])
        assert [fb["model"] for fb in agent._fallback_chain] == ["model-x", "model-b", "gpt-4o"]
        assert agent._fallback_chain[agent._fallback_index]["model"] == "model-b"

    def test_already_walked_model_is_not_reinjected(self):
        """An alternate that was already tried this turn (now behind the
        cursor — e.g. it returned its own fair-share 429) must not be
        spliced in again for an immediate retry."""
        from agent.chat_completion_helpers import inject_fairshare_alternates

        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        assert inject_fairshare_alternates(agent, ["model-b", "model-c"]) == 2
        # Simulate having activated model-b, which then 429'd and named
        # model-b/model-c as alternates again.
        agent._fallback_index = 1
        agent.model = "model-b"
        assert inject_fairshare_alternates(agent, ["model-b", "model-c", "model-d"]) == 1
        assert [fb["model"] for fb in agent._fallback_chain] == [
            "model-b", "model-d", "model-c", "gpt-4o",
        ]

    def test_activation_walks_alternates_first_then_static_chain(self):
        from agent.chat_completion_helpers import inject_fairshare_alternates

        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        inject_fairshare_alternates(agent, ["model-b", "model-c"])
        activated = []

        def _resolve(provider, model, *a, **k):
            activated.append((provider, model))
            return _mock_client(), model

        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            assert agent._try_activate_fallback(reason=FailoverReason.upstream_rate_limit)
            assert agent.model == "model-b"
            assert agent._try_activate_fallback(reason=FailoverReason.upstream_rate_limit)
            assert agent.model == "model-c"
            assert agent._try_activate_fallback(reason=FailoverReason.upstream_rate_limit)
            assert agent.model == "gpt-4o"
            assert not agent._try_activate_fallback(reason=FailoverReason.upstream_rate_limit)
        assert [m for _, m in activated] == ["model-b", "model-c", "gpt-4o"]

    def test_strip_removes_only_injected_entries_and_fixes_index(self):
        from agent.chat_completion_helpers import (
            inject_fairshare_alternates,
            strip_fairshare_alternates,
        )

        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        inject_fairshare_alternates(agent, ["model-b", "model-c"])
        agent._fallback_index = 2  # both alternates consumed; gpt-4o is next
        assert strip_fairshare_alternates(agent) == 2
        assert [fb["model"] for fb in agent._fallback_chain] == ["gpt-4o"]
        assert agent._fallback_index == 0
        assert strip_fairshare_alternates(agent) == 0

    def test_restore_primary_strips_alternates_between_turns(self):
        from agent.chat_completion_helpers import inject_fairshare_alternates

        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        inject_fairshare_alternates(agent, ["model-b"])
        assert not agent._fallback_activated
        agent._restore_primary_runtime()
        assert [fb["model"] for fb in agent._fallback_chain] == ["gpt-4o"]
        assert agent._fallback_index == 0

    def test_fairshare_skips_credential_rotation(self):
        """upstream_rate_limit must skip pool rotation (the key is healthy)."""
        from agent.agent_runtime_helpers import recover_with_credential_pool

        class _Pool:
            provider = "nous"
            mark_exhausted_and_rotate = MagicMock()

            def entries(self):
                return []

        pool = _Pool()
        agent = SimpleNamespace(
            provider="nous",
            base_url="https://inference-api.nousresearch.com/v1",
            api_key="k",
            _credential_pool=pool,
            _credential_pool_entry_id=None,
            _swap_credential=MagicMock(),
        )
        ctx = classify_api_error(_fairshare_error(), provider="nous").error_context
        recovered, _ = recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.upstream_rate_limit,
            error_context=ctx,
        )
        assert recovered is False
        pool.mark_exhausted_and_rotate.assert_not_called()


# ── Recovery: retry_after wait path when no fallback exists ─────────────


class TestRetryAfterWaitPath:
    """Mirror the loop's retry-wait derivation: header first, then the
    fairshare body value, then jittered backoff."""

    @staticmethod
    def _derive_wait(classified, headers=None):
        _retry_after = None
        if headers:
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw:
                try:
                    _retry_after = min(float(raw), 600)
                except (TypeError, ValueError):
                    pass
        ctx = classified.error_context or {}
        fs_ra = ctx.get("retry_after")
        if _retry_after is None and ctx.get("fairshare") and isinstance(fs_ra, (int, float)) and fs_ra > 0:
            _retry_after = min(float(fs_ra), 600)
        return _retry_after

    def test_body_retry_after_is_honored_without_header(self):
        classified = classify_api_error(_fairshare_error(), provider="nous")
        assert self._derive_wait(classified) == 51.0

    def test_long_retry_after_is_kept_up_to_cap(self):
        classified = classify_api_error(
            _fairshare_error(dict(FAIRSHARE_BODY, retry_after=301)), provider="nous"
        )
        assert self._derive_wait(classified) == 301.0

    def test_retry_after_zero_falls_through_to_backoff(self):
        classified = classify_api_error(
            _fairshare_error(dict(FAIRSHARE_BODY, retry_after=0)), provider="nous"
        )
        assert self._derive_wait(classified) is None

    def test_header_still_wins_when_present(self):
        classified = classify_api_error(_fairshare_error(), provider="nous")
        assert self._derive_wait(classified, headers={"retry-after": "9"}) == 9.0

    def test_plain_429_without_header_uses_backoff(self):
        plain = classify_api_error(
            MockAPIError("rpm", status_code=429, body=PLAIN_NOUS_BODY), provider="nous"
        )
        assert self._derive_wait(plain) is None


# ── UX ──────────────────────────────────────────────────────────────────


class TestUpgradeHint:
    def test_upgrade_url_printed_once_per_agent(self):
        from agent.conversation_loop import _print_fairshare_upgrade_hint

        agent = MagicMock()
        agent.log_prefix = ""
        agent._fairshare_upgrade_hinted_url = None
        ctx = classify_api_error(_fairshare_error(), provider="nous").error_context
        assert _print_fairshare_upgrade_hint(agent, ctx) is True
        assert _print_fairshare_upgrade_hint(agent, ctx) is False
        agent._vprint.assert_called_once()
        assert "https://portal.nousresearch.com/upgrade" in agent._vprint.call_args[0][0]

    def test_upgrade_hint_rearms_on_primary_restore(self):
        from agent.conversation_loop import _print_fairshare_upgrade_hint

        agent = _make_agent(fallback_model=None)
        agent._vprint = MagicMock()
        ctx = classify_api_error(_fairshare_error(), provider="nous").error_context
        assert _print_fairshare_upgrade_hint(agent, ctx) is True
        assert _print_fairshare_upgrade_hint(agent, ctx) is False
        agent._restore_primary_runtime()  # new turn
        assert _print_fairshare_upgrade_hint(agent, ctx) is True
        assert agent._vprint.call_count == 2

    def test_absent_upgrade_url_prints_nothing(self):
        from agent.conversation_loop import _print_fairshare_upgrade_hint

        agent = MagicMock()
        body = dict(FAIRSHARE_BODY)
        del body["upgrade_url"]
        ctx = classify_api_error(_fairshare_error(body), provider="nous").error_context
        assert _print_fairshare_upgrade_hint(agent, ctx) is False
        agent._vprint.assert_not_called()
