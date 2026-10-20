"""Regression: a terminal failure must name the WHOLE fallback walk, not only the hop it
ended on.

The reported shape: a cron job pinned to ``nous``/``deepseek-v4.1-flash`` failed with
``RuntimeError: Gemini HTTP 429`` while a live probe of its pinned model served. The error
was the *last* hop's message; nothing in the surfaced result said why the earlier hops had
not served — the primary's fair-share 429 and any skipped entry lived only in ``logger.*``
lines. ``cron/scheduler.py`` raises ``RuntimeError(result["error"])``, so that masked text
is what the operator sees.

``agent.fallback_trail`` records the hop decisions and both terminal result builders in
``agent.turn_recovery`` render them beside the error. The cron delivery half of this fix
lives in ``tests/cron/test_cron_failure_walk_clause.py`` (placement mirrors the source tree).
"""
from unittest.mock import MagicMock, patch

from agent import fallback_trail as ft
from agent.chat_completion_helpers import (
    _fallback_chain_exhausted, _fallback_entry_key, _should_skip_fallback_candidate,
    try_activate_fallback,
)
from agent.error_classifier import FailoverReason, classify_api_error


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _agent(provider="nous", model="deepseek-v4.1-flash", base_url="https://inference-api.nousresearch.com/v1"):
    """Minimal AIAgent-shaped object for the fallback walk (explicit attrs, no MagicMock
    auto-attributes for anything a code path reads as a value)."""
    agent = MagicMock()
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.api_mode = "chat_completions"
    agent._fallback_index = 0
    agent._fallback_activated = False
    agent._fallback_chain = []
    agent._rate_limit_backoff_count = 0
    agent._rate_limited_until = 0
    agent._entitlement_rejected_models = ()
    agent._unavailable_fallback_keys = set()
    agent._primary_runtime = {
        "provider": provider, "model": model, "base_url": base_url,
        "api_mode": "chat_completions", "api_key": "primary-key",
    }
    return agent


def _record_incident_walk(agent):
    """The exact walk the reported incident produced: primary rate-limited, then the
    terminal hop rate-limited."""
    ft.reset_fallback_trail(agent)
    ft.record_hop_failure(agent, provider="nous", model="stepfun/step-3.7-flash:free",
                          role="primary", reason="rate limit")
    ft.record_hop_failure(agent, provider="gemini", model="gemini-2.5-flash-lite",
                          role="fallback", reason="rate limit")
    return agent


# ── The trail module ─────────────────────────────────────────────────────────────

def test_walk_renders_every_hop_with_its_own_reason():
    agent = _agent()
    ft.reset_fallback_trail(agent)
    ft.record_hop_failure(agent, provider="nous", model="deepseek-v4.1-flash",
                          role="primary", reason="rate limit")
    ft.record_hop_skip(agent, provider="gemini", model="gemini-2.5-flash-lite",
                       reason="resolves to the same backend that just failed (would loop the failure)")
    ft.record_hop_failure(agent, provider="custom", model="cascade-free", reason="provider server error")
    ft.record_chain_end(agent, entries=2)

    text = ft.format_fallback_trail(agent)
    assert "hop 1 primary nous/deepseek-v4.1-flash — failed: rate limit" in text
    assert (
        "hop 2 fallback gemini/gemini-2.5-flash-lite — skipped: "
        "resolves to the same backend that just failed (would loop the failure)"
    ) in text
    assert "hop 3 fallback custom/cascade-free — failed: provider server error" in text
    assert "hop 4 chain" in text and "stopped: end of chain" in text


def test_single_backend_failure_renders_nothing():
    """No hop was taken, so the terminal error already names that backend — a walk block
    would be pure noise on every failed turn."""
    agent = _agent()
    ft.reset_fallback_trail(agent)
    ft.record_hop_failure(agent, provider="nous", model="deepseek-v4.1-flash", role="primary", reason="timeout")
    assert ft.format_fallback_trail(agent) == ""


def test_failure_dedupe_keeps_one_line_per_backend():
    """The retry loop re-enters try_activate_fallback for the same still-failing backend
    once the chain is exhausted; that must not append a second identical line."""
    agent = _agent()
    ft.reset_fallback_trail(agent)
    for _ in range(3):
        ft.record_hop_failure(agent, provider="nous", model="deepseek-v4.1-flash",
                              role="primary", reason="rate limit")
    assert len(ft.fallback_trail(agent)) == 1


def test_chained_failures_are_not_deduped_away():
    agent = _agent()
    ft.reset_fallback_trail(agent)
    ft.record_hop_failure(agent, provider="nous", model="a", role="primary", reason="rate limit")
    ft.record_hop_failure(agent, provider="nous", model="b", role="fallback", reason="rate limit")
    assert [e["backend"] for e in ft.fallback_trail(agent)] == ["nous/a", "nous/b"]


# ── The terminal error surface ───────────────────────────────────────────────────

def test_terminal_error_carries_the_walk_not_just_the_last_hop():
    agent = _record_incident_walk(_agent())
    result = {
        "final_response": "API call failed after 3 retries: Gemini HTTP 429",
        "messages": [], "api_calls": 5, "completed": False, "failed": True,
        "error": "Gemini HTTP 429 (RESOURCE_EXHAUSTED): You exceeded your current quota",
    }

    out = ft.attach_fallback_trail(result, agent)

    # The last hop is still named first — nothing is replaced.
    assert out["error"].startswith("Gemini HTTP 429 (RESOURCE_EXHAUSTED)")
    # …but the walk now says why the route got there. Before the fix the error contained
    # no hop-1 entry at all: that was the masking.
    assert "hop 1 primary nous/stepfun/step-3.7-flash:free — failed: rate limit" in out["error"]
    assert "hop 2 fallback gemini/gemini-2.5-flash-lite — failed: rate limit" in out["error"]
    assert out["fallback_trail"] == ft.fallback_trail(agent)
    assert [e["outcome"] for e in out["fallback_trail"]] == ["failed", "failed"]


def test_attach_is_a_no_op_without_a_walk():
    agent = _agent()
    ft.reset_fallback_trail(agent)
    ft.record_hop_failure(agent, provider="nous", model="deepseek-v4.1-flash", role="primary", reason="timeout")
    result = {"error": "Connection error.", "failed": True, "completed": False}

    out = ft.attach_fallback_trail(result, agent)

    assert out["error"] == "Connection error."
    assert "fallback_trail" not in out


def test_attach_tolerates_non_dict_results():
    agent = _record_incident_walk(_agent())
    assert ft.attach_fallback_trail(None, agent) is None


def test_walk_without_an_existing_error_becomes_the_error():
    agent = _record_incident_walk(_agent())
    out = ft.attach_fallback_trail({"error": "", "failed": True, "completed": False}, agent)
    assert out["error"].startswith("Fallback chain walk")


# ── The walk as recorded by try_activate_fallback ────────────────────────────────

def test_skip_reason_is_text_so_a_passed_over_hop_can_be_explained():
    agent = _agent()
    fb = {"provider": "custom", "model": "cascade-free", "base_url": "http://127.0.0.1:8319/v1"}
    agent._unavailable_fallback_keys = {_fallback_entry_key(fb)}

    reason = _should_skip_fallback_candidate(
        agent, fb, _fallback_entry_key(fb), "custom", "cascade-free", agent._unavailable_fallback_keys
    )

    assert reason == "already marked unavailable this session"
    assert _should_skip_fallback_candidate(
        agent, {"provider": "", "model": ""}, ("", "", ""), "", "", set()
    ) == "entry is missing a provider or model"


def test_exhausted_chain_is_recorded_with_its_length():
    agent = _agent()
    agent._fallback_chain = [{"provider": "custom", "model": "cascade-free"}]
    ft.reset_fallback_trail(agent)
    ft.record_hop_failure(agent, provider="nous", model="deepseek-v4.1-flash", role="primary", reason="rate limit")

    assert _fallback_chain_exhausted(agent, FailoverReason.rate_limit) is False

    trail = ft.fallback_trail(agent)
    assert trail[-1]["outcome"] == "exhausted"
    assert "end of chain (1 entry)" in trail[-1]["reason"]


def test_empty_chain_records_no_end_marker_so_failed_turns_stay_quiet():
    agent = _agent()
    agent._fallback_chain = []
    ft.reset_fallback_trail(agent)
    ft.record_hop_failure(agent, provider="nous", model="deepseek-v4.1-flash", role="primary", reason="rate limit")

    _fallback_chain_exhausted(agent, FailoverReason.rate_limit)

    assert [e["outcome"] for e in ft.fallback_trail(agent)] == ["failed"]
    assert ft.format_fallback_trail(agent) == ""


def test_try_activate_records_the_failing_backend_then_the_skipped_hop():
    agent = _agent(provider="nous", model="deepseek-v4.1-flash")
    fb = {"provider": "custom", "model": "cascade-free", "base_url": "http://127.0.0.1:8319/v1"}
    agent._fallback_chain = [fb]
    agent._unavailable_fallback_keys = {_fallback_entry_key(fb)}

    assert try_activate_fallback(agent, FailoverReason.rate_limit) is False

    trail = ft.fallback_trail(agent)
    assert [e["outcome"] for e in trail] == ["failed", "skipped", "exhausted"]
    assert trail[0]["role"] == "primary"
    assert trail[0]["reason"] == "rate limit"
    assert trail[1]["backend"] == "custom/cascade-free"
    assert trail[1]["reason"] == "already marked unavailable this session"


def test_try_activate_records_an_unactivatable_hop_with_the_cause():
    agent = _agent()
    agent._fallback_chain = [{"provider": "custom", "model": "cascade-free"}]

    with patch("agent.chat_completion_helpers._should_skip_fallback_candidate", return_value=None), \
         patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)):
        assert try_activate_fallback(agent, FailoverReason.server_error) is False

    trail = ft.fallback_trail(agent)
    assert [e["outcome"] for e in trail] == ["failed", "skipped", "exhausted"]
    assert trail[1]["reason"] == "provider not configured"
    assert ft.format_fallback_trail(agent)


def test_a_new_walk_does_not_inherit_the_previous_walk():
    agent = _record_incident_walk(_agent())
    assert len(ft.fallback_trail(agent)) == 2

    # _fallback_index == 0 is the retry loop re-arming the chain: a fresh failure.
    agent._fallback_chain = [{"provider": "custom", "model": "cascade-free"}]
    agent._unavailable_fallback_keys = {
        _fallback_entry_key({"provider": "custom", "model": "cascade-free"})
    }
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)):
        try_activate_fallback(agent, FailoverReason.rate_limit)

    assert [e["backend"] for e in ft.fallback_trail(agent)][0] == "nous/deepseek-v4.1-flash"


# ── Wiring: the terminal result builder ──────────────────────────────────────────

class _QuotaError(Exception):
    """HTTP-status-bearing error stand-in (RuntimeError's type is immutable)."""

    status_code = 429


class _FakeAgent:
    """Enough surface for max_retries_exhausted_result to run for real."""

    log_prefix = ""

    def __init__(self):
        self.provider = "gemini"
        self.model = "gemini-2.5-flash-lite"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self._fallback_chain = [{"provider": "custom", "model": "cascade-free"}]
        self._fallback_index = 1
        self._fallback_activated = True
        self._primary_runtime = {
            "provider": "nous", "model": "stepfun/step-3.7-flash:free",
            "base_url": "https://inference-api.nousresearch.com/v1",
            "api_mode": "chat_completions", "api_key": "k",
        }
        self.lines = []
        self.status = []

    def _flush_status_buffer(self):
        pass

    def _summarize_api_error(self, error):
        return str(error)

    def _emit_status(self, line):
        self.status.append(str(line))

    def _emit_diagnostic_status(self, line):
        self.status.append(str(line))

    def _vprint(self, line, force=False, diagnostic=False):
        self.lines.append(str(line))

    def _persist_session(self, *a, **k):
        pass

    def _dump_api_request_debug(self, *a, **k):
        pass


def test_max_retries_exhausted_result_renders_the_walk():
    from agent.turn_recovery import max_retries_exhausted_result

    agent = _FakeAgent()
    err = _QuotaError("Gemini HTTP 429 (RESOURCE_EXHAUSTED): You exceeded your current quota")
    classified = classify_api_error(
        err, provider="gemini", model="gemini-2.5-flash-lite", approx_tokens=1200,
        context_length=1_000_000, num_messages=3,
    )
    ft.reset_fallback_trail(agent)
    ft.record_hop_failure(agent, provider="nous", model="stepfun/step-3.7-flash:free",
                          role="primary", reason="rate limit")
    ft.record_hop_failure(agent, provider="gemini", model="gemini-2.5-flash-lite",
                          role="fallback", reason="rate limit")

    result = max_retries_exhausted_result(
        agent, err, classified, max_retries=3, is_rate_limited=True,
        error_msg=str(err).lower(), api_kwargs=None, api_messages=[], messages=[],
        conversation_history=[], api_call_count=5, approx_tokens=1200,
        provider="gemini", base_url=agent.base_url, model=agent.model,
    )

    assert "hop 1 primary nous/stepfun/step-3.7-flash:free — failed: rate limit" in result["error"]
    assert result["fallback_trail"][1]["reason"] == "rate limit"
    assert any("hop 1 primary" in line for line in agent.lines)
