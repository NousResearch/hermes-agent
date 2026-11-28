"""A transient provider overload must retry the whole backoff budget before falling back.

`classify_api_error` deliberately marks `overloaded` as `should_fallback=False`: the cascade
for that reason is documented to run *after* the backoff budget, not immediately.
`route_classified_error` contradicted it with a blanket ``retry_count >= 2`` applied to every
transport reason (timeout AND overloaded), so a 429/503 that merely meant "provider busy, try
again shortly" walked to the next provider on the second retry while attempts remained.

This is the defect Z.AI's workaround and Ark's ``ServerOverloaded`` both ran into. Note Ark
needed NO classifier change: upstream's ``_OVERLOADED_PATTERNS`` already contains the
substring ``"overloaded"`` and matches ``ServerOverloaded`` case-insensitively — the
classification was right and the retry THRESHOLD was wrong, which is why the fix is general
and lives at the threshold rather than in a provider-specific branch.

These tests drive ``route_classified_error`` itself, because it is the only seam where the
two thresholds (``>= 2`` for a plain transport failure, ``>= max_retries`` for an overload)
produce different outcomes. The later ``settle_unrecovered_error`` stage cannot observe the
difference: by then retries are spent and both thresholds collapse to the same result, so a
test written against it passes with and without the fix.
"""
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_retry_state import TurnRetryState


class _Err(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = None
        self.body = body if body is not None else {"error": {"message": message}}


class _Agent:
    """Only the fallback seam is real; every other helper the path touches is a no-op."""

    log_prefix = ""
    verbose = False
    provider = "openai"
    _fallback_chain = [object(), object()]
    _fallback_index = 0
    _credential_pool = None

    def __init__(self):
        self.activated = []

    def _has_pending_fallback(self):
        return True

    def _try_activate_fallback(self, **kwargs):
        self.activated.append(kwargs.get("reason"))
        return True

    def _summarize_api_error(self, error):
        return str(error)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _route(agent, err, *, retry_count, max_retries, provider="openai"):
    """Call the real routing function.

    ``route_classified_error`` imports its collaborators with FUNCTION-LOCAL imports; the
    ``agent.conversation_loop`` one drags in process bootstrap and interrupted-update
    recovery, which probes a path under the real hermes home (the suite's home-IO guard
    rightly refuses it, and from a git worktree that path lands under the guarded root while
    a plain checkout hides it). The no-op agent declines every branch that would use those
    helpers, so the module is pre-seeded with stand-ins to keep this test off that chain.
    """
    if "agent.conversation_loop" not in sys.modules:
        stub = ModuleType("agent.conversation_loop")
        stub._arm_fallback_restart = lambda *a, **k: None
        stub._is_copilot_provider = lambda a: False
        stub._is_stale_copilot_credential_error = lambda *a, **k: False
        # _ra() hands back a router whose pool check the no-op agent always declines.
        stub._pool_may_recover_from_rate_limit = lambda *a, **k: False
        stub._ra = lambda: stub
        with patch.dict(sys.modules, {"agent.conversation_loop": stub}):
            return _route_call(agent, err, retry_count, max_retries, provider)
    return _route_call(agent, err, retry_count, max_retries, provider)


def _route_call(agent, err, retry_count, max_retries, provider):
    from agent.turn_recovery import route_classified_error

    classified = classify_api_error(err, provider=provider)
    return route_classified_error(
        agent, err, classified, TurnRetryState(),
        error_msg=str(err), error_context={}, recovered_with_pool=False,
        base_url="https://api.example.com/v1", model="gpt-x",
        messages=[{"role": "user", "content": "hi"}], api_messages=None,
        system_message=None, active_system_prompt="", conversation_history=None,
        retry_count=retry_count, max_retries=max_retries,
        compression_attempts=0, max_compression_attempts=3, api_call_count=1,
        effective_task_id="t-test",
    )


def _overload():
    """Ark's wording for transient capacity, which classifies via the substring pattern."""
    err = _Err(
        "Error code: 503 - ServerOverloaded", status_code=503,
        body={"error": {"message": "Error code: 503 - ServerOverloaded", "type": "server_error"}},
    )
    verdict = classify_api_error(err, provider="openai")
    assert verdict.reason is FailoverReason.overloaded, "premise: this wording must classify as overloaded"
    assert verdict.should_fallback is False, (
        "premise: the classifier says do NOT fall back for an overload — the retry threshold "
        "must not overrule that before the budget is spent"
    )
    return err


def _timeout():
    """A read timeout arrives as a message, with no error-type body to steer the classifier."""
    err = _Err("Request timed out", status_code=None,
               body={"error": {"message": "Request timed out"}})
    assert classify_api_error(err, provider="openai").reason is FailoverReason.timeout
    return err


def test_overload_does_not_fall_back_before_the_budget_is_spent():
    """The regression this fix prevents: at the 2nd retry an overload must keep retrying."""
    for retry_count in (1, 2):
        agent = _Agent()
        _route(agent, _overload(), retry_count=retry_count, max_retries=3)
        assert agent.activated == [], (
            f"an overload fell back at retry_count={retry_count}/3 — the classifier's "
            "should_fallback=False says it should keep retrying until the budget is spent"
        )


def test_overload_falls_back_at_the_budget():
    agent = _Agent()
    _route(agent, _overload(), retry_count=3, max_retries=3)
    assert agent.activated == [FailoverReason.overloaded], (
        "an exhausted overload budget must still reach the fallback"
    )


def test_timeout_keeps_the_short_threshold():
    """The asymmetry is the point of the fix: a slow provider is not a busy one, so a timeout
    still falls back after 2 retries rather than waiting out the whole budget."""
    agent = _Agent()
    _route(agent, _timeout(), retry_count=2, max_retries=6)
    assert agent.activated == [FailoverReason.timeout], (
        "a timeout must keep falling back at the short threshold even when the budget is larger"
    )

    agent = _Agent()
    _route(agent, _timeout(), retry_count=1, max_retries=6)
    assert agent.activated == [], "a timeout still retries before its short threshold"


@pytest.mark.parametrize("budget", [3, 4, 6])
def test_overload_threshold_tracks_the_configured_budget(budget):
    """`max_retries` comes from `agent.api_max_retries`, so the overload threshold moves with
    it. The pre-fix code pinned every transport reason to the constant 2 instead."""
    agent = _Agent()
    _route(agent, _overload(), retry_count=budget - 1, max_retries=budget)
    assert agent.activated == [], f"an overload fell back early with max_retries={budget}"

    agent = _Agent()
    _route(agent, _overload(), retry_count=budget, max_retries=budget)
    assert agent.activated == [FailoverReason.overloaded], (
        f"an overload must fall back once max_retries={budget} is spent"
    )
