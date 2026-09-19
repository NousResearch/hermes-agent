"""Bounded post-exhaustion auto-recovery ladder (#85426, #107307): fallback first, transient
outages only, visible on every surface, interrupt cancels."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from agent.turn_recovery_autorecover import ladder_wait_seconds
from agent.turn_retry_state import TurnRetryState


class _Err(Exception):
    def __init__(self, status_code, message, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.response = SimpleNamespace(headers=headers or {})
        self.body = None


class _Agent:
    """Real seams: fallback state, the status/wait callbacks the surfaces read, the streamed-text
    gate and the interrupt flag; everything else the terminal path touches is a no-op."""
    log_prefix = ""
    verbose = False
    provider = "custom"
    platform = "cli"
    _credential_pool = None
    _current_streamed_assistant_text = ""
    _interrupt_requested = False
    sleep_outcome = None  # what the (patched) interruptible wait returns

    def __init__(self, *, fallback_left=False, cycles=5, platform="cli"):
        self.statuses, self.waits, self.activated = [], [], []
        self._fallback_left = fallback_left
        self._auto_recovery_cycles = cycles
        self.platform = platform

    def _has_pending_fallback(self):
        return self._fallback_left

    def _try_activate_fallback(self, **kwargs):
        self.activated.append(True)
        return self._fallback_left

    def _try_recover_primary_transport(self, *a, **k):
        return False

    def _has_content_after_think_block(self, text):
        return bool(text.strip())

    def _emit_diagnostic_status(self, text):
        self.statuses.append(str(text))

    def _emit_diagnostic_wait(self, text):
        self.waits.append(str(text))

    def _summarize_api_error(self, error):
        return str(error)

    def _client_log_context(self):
        return ""

    def clear_interrupt(self, **kwargs):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _settle(agent, err, retry=None):
    retry = retry or TurnRetryState(primary_recovery_attempted=True)
    classified = classify_api_error(err, provider=agent.provider)
    with patch("agent.conversation_loop._is_copilot_provider", lambda a: False), \
            patch("agent.conversation_loop._arm_fallback_restart", lambda a, m, s, r: s), \
            patch("agent.turn_recovery.interruptible_backoff_sleep", lambda agent, *a, **k: agent.sleep_outcome), \
            patch("agent.turn_api_error.max_retries_exhausted_result", lambda *a, **k: {"failed": True}), \
            patch("agent.turn_api_error.nonretryable_client_error_result", lambda *a, **k: {"failed": True, "nonretryable": True}):
        return settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=err.status_code,
            error_msg=str(err).lower(), is_context_length_error=False, is_rate_limited=False,
            _is_zai_coding_overload=False, _provider=agent.provider, _base="http://x", _model="m",
            messages=[], api_messages=[], api_kwargs=None, active_system_prompt=None,
            conversation_history=None, approx_tokens=0, retry_count=3, max_retries=3,
            compression_attempts=0, api_call_count=1,
        ), retry


@pytest.mark.parametrize("status,message", [(503, "Service temporarily unavailable"), (529, "Overloaded")])
def test_transient_exhaustion_without_fallback_enters_ladder_and_retries(status, message):
    """No fallback left + 5xx/overloaded + nothing delivered -> one visible cycle, then retry from 0."""
    agent = _Agent()
    verdict, retry = _settle(agent, _Err(status, message))
    assert (verdict.action, verdict.retry_count, retry.auto_recovery_cycles_used) == ("continue", 0, 1)
    assert agent.statuses == agent.waits and len(agent.statuses) == 1
    line = agent.statuses[0]
    assert "retrying automatically in" in line and "(cycle 1/5)" in line and "press Esc to stop" in line


def test_ladder_is_bounded_and_yields_to_fallback_and_shuns_nonretryable():
    agent = _Agent()
    spent = TurnRetryState(primary_recovery_attempted=True, auto_recovery_cycles_used=5)
    verdict, _ = _settle(agent, _Err(503, "down"), retry=spent)
    assert verdict.action == "return" and verdict.result == {"failed": True}
    assert any("gave up after 5 cycles" in s for s in agent.statuses)

    # Fallback first: a remaining fallback engages and the ladder never runs.
    fb = _Agent(fallback_left=True)
    verdict, retry = _settle(fb, _Err(503, "down"))
    assert (verdict.action, fb.activated, retry.auto_recovery_cycles_used) == ("break", [True], 0)

    # Non-retryable classes never enter (401 here) and delivered text is never replayed.
    auth = _Agent()
    verdict, retry = _settle(auth, _Err(401, "Incorrect API key provided"))
    assert verdict.result.get("nonretryable") and retry.auto_recovery_cycles_used == 0
    streamed = _Agent()
    streamed._current_streamed_assistant_text = "Here is the first half of the answer"
    verdict, retry = _settle(streamed, _Err(503, "down"))
    assert (verdict.action, retry.auto_recovery_cycles_used) == ("return", 0)

    # Disabled by config.
    off = _Agent(cycles=0)
    verdict, retry = _settle(off, _Err(503, "down"))
    assert (verdict.action, retry.auto_recovery_cycles_used, off.statuses) == ("return", 0, [])


def test_ladder_schedule_honours_retry_after_and_platform_stop_hint(monkeypatch):
    import agent.retry_utils as retry_utils
    calls = []

    def _recording_backoff(attempt, *, base_delay, max_delay, jitter_ratio):
        calls.append((attempt, base_delay, max_delay))
        return min(base_delay * 2 ** (attempt - 1), max_delay)
    monkeypatch.setattr(retry_utils, "jittered_backoff", _recording_backoff)

    plain = _Err(503, "down")
    assert [ladder_wait_seconds(c, plain) for c in (1, 2, 3, 4, 5)] == [15, 30, 60, 60, 60]
    assert calls[0] == (1, 15.0, 60.0)
    assert ladder_wait_seconds(1, _Err(529, "Overloaded", {"Retry-After": "4"})) == 4
    # Past the 60 s cap the provider's own cooldown is honoured, up to 120 s.
    assert ladder_wait_seconds(1, _Err(529, "Overloaded", {"Retry-After": "90"})) == 90
    assert ladder_wait_seconds(1, _Err(529, "Overloaded", {"Retry-After": "900"})) == 120

    tg = _Agent(platform="telegram")
    _settle(tg, plain)
    assert "send /stop to cancel" in tg.statuses[0]
    cron = _Agent(platform="cron")
    _settle(cron, plain)
    assert cron.statuses[0].endswith("(cycle 1/5)")


def test_interrupt_during_ladder_wait_returns_interrupted_result():
    agent = _Agent()
    agent.sleep_outcome = {"interrupted": True, "completed": False}
    verdict, retry = _settle(agent, _Err(503, "down"))
    assert verdict.action == "return" and verdict.result["interrupted"] is True
    assert retry.auto_recovery_cycles_used == 1
