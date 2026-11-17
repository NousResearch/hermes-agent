"""The codex app-server runtime consumes the plan windows codex pushes (``account/rateLimits/updated``).

A codex app-server login lives in codex's own home, so Hermes' usage API fetch has no credential for
it: ``/usage`` and the picker gauge showed nothing, and a usage-limit failure named only codex's bare
"try again at 7:33 AM". Payload shapes are the codex-cli 0.147 app-server protocol schema
(``AccountRateLimitsUpdatedNotification``, ``TurnError.codexErrorInfo``), as the real binary emits them.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import account_usage, account_usage_cache
from agent.codex_runtime import make_codex_app_server_event_bridge, run_codex_app_server_turn

LIMIT_MESSAGE = ("You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage to purchase "
                 "more credits or try again at 7:33 AM.")


def _rate_limits(primary_used: int, primary_reset: int, secondary_used: int, secondary_reset: int) -> dict:
    return {"method": "account/rateLimits/updated", "params": {"rateLimits": {
        "limitId": "codex", "limitName": None,
        "primary": {"usedPercent": primary_used, "windowDurationMins": 300, "resetsAt": primary_reset},
        "secondary": {"usedPercent": secondary_used, "windowDurationMins": 10080, "resetsAt": secondary_reset},
        "credits": None, "individualLimit": None, "spendControlReached": None, "planType": None,
        "rateLimitReachedType": None,
    }}}


@pytest.fixture(autouse=True)
def _empty_usage_cache(monkeypatch):
    monkeypatch.setattr(account_usage_cache, "_snapshots", {})
    monkeypatch.setattr(account_usage_cache, "_started", {})


def _codex_agent(notes: list[dict], turn_error):
    """A seeded codex session whose turn streams ``notes`` through the real event bridge."""
    agent = MagicMock()
    agent._codex_session_prompt = None
    agent.tool_progress_callback = None
    agent._iters_since_skill = agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent._hold_stream = False
    on_event = make_codex_app_server_event_bridge(agent)

    def run_turn(**_kw):
        for note in notes:
            on_event(note)
        return SimpleNamespace(interrupted=False, error=turn_error, thread_id="th", turn_id="tu", tool_iterations=0,
                               projected_messages=[], final_text="", should_retire=False)

    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.side_effect = run_turn
    return agent


def test_pushed_windows_reach_usage_when_the_usage_api_has_no_login(monkeypatch):
    def no_hermes_login(_base_url, _api_key):
        raise RuntimeError("No Codex credentials stored")

    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "openai-codex", no_hermes_login)
    now = int(time.time())
    on_event = make_codex_app_server_event_bridge(SimpleNamespace())
    on_event(_rate_limits(37, now + 4 * 3600, 12, now + 5 * 86400))

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert [(w.label, w.used_percent, int(w.reset_at.timestamp())) for w in snapshot.windows] == [
        ("Session", 37.0, now + 4 * 3600), ("Weekly", 12.0, now + 5 * 86400)]
    lines = account_usage.account_limit_lines("openai-codex")
    assert any(line.startswith("Session: 63% remaining (37% used)") for line in lines)


def test_usage_limit_failure_names_the_reset_and_hands_it_to_the_cooldown():
    now = int(time.time())
    reset = now + 2 * 3600 + 17 * 60
    error = {"method": "error", "params": {
        "error": {"message": LIMIT_MESSAGE, "codexErrorInfo": "usageLimitExceeded", "additionalDetails": None},
        "willRetry": False, "threadId": "th", "turnId": "tu"}}
    agent = _codex_agent([_rate_limits(100, reset, 64, now + 3 * 86400), error],
                         turn_error=f"turn ended status=failed: {LIMIT_MESSAGE}")

    result = run_codex_app_server_turn(agent, user_message="hi", original_user_message="hi",
                                       messages=[{"role": "user", "content": "hi"}], effective_task_id="t")

    assert result["error"].startswith("Codex usage limit reached — it resets in 2h 1")
    assert LIMIT_MESSAGE in result["error"]
    assert result["rate_limit_reset_at"] == float(reset)
