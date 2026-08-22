"""Unit tests for the native ``watch`` tool helpers (#56694).

These cover the pure, side-effect-free helpers in ``tools/watch_tool.py``
so the condition language, duration parsing, and tick planning can be
verified without booting the full agent.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.watch_tool import (
    WATCH_SCHEMA,
    _eval_condition,
    _parse_duration,
    _plan_ticks,
    _should_notify,
    run_once,
    stop_all_watches,
)


def test_eval_condition_contains():
    assert _eval_condition('contains "down"', "service is down") is True
    assert _eval_condition('contains "down"', "service is up") is False


def test_eval_condition_not_contains():
    assert _eval_condition('not contains "down"', "service is up") is True
    assert _eval_condition('not contains "down"', "service is down") is False


def test_eval_condition_equals_strips():
    assert _eval_condition('equals "x"', " x ") is True
    assert _eval_condition('equals "x"', "y") is False


def test_eval_condition_matches():
    assert _eval_condition('matches "[0-9]+"', "id=42") is True
    assert _eval_condition('matches "[0-9]+"', "no digits") is False
    # invalid regex must not raise; treated as no-match
    assert _eval_condition('matches "([0-9"', "anything") is False


def test_eval_condition_bare_substring():
    assert _eval_condition('down', "service is down") is True
    assert _eval_condition('"down"', "service is down") is True
    assert _eval_condition('down', "up") is False


def test_eval_condition_empty_is_unconditional():
    assert _eval_condition("", "anything") is True
    assert _eval_condition("   ", "anything") is True


def test_parse_duration_units():
    assert _parse_duration("24h") == 86400
    assert _parse_duration("30m") == 1800
    assert _parse_duration("45s") == 45
    assert _parse_duration("120") == 120


def test_parse_duration_numeric():
    assert _parse_duration(90) == 90
    assert _parse_duration(120.0) == 120


def test_parse_duration_invalid():
    assert _parse_duration("garbage") == 0
    assert _parse_duration("10x") == 0
    assert _parse_duration(None) == 0
    assert _parse_duration("") == 0


def test_plan_ticks_basic():
    assert _plan_ticks(60, 3600) == 60
    assert _plan_ticks(60, 0) == 1
    assert _plan_ticks(60, None) == 1
    assert _plan_ticks(60, -1) == 1
    # rounding up for remainder
    assert _plan_ticks(10, 25) == 3


def test_run_once_echo():
    out = run_once("echo hello-watch-test", timeout=10)
    assert "hello-watch-test" in out


def test_run_once_timeout_safe():
    out = run_once('python -c "import time; time.sleep(5)"', timeout=1)
    assert "timeout after 1s" in out


def test_should_notify_rising_edge_and_unconditional_once():
    assert _should_notify(condition='contains "x"', triggered=True, prev_triggered=False, tick=2) is True
    assert _should_notify(condition='contains "x"', triggered=True, prev_triggered=True, tick=3) is False
    assert _should_notify(condition="", triggered=True, prev_triggered=False, tick=1) is True
    assert _should_notify(condition="", triggered=True, prev_triggered=True, tick=2) is False


def test_dispatch_end_to_end_registers_and_runs():
    """Registry path: handler must be invoked via registry.dispatch (args dict,
    async bridge, JSON-string result). Mirrors how the real agent calls tools."""
    import json as _json
    import tools.watch_tool  # noqa: F401  (self-register side effect)
    from tools.registry import registry

    assert "watch" in registry.get_all_tool_names()
    res = registry.dispatch("watch", {"command": "echo integrated", "interval": 5})
    data = _json.loads(res)
    assert data["status"] == "completed"
    assert data["observations"][0]["output"].strip() == "integrated"
    assert data["observations"][0]["triggered"] is True


def test_dispatch_condition_match_and_no_match():
    import json as _json
    import tools.watch_tool  # noqa: F401
    from tools.registry import registry

    hit = _json.loads(registry.dispatch("watch", {
        "command": "echo service is down", "condition": 'contains "down"', "interval": 5
    }))
    assert hit["observations"][0]["triggered"] is True

    miss = _json.loads(registry.dispatch("watch", {
        "command": "echo all good", "condition": 'contains "down"', "interval": 5
    }))
    assert miss["observations"][0]["triggered"] is False


def test_background_poll_survives_dispatch_without_agent_loop():
    """Production agents have no ``_loop``. Polling must live on a worker
    thread that keeps ticking after ``registry.dispatch`` returns — the
    gateway ``_run_async`` worker loop would cancel an asyncio task here.
    """
    import json as _json
    import time as _time
    import tools.watch_tool  # noqa: F401
    from tools.registry import registry

    class FakeAgent:
        def __init__(self):
            self.status_calls = []
        def _emit_status(self, msg):
            self.status_calls.append(msg)

    agent = FakeAgent()
    hold = registry.dispatch(
        "watch",
        {
            "command": 'python -c "import time; time.sleep(2); print(\'hold\')"',
            "interval": 5,
            "duration": "30s",
            "condition": "NEVER",
        },
        agent=agent,
    )
    hold_data = _json.loads(hold)
    assert hold_data["status"] == "running"
    assert "_task" not in hold_data
    hold_session = agent._watch_sessions[hold_data["watch_id"]]
    assert hold_session["_task"].is_alive(), "watch worker must outlive dispatch"
    stop_all_watches(agent)

    res = registry.dispatch(
        "watch",
        {"command": "echo STATUS_UP", "interval": 5, "condition": "UP", "duration": "30s"},
        agent=agent,
    )
    data = _json.loads(res)
    assert data["status"] == "running"
    assert "_task" not in data
    session = next(iter(agent._watch_sessions.values()))
    worker = session.get("_task")
    assert worker is not None
    assert worker.is_alive(), "watch worker must outlive dispatch"
    deadline = _time.time() + 3
    while _time.time() < deadline and session.get("status") == "running":
        _time.sleep(0.05)
    assert session["status"] == "matched"
    assert session["observations"]
    assert agent.status_calls, "match must go through agent._emit_status"
    stop_all_watches(agent)


def test_watch_list_and_stop_lifecycle():
    import json as _json
    import tools.watch_tool  # noqa: F401
    from tools.registry import registry

    class FakeAgent:
        pass

    agent = FakeAgent()
    registry.dispatch("watch", {"command": "echo a", "interval": 5, "duration": "60s"}, agent=agent)
    registry.dispatch("watch", {"command": "echo b", "interval": 5, "duration": "60s"}, agent=agent)
    listed = _json.loads(registry.dispatch("watch", {"action": "list"}, agent=agent))
    assert listed["count"] == 2
    wid = listed["watches"][0]["watch_id"]
    stopped = _json.loads(
        registry.dispatch("watch", {"action": "stop", "watch_id": wid}, agent=agent)
    )
    assert wid in stopped["stopped"]
    listed2 = _json.loads(registry.dispatch("watch", {"action": "list"}, agent=agent))
    assert listed2["count"] == 1
    registry.dispatch("watch", {"action": "stop", "watch_id": "all"}, agent=agent)
    listed3 = _json.loads(registry.dispatch("watch", {"action": "list"}, agent=agent))
    assert listed3["count"] == 0


def test_watch_interval_parses_duration_strings():
    import json as _json
    import tools.watch_tool  # noqa: F401
    from tools.registry import registry

    res = _json.loads(
        registry.dispatch("watch", {"command": "echo ok", "interval": "30s", "timeout": "oops"})
    )
    assert res["interval"] == 30
    assert res["observations"][0]["output"].strip() == "ok"


def test_watch_schema_does_not_require_command():
    required = WATCH_SCHEMA["function"]["parameters"].get("required") or []
    assert "command" not in required
    import json as _json
    import tools.watch_tool  # noqa: F401
    from tools.registry import registry

    listed = _json.loads(registry.dispatch("watch", {"action": "list"}))
    assert listed["action"] == "list"
    assert listed["count"] == 0
