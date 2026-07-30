"""``pre_goal_turn`` hook at the kanban goal-loop turn boundary.

Fired before each CONTINUATION turn (never the one-shot finalize nudge) with
the loop's state and the caller's runtime dict. A callback may return
``{"prompt": ..., "handed_off": bool}`` to rewrite the continuation prompt
and/or record a context-budget handoff to a fresh session; a raising hook
degrades to a normal in-session turn.
"""
from hermes_cli import goals
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


def test_pre_goal_turn_is_a_valid_hook():
    assert "pre_goal_turn" in VALID_HOOKS


def _run_loop(monkeypatch, verdicts, hook, max_turns=6, statuses=None):
    """Drive the loop with scripted judge verdicts and a test hook."""
    state = {"prompts": [], "judge_i": 0, "turn_i": 0}

    def fake_judge(goal_text, response):
        i = min(state["judge_i"], len(verdicts) - 1)
        state["judge_i"] += 1
        # (verdict, reason, parse_failed, wait, transport_failed)
        return verdicts[i], f"scripted verdict #{i}", False, False, False

    def fake_status():
        state["turn_i"] += 1
        if statuses:
            return statuses[min(state["turn_i"] - 1, len(statuses) - 1)]
        return "done" if state["judge_i"] >= len(verdicts) else "running"

    monkeypatch.setattr(goals, "judge_goal", fake_judge)
    mgr = get_plugin_manager()
    saved = list(mgr._hooks.get("pre_goal_turn", []))
    mgr._hooks["pre_goal_turn"] = [hook] if hook else []
    try:
        result = goals.run_kanban_goal_loop(
            task_id="t_test", goal_text="do the thing",
            run_turn=lambda p: state["prompts"].append(p) or "worked on it",
            task_status_fn=fake_status, block_fn=lambda r: None,
            max_turns=max_turns, first_response="first",
            runtime={"marker": "RUNTIME-OK"},
        )
    finally:
        mgr._hooks["pre_goal_turn"] = saved
    return result, state


def test_hook_fires_per_continuation_turn_with_state(monkeypatch):
    captured = []
    result, _state = _run_loop(
        monkeypatch, ["continue", "continue"],
        lambda **kw: captured.append(kw))

    assert result["outcome"] == "completed_by_worker"
    assert len(captured) == 2
    payload = captured[0]
    assert payload["task_id"] == "t_test"
    assert payload["goal_text"] == "do the thing"
    assert payload["progress"] == "first"
    assert payload["handoffs_done"] == 0
    assert payload["runtime"]["marker"] == "RUNTIME-OK"


def test_finalize_nudge_is_never_intercepted(monkeypatch):
    captured = []
    _run_loop(monkeypatch, ["done"], lambda **kw: captured.append(kw),
              statuses=["running", "done"])
    assert captured == []


def test_prompt_rewrite_and_handoff_accounting(monkeypatch):
    captured = []

    def handoff_hook(**kw):
        captured.append(kw)
        if len(captured) == 1:
            return {"prompt": "FRESH-SESSION PROMPT", "handed_off": True}
        return None

    _result, state = _run_loop(
        monkeypatch, ["continue", "continue"], handoff_hook)

    assert state["prompts"][0] == "FRESH-SESSION PROMPT"
    assert len(captured) == 2
    assert captured[1]["handoffs_done"] == 1


def test_raising_hook_degrades_to_in_session_turn(monkeypatch):
    def boom(**kw):
        raise RuntimeError("boom")

    result, state = _run_loop(monkeypatch, ["continue", "continue"], boom)
    assert result["outcome"] == "completed_by_worker"
    assert len(state["prompts"]) >= 1
    assert state["prompts"][0] != "FRESH-SESSION PROMPT"
