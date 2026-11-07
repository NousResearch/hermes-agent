"""The /goal judge's host-observed evidence block, read back from a real SessionDB (#117222)."""

import json

import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    from hermes_cli import goals

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()  # re-resolve the store under this HERMES_HOME
    handle = goals._get_session_db()
    assert handle is not None
    yield handle
    goals._DB_CACHE.clear()


def _calls(*specs):
    return json.dumps([{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                       for cid, name, args in specs])


def _seed_edit_turn(db, sid):
    db.ensure_session(sid)
    db.append_message(sid, "user", "fix the bug and make sure the tests pass")
    db.append_message(sid, "assistant", "", tool_calls=_calls(("e1", "patch", {"path": "src/app.py"})))
    db.append_message(sid, "tool", json.dumps({"success": True, "files_modified": ["src/app.py"]}), tool_call_id="e1")


def test_claimed_pass_with_no_check_after_the_edit_is_flagged(db):
    from hermes_cli.goals_evidence import gather_tool_activity

    _seed_edit_turn(db, "sid-unchecked")
    db.append_message("sid-unchecked", "assistant", "Fixed. All tests pass.")

    block = gather_tool_activity("sid-unchecked")
    assert "- patch path=src/app.py" in block
    assert "NOT VERIFIED: no test/build/CI command ran after the last edit (patch path=src/app.py)" in block


def test_check_in_an_earlier_turn_still_counts_and_a_failing_one_is_named(db):
    from hermes_cli.goals_evidence import gather_tool_activity

    sid = "sid-multi"
    _seed_edit_turn(db, sid)
    db.append_message(sid, "assistant", "", tool_calls=_calls(("t1", "terminal", {"command": "pytest -q"})))
    db.append_message(sid, "tool", json.dumps({"output": "1 failed", "exit_code": 1}), tool_call_id="t1")
    db.append_message(sid, "assistant", "One failure left.")
    # A response-only continuation turn: no calls of its own, but the session evidence still applies.
    db.append_message(sid, "user", "[Continuing toward your standing goal]")
    db.append_message(sid, "assistant", "All tests pass now.")

    block = gather_tool_activity(sid)
    assert "This turn:\n- (no tool calls)" in block
    assert "the last check after the edit failed (terminal command=pytest -q → exit 1)" in block

    db.append_message(sid, "user", "[Continuing toward your standing goal]")
    db.append_message(sid, "assistant", "", tool_calls=_calls(("t2", "terminal", {"command": "pytest -q"})))
    db.append_message(sid, "tool", json.dumps({"output": "3 passed", "exit_code": 0}), tool_call_id="t2")
    db.append_message(sid, "assistant", "All tests pass now.")
    block = gather_tool_activity(sid)
    assert "- terminal command=pytest -q → exit 0" in block
    assert "terminal command=pytest -q passed after the last edit" in block
    assert "NOT VERIFIED" not in block


def test_failed_write_is_shown_as_failed(db):
    from hermes_cli.goals_evidence import gather_tool_activity

    db.ensure_session("sid-fail")
    db.append_message("sid-fail", "user", "save the summary to notes.md")
    db.append_message("sid-fail", "assistant", "", tool_calls=_calls(("w1", "write_file", {"path": "notes.md"})))
    db.append_message("sid-fail", "tool", json.dumps({"error": "Permission denied"}), tool_call_id="w1")
    db.append_message("sid-fail", "assistant", "Saved notes.md.")
    assert "- write_file path=notes.md → FAILED (nothing written)" in gather_tool_activity("sid-fail")


def test_session_without_tool_calls_leaves_the_judge_prompt_unchanged(db):
    from hermes_cli.goals_evidence import gather_tool_activity

    db.ensure_session("sid-prose")
    db.append_message("sid-prose", "user", "write me a haiku")
    db.append_message("sid-prose", "assistant", "An old silent pond...")
    assert gather_tool_activity("sid-prose") == ""
    assert gather_tool_activity(None) == ""


def test_judge_prompt_carries_the_block(monkeypatch):
    from hermes_cli import goals

    seen = {}

    def fake_judge_llm(_call_llm, system_prompt, user_prompt, _timeout):
        seen["user"] = user_prompt
        return '{"verdict": "continue", "reason": "no test ran after the edit"}'

    monkeypatch.setattr(goals, "_call_goal_judge_llm", fake_judge_llm)
    goals.judge_goal("fix the bug", "All tests pass.", tool_activity="Tool activity the host observed\nNOT VERIFIED\n\n")
    assert "NOT VERIFIED" in seen["user"]
