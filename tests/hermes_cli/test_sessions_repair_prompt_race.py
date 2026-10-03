"""Repair consent applies to the scanned prompt and pin, never a live replacement."""

from types import SimpleNamespace

import pytest

from hermes_cli import sessions_cmd
from hermes_state import SessionDB


@pytest.mark.parametrize("change", ["prompt", "tools"])
def test_repair_keeps_state_changed_after_scan(tmp_path, monkeypatch, capsys, change):
    db = SessionDB(tmp_path / "state.db")
    sid = db.create_session("race", "telegram", system_prompt="degraded prompt")
    db.update_session_tool_names(sid, ["skill_manage"])
    expected = []

    def confirm(_message):
        if change == "prompt":
            db.update_system_prompt(sid, "healthy rebuilt prompt")
        else:
            db.update_session_tool_names(sid, ["memory"])
        expected.append(db.get_session(sid))
        return True

    monkeypatch.setattr(sessions_cmd, "_confirm_prompt", confirm)
    try:
        args = SimpleNamespace(session_id=None, apply=True, json=False)
        assert sessions_cmd._cmd_repair_prompts(db, args) == 0
        assert db.get_session(sid) == expected[0]
        assert "changed since" in capsys.readouterr().out
    finally:
        db.close()
