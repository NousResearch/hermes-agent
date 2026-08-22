"""CLI session-source ContextVar binding regressions."""

import json
from types import SimpleNamespace


def test_bind_cli_session_source_sets_task_local_surface(monkeypatch):
    from gateway.session_context import (
        clear_session_vars,
        get_session_var,
        reset_session_vars,
    )
    from hermes_cli.main import _bind_cli_session_source

    reset_session_vars()
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "stale-environment-value")
    tokens = _bind_cli_session_source(SimpleNamespace(source="cli"))
    try:
        assert get_session_var("HERMES_SESSION_SOURCE") == "cli"
    finally:
        clear_session_vars(tokens)


def test_oneshot_stages_skill_write_with_task_local_cli_surface(
    tmp_path, monkeypatch
):
    from gateway.session_context import get_session_var, reset_session_vars
    from hermes_cli import oneshot
    from tools import skill_manager_tool as sm
    from tools import write_approval as wa

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sm, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(
        wa,
        "write_approval_enabled",
        lambda subsystem: subsystem == wa.SKILLS,
    )
    reset_session_vars()
    captured = {}

    def _run_agent(*_args, **_kwargs):
        captured["surface"] = get_session_var("HERMES_SESSION_SOURCE")
        captured["result"] = json.loads(
            sm.skill_manage(
                action="create",
                name="oneshot-demo",
                content="---\nname: oneshot-demo\ndescription: demo\n---\n\n# Demo\n",
                session_id="session-oneshot",
                tool_call_id="call-oneshot",
            )
        )
        return "done", {"completed": True, "failed": False}

    monkeypatch.setattr(oneshot, "_run_agent", _run_agent)

    assert oneshot.run_oneshot("create the demo skill") == 0
    assert captured["surface"] == "cli"
    assert get_session_var("HERMES_SESSION_SOURCE") == ""
    assert captured["result"]["success"] is True
    assert captured["result"]["staged"] is True
    record = wa.get_pending(wa.SKILLS, captured["result"]["pending_id"])
    assert record is not None
    assert record["session_context"] == {
        "profile": "default",
        "session_id": "session-oneshot",
        "surface": "cli",
        "tool_call_id": "call-oneshot",
    }
