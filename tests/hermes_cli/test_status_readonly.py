"""Status must not bootstrap the session writer it merely observes."""

import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("legacy", [False, True])
def test_session_status_does_not_create_database(tmp_path, monkeypatch, capsys, legacy):
    from hermes_cli import status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status, "get_hermes_home", lambda: tmp_path)
    if legacy:
        directory = tmp_path / "sessions"
        directory.mkdir()
        (directory / "sessions.json").write_text(json.dumps({"fixture": {}}))
    status._render_sessions(SimpleNamespace(config={}))
    output = capsys.readouterr().out
    assert "1 session(s)" in output if legacy else "Active:" in output
    assert not (tmp_path / "state.db").exists()
