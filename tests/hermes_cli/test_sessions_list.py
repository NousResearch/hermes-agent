import sys

import pytest


@pytest.mark.parametrize(
    ("argv_tail", "session_fields", "workspace_label"),
    [
        ([], {}, None),
        (
            ["--workspace", "project"],
            {"git_repo_root": "/work/project", "cwd": "/work/project/src"},
            "project",
        ),
    ],
)
@pytest.mark.parametrize("has_title", [True, False])
def test_sessions_list_layout_keeps_source(
    monkeypatch, capsys, tmp_path, argv_tail, session_fields, workspace_label, has_title
):
    import hermes_cli.main as main_mod
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with hermes_state.SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("listed-session", source="telegram", **session_fields)
        db.append_message("listed-session", role="user", content="hello from telegram")
        if has_title:
            db.set_session_title("listed-session", "Named chat")
        db.create_session("hidden-tool-session", source="tool", **session_fields)
        db.append_message("hidden-tool-session", role="user", content="hidden tool preview")
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "list", *argv_tail])

    main_mod.main()

    output = capsys.readouterr().out
    lines = output.splitlines()
    assert lines, "expected sessions list output"
    header = lines[0]
    assert ("Title" in header) is has_title
    assert "Src" in header
    assert ("Named chat" in output) is has_title
    assert "telegram" in output
    assert "listed-session" in output
    assert "hidden-tool-session" not in output
    if not workspace_label or not has_title:
        assert "hello from telegram" in output
    if has_title:
        assert len(lines[1]) == len(header)
    if workspace_label:
        assert "Workspace" in header
        assert workspace_label in output


def test_sessions_list_empty_result_closes_db(monkeypatch, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    closed = {"value": False}

    class FakeDB:
        def list_sessions_rich(self, **kwargs):
            return []

        def close(self):
            closed["value"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda **_kwargs: FakeDB())
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "list"])

    main_mod.main()

    assert "No sessions found." in capsys.readouterr().out
    assert closed["value"] is True
