import sys
import time

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
def test_sessions_list_titled_layout_shows_source(
    monkeypatch, capsys, argv_tail, session_fields, workspace_label
):
    import hermes_cli.main as main_mod
    import hermes_state

    class FakeDB:
        def list_sessions_rich(self, **kwargs):
            assert kwargs["source"] is None
            assert kwargs["exclude_sources"] == ["tool"]
            return [
                {
                    "id": "20260401_201329_d85961",
                    "source": "telegram",
                    "title": "Named chat",
                    "preview": "hello from telegram",
                    "last_active": time.time(),
                    **session_fields,
                }
            ]

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "list", *argv_tail])

    main_mod.main()

    output = capsys.readouterr().out
    lines = output.splitlines()
    assert lines, "expected sessions list output"
    header = lines[0]
    assert "Title" in header
    assert "Src" in header
    assert "Named chat" in output
    assert "telegram" in output
    assert "20260401_201329_d85961" in output
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

    monkeypatch.setattr(hermes_state, "SessionDB", lambda: FakeDB())
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "list"])

    main_mod.main()

    assert "No sessions found." in capsys.readouterr().out
    assert closed["value"] is True
