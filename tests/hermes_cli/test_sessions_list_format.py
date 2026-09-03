import json
from types import SimpleNamespace

import hermes_state
from hermes_cli import sessions_cmd


def _seed_session(
    db_path,
    *,
    session_id="session-alpha",
    workspace="alpha",
    preview="hello world",
):
    with hermes_state.SessionDB(db_path=db_path) as db:
        db.create_session(
            session_id,
            source="cli",
            model="test-model",
            git_repo_root=f"/work/{workspace}",
            system_prompt="private prompt content must not appear in list output",
        )
        db.set_session_title(session_id, f"Project {workspace.title()}")
        db.append_message(session_id, role="user", content=preview)


def _run_list(monkeypatch, capsys, db_path, **overrides):
    session_db_type = hermes_state.SessionDB
    monkeypatch.setattr(
        hermes_state,
        "SessionDB",
        lambda: session_db_type(db_path=db_path),
    )
    args = SimpleNamespace(
        sessions_action="list",
        source=None,
        limit=20,
        workspace=None,
        format="table",
    )
    for name, value in overrides.items():
        setattr(args, name, value)

    sessions_cmd.cmd_sessions(args)
    return capsys.readouterr().out


def test_sessions_list_json_is_stable_filtered_and_non_sensitive(
    tmp_path, monkeypatch, capsys
):
    db_path = tmp_path / "state.db"
    _seed_session(db_path, workspace="alpha")
    _seed_session(db_path, session_id="session-beta", workspace="beta")

    output = _run_list(
        monkeypatch,
        capsys,
        db_path,
        workspace="alpha",
        format="json",
    )

    data = json.loads(output)
    assert len(data) == 1
    record = data[0]
    assert set(record) == {"id", "title", "preview", "last_active", "source"}
    assert record == {
        "id": "session-alpha",
        "title": "Project Alpha",
        "preview": "hello world",
        "last_active": record["last_active"],
        "source": "cli",
    }
    assert isinstance(record["last_active"], (int, float))
    assert "system_prompt" not in output
    assert "private prompt" not in output
    assert output.endswith("\n")


def test_sessions_list_json_empty_filter_is_valid_array(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "state.db"
    _seed_session(db_path)

    output = _run_list(
        monkeypatch,
        capsys,
        db_path,
        workspace="missing",
        format="json",
    )

    assert json.loads(output) == []
    assert output == "[]\n"


def test_sessions_list_tsv_has_stable_columns_and_single_line_cells(
    tmp_path, monkeypatch, capsys
):
    db_path = tmp_path / "state.db"
    _seed_session(db_path, preview="hello\tworld\nagain")

    output = _run_list(monkeypatch, capsys, db_path, format="tsv")
    lines = output.splitlines()

    assert lines[0] == "id\ttitle\tpreview\tlast_active\tsource"
    assert len(lines) == 2
    assert lines[1].startswith("session-alpha\tProject Alpha\thello world again\t")
    assert lines[1].endswith("\tcli")


def test_sessions_list_tsv_empty_filter_emits_header_only(
    tmp_path, monkeypatch, capsys
):
    db_path = tmp_path / "state.db"
    _seed_session(db_path)

    output = _run_list(
        monkeypatch,
        capsys,
        db_path,
        workspace="missing",
        format="tsv",
    )

    assert output == "id\ttitle\tpreview\tlast_active\tsource\n"


def test_sessions_list_table_preserves_workspace_columns(
    tmp_path, monkeypatch, capsys
):
    db_path = tmp_path / "state.db"
    _seed_session(db_path)
    monkeypatch.setattr(sessions_cmd, "_relative_time", lambda _timestamp: "2h ago")

    output = _run_list(monkeypatch, capsys, db_path, format="table")

    assert output == (
        f"{'Title':<28} {'Workspace':<18} {'Last Active':<13} {'ID'}\n"
        + "─" * 110
        + "\n"
        + f"{'Project Alpha':<28} {'alpha':<18} {'2h ago':<13} session-alpha\n"
    )
