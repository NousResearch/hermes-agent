"""Machine output shares list selection without table decoration or internal metadata."""

import argparse
import json

import pytest

from hermes_cli import sessions_cmd
from hermes_cli.subcommands.sessions import build_sessions_parser
from hermes_state import SessionDB


def _run_list(capsys, *options):
    parser = argparse.ArgumentParser()
    build_sessions_parser(parser.add_subparsers(), cmd_sessions=sessions_cmd.cmd_sessions)
    args = parser.parse_args(["sessions", "list", *options])
    assert sessions_cmd.cmd_sessions(args) in (None, 0)
    return capsys.readouterr().out


def _seed(db, session_id, *, source="cli", workspace=None, title=None):
    db.create_session(
        session_id, source=source, git_repo_root=workspace,
        system_prompt="internal prompt sentinel", model="test-model",
    )
    if title:
        db.set_session_title(session_id, title)
    db.append_message(session_id, "user", "hello\tworld\nagain\r終")


@pytest.mark.parametrize("output_format", ["json", "tsv"])
def test_machine_output_selection_schema_and_empty_store(tmp_path, monkeypatch, capsys, output_format):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    header = "id\ttitle\tpreview\tlast_active\tsource\n"
    empty = "[]\n" if output_format == "json" else header
    assert _run_list(capsys, "--format", output_format) == empty
    assert not (tmp_path / "state.db").exists()

    with SessionDB() as db:
        assert _run_list(capsys, "--format", output_format) == empty
        _seed(db, "session-alpha", workspace="/work/alpha", title="Project Alpha")
        _seed(db, "session-beta", workspace="/work/beta")
        _seed(db, "session-tool", source="tool")
        for options, expected_ids in [
            ([], {"session-alpha", "session-beta"}),
            (["--workspace", "alpha"], {"session-alpha"}),
            (["--workspace", "missing"], set()),
            (["--source", "tool"], {"session-tool"}),
            (["--limit", "0"], set()),
            (["--limit", "1"], None),
        ]:
            output = _run_list(capsys, "--format", output_format, *options)
            if output_format == "json":
                records = json.loads(output)
                for record in records:
                    assert set(record) == {"id", "title", "preview", "last_active", "source"}
                    assert record["preview"] == "hello\tworld again 終"
                    assert isinstance(record["last_active"], (int, float))
                    assert record["title"] == ("Project Alpha" if record["id"] == "session-alpha" else None)
                ids = {record["id"] for record in records}
            else:
                assert output.startswith(header)
                rows = [line.split("\t") for line in output.splitlines()[1:]]
                for row in rows:
                    assert len(row) == 5
                    assert row[2] == "hello world again 終"
                    assert row[1] == ("Project Alpha" if row[0] == "session-alpha" else "")
                ids = {row[0] for row in rows}
            assert len(ids) == 1 if expected_ids is None else ids == expected_ids
            assert "internal prompt sentinel" not in output
            assert "system_prompt" not in output
            assert "/work/" not in output
            assert "more not shown" not in output
            assert output.endswith("\n")


@pytest.mark.parametrize("workspace", [None, "/work/alpha"])
@pytest.mark.parametrize("title", [None, "Project Alpha"])
def test_table_default_matches_explicit_format(tmp_path, monkeypatch, capsys, workspace, title):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert _run_list(capsys) == _run_list(capsys, "--format", "table") == "No sessions found.\n"
    monkeypatch.setattr(sessions_cmd, "_relative_time", lambda *a, **kw: "2h ago")
    with SessionDB() as db:
        _seed(db, "session-alpha", workspace=workspace, title=title)
        _seed(db, "session-beta", workspace=workspace)
    default = _run_list(capsys, "--limit", "1")
    assert default == _run_list(capsys, "--format", "table", "--limit", "1")
    assert "--limit 2" in default
    assert ("Workspace" in default) == bool(workspace)
    assert "2h ago" in default
