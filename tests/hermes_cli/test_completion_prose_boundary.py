"""Prose is not an artifact declaration in persistent workspaces."""
import sqlite3

import pytest

from hermes_cli.kanban_db_artifacts import _merge_completion_prose_artifacts


@pytest.mark.parametrize("kind", ["dir", "worktree"])
@pytest.mark.parametrize("filename", [".env", "report.txt"])
def test_persistent_workspace_prose_does_not_declare_artifacts(tmp_path, kind, filename):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE tasks (id TEXT, workspace_path TEXT, workspace_kind TEXT)")
    conn.execute("INSERT INTO tasks VALUES (?, ?, ?)", ("t", str(tmp_path), kind))
    target = tmp_path / filename
    target.write_text("FAKE_TEST_VALUE")
    metadata = {"review": "pending"}
    result = _merge_completion_prose_artifacts(conn, "t", metadata, summary=f"Reviewed {target}", result=None)
    assert result == metadata
    conn.close()


def _scratch_conn(workspace):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE tasks (id TEXT, workspace_path TEXT, workspace_kind TEXT)")
    conn.execute("INSERT INTO tasks VALUES (?, ?, ?)", ("t", str(workspace), "scratch"))
    return conn


@pytest.mark.parametrize("relative", [
    ".env", ".npmrc", ".ssh/config", "keys/server.pem", "id_ed25519", "out/credentials.json",
])
def test_scratch_prose_never_infers_hidden_or_credential_files(tmp_path, relative):
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("FAKE_TEST_VALUE")
    conn = _scratch_conn(tmp_path)
    metadata = {"review": "pending"}
    assert _merge_completion_prose_artifacts(conn, "t", metadata, summary=f"Wrote {target}.", result=None) == metadata
    conn.close()


def test_scratch_prose_still_promotes_ordinary_deliverable(tmp_path):
    report = tmp_path / "out" / "report.md"
    report.parent.mkdir()
    report.write_text("deliverable")
    conn = _scratch_conn(tmp_path)
    result = _merge_completion_prose_artifacts(conn, "t", None, summary=f"Report at {report}.", result=None)
    assert result == {"artifacts": [str(report.resolve())]}
    conn.close()
