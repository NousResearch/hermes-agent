from __future__ import annotations

import pytest

from hermes_cli import plugins, projects_db
from hermes_cli.plugins_manifest import PluginManifest
from hermes_state import SessionDB


@pytest.fixture(autouse=True)
def reset_plugins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugins._reset_plugin_managers_for_tests()
    yield
    plugins._reset_plugin_managers_for_tests()


def _context(tmp_path):
    manager = plugins.get_plugin_manager()
    manager._discovered = True
    return plugins.PluginContext(PluginManifest(name="reader"), manager)


def test_plugin_project_affinity_surface_is_read_only_and_bound(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with projects_db.connect_closing(db_path=tmp_path / "projects.db") as conn:
        project_id = projects_db.create_project(conn, name="Project", primary_path=str(root))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="test", cwd=str(root))
        db.update_session_project_affinity(
            "s1", project_id=project_id, project_root=str(root.resolve()),
            project_context_hash="sha256:context",
        )
    finally:
        db.close()

    snapshot = _context(tmp_path).get_session_project_affinity("s1")

    assert snapshot["status"] == "bound"
    assert snapshot["project_id"] == project_id
    assert snapshot["project_root"] == str(root.resolve())
    with pytest.raises(TypeError):
        snapshot["project_id"] = "replacement"


def test_plugin_project_affinity_surface_reports_partial_and_stale(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with projects_db.connect_closing(db_path=tmp_path / "projects.db"):
        pass
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("partial", source="test", cwd=str(root))
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET project_id = ?, project_affinity_generation = 4 WHERE id = ?",
            ("partial-project", "partial"),
        ))
        db.create_session("stale", source="test", cwd=str(root))
        db.update_session_project_affinity(
            "stale", project_id="missing-project", project_root=str(root.resolve()),
            project_context_hash="sha256:stale",
        )
    finally:
        db.close()

    context = _context(tmp_path)
    partial = context.get_session_project_affinity("partial")
    stale = context.get_session_project_affinity("stale")

    assert partial["status"] == "partial"
    assert partial["project_generation"] == 4
    assert stale["status"] == "stale"
