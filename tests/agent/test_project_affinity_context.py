from __future__ import annotations

from agent.project_affinity import (
    ensure_agent_project_affinity,
    load_project_affinity_candidate,
    resolve_project_affinity_for_cwd,
)
from agent.system_prompt import _plugin_session_info
from hermes_cli import projects_db
from hermes_state import SessionDB


class _Agent:
    def __init__(self, db, session_id):
        self._session_db = db
        self._profile_home = db.db_path.parent
        self.session_id = session_id
        self._persist_disabled = False
        self.context_compressor = type("Compressor", (), {"context_length": 0})()
        self.model = "test-model"
        self.provider = "test-provider"
        self.platform = "test"
        self.active_profile = "default"


def _create_project(tmp_path, name="Project"):
    root = tmp_path / name.lower()
    root.mkdir()
    (root / "AGENTS.md").write_text(f"{name.upper()}-RULE\n", encoding="utf-8")
    with projects_db.connect_closing(db_path=tmp_path / "projects.db") as conn:
        project_id = projects_db.create_project(conn, name=name, primary_path=str(root))
    return project_id, root


def test_candidate_contains_only_identity_and_context_hash(tmp_path):
    project_id, root = _create_project(tmp_path)

    candidate = load_project_affinity_candidate(project_id=project_id, project_root=str(root))

    assert candidate is not None
    assert candidate.project_id == project_id
    assert candidate.project_root == str(root.resolve())
    assert candidate.context_hash.startswith("sha256:")
    assert not hasattr(candidate, "context")


def test_cwd_resolution_uses_innermost_project(tmp_path):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    leaf = inner / "src"
    leaf.mkdir(parents=True)
    (inner / "AGENTS.md").write_text("INNER-RULE\n", encoding="utf-8")
    with projects_db.connect_closing(db_path=tmp_path / "projects.db") as conn:
        projects_db.create_project(conn, name="Outer", primary_path=str(outer))
        inner_id = projects_db.create_project(conn, name="Inner", primary_path=str(inner))
        candidate = resolve_project_affinity_for_cwd(str(leaf), projects_conn=conn)

    assert candidate is not None
    assert candidate.project_id == inner_id
    assert candidate.project_root == str(inner.resolve())


def test_new_root_session_binds_at_prompt_generation_and_exposes_frozen_info(tmp_path):
    project_id, root = _create_project(tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("new", source="test", cwd=str(root))
        info = _plugin_session_info(_Agent(db, "new"))
        row = db.get_session("new")

        assert info["project_affinity_status"] == "bound"
        assert info["project_id"] == project_id
        assert info["project_root"] == str(root.resolve())
        assert info["project_generation"] == "1"
        assert info["project_context_hash"].startswith("sha256:")
        assert row["project_id"] == project_id
        assert row["project_affinity_generation"] == 1
    finally:
        db.close()


def test_existing_or_parent_linked_unowned_session_does_not_infer_owner(tmp_path):
    _, root = _create_project(tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("existing", source="test", cwd=str(root))
        db.append_message("existing", role="user", content="older turn")
        db.create_session("parent", source="test", cwd=str(root))
        db.create_session("child", source="test", parent_session_id="parent")

        existing = ensure_agent_project_affinity(_Agent(db, "existing"))
        child = ensure_agent_project_affinity(_Agent(db, "child"))

        assert existing["status"] == "unbound"
        assert child["status"] == "unbound"
        assert db.get_session("existing")["project_id"] is None
        assert db.get_session("child")["project_id"] is None
    finally:
        db.close()


def test_bound_identity_ignores_cwd_drift_and_refreshes_only_at_prompt_boundary(tmp_path):
    project_id, root = _create_project(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("bound", source="test", cwd=str(root))
        agent = _Agent(db, "bound")
        first = ensure_agent_project_affinity(agent)
        db.update_session_cwd("bound", str(other))
        (root / "AGENTS.md").write_text("CHANGED-RULE\n", encoding="utf-8")
        refreshed = ensure_agent_project_affinity(agent)
        repeated = ensure_agent_project_affinity(agent)

        assert first["project_id"] == project_id
        assert refreshed["project_id"] == project_id
        assert refreshed["project_root"] == str(root.resolve())
        assert refreshed["project_generation"] == 2
        assert repeated["project_generation"] == 2
        assert refreshed["project_context_hash"] != first["project_context_hash"]
    finally:
        db.close()


def test_partial_tuple_is_reported_and_never_auto_repaired(tmp_path):
    _, root = _create_project(tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("partial", source="test", cwd=str(root))
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET project_id = ?, project_affinity_generation = 7 WHERE id = ?",
            ("legacy-project", "partial"),
        ))

        snapshot = ensure_agent_project_affinity(_Agent(db, "partial"))

        assert snapshot["status"] == "partial"
        assert snapshot["project_id"] == "legacy-project"
        assert snapshot["project_root"] == ""
        assert db.get_session("partial")["project_affinity_generation"] == 7
    finally:
        db.close()
