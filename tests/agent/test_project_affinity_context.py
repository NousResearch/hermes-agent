from __future__ import annotations

from agent.project_affinity import (
    collect_turn_project_affinity,
    load_project_affinity_candidate,
    render_project_affinity_context,
    resolve_project_affinity_for_cwd,
)
from hermes_cli import projects_db
from hermes_state import SessionDB


def test_explicit_candidate_hashes_exact_project_context(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENTS.md").write_text("PROJECT-RULE\n", encoding="utf-8")

    candidate = load_project_affinity_candidate(project_id="p-1", project_root=str(root))

    assert candidate is not None
    assert candidate.project_id == "p-1"
    assert candidate.project_root == str(root.resolve())
    assert "PROJECT-RULE" in candidate.context
    assert candidate.context_hash.startswith("sha256:")
    assert candidate == load_project_affinity_candidate(project_id="p-1", project_root=str(root))


def test_cwd_resolution_uses_innermost_project(tmp_path):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    leaf = inner / "src"
    leaf.mkdir(parents=True)
    (inner / "AGENTS.md").write_text("INNER-RULE\n", encoding="utf-8")
    conn = projects_db.connect(db_path=tmp_path / "projects.db")
    try:
        projects_db.create_project(conn, name="Outer", primary_path=str(outer))
        inner_id = projects_db.create_project(conn, name="Inner", primary_path=str(inner))

        candidate = resolve_project_affinity_for_cwd(str(leaf), projects_conn=conn)

        assert candidate is not None
        assert candidate.project_id == inner_id
        assert candidate.project_root == str(inner.resolve())
        assert "INNER-RULE" in candidate.context
    finally:
        conn.close()


class _Agent:
    def __init__(self, db, session_id):
        self._session_db = db
        self.session_id = session_id
        self._persist_disabled = False
        self.skip_context_files = False
        self.context_compressor = type("Compressor", (), {"context_length": 0})()


def test_new_session_auto_binds_once_and_replays_from_sidecar(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENTS.md").write_text("AUTO-RULE\n", encoding="utf-8")
    with projects_db.connect_closing(db_path=tmp_path / "projects.db") as conn:
        project_id = projects_db.create_project(conn, name="Project", primary_path=str(root))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("new", source="test", cwd=str(root))
        agent = _Agent(db, "new")

        first = collect_turn_project_affinity(agent, messages=[], active_system_prompt="SYSTEM")
        row = db.get_session("new")
        replay = [{"role": "user", "content": "hello", "api_content": "hello\n\n" + first}]
        second = collect_turn_project_affinity(agent, messages=replay, active_system_prompt="SYSTEM")

        assert "AUTO-RULE" in first
        assert row["project_id"] == project_id
        assert row["project_root"] == str(root.resolve())
        assert row["project_affinity_generation"] == 1
        assert second == ""
    finally:
        db.close()


def test_existing_unowned_session_does_not_auto_bind_from_cwd(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENTS.md").write_text("MUST-NOT-BIND\n", encoding="utf-8")
    with projects_db.connect_closing(db_path=tmp_path / "projects.db") as conn:
        projects_db.create_project(conn, name="Project", primary_path=str(root))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("existing", source="test", cwd=str(root))
        db.append_message("existing", role="user", content="older turn")

        context = collect_turn_project_affinity(
            _Agent(db, "existing"), messages=[], active_system_prompt="SYSTEM",
        )

        row = db.get_session("existing")
        assert context == ""
        assert row["project_id"] is None
        assert row["project_affinity_generation"] == 0
    finally:
        db.close()


def test_parent_linked_empty_session_never_auto_binds_from_inherited_cwd(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENTS.md").write_text("MUST-NOT-REHOME\n", encoding="utf-8")
    with projects_db.connect_closing(db_path=tmp_path / "projects.db") as conn:
        projects_db.create_project(conn, name="Project", primary_path=str(root))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="test", cwd=str(root))
        db.create_session("child", source="test", parent_session_id="parent")

        context = collect_turn_project_affinity(
            _Agent(db, "child"), messages=[], active_system_prompt="SYSTEM",
        )

        child = db.get_session("child")
        assert child["message_count"] == 0
        assert child["parent_session_id"] == "parent"
        assert context == ""
        assert child["project_id"] is None
        assert child["project_affinity_generation"] == 0
    finally:
        db.close()


def test_compression_child_resume_keeps_full_affinity_and_can_reinject(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENTS.md").write_text("RESUMED-PROJECT-RULE\n", encoding="utf-8")
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="test", cwd=str(root))
        candidate = load_project_affinity_candidate(project_id="project-1", project_root=str(root))
        assert candidate is not None
        db.update_session_project_affinity(
            "parent", project_id=candidate.project_id, project_root=candidate.project_root,
            project_context_hash=candidate.context_hash,
        )
        db.publish_compression_child(
            parent_session_id="parent", child_session_id="child", source="test",
            messages=[{"role": "user", "content": "compacted handoff"}],
            require_compression_lease=False,
        )

        child = db.get_session("child")
        context = collect_turn_project_affinity(
            _Agent(db, "child"),
            messages=[{"role": "user", "content": "compacted handoff"}],
            active_system_prompt="SYSTEM",
        )

        assert child["project_id"] == candidate.project_id
        assert child["project_root"] == candidate.project_root
        assert child["project_context_hash"] == candidate.context_hash
        assert "RESUMED-PROJECT-RULE" in context
    finally:
        db.close()


def test_skip_context_files_disables_affinity_reads_and_binding(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENTS.md").write_text("MUST-NOT-LOAD\n", encoding="utf-8")
    with projects_db.connect_closing(db_path=tmp_path / "projects.db") as conn:
        projects_db.create_project(conn, name="Project", primary_path=str(root))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("skipped", source="test", cwd=str(root))
        agent = _Agent(db, "skipped")
        agent.skip_context_files = True

        context = collect_turn_project_affinity(agent, messages=[], active_system_prompt="SYSTEM")

        assert context == ""
        assert db.get_session("skipped")["project_id"] is None
    finally:
        db.close()


def test_empty_project_context_emits_a_tombstone_for_old_cached_rules(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    candidate = load_project_affinity_candidate(project_id="p-1", project_root=str(root))

    rendered = render_project_affinity_context(candidate, 2)

    assert candidate is not None
    assert "supersedes any older Project Context" in rendered
    assert "No project context files" in rendered
    assert candidate.context_hash in rendered
