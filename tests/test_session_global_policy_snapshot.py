"""Session-owned GLOBAL policy snapshots persist independently of prompt blobs."""

from __future__ import annotations

import sqlite3

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_SQL


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key", "test")
        self.base_url = kwargs.get("base_url", "http://test")

    def close(self):
        pass


def _new_agent(monkeypatch, db, session_id, policy):
    """Build a real AIAgent with a real SessionDB but no network/tool discovery."""
    from run_agent import AIAgent

    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **_kwargs: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id=session_id,
        session_db=db,
        global_policy_snapshot=policy,
        enabled_toolsets=["file"],
    )


# This test builds a writable database with the immediately previous sessions
# definition, rather than using SessionDB to initialize a current store first.
_LEGACY_SESSIONS_SCHEMA_SQL = SCHEMA_SQL.replace(
    "    global_policy_snapshot TEXT,\n", ""
)


def test_writable_legacy_sessions_table_is_reconciled_with_null_snapshot(tmp_path):
    """Opening a pre-snapshot store adds the nullable column without rewriting rows."""
    path = tmp_path / "legacy-state.db"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_LEGACY_SESSIONS_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("legacy-session", "cli", 1.0),
        )
        conn.commit()
    finally:
        conn.close()

    db = SessionDB(db_path=path)
    try:
        columns = {
            row["name"]
            for row in db._conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        assert "global_policy_snapshot" in columns
        assert db.get_session("legacy-session")["global_policy_snapshot"] is None
    finally:
        db.close()


def test_create_session_upsert_omitting_snapshot_preserves_stored_value(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(
            "snapshot-session",
            "cli",
            system_prompt="first prompt",
            global_policy_snapshot="stored policy",
        )

        # Upsert metadata without supplying a replacement snapshot.
        db.create_session("snapshot-session", "cli", system_prompt="second prompt")

        assert db.get_session("snapshot-session")["global_policy_snapshot"] == "stored policy"
    finally:
        db.close()


def test_global_policy_snapshot_distinguishes_value_empty_and_null_across_reopen(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db.create_session(
            "snapshot-session",
            "cli",
            system_prompt="first prompt",
            global_policy_snapshot="A",
        )
        assert db.get_session("snapshot-session")["system_prompt"] == "first prompt"
        assert db.get_session("snapshot-session")["global_policy_snapshot"] == "A"

        db.update_system_prompt(
            "snapshot-session", "second prompt", global_policy_snapshot=""
        )
        assert db.get_session("snapshot-session")["system_prompt"] == "second prompt"
        assert db.get_session("snapshot-session")["global_policy_snapshot"] == ""

        # Omission preserves the explicit empty snapshot rather than treating it
        # as an instruction to reread or clear the policy.
        db.update_system_prompt("snapshot-session", "third prompt")
        assert db.get_session("snapshot-session")["global_policy_snapshot"] == ""

        # NULL remains the legacy/uninitialized state and differs from "".
        db.update_system_prompt(
            "snapshot-session", "fourth prompt", global_policy_snapshot=None
        )
    finally:
        db.close()

    reopened = SessionDB(db_path=path)
    try:
        session = reopened.get_session("snapshot-session")
        assert session["system_prompt"] == "fourth prompt"
        assert session["global_policy_snapshot"] is None
    finally:
        reopened.close()


def test_initial_agent_session_create_publishes_frozen_policy_snapshot(monkeypatch, tmp_path):
    """The first row must atomically contain the prompt and its frozen policy."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        agent = _new_agent(monkeypatch, db, "initial-snapshot", "Policy A")
        agent._cached_system_prompt = "prompt rendered with Policy A"

        agent._ensure_db_session()

        stored = db.get_session("initial-snapshot")
        assert stored["system_prompt"] == "prompt rendered with Policy A"
        assert stored["global_policy_snapshot"] == "Policy A"
    finally:
        db.close()


def test_real_db_policy_snapshot_lifecycle_survives_resume_rebuild_and_rotation(
    monkeypatch, tmp_path
):
    """Real SessionDB rows keep policy A across every resume-derived session.

    This intentionally changes GLOBAL.md between constructions: only a new
    session may use B. Resume, legacy/missing recovery, stale model rebuild,
    rotated child reconstruction, and a delegated child must stay frozen to A.
    """
    from agent.conversation_loop import _restore_or_build_system_prompt

    home = tmp_path / "hermes-home"
    policy_file = home / "memories" / "GLOBAL.md"
    policy_file.parent.mkdir(parents=True)
    policy_file.write_text("Policy A", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=tmp_path / "state.db")
    delegated = None
    try:
        parent = _new_agent(monkeypatch, db, "parent", None)
        parent._cached_system_prompt = parent._build_system_prompt(None)
        parent._ensure_db_session()
        parent_prompt = parent._cached_system_prompt
        parent_row = db.get_session("parent")
        policy_a = parent._global_policy_snapshot
        assert "Policy A" in parent_prompt
        assert "Policy B" not in parent_prompt
        assert parent_prompt.count("GLOBAL POLICY (shared across all Hermes profiles)") == 1
        assert parent_row["system_prompt"].encode("utf-8") == parent_prompt.encode("utf-8")
        assert parent_row["global_policy_snapshot"] == policy_a

        # Disk policy mutates after the first-turn atomic publication.
        policy_file.write_text("Policy B", encoding="utf-8")
        resumed = _new_agent(monkeypatch, db, "parent", None)
        _restore_or_build_system_prompt(resumed, None, [{"role": "user", "content": "resume"}])
        assert resumed._global_policy_snapshot == policy_a
        assert resumed._cached_system_prompt == parent_prompt

        # The real delegation constructor must pass the resumed parent snapshot
        # into the normal child AIAgent prompt and its durable child row.
        from tools.delegate_tool import _build_child_agent

        monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {})
        delegated = _build_child_agent(
            task_index=0,
            goal="Verify frozen policy propagation",
            context=None,
            toolsets=["file"],
            model=None,
            max_iterations=1,
            task_count=1,
            parent_agent=resumed,
        )
        delegated._cached_system_prompt = delegated._build_system_prompt(None)
        delegated._ensure_db_session()
        delegated_prompt = delegated._cached_system_prompt
        delegated_row = db.get_session(delegated.session_id)
        assert "Policy A" in delegated_prompt and "Policy B" not in delegated_prompt
        assert delegated_prompt.count("GLOBAL POLICY (shared across all Hermes profiles)") == 1
        assert delegated_row["system_prompt"].encode("utf-8") == delegated_prompt.encode("utf-8")
        assert delegated._global_policy_snapshot == policy_a
        assert delegated_row["global_policy_snapshot"] == policy_a

        # Missing prompt recovery must rebuild from stored A and publish A.
        db.update_system_prompt("parent", None)
        missing = _new_agent(monkeypatch, db, "parent", None)
        _restore_or_build_system_prompt(missing, None, [{"role": "user", "content": "recover"}])
        assert missing._global_policy_snapshot == policy_a
        assert "Policy A" in db.get_session("parent")["system_prompt"]
        assert db.get_session("parent")["global_policy_snapshot"] == policy_a

        # A legacy row with no GLOBAL header never adopts B.
        db.create_session("legacy", "cli", system_prompt="legacy prompt")
        legacy = _new_agent(monkeypatch, db, "legacy", None)
        _restore_or_build_system_prompt(legacy, None, [{"role": "user", "content": "resume"}])
        assert legacy._global_policy_snapshot == ""

        # A genuinely new session after the disk edit does receive B.
        fresh = _new_agent(monkeypatch, db, "fresh-after-b", None)
        fresh._cached_system_prompt = fresh._build_system_prompt(None)
        fresh._ensure_db_session()
        assert "Policy B" in fresh._cached_system_prompt
        assert db.get_session("fresh-after-b")["global_policy_snapshot"] == fresh._global_policy_snapshot

        # A stale runtime/model rebuild restores A before it renders and writes.
        db.update_system_prompt("parent", parent_prompt, global_policy_snapshot=policy_a)
        rebuilt = _new_agent(monkeypatch, db, "parent", None)
        rebuilt.model = "different-model"
        _restore_or_build_system_prompt(rebuilt, None, [{"role": "user", "content": "switch"}])
        assert rebuilt._global_policy_snapshot == policy_a
        assert "Policy A" in db.get_session("parent")["system_prompt"]
        assert db.get_session("parent")["global_policy_snapshot"] == policy_a

        # Rotation persists A into the child; reconstruction and a subsequent
        # delegated child use that DB value after disk has changed to B.
        db.publish_compression_child(
            parent_session_id="parent",
            child_session_id="rotated",
            source="cli",
            model=parent.model,
            system_prompt=parent_prompt,
            global_policy_snapshot=policy_a,
            messages=[{"role": "user", "content": "compressed handoff"}],
            require_compression_lease=False,
        )
        rotated = _new_agent(monkeypatch, db, "rotated", None)
        _restore_or_build_system_prompt(rotated, None, [{"role": "user", "content": "continue"}])
        assert rotated._global_policy_snapshot == policy_a
        rotated_child = _new_agent(monkeypatch, db, "rotated-child", rotated._global_policy_snapshot)
        rotated_child_prompt = rotated_child._build_system_prompt(None)
        assert "Policy A" in rotated_child_prompt and "Policy B" not in rotated_child_prompt
        assert rotated_child_prompt.count("GLOBAL POLICY (shared across all Hermes profiles)") == 1
    finally:
        if delegated is not None:
            delegated.close()
        db.close()
