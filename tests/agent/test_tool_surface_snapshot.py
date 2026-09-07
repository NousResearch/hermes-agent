"""Cold-resume tool-surface snapshot tests — TDD cases for the #103579 族 fix.

Background (observed 2026-09-07 22:11:57, session 20260907_220529_5021de):
  After a dashboard/gateway restart the cold-resumed agent rebuilds its tool
  surface via ``assemble_tool_defs`` from the *live registry state*, so the
  ``tools[]`` bytes change with the deferrable set (the ``tool_search`` bridge
  description embeds "Search {N} additional tools" + a catalog listing) →
  the cache prefix breaks right after the system prompt (observed hit residue
  == system-prompt tokens, 14,592 in this environment).

Fix: **session-scoped tool-surface snapshot** — after a session's first build
store the final ``agent.tools`` bytes in state.db (``agent_tools`` table,
content-addressed, linked via ``sessions.tools_hash`` /
``sessions.tools_fingerprint``); on cold resume decide by fingerprint:
  * fingerprint match (model/toolsets unchanged) → reuse the snapshot bytes
    unconditionally (byte-stable, ignores the live registry) → cache prefix
    stays stable across restarts
  * fingerprint mismatch / no snapshot → build live and store a new snapshot

The fingerprint covers only **user-visible configuration** (model +
enabled/disabled toolsets), never runtime registry state — that is the crux:
plugin registration order / check_fn pass-through / MCP connection state
changes must NOT trigger a rebuild.
"""

from __future__ import annotations

import json
import os
import sys

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _td(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


_SNAPSHOT_TOOLS = [_td("terminal", "Run shell commands."), _td("read_file", "Read a file.")]


def _tmp_session_db(tmp_path):
    """Build an isolated SessionDB (never touches the production state.db)."""
    from hermes_state import SessionDB

    return SessionDB(db_path=tmp_path / "state.db")


class TestFingerprint:
    """Fingerprint = f(user-visible configuration) contract."""

    def test_stable_for_same_input(self):
        from agent.tool_surface_snapshot import tool_surface_fingerprint

        fp1 = tool_surface_fingerprint("deepseek-v4", ["cli"], [])
        fp2 = tool_surface_fingerprint("deepseek-v4", ["cli"], [])
        assert fp1 == fp2

    def test_changes_with_model(self):
        from agent.tool_surface_snapshot import tool_surface_fingerprint

        assert tool_surface_fingerprint("model-a", ["cli"], []) != tool_surface_fingerprint(
            "model-b", ["cli"], []
        )

    def test_changes_with_toolsets(self):
        from agent.tool_surface_snapshot import tool_surface_fingerprint

        assert tool_surface_fingerprint("m", ["cli"], []) != tool_surface_fingerprint(
            "m", ["cli", "coding"], []
        )
        assert tool_surface_fingerprint("m", ["cli"], []) != tool_surface_fingerprint(
            "m", ["cli"], ["web"]
        )

    def test_order_insensitive(self):
        """toolsets participate as sets (enabled/disabled), not ordered lists."""
        from agent.tool_surface_snapshot import tool_surface_fingerprint

        assert tool_surface_fingerprint("m", ["a", "b"], []) == tool_surface_fingerprint(
            "m", ["b", "a"], []
        )


class TestSnapshotStore:
    """DB round-trip: stored bytes must come back exactly (byte-fidelity contract)."""

    def test_roundtrip(self, tmp_path):
        from agent.tool_surface_snapshot import load_snapshot, store_snapshot

        db = _tmp_session_db(tmp_path)
        db.create_session("sid-test", "cli", model="m")
        tools = _SNAPSHOT_TOOLS
        h = store_snapshot(db, "sid-test", "FP-1", tools)
        assert isinstance(h, str) and len(h) == 64
        loaded = load_snapshot(db, "sid-test", "FP-1")
        assert loaded is not None
        assert json.dumps(loaded, sort_keys=True) == json.dumps(tools, sort_keys=True)

    def test_reload_with_new_db_instance(self, tmp_path):
        """Simulate a restart: a new SessionDB instance reads the same DB identically."""
        from agent.tool_surface_snapshot import load_snapshot, store_snapshot

        db1 = _tmp_session_db(tmp_path)
        db1.create_session("sid-restart", "cli", model="m")
        store_snapshot(db1, "sid-restart", "FP-1", _SNAPSHOT_TOOLS)

        db2 = _tmp_session_db(tmp_path)
        loaded = load_snapshot(db2, "sid-restart", "FP-1")
        assert loaded is not None
        assert json.dumps(loaded, sort_keys=True) == json.dumps(_SNAPSHOT_TOOLS, sort_keys=True)


class TestColdResume:
    """Capture cases (FAIL before the fix / PASS after) — cold-resume byte stability."""

    def test_load_returns_none_for_unknown_session(self, tmp_path):
        from agent.tool_surface_snapshot import load_snapshot

        db = _tmp_session_db(tmp_path)
        assert load_snapshot(db, "sid-unknown", "FP-1") is None

    def test_load_returns_none_on_fingerprint_mismatch(self, tmp_path):
        """User changed config (model/toolsets) → fingerprint mismatch → rebuild."""
        from agent.tool_surface_snapshot import load_snapshot, store_snapshot

        db = _tmp_session_db(tmp_path)
        db.create_session("sid-fp", "cli", model="m")
        store_snapshot(db, "sid-fp", "FP-OLD", _SNAPSHOT_TOOLS)
        assert load_snapshot(db, "sid-fp", "FP-NEW") is None

    def test_registry_changes_do_not_invalidate_snapshot(self, tmp_path, monkeypatch):
        """★ Core capture case: registry-state changes on cold resume must not alter bytes.

        Bug scenario (22:11:57): plugin registration order / check_fn pass-through
        differs from the previous process → assembled bytes differ → prefix break.
        Snapshot semantics: when the fingerprint matches (config unchanged) the
        stored bytes must be returned regardless of live registry state.
        """
        from agent.tool_surface_snapshot import load_snapshot, store_snapshot

        db = _tmp_session_db(tmp_path)
        db.create_session("sid-resume", "cli", model="m")
        # First build: full tool surface (incl. 2 plugin tools)
        original = _SNAPSHOT_TOOLS + [
            _td("feishu_doc_read", "Read a Feishu doc."),
            _td("video_analyze", "Analyze a video."),
        ]
        store_snapshot(db, "sid-resume", "FP-1", original)

        # Simulate a restart where the registry lost 2 plugin tools (check_fn
        # not passed / not registered) — a live assemble would produce different
        # bytes; the snapshot must still return the original bytes.
        loaded = load_snapshot(db, "sid-resume", "FP-1")
        assert loaded is not None
        assert json.dumps(loaded, sort_keys=True) == json.dumps(original, sort_keys=True), (
            "Bug: cold resume changed the tool surface — snapshot must be used "
            "regardless of live registry state when the config fingerprint matches"
        )

    def test_store_overwrites_newer_fingerprint(self, tmp_path):
        """After a config change the rebuild must let the new fingerprint take over."""
        from agent.tool_surface_snapshot import load_snapshot, store_snapshot

        db = _tmp_session_db(tmp_path)
        db.create_session("sid-evolve", "cli", model="m")
        store_snapshot(db, "sid-evolve", "FP-1", _SNAPSHOT_TOOLS)
        store_snapshot(
            db,
            "sid-evolve",
            "FP-2",
            _SNAPSHOT_TOOLS + [_td("new_tool", "New.")],
        )
        assert load_snapshot(db, "sid-evolve", "FP-1") is None
        assert load_snapshot(db, "sid-evolve", "FP-2") is not None


class TestNewSessionBuildLink:
    """Capture cases (FAIL before the fix) — new-session row creation must link the snapshot.

    Incident (2026-09-07, session 20260907_231241_8d9da0): a new session's agent
    build happens BEFORE the sessions row is created (row creation is deferred to
    the first turn of run_conversation). During build,
    resolve_tool_surface → store_snapshot runs ``UPDATE sessions SET
    tools_hash=... WHERE id=?`` — the row doesn't exist, so the UPDATE hits 0
    rows (SQLite is silent), the snapshot is stored but the link is silently
    lost → sessions.tools_hash stays NULL → a cold resume has no snapshot to
    reuse → the tool-surface bytes can drift across restarts (the #103579 族
    fix would be defeated).

    system_prompt has no such problem: ``_ensure_db_session`` passes
    system_prompt to create_session (writing system_prompts + linking the hash).
    The tools snapshot is missing exactly that symmetric link.
    """

    def test_create_session_persists_tools_link(self, tmp_path):
        """DB contract: create_session must accept tools_hash/tools_fingerprint.

        This is the storage-layer support for "carry the snapshot link at row
        creation" (isomorphic to system_prompt_hash); before the fix,
        _insert_session_row raised TypeError on unknown kwargs.
        """
        from agent.tool_surface_snapshot import _tools_hash

        db = _tmp_session_db(tmp_path)
        h = _tools_hash(_SNAPSHOT_TOOLS)
        db.create_session(
            "sid-tools-link",
            "cli",
            model="m",
            tools_hash=h,
            tools_fingerprint="FP-1",
        )
        row = db.get_session("sid-tools-link")
        assert row.get("tools_hash") == h, (
            "Bug: create_session dropped tools_hash — the session row must "
            "carry the built tool-surface snapshot link so a cold resume can "
            "reuse it"
        )
        assert row.get("tools_fingerprint") == "FP-1", (
            "Bug: create_session dropped tools_fingerprint — fingerprint is "
            "required to decide whether the snapshot is still valid"
        )

    def test_store_before_row_creation_leaves_snapshot_row_but_no_link(
        self, tmp_path
    ):
        """Guard: store before the row exists (new-session build semantics) must not
        raise, and the agent_tools row is persisted — only the sessions link is
        missing (snapshot body intact)."""
        from agent.tool_surface_snapshot import load_snapshot, store_snapshot

        db = _tmp_session_db(tmp_path)
        # Build time: no session row yet (deferred); store only writes agent_tools
        store_snapshot(db, "sid-late", "FP-1", _SNAPSHOT_TOOLS)
        # Row still absent: UPDATE of 0 rows was silently swallowed (no exception)
        # — pre-fix behavior, do not assert success.
        row = db.get_session("sid-late")
        assert row is None or row.get("tools_hash") is None

    def test_ensure_db_session_links_built_tool_surface(self, tmp_path):
        """★ Core capture case: after the first turn creates the row
        (_ensure_db_session), the sessions row must carry the build-time
        snapshot's tools_hash/tools_fingerprint.

        Bug scenario (23:12:41): during build the store's UPDATE hit 0 rows; row
        creation (first turn of run_conversation) did not carry the snapshot link
        → tools_hash stayed NULL forever → cold resume had no snapshot to reuse.
        system_prompt travels the same path; the tools snapshot lacked the
        symmetric piece.
        """
        from types import SimpleNamespace

        from run_agent import AIAgent

        db = _tmp_session_db(tmp_path)
        # Minimal agent stand-in: only the attributes _ensure_db_session reads
        # (building a real AIAgent pulls full tool registration/MCP discovery).
        fake = SimpleNamespace(
            _persist_disabled=False,
            _session_db_created=False,
            _session_db=db,
            platform="desktop",
            session_id="sid-newbuild",
            _session_init_model_config=None,
            _cached_system_prompt="system prompt",
            _parent_session_id=None,
            model="m",
            # After the fix: init_agent's snapshot block stashes the build-time
            # hash/fingerprint here.
            _tool_surface_hash="dummy-snapshot-hash",
            _tool_surface_fingerprint="FP-NEW",
        )
        # Upstream _ensure_db_session delegates model_config to this method.
        fake._session_row_model_config = lambda: None
        # Simulate the first-turn row creation call from run_conversation
        AIAgent._ensure_db_session(fake)

        row = db.get_session("sid-newbuild")
        assert row.get("tools_hash") == "dummy-snapshot-hash", (
            "Bug: new-session row creation dropped the built tool-surface "
            "snapshot link — the snapshot was stored before the row existed "
            "(UPDATE 0 rows, silent), so without this link a cold resume "
            "rebuilds tools[] from the live registry and can break the "
            "prefix cache (tools_hash must travel with row creation, same as "
            "system_prompt_hash)"
        )
        assert row.get("tools_fingerprint") == "FP-NEW", (
            "Bug: tools_fingerprint missing on new-session row — without it "
            "a resume cannot decide whether the stored snapshot is still valid"
        )
