"""``SQLiteSessionMaintenance`` against the legacy ``SessionDB``: each case seeds one store with ``SessionDB``, copies
it, runs the legacy method on one copy and the backend verb on the other, and compares the results and the tables. The
cases that check the backend alone run the same seeds through the backend's own writes."""

from __future__ import annotations

import importlib.util
import sqlite3
import time
from functools import partial
from types import SimpleNamespace

import pytest

from persistence.core.errors import ExportTooLarge
from persistence.housekeeping.maintenance import SESSION_PROJECTION_FIELDS, PruneCriteria
from persistence.sqlite.conversation._store.sessions import create_session
from tests.persistence_sqlite._legacy_store import legacy_store, oracle_tree
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

NOW = time.time()
DAY = 86400.0
OLD = NOW - 200 * DAY
MEDIA_ROOT = "/nonexistent/native-inputs"

_SESSIONS = "SELECT id, parent_session_id, end_reason, ended_at IS NOT NULL, archived, auto_archived, pinned FROM sessions"
_TABLES = (
    _SESSIONS,
    "SELECT session_id, role, content, active FROM messages",
    "SELECT hash FROM system_prompts",
    "SELECT key FROM state_meta",
    "SELECT session_id, holder FROM compression_locks",
    "SELECT conversation_id, holder FROM session_turn_leases",
    "SELECT backend_id, pid, started_at, last_heartbeat, profile, host FROM gateway_heartbeats",
)


def _sql(db, sql, params=()):
    db._execute_write(lambda conn: conn.execute(sql, params))


def _session(db, sid, *, source="cli", started=OLD, ended=OLD + DAY, **cols):
    """A session row with the given columns; ``ended=None`` leaves it open."""
    db.create_session(sid, source)
    values = {"started_at": started, "ended_at": ended, "end_reason": "user_exit" if ended else None,
              "last_activity_at": None, **cols}
    _sql(db, f"UPDATE sessions SET {', '.join(f'{k} = ?' for k in values)} WHERE id = ?", (*values.values(), sid))


def _message(db, sid, at=OLD, *, active=1):
    _sql(db, "INSERT INTO messages (session_id, role, content, timestamp, active) VALUES (?, 'user', 'x', ?, ?)",
         (sid, at, active))


def _rows(path, sql):
    conn = sqlite3.connect(path)
    try:
        return sorted(conn.execute(sql).fetchall(), key=repr)
    finally:
        conn.close()


def _assert_same_tables(legacy, store):
    for sql in _TABLES:
        assert _rows(legacy.db_path, sql) == _rows(store.store_path, sql), sql


@pytest.fixture
def pair(tmp_path, backend, scope_a, legacy_store):
    """``pair(seed)``: a legacy ``SessionDB`` and the backend handle, each over its own copy of the seeded store."""
    from hermes_state import SessionDB

    opened = []

    def build(seed):
        seed_path, legacy_path = tmp_path / "seed.db", tmp_path / "legacy.db"
        db = SessionDB(db_path=seed_path)
        seed(db)
        db.close()
        for target in (legacy_path, store_for(scope_a)):
            source, copy = sqlite3.connect(seed_path), sqlite3.connect(target)
            source.backup(copy)
            source.close()
            copy.close()
        opened.append(SessionDB(db_path=legacy_path))
        return opened[-1], backend.open(scope_a)

    yield build
    for db in opened:
        db.close()


@pytest.fixture
def seeded(backend, scope_a):
    """``seeded(seed)``: the backend handle over its store, the seed run on the backend's own session and SQL writes
    under the ``SessionDB`` names it calls."""
    def build(seed):
        store = backend.open(scope_a)
        holder = store.holder()
        seed(SimpleNamespace(create_session=partial(create_session, holder), _execute_write=holder._execute_write))
        return store

    return build


def _seed_selection(db):
    _session(db, "old-a", title="Alpha Report", model="GPT-X", billing_provider="OpenAI", user_id="u1", chat_id="c1",
             chat_type="dm", git_branch="feat/Login", cwd="/work/proj", input_tokens=100, output_tokens=50,
             actual_cost_usd=0.5, tool_call_count=3, message_count=4)
    _message(db, "old-a")
    _session(db, "old-b", source="telegram", title="beta", cwd="/work/proj_other", estimated_cost_usd=2.0, archived=1)
    _session(db, "old-c", pinned=1, end_reason="compression")
    _session(db, "tail-c", parent_session_id="old-c")
    _session(db, "root-d", end_reason="compression")
    _session(db, "tip-d", parent_session_id="root-d", started=NOW - DAY, ended=NOW - DAY / 2)
    _session(db, "open-e", ended=None)
    _session(db, "orphan-f", end_reason="startup_orphan_reap", ended=NOW - DAY)
    _session(db, "recent-g", started=NOW - 2 * DAY, ended=NOW - DAY)
    _session(db, "cwd-h", cwd="/work/proj/sub", title="Gamma", message_count=1)


_CRITERIA = (
    {}, {"title_like": "ALPHA"}, {"cwd_prefix": "/work/proj"}, {"cwd_prefix": "/work/proj/"}, {"min_messages": 1},
    {"max_messages": 0}, {"model_like": "gpt"}, {"provider": "openai"}, {"user_id": "u1"}, {"chat_id": "c1"},
    {"chat_type": "dm"}, {"branch_like": "login"}, {"min_tokens": 150}, {"max_tokens": 0}, {"min_cost": 1.0},
    {"max_cost": 0.5}, {"min_tool_calls": 1}, {"max_tool_calls": 0}, {"archived": True}, {"archived": False},
    {"include_pinned": True}, {"lineage_tips_only": True}, {"last_active_after": NOW - 3 * DAY},
    {"started_before": NOW - 100 * DAY}, {"started_after": NOW - 3 * DAY}, {"end_reason": "compression"},
    {"title_like": ""}, {"last_active_before": NOW - 2 * DAY},
)


def _candidate(row):
    return (row.session_id, row.source, row.title, row.model, row.started_at, row.last_active, row.ended_at,
            row.message_count, row.archived)


def _legacy_candidate(row):
    return tuple(row[key] for key in ("id", "source", "title", "model", "started_at", "last_active", "ended_at",
                                      "message_count", "archived"))


@pytest.mark.parametrize(("days", "source"), [(None, None), (90, None), (90, "cli"), (0, "telegram")])
def test_prune_selection_reads_match_legacy(pair, days, source):
    legacy, store = pair(_seed_selection)
    maintenance = store.maintenance
    for filters in _CRITERIA:
        criteria = PruneCriteria(**filters)
        for whole in (False, True):
            expected = [_legacy_candidate(row) for row in legacy.list_prune_candidates(
                days, source, whole_lineages=whole, **dict(filters))]
            got = maintenance.list_prune_candidates(days, source, criteria=criteria, whole_lineages=whole)
            assert [_candidate(row) for row in got] == expected, (filters, whole)
        for pinned_only in (False, True):
            assert maintenance.count_prune_matches(days, source, criteria=criteria, pinned_only=pinned_only) == (
                legacy.count_prune_matches(days, source, pinned_only=pinned_only, **dict(filters))), filters
        assert maintenance.count_open_prune_matches(days, source, criteria=criteria) == (
            legacy.count_open_prune_matches(days, source, **dict(filters))), filters
    assert maintenance.count_open_prune_matches(criteria=PruneCriteria(include_open=True)) == 0


def test_prune_refuses_open_rows_and_negative_age(seeded):
    store = seeded(_seed_selection)
    with pytest.raises(ValueError, match="include_open is archive-only"):
        store.maintenance.prune_sessions(criteria=PruneCriteria(include_open=True), media_root=MEDIA_ROOT)
    with pytest.raises(ValueError, match="older_than_days must be >= 0"):
        store.maintenance.prune_sessions(-1, media_root=MEDIA_ROOT)
    with pytest.raises(ValueError, match="older_than_days must be >= 0"):
        store.maintenance.list_prune_candidates(-1)


def test_prune_deletes_whole_lineages_and_spares_pinned_tails(pair):
    def seed(db):
        _seed_selection(db)
        db.create_session("child-of-a", "cli", parent_session_id="old-a")
        _sql(db, "UPDATE sessions SET started_at = ? WHERE id = 'child-of-a'", (NOW,))
        _sql(db, "INSERT INTO system_prompts (hash, prompt) VALUES ('h-a', 'p')")
        _sql(db, "UPDATE sessions SET system_prompt_hash = 'h-a' WHERE id = 'old-a'")
    legacy, store = pair(seed)
    out: list[str] = []
    expected = legacy.prune_sessions(90, deleted_ids=out)
    result = store.maintenance.prune_sessions(90, media_root=MEDIA_ROOT)
    assert result.count == expected
    assert sorted(result.deleted_ids) == sorted(out)
    assert {"old-a", "old-b"} <= set(result.deleted_ids)
    # A compression root with a live continuation, and a pinned segment with its unpinned tail, stay.
    assert not {"root-d", "old-c", "tail-c", "open-e"} & set(result.deleted_ids)
    _assert_same_tables(legacy, store)


def _lock(db, table, key, holder, expires):
    column = "session_id" if table == "compression_locks" else "conversation_id"
    _sql(db, f"INSERT INTO {table} ({column}, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
         (key, holder, NOW - 60, expires))


def _admission(db, sid, status="queued"):
    _sql(db, "INSERT INTO session_admissions (admission_id, request_id, principal_id, target_session_id, lineage_json,"
             " payload_json, payload_digest, intent, status) VALUES (?, ?, 'p', ?, '[]', '{}', 'd', 'queue', ?)",
         (f"adm-{sid}", f"req-{sid}", sid, status))


def _seed_guards(db):
    for sid in ("live-lock", "live-lease", "expired-lock", "expired-lease", "busy-ledger", "plain"):
        _session(db, sid)
        _message(db, sid)
    _lock(db, "compression_locks", "live-lock", "other", NOW + 3600)
    _lock(db, "session_turn_leases", "live-lease", "other", NOW + 3600)
    _lock(db, "compression_locks", "expired-lock", "other", NOW - 10)
    _lock(db, "session_turn_leases", "expired-lease", "other", NOW - 10)
    _admission(db, "busy-ledger")


@pytest.mark.parametrize("exclude_guards", [True, False])
def test_prune_guard_skip_reclaims_expired_guards_and_skips_busy_ledgers(pair, exclude_guards):
    legacy, store = pair(_seed_guards)
    out: list[str] = []
    expected = legacy.prune_sessions(90, exclude_active_write_guards=exclude_guards, deleted_ids=out)
    result = store.maintenance.prune_sessions(90, exclude_active_write_guards=exclude_guards, media_root=MEDIA_ROOT)
    assert (result.count, sorted(result.deleted_ids)) == (expected, sorted(out))
    assert "busy-ledger" not in result.deleted_ids
    assert ({"live-lock", "live-lease"} & set(result.deleted_ids) == set()) is exclude_guards
    assert {"expired-lock", "expired-lease", "plain"} <= set(result.deleted_ids)
    _assert_same_tables(legacy, store)


def test_ghost_prune_matches_legacy(pair):
    def seed(db):
        _session(db, "ghost", source="tui")
        _session(db, "ghost-2", source="tui")
        _session(db, "titled", source="tui", title="Kept")
        _session(db, "with-message", source="tui")
        _message(db, "with-message")
        _session(db, "open", source="tui", ended=None)
        _session(db, "young", source="tui", started=NOW - 3600, ended=NOW - 1800)
        _session(db, "cli-row", source="cli")
        _session(db, "busy", source="tui")
        _admission(db, "busy")
    legacy, store = pair(seed)
    expected = legacy.prune_empty_ghost_sessions()
    removed = store.maintenance.prune_empty_ghost_sessions(media_root=MEDIA_ROOT)
    assert len(removed) == expected
    assert sorted(removed) == ["ghost", "ghost-2"]
    _assert_same_tables(legacy, store)


def _seed_orphans(db):
    for sid in ("orphan", "orphan-2", "pinned", "excluded", "leased"):
        _session(db, sid, source="tui", ended=None)
    _sql(db, "UPDATE sessions SET pinned = 1 WHERE id = 'pinned'")
    _session(db, "fresh", source="tui", started=NOW - 60, ended=None)
    _session(db, "telegram-open", source="telegram", ended=None)
    _session(db, "closed", source="tui")
    _lock(db, "session_turn_leases", "leased", "other", NOW + 3600)
    _lock(db, "compression_locks", "orphan-2", "other", NOW - 10)


@pytest.mark.parametrize("exclude_pinned", [True, False])
def test_orphan_sweep_respects_heartbeats_guards_and_exclusions(pair, exclude_pinned):
    legacy, store = pair(_seed_orphans)
    maintenance = store.maintenance
    kwargs = {"max_idle_seconds": 10 * DAY, "exclude_ids": ("excluded",), "exclude_pinned": exclude_pinned}
    for db in (legacy, maintenance):
        db.register_backend_heartbeat(backend_id="gw", pid=1, started_at=OLD + 5 * DAY, last_heartbeat=NOW)
    # A live backend that could own the rows spares them.
    assert maintenance.sweep_orphaned_sessions(**kwargs) == tuple(legacy.sweep_orphaned_sessions(**kwargs)) == ()
    # A backend that started after every row (beyond the grace) owns none of them.
    for db in (legacy, maintenance):
        db.register_backend_heartbeat(backend_id="gw", pid=1, started_at=NOW, last_heartbeat=NOW)
    got = maintenance.sweep_orphaned_sessions(**kwargs, heartbeat_ownership_grace_seconds=0)
    assert sorted(got) == sorted(legacy.sweep_orphaned_sessions(**kwargs, heartbeat_ownership_grace_seconds=0))
    assert set(got) == ({"orphan", "orphan-2"} if exclude_pinned else {"orphan", "orphan-2", "pinned"})
    _assert_same_tables(legacy, store)
    assert _rows(store.store_path, "SELECT DISTINCT end_reason FROM sessions WHERE id = 'orphan'") == [
        ("startup_orphan_reap",)]
    assert maintenance.sweep_orphaned_sessions(max_idle_seconds=0) == ()
    assert maintenance.sweep_orphaned_sessions(max_idle_seconds=DAY, sources=("",)) == ()


def test_orphan_sweep_without_heartbeat_gate(pair):
    legacy, store = pair(_seed_orphans)
    for db in (legacy, store.maintenance):
        db.register_backend_heartbeat(backend_id="gw", pid=1, started_at=OLD, last_heartbeat=NOW)
    kwargs = {"max_idle_seconds": 10 * DAY, "sources": ("tui", "telegram"), "respect_gateway_heartbeats": False}
    got = store.maintenance.sweep_orphaned_sessions(**kwargs)
    assert sorted(got) == sorted(legacy.sweep_orphaned_sessions(**kwargs))
    assert "telegram-open" in got and "leased" not in got
    _assert_same_tables(legacy, store)


@pytest.mark.parametrize("exclude_pinned", [True, False])
def test_idle_archive_matches_legacy_sweep(pair, exclude_pinned):
    def seed(db):
        _session(db, "idle", ended=None)
        _session(db, "root", end_reason="compression", archived=1)
        _session(db, "tip", parent_session_id="root", started=OLD + 2 * DAY, ended=OLD + 3 * DAY)
        _session(db, "bot", hidden=1, title="Bot Chat")
        _session(db, "pinned", pinned=1, started=OLD - DAY)
        _session(db, "fresh", started=NOW - 60, ended=None)
        _session(db, "archived", archived=1)
    legacy, store = pair(seed)
    expected = legacy.archive_stale_sessions(3, exclude_pinned=exclude_pinned)
    candidates = store.maintenance.list_idle_archive_candidates(3, exclude_pinned=exclude_pinned)
    assert len(candidates) == expected
    assert list(candidates) == (["pinned"] if not exclude_pinned else []) + ["idle", "tip"]
    assert all(store.maintenance.auto_archive_lineage(sid) for sid in candidates)
    _assert_same_tables(legacy, store)
    # A deliberately archived ancestor keeps its provenance; the rows the sweep hid are stamped.
    assert dict(_rows(store.store_path, "SELECT id, auto_archived FROM sessions WHERE id IN ('root', 'tip')")) == {
        "root": 0, "tip": 1}
    assert store.maintenance.auto_archive_lineage("missing") is False
    assert store.maintenance.list_idle_archive_candidates(-1) == ()


def test_backend_heartbeats_match_legacy(pair):
    legacy, store = pair(lambda db: None)
    maintenance = store.maintenance
    for db in (legacy, maintenance):
        db.register_backend_heartbeat(backend_id="a@h:1", pid=1, started_at=OLD, last_heartbeat=NOW, profile="p",
                                      host="h")
        db.register_backend_heartbeat(backend_id="a@h:1", pid=2, started_at=OLD + 1, last_heartbeat=NOW + 1)
        db.register_backend_heartbeat(backend_id="b@h:3", pid=3, started_at=OLD, last_heartbeat=NOW)
        db.register_backend_heartbeat(backend_id="", pid=4, started_at=OLD)
    _assert_same_tables(legacy, store)
    assert maintenance.clear_backend_heartbeat("b@h:3") is legacy.clear_backend_heartbeat("b@h:3") is True
    assert maintenance.clear_backend_heartbeat("b@h:3") is legacy.clear_backend_heartbeat("b@h:3") is False
    assert maintenance.clear_backend_heartbeat("") is False
    _assert_same_tables(legacy, store)
    before = time.time()
    maintenance.register_backend_heartbeat(backend_id="c", pid=5, started_at=OLD)
    assert _rows(store.store_path, "SELECT last_heartbeat FROM gateway_heartbeats WHERE backend_id = 'c'")[0][0] >= before


def test_assert_export_safe_counts_every_stored_row(pair):
    from hermes_state import SessionExportTooLargeError

    def seed(db):
        _session(db, "s")
        for active in (1, 1, 0):
            _message(db, "s", active=active)
    legacy, store = pair(seed)
    maintenance = store.maintenance
    for limit in (3, 5):
        assert maintenance.assert_export_safe("s", limit) == legacy.assert_export_safe("s", max_messages=limit) == 3
    assert maintenance.assert_export_safe("s", 0) == 0
    assert maintenance.assert_export_safe("missing", 2) == legacy.assert_export_safe("missing", max_messages=2) == 0
    with pytest.raises(SessionExportTooLargeError) as legacy_refusal:
        legacy.assert_export_safe("s", max_messages=2)
    with pytest.raises(ExportTooLarge) as refusal:
        maintenance.assert_export_safe("s", 2)
    assert (refusal.value.session_id, refusal.value.message_count, refusal.value.limit) == (
        legacy_refusal.value.session_id, legacy_refusal.value.message_count, legacy_refusal.value.limit) == ("s", 3, 2)
    assert str(refusal.value) == str(legacy_refusal.value)
    with pytest.raises(ValueError, match="non-negative"):
        maintenance.assert_export_safe("s", -1)


def test_session_projection_digest_matches_legacy_probe(seeded):
    from tui_gateway.change_watcher import _session_db_content_sig

    store = seeded(_seed_selection)
    maintenance = store.maintenance
    digest = maintenance.session_projection_digest()
    assert digest is not None and digest[0] == SESSION_PROJECTION_FIELDS
    assert digest == _session_db_content_sig(store.store_path)
    # The activity columns the heartbeat restamps are not part of it; a projected column is.
    store.holder()._write_sql("UPDATE sessions SET last_activity_at = ? WHERE id = 'old-a'", (NOW,))
    assert maintenance.session_projection_digest(busy_timeout_s=1.0) == digest
    store.holder()._write_sql("UPDATE sessions SET title = 'renamed' WHERE id = 'old-a'")
    changed = maintenance.session_projection_digest()
    assert changed != digest and changed == _session_db_content_sig(store.store_path)
    # The read connection's own busy budget is restored after the probe.
    with store.holder()._read_ctx() as conn:
        budget = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    maintenance.session_projection_digest(busy_timeout_s=0.01)
    with store.holder()._read_ctx() as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == budget


def _main_worktree_ops():
    """Main's ``hermes_cli/worktree_ops.py`` from the pinned checkout, under a name of its own."""
    spec = importlib.util.spec_from_file_location("main_worktree_ops", oracle_tree() / "hermes_cli" / "worktree_ops.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worktree_reads_match_legacy(seeded, tmp_path):
    main = _main_worktree_ops()
    _retained_tree_is_live, _tree_still_served = main._retained_tree_is_live, main._tree_still_served

    tree, other = tmp_path / "tree", tmp_path / "tree-other"
    tree.mkdir()
    other.mkdir()

    def seed(db):
        _session(db, "history", cwd=str(tree / "sub"), message_count=2)
        _session(db, "pending", cwd=str(tree), chat_id="chat-pending")
        _admission(db, "pending")
        _session(db, "empty", cwd=str(tree) + "/", chat_id="chat-empty")
        _session(db, "outside", cwd=str(other), message_count=5, ended=None, chat_id="chat-outside")
    store = seeded(seed)
    maintenance = store.maintenance
    path = str(store.store_path)
    for pending_only in (False, True):
        assert maintenance.retained_tree_is_live(str(tree), pending_only=pending_only) is (
            _retained_tree_is_live(path, str(tree), pending_only=pending_only)) is True
    store.holder()._write_sql("UPDATE session_admissions SET status = 'terminal'")
    assert maintenance.retained_tree_is_live(str(tree), pending_only=True) is (
        _retained_tree_is_live(path, str(tree), pending_only=True)) is False
    assert maintenance.retained_tree_is_live(str(tree)) is _retained_tree_is_live(path, str(tree)) is True

    facts = maintenance.tree_served_sessions(str(tree))
    assert _tree_still_served(path, str(tree)) is None
    assert (facts.served, facts.chat_ids) == (False, frozenset({"chat-pending", "chat-empty"}))
    for value, served in (('{"status": "done"}', False), ("{}", True), ("not json", True)):
        store.holder().set_meta("goal:empty", value)
        assert maintenance.tree_served_sessions(str(tree)).served is served
        assert (_tree_still_served(path, str(tree)) == "session") is served
    store.holder().set_meta("goal:empty", '{"status": "done"}')
    store.holder()._write_sql("UPDATE sessions SET ended_at = NULL WHERE id = 'history'")
    assert maintenance.tree_served_sessions(str(tree)).served is (_tree_still_served(path, str(tree)) == "session")
    assert maintenance.tree_served_sessions(str(other)).served is True
