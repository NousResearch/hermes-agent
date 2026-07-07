import ast
import json
import os
import shutil
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest

import hermes_state
from hermes_state import SessionDB, _session_list_denorm_enabled
from hermes_cli.config import DEFAULT_CONFIG, OPTIONAL_ENV_VARS


def _write_dashboard_flag(enabled: bool) -> None:
    config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config_path.write_text(
        "dashboard:\n"
        f"  session_list_denorm: {str(enabled).lower()}\n",
        encoding="utf-8",
    )


def _make_db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def _set_started_at(db: SessionDB, session_id: str, started_at: float) -> None:
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (started_at, session_id),
        )
    )
    db.recompute_effective_last_active(session_id)


def _seed_session(
    db: SessionDB,
    session_id: str,
    *,
    source: str = "cli",
    started_at: float,
    message_ts: float,
    archived: bool = False,
) -> None:
    db.create_session(session_id, source=source, model="test-model")
    _set_started_at(db, session_id, started_at)
    db.append_message(
        session_id,
        role="user",
        content=f"hello from {session_id}",
        timestamp=message_ts,
    )
    if archived:
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET archived = 1 WHERE id = ?",
                (session_id,),
            )
        )


def _seed_compression_chain(db: SessionDB) -> None:
    _seed_session(db, "compressed-root", started_at=80.0, message_ts=90.0)
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET ended_at = ?, end_reason = 'compression' WHERE id = ?",
            (95.0, "compressed-root"),
        )
    )
    db.create_session(
        "compressed-tip",
        source="cli",
        model="test-model",
        parent_session_id="compressed-root",
    )
    _set_started_at(db, "compressed-tip", 96.0)
    db.append_message(
        "compressed-tip",
        role="user",
        content="hello from compressed tip",
        timestamp=500.0,
    )
    db.recompute_effective_last_active("compressed-root")


def _seed_listing_fixture(db: SessionDB) -> None:
    _seed_session(db, "old-but-active", started_at=100.0, message_ts=300.0)
    _seed_session(db, "newer-start", started_at=200.0, message_ts=210.0)
    _seed_session(db, "discord-row", source="discord", started_at=150.0, message_ts=250.0)
    _seed_session(db, "archived-row", started_at=50.0, message_ts=400.0, archived=True)
    _seed_compression_chain(db)


def _normalized(rows):
    """JSON round-trip like an RPC boundary; catches byte-shape drift in values."""
    return json.loads(json.dumps(rows, sort_keys=True, default=str))


def test_dashboard_session_list_denorm_default_is_false_config_only():
    assert DEFAULT_CONFIG["dashboard"]["session_list_denorm"] is False
    assert isinstance(DEFAULT_CONFIG["dashboard"]["session_list_denorm"], bool)
    assert not any("SESSION_LIST_DENORM" in name for name in OPTIONAL_ENV_VARS)


def test_session_list_denorm_flag_is_live_config_only(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_LIST_DENORM", "1")

    _write_dashboard_flag(False)
    assert _session_list_denorm_enabled() is False

    _write_dashboard_flag(True)
    assert _session_list_denorm_enabled() is True


def test_flag_off_keeps_cte_oracle_output_identical_for_existing_filters(tmp_path):
    _write_dashboard_flag(False)
    db = _make_db(tmp_path)
    try:
        _seed_listing_fixture(db)
        cases = [
            {"order_by_last_active": True},
            {"order_by_last_active": True, "include_archived": True},
            {"order_by_last_active": True, "source": "discord"},
            {"order_by_last_active": True, "id_query": "old-but"},
            {"order_by_last_active": True, "id_query": "compressed-tip"},
        ]
        for kwargs in cases:
            got = db.list_sessions_rich(limit=20, **kwargs)
            oracle = db.list_sessions_rich(limit=20, _force_cte_oracle=True, **kwargs)
            assert _normalized(got) == _normalized(oracle), kwargs
    finally:
        db.close()


def test_flag_on_denorm_path_matches_cte_oracle_byte_for_byte(tmp_path):
    _write_dashboard_flag(True)
    db = _make_db(tmp_path)
    try:
        _seed_listing_fixture(db)
        cases = [
            {"order_by_last_active": True},
            {"order_by_last_active": True, "include_archived": True},
            {"order_by_last_active": True, "source": "discord"},
            {"order_by_last_active": True, "id_query": "archived-row", "include_archived": True},
            {"order_by_last_active": True, "id_query": "compressed-tip"},
        ]
        for kwargs in cases:
            got = db.list_sessions_rich(limit=20, **kwargs)
            oracle = db.list_sessions_rich(limit=20, _force_cte_oracle=True, **kwargs)
            assert _normalized(got) == _normalized(oracle), kwargs
    finally:
        db.close()


def test_backfill_version_is_strictly_newer_than_every_shipped_marker():
    """The cutover-repair fires only when marker != VERSION, so VERSION MUST be
    greater than every marker value ever stamped onto a live DB — otherwise a
    build no-ops its own repair on a DB already carrying that marker.

    "4" is a POISONED value: an uncommitted VERSION="4" build was run against
    the live state.db during development and burned marker "4" onto production
    (git history only ever committed 1->2->3). A shipped VERSION of "4" would
    therefore skip the cutover repair on Ace's real DB. VERSION must be > 4.
    """
    version = hermes_state._EFFECTIVE_LAST_ACTIVE_BACKFILL_VERSION
    assert version.isdigit(), f"backfill version must be an integer string, got {version!r}"
    assert int(version) >= 5, (
        f"BACKFILL_VERSION={version!r} is not strictly newer than the poisoned "
        "marker '4' burned onto the live DB — the open-path repair would be "
        "skipped on a DB that already carries this marker. Bump the version."
    )


@pytest.mark.parametrize("stale_marker", ["1", "2", "3", "4", None])
def test_stale_backfill_marker_re_repairs_on_open(tmp_path, stale_marker):
    """Any marker OLDER than the current VERSION (or absent) must re-repair the
    stored recency on open — proven against every marker ever shipped, incl. the
    poisoned "4". This is the invariant the brittle v3-only test missed: it
    hardcoded "3" and so would have gone green while a VERSION=="4" build
    silently skipped the repair on the "4"-stamped production DB.
    """
    _write_dashboard_flag(True)
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        _seed_session(db, "stale-row", started_at=100.0, message_ts=500.0)
        _seed_session(db, "fresh-row", started_at=200.0, message_ts=300.0)

        def _force_stale_marker(conn):
            conn.execute(
                "UPDATE sessions SET effective_last_active = ? WHERE id = ?",
                (1.0, "stale-row"),
            )
            if stale_marker is None:
                conn.execute(
                    "DELETE FROM state_meta WHERE key = ?",
                    (hermes_state._EFFECTIVE_LAST_ACTIVE_BACKFILL_META_KEY,),
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?)",
                    (hermes_state._EFFECTIVE_LAST_ACTIVE_BACKFILL_META_KEY, stale_marker),
                )

        db._execute_write(_force_stale_marker)
    finally:
        db.close()

    # Precondition: the marker we forced is genuinely older than VERSION, so the
    # repair is EXPECTED to fire. Guards against a future VERSION regressing to
    # <= a value in this list (which would make the case a no-op false-pass).
    if stale_marker is not None:
        assert int(stale_marker) < int(
            hermes_state._EFFECTIVE_LAST_ACTIVE_BACKFILL_VERSION
        ), "test precondition broken: forced marker is not older than VERSION"

    reopened = SessionDB(db_path=db_path)
    try:
        assert reopened._conn is not None
        stored = reopened._conn.execute(
            "SELECT effective_last_active FROM sessions WHERE id = ?",
            ("stale-row",),
        ).fetchone()[0]
        assert stored == 500.0
        marker = reopened._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._EFFECTIVE_LAST_ACTIVE_BACKFILL_META_KEY,),
        ).fetchone()[0]
        assert marker == hermes_state._EFFECTIVE_LAST_ACTIVE_BACKFILL_VERSION
        got = reopened.list_sessions_rich(order_by_last_active=True, limit=10)
        oracle = reopened.list_sessions_rich(
            order_by_last_active=True,
            limit=10,
            _force_cte_oracle=True,
        )
        assert _normalized(got) == _normalized(oracle)
    finally:
        reopened.close()


class TestGatewayFlushMaintainsEffectiveLastActive:
    def _make_agent(self, db: SessionDB, session_id: str):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent

            return AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=db,
                session_id=session_id,
                platform="desktop",
                skip_context_files=True,
                skip_memory=True,
            )

    def test_new_agent_flush_session_orders_like_cte_oracle_and_has_denorm_value(self, tmp_path):
        _write_dashboard_flag(True)
        db = _make_db(tmp_path)
        try:
            _seed_session(db, "newer-than-live", started_at=10.0, message_ts=1200.0)
            _seed_session(db, "older-than-live", started_at=20.0, message_ts=800.0)

            agent = self._make_agent(db, "live-gateway-new-session")
            turn_messages = [
                {"role": "user", "content": "live user", "timestamp": 1000.0},
                {"role": "assistant", "content": "live assistant", "timestamp": 1000.5},
            ]
            agent._flush_messages_to_session_db(turn_messages, conversation_history=[])

            stored = db._conn.execute(
                "SELECT effective_last_active FROM sessions WHERE id = ?",
                ("live-gateway-new-session",),
            ).fetchone()[0]
            assert stored is not None

            got = db.list_sessions_rich(order_by_last_active=True, limit=10)
            oracle = db.list_sessions_rich(
                order_by_last_active=True,
                limit=10,
                _force_cte_oracle=True,
            )
            assert _normalized(got) == _normalized(oracle)
            ids = [row["id"] for row in got]
            assert ids[:3] == [
                "newer-than-live",
                "live-gateway-new-session",
                "older-than-live",
            ]
        finally:
            db.close()


def _function_sources_by_qualname(source_path: Path) -> Dict[str, str]:
    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    found: Dict[str, str] = {}

    def visit_body(body, prefix: str = "") -> None:
        for item in body:
            if isinstance(item, ast.ClassDef):
                visit_body(item.body, f"{prefix}{item.name}.")
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[f"{prefix}{item.name}"] = "\n".join(
                    lines[item.lineno - 1 : item.end_lineno]
                )

    visit_body(tree.body)
    return found


def _sessiondb_method_sources() -> Dict[str, str]:
    source_path = Path(hermes_state.__file__)
    prefix = "SessionDB."
    return {
        name[len(prefix):]: src
        for name, src in _function_sources_by_qualname(source_path).items()
        if name.startswith(prefix) and "." not in name[len(prefix):]
    }


def test_every_sessiondb_message_insert_path_is_effective_last_active_adjacent():
    all_functions = _function_sources_by_qualname(Path(hermes_state.__file__))
    insert_functions = {
        name: src
        for name, src in all_functions.items()
        if "INSERT INTO messages (" in src
    }
    assert insert_functions == {
        "_db_opens_cleanly": insert_functions.get("_db_opens_cleanly"),
        "SessionDB.append_message": insert_functions.get("SessionDB.append_message"),
        "SessionDB._insert_message_rows": insert_functions.get("SessionDB._insert_message_rows"),
    }

    methods = _sessiondb_method_sources()
    inserting_methods = {
        name: src
        for name, src in methods.items()
        if "INSERT INTO messages (" in src
    }
    assert inserting_methods, "source contract must see production message INSERTs"
    offenders = [
        name
        for name, src in inserting_methods.items()
        if "_bump_effective_last_active_for_message" not in src
    ]
    assert offenders == []

    recompute_required = [
        "replace_messages",
        "archive_and_compact",
        "update_session_meta",
    ]
    missing_recompute = [
        name
        for name in recompute_required
        if "_recompute_effective_last_active_for_session" not in methods[name]
    ]
    assert missing_recompute == []


def test_update_session_meta_visibility_flip_matches_cte_oracle(tmp_path):
    """Greptile P1 regression: rewriting model_config via update_session_meta can
    flip a row's session.list visibility (it carries the _delegate_from marker).
    The denorm path keys visibility on the stored effective_last_active column, so
    if update_session_meta doesn't recompute it, a row made delegate-only keeps a
    stale non-NULL value and stays visible (or a cleared row stays hidden) — the
    flag-on denorm output then diverges from the CTE oracle. Both directions must
    stay byte-identical to the oracle after the mutation.
    """
    _write_dashboard_flag(True)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _seed_session(db, "row-A", started_at=100.0, message_ts=1000.0)
        _seed_session(db, "row-B", started_at=200.0, message_ts=2000.0)

        def _assert_matches_oracle(context):
            got = db.list_sessions_rich(order_by_last_active=True, limit=10)
            oracle = db.list_sessions_rich(
                order_by_last_active=True, limit=10, _force_cte_oracle=True
            )
            assert _normalized(got) == _normalized(oracle), context

        # Baseline: both visible, both paths agree.
        _assert_matches_oracle("baseline")

        # Make row-B delegate-only AFTER write → must vanish from BOTH paths.
        db.update_session_meta("row-B", json.dumps({"_delegate_from": "parent-x"}))
        assert "row-B" not in [
            r["id"] for r in db.list_sessions_rich(order_by_last_active=True, limit=10)
        ]
        _assert_matches_oracle("after making row-B delegate-only")

        # Clear the marker → row-B must reappear in BOTH paths.
        db.update_session_meta("row-B", json.dumps({}))
        assert "row-B" in [
            r["id"] for r in db.list_sessions_rich(order_by_last_active=True, limit=10)
        ]
        _assert_matches_oracle("after clearing row-B delegate marker")
    finally:
        db.close()


def test_update_session_meta_recomputes_previous_root(tmp_path):
    """Greptile P4 regression: a model_config marker flip can MOVE a row between
    compression roots (a continuation child branching away from its root). The P1
    fix recomputed the row's NEW root but not its PREVIOUS root, so the old root
    kept an effective_last_active that still folded in the departed child's fresh
    message and sorted ahead of the CTE path. update_session_meta must capture the
    previous root BEFORE the write and recompute it after (like the other
    linkage-changing paths). Mutation-proven: drop the previous-root recompute → RED.
    """
    _write_dashboard_flag(True)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        # root (compression) -> child continuation carrying the fresh message.
        _seed_session(db, "root", started_at=1000.0, message_ts=1000.0)
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = 'compression' WHERE id = ?",
                (1001.0, "root"),
            )
        )
        db.create_session("child", source="cli", model="test-model", parent_session_id="root")
        db.append_message("child", role="user", content="c", timestamp=5000.0)
        db.recompute_effective_last_active("root")
        # rival timestamp sits between root's own recency (1000) and the child's (5000).
        _seed_session(db, "rival", started_at=1500.0, message_ts=3000.0)

        # Precondition: root's stored recency currently folds in the child (5000).
        assert (
            db._conn.execute(
                "SELECT effective_last_active FROM sessions WHERE id = ?", ("root",)
            ).fetchone()[0]
            == 5000.0
        )

        # Branch the child away from root → root's recency must drop back to 1000.
        db.update_session_meta("child", json.dumps({"_branched_from": "root"}))
        assert (
            db._conn.execute(
                "SELECT effective_last_active FROM sessions WHERE id = ?", ("root",)
            ).fetchone()[0]
            == 1000.0
        ), "previous root kept stale recency including the departed child's message"

        got = db.list_sessions_rich(order_by_last_active=True, limit=10)
        oracle = db.list_sessions_rich(
            order_by_last_active=True, limit=10, _force_cte_oracle=True
        )
        assert _normalized(got) == _normalized(oracle)
    finally:
        db.close()


def test_id_query_deep_chain_matches_cte_oracle(tmp_path):
    """Greptile P2 regression: the denorm id_query recursion previously capped at
    depth < 100 while the legacy CTE search is unbounded, so a compression chain
    whose matching tip sits beyond depth 100 was found by the oracle but missed by
    the flag-on denorm path — the same session.list search returning different rows
    once the flag is enabled. Compression edges are tree-structured (one parent per
    child), so the recursion terminates without the magic cap. A >100-deep chain
    searched by its deep tip id must resolve identically in both paths.
    """
    _write_dashboard_flag(True)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        depth = 130
        prev = None
        for i in range(depth):
            sid = f"c{i:03d}"
            db.create_session(sid, source="cli", model="test-model", parent_session_id=prev)
            db.append_message(sid, role="user", content=f"turn {i}", timestamp=1000.0 + i)
            if prev is not None:
                db._execute_write(
                    lambda conn, p=prev: conn.execute(
                        "UPDATE sessions SET ended_at = ?, end_reason = 'compression' WHERE id = ?",
                        (999.0, p),
                    )
                )
            prev = sid

        deep_tip = f"c{depth - 1:03d}"  # depth 129, well beyond the old cap
        got = db.list_sessions_rich(order_by_last_active=True, limit=10, id_query=deep_tip)
        oracle = db.list_sessions_rich(
            order_by_last_active=True, limit=10, id_query=deep_tip, _force_cte_oracle=True
        )
        assert _normalized(got) == _normalized(oracle)
        # And the deep-tip search actually resolves to a row (not an empty result
        # from a truncated recursion). Before the fix the denorm path returned []
        # for a tip beyond depth 100 while the oracle found it.
        assert len(got) == 1, got
    finally:
        db.close()


def test_deep_chain_stored_recency_matches_unbounded_oracle(tmp_path):
    """Greptile P3 regression: the id_query search recursion was unbounded (P2 fix)
    but the CTEs that compute the STORED effective_last_active (root-resolve,
    _expected_effective_last_active, and the backfill recompute-all) still capped at
    depth < 100. So a >100-deep compression chain whose freshest message is past hop
    100 had its stored recency truncated to the 100-hop max — the denorm path then
    ordered/paginated on a stale value that diverges from the unbounded CTE oracle.
    A recompute (backfill-on-open, visibility flip, archive_and_compact) would even
    REGRESS an already-correct value back to the truncated one. All recency CTEs must
    be unbounded to match the oracle. Mutation-proven: restore any cap → this REDs.
    """
    _write_dashboard_flag(True)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        depth = 130
        prev = None
        for i in range(depth):
            sid = f"c{i:03d}"
            if prev is not None:
                db._execute_write(
                    lambda conn, p=prev: conn.execute(
                        "UPDATE sessions SET ended_at = ?, end_reason = 'compression' WHERE id = ?",
                        (999.0, p),
                    )
                )
            db.create_session(sid, source="cli", model="test-model", parent_session_id=prev)
            # Deepest hop (past 100) carries the freshest timestamp.
            db.append_message(sid, role="user", content=f"t{i}", timestamp=1000.0 + i * 10.0)
            prev = sid

        deep_tip_ts = 1000.0 + (depth - 1) * 10.0  # 2290.0, at hop 129
        root = "c000"

        # The fresh per-row oracle must see the deep tip (unbounded).
        assert db.expected_effective_last_active(root) == deep_tip_ts

        # And a recompute (what backfill-on-open / visibility-flip / compaction do)
        # must NOT truncate the stored value back to the 100-hop max.
        db.recompute_effective_last_active(root)
        stored = db._conn.execute(
            "SELECT effective_last_active FROM sessions WHERE id = ?", (root,)
        ).fetchone()[0]
        assert stored == deep_tip_ts, f"stored recency truncated to {stored}, expected {deep_tip_ts}"

        # A rival session whose timestamp sits between hop-100 and the deep tip must
        # rank BELOW the deep chain's root in both paths.
        db.create_session("rival", source="cli", model="test-model")
        db.append_message("rival", role="user", content="rival", timestamp=1000.0 + 115 * 10.0)
        got = db.list_sessions_rich(order_by_last_active=True, limit=10)
        oracle = db.list_sessions_rich(
            order_by_last_active=True, limit=10, _force_cte_oracle=True
        )
        assert _normalized(got) == _normalized(oracle)
    finally:
        db.close()


@pytest.mark.skipif(
    not os.environ.get("SESSION_LIST_REAL_COPY_DB"),
    reason="set SESSION_LIST_REAL_COPY_DB=/path/to/state.db for real-copy churn check",
)
def test_real_copy_denorm_listing_matches_cte_oracle(tmp_path):
    _write_dashboard_flag(True)
    src = Path(os.environ["SESSION_LIST_REAL_COPY_DB"])
    dst = tmp_path / "real-copy-state.db"
    shutil.copy2(src, dst)
    for suffix in ("-wal", "-shm"):
        sidecar = src.with_name(src.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, dst.with_name(dst.name + suffix))

    db = SessionDB(db_path=dst)
    try:
        sample = db.list_sessions_rich(
            order_by_last_active=True,
            limit=1,
            _force_cte_oracle=True,
        )
        if not sample:
            pytest.skip("real-copy DB has no sessions to diff")
        sample_id = sample[0]["id"]
        sample_source = sample[0]["source"]
        cases = [
            {"order_by_last_active": True, "limit": 200},
            {"order_by_last_active": True, "limit": 200, "include_archived": True},
            {"order_by_last_active": True, "limit": 200, "id_query": sample_id},
            {"order_by_last_active": True, "limit": 200, "source": sample_source},
        ]
        for kwargs in cases:
            assert _normalized(db.list_sessions_rich(**kwargs)) == _normalized(
                db.list_sessions_rich(_force_cte_oracle=True, **kwargs)
            )
    finally:
        db.close()
