"""HRM-T0a step 4 — inbound active-topic-pointer routing pre-pass.

Covers:
- ``resolve_topic_session_key`` returns a topic-routed key when a
  pointer exists, the registry checker is wired, and the topic is
  registered.
- Legacy fall-through when (pointer mode off, no app_id, no pointer,
  no registry checker, registry rejects).
- Fail-closed contract: an unwired registry checker NEVER promotes
  the routing flip — Critic finding #5.
- ``SessionStore._generate_session_key`` consults the pre-pass and
  honours the topic key it returns.
- Concurrent ``move_turns`` calls with the same idempotency key
  return a consistent body — Critic finding #2.
- The HRM-T0a migration stamp is monotonic above ``max(started_at)``
  across many pre-existing sessions — Critic finding #3.
- ``_resolve_move_range`` excludes already-tombstoned rows — Critic
  finding #4.
- The same registry checker that guards slash-write also guards the
  routing flip — Critic finding #5.
- The lock-window boundary is documented in code and respected by
  ``resolve_topic_session_key_async`` — Critic finding #1.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gateway.active_topic as active_topic_module
from gateway.active_topic import (
    PlatformPrincipal,
    build_topic_session_key,
    resolve_topic_session_key,
    resolve_topic_session_key_async,
    set_registered_check,
)
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_state import SessionDB


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_module_state():
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    yield
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)


def _source(**overrides) -> SessionSource:
    """Build a default Telegram-DM SessionSource with overrides."""
    defaults = dict(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
        thread_id="1234",
    )
    defaults.update(overrides)
    return SessionSource(**defaults)


def _config(*, pointer_mode_enabled=True, default_app_id="hermes-agent") -> GatewayConfig:
    return GatewayConfig(
        topic_pointer_mode_enabled=pointer_mode_enabled,
        topic_default_app_id=default_app_id,
    )


def _ok_checker():
    async def _check(app_id, topic_id):
        return True
    return _check


def _reject_checker():
    async def _check(app_id, topic_id):
        return False
    return _check


# ── build_topic_session_key ────────────────────────────────────────────


def test_build_topic_session_key_default_namespace():
    p = PlatformPrincipal("telegram", "u1", "c1", "app1")
    key = build_topic_session_key(p, topic_id="research")
    assert key == "agent:main:topic:telegram:c1:u1:app1:research"


def test_build_topic_session_key_named_profile():
    p = PlatformPrincipal("telegram", "u1", "c1", "app1")
    key = build_topic_session_key(p, topic_id="research", profile="coder")
    assert key.startswith("agent:coder:topic:")


def test_build_topic_session_key_requires_topic_id():
    p = PlatformPrincipal("telegram", "u1", "c1", "app1")
    with pytest.raises(ValueError, match="topic_id is required"):
        build_topic_session_key(p, topic_id="")


def test_topic_session_key_isolates_apps_and_topics_and_users():
    p1 = PlatformPrincipal("telegram", "u1", "c1", "app1")
    p2 = PlatformPrincipal("telegram", "u1", "c1", "app2")
    p3 = PlatformPrincipal("telegram", "u2", "c1", "app1")
    assert build_topic_session_key(p1, topic_id="t1") != build_topic_session_key(
        p2, topic_id="t1"
    )
    assert build_topic_session_key(p1, topic_id="t1") != build_topic_session_key(
        p1, topic_id="t2"
    )
    assert build_topic_session_key(p1, topic_id="t1") != build_topic_session_key(
        p3, topic_id="t1"
    )


# ── resolve_topic_session_key (sync) — fall-through paths ─────────────


def test_resolve_pre_pass_returns_none_when_pointer_mode_disabled(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    try:
        assert (
            resolve_topic_session_key(
                _source(),
                db,
                app_id="hermes-agent",
                pointer_mode_enabled=False,
            )
            is None
        )
    finally:
        db.close()


def test_resolve_pre_pass_returns_none_when_no_app_id(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert resolve_topic_session_key(_source(), db, app_id=None) is None
        assert resolve_topic_session_key(_source(), db, app_id="") is None
    finally:
        db.close()


def test_resolve_pre_pass_returns_none_when_no_pointer(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    try:
        key = resolve_topic_session_key(_source(), db, app_id="hermes-agent")
        assert key is None
    finally:
        db.close()


def test_resolve_pre_pass_returns_topic_key_when_pointer_and_registered(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    try:
        key = resolve_topic_session_key(_source(), db, app_id="hermes-agent")
        assert key == (
            "agent:main:topic:telegram:208214988:208214988:hermes-agent:research"
        )
    finally:
        db.close()


# ── Critic finding #5: fail-closed routing ────────────────────────────


def test_resolve_pre_pass_fails_closed_when_no_checker_wired(tmp_path):
    """Pointer exists, but no registry checker → MUST fall through.

    The slash side rejects writes without a wired checker (step 3).
    The routing side mirrors that contract: a pointer that was written
    *before* the checker was wired could route messages to a topic
    that's no longer registered — fail closed.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    # Write the pointer with require_registered=False (compensating-
    # rollback path) so we can exercise the read side without a checker.
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    try:
        # No checker wired.
        assert active_topic_module._REGISTERED_CHECK is None
        key = resolve_topic_session_key(
            _source(),
            db,
            app_id="hermes-agent",
            require_registered_check=True,
        )
        assert key is None, "fail-closed: no checker ⇒ no routing flip"
    finally:
        db.close()


def test_resolve_pre_pass_falls_through_when_registry_rejects(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    set_registered_check(_reject_checker())
    try:
        assert (
            resolve_topic_session_key(_source(), db, app_id="hermes-agent")
            is None
        )
    finally:
        db.close()


def test_resolve_pre_pass_can_skip_registry_check(tmp_path):
    """``require_registered_check=False`` skips the check.

    Reserved for compensating-rollback paths and tests — production
    routing must always pass True.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    try:
        # No checker wired; require_registered_check=False bypasses the gate.
        key = resolve_topic_session_key(
            _source(),
            db,
            app_id="hermes-agent",
            require_registered_check=False,
        )
        assert key is not None
        assert "research" in key
    finally:
        db.close()


# ── resolve_topic_session_key_async (locked) ──────────────────────────


def test_resolve_async_wraps_with_lock_and_returns_key(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )

    async def run():
        key = await resolve_topic_session_key_async(
            _source(), db, app_id="hermes-agent"
        )
        return key

    try:
        key = asyncio.run(run())
        assert key is not None and "research" in key
    finally:
        db.close()


def test_resolve_async_fails_closed_without_checker(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )

    async def run():
        return await resolve_topic_session_key_async(
            _source(), db, app_id="hermes-agent"
        )

    try:
        assert asyncio.run(run()) is None
    finally:
        db.close()


def test_resolve_async_lock_released_before_response_decision(tmp_path):
    """Lock-window contract (Critic #1) — async resolver releases lock
    on return so callers can serialise the next operation without
    holding the lock through agent generation.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    set_registered_check(_ok_checker())
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    principal = PlatformPrincipal.from_source(_source(), app_id="hermes-agent")

    async def run():
        key = await resolve_topic_session_key_async(
            _source(), db, app_id="hermes-agent"
        )
        # After the resolver returns, the per-key lock must NOT still
        # be held. Acquiring it again from the same task must succeed
        # immediately.
        lock = await active_topic_module.acquire_pointer_lock(principal.key)
        assert not lock.locked(), "lock must be released before resolver returns"
        return key

    try:
        assert asyncio.run(run()) is not None
    finally:
        db.close()


# ── SessionStore._generate_session_key wiring ──────────────────────────


def test_session_store_uses_topic_pre_pass_when_pointer_present(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    set_registered_check(_ok_checker())
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    db.close()

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    # Point the store at the same SessionDB the pointer lives in.
    store._db = SessionDB(db_path=db_path)
    try:
        set_registered_check(_ok_checker())
        key = store._generate_session_key(_source())
        assert "topic:" in key, f"expected topic-routed key, got: {key}"
        assert key.endswith(":research")
    finally:
        store._db.close()


def test_session_store_falls_back_to_legacy_when_no_pointer(tmp_path):
    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=tmp_path / "state.db")
    try:
        set_registered_check(_ok_checker())
        key = store._generate_session_key(_source())
        # Same as build_session_key with no pre-pass.
        expected = build_session_key(_source())
        assert key == expected
        assert "topic:" not in key
    finally:
        store._db.close()


def test_session_store_falls_back_when_pointer_mode_disabled(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    db.close()

    cfg = _config(pointer_mode_enabled=False)
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=db_path)
    try:
        set_registered_check(_ok_checker())
        key = store._generate_session_key(_source())
        assert "topic:" not in key
    finally:
        store._db.close()


def test_session_store_falls_back_when_no_default_app_id(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="research",
        updated_by="x",
    )
    db.close()

    cfg = _config(default_app_id=None)
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=db_path)
    try:
        set_registered_check(_ok_checker())
        key = store._generate_session_key(_source())
        assert "topic:" not in key
    finally:
        store._db.close()


def test_session_store_falls_back_when_unregistered_switch_routed(tmp_path):
    """Pointer present but registry rejects → legacy key.

    Critic finding #5: no unregistered topic ever routes — a topic
    that fails registry at routing time degrades to legacy thread mode,
    NOT to the topic session.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    # Bypass the registry on write to seed an "orphaned" pointer.
    db.set_active_topic(
        platform="telegram",
        user_id="208214988",
        chat_id="208214988",
        app_id="hermes-agent",
        topic_id="orphan",
        updated_by="x",
    )
    db.close()

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=db_path)
    try:
        set_registered_check(_reject_checker())
        key = store._generate_session_key(_source())
        assert "topic:" not in key
    finally:
        store._db.close()


def test_two_topics_same_principal_route_to_different_session_keys(tmp_path):
    """Switching topics flips the routing — same principal, two keys.

    Charter T0 acceptance: same chat, two topics, two sessions.
    """
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    set_registered_check(_ok_checker())

    cfg = _config()
    cfg.sessions_dir = tmp_path / "sessions"
    store = SessionStore(cfg.sessions_dir, cfg)
    store._db = SessionDB(db_path=db_path)

    try:
        # Switch to topic A.
        db.set_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
            topic_id="topic-A",
            updated_by="x",
        )
        key_a = store._generate_session_key(_source())

        # Switch to topic B.
        db.set_active_topic(
            platform="telegram",
            user_id="208214988",
            chat_id="208214988",
            app_id="hermes-agent",
            topic_id="topic-B",
            updated_by="x",
        )
        key_b = store._generate_session_key(_source())

        assert key_a != key_b
        assert key_a.endswith(":topic-A")
        assert key_b.endswith(":topic-B")
    finally:
        store._db.close()
        db.close()


# ── Critic finding #2: concurrent move_turns idempotency ──────────────


def test_concurrent_move_turns_same_idempotency_key_replays_consistently(tmp_path):
    """Two threads racing on the same idempotency key produce one move.

    One write wins (PK constraint on move_log keeps it at-most-once)
    and the loser observes a byte-equal replay body. Neither raises
    and neither double-inserts into dst.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="src", source="telegram", user_id="u")
    db.create_session(session_id="dst", source="telegram", user_id="u")
    for i in range(5):
        db.append_message("src", "user", f"src-message-{i}")

    barrier = threading.Barrier(8)
    results: list = []
    errors: list = []

    def race():
        barrier.wait()
        try:
            resp = db.move_turns(
                src_session_id="src",
                dst_session_id="dst",
                range_spec="last:3",
                idempotency_key="race-key",
            )
            results.append(resp)
        except Exception as e:  # noqa: BLE001 — captured for assertion below
            errors.append(e)

    threads = [threading.Thread(target=race) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # No thread saw an error (PK conflict is handled by replay path).
    assert errors == [], f"unexpected errors: {errors}"
    assert len(results) == 8

    # Exactly one is the original commit; the rest are replays.
    primaries = [r for r in results if not r["replay"]]
    replays = [r for r in results if r["replay"]]
    assert len(primaries) == 1, f"expected 1 primary, saw {len(primaries)}"
    assert len(replays) == 7

    primary = primaries[0]
    # Every replay body equals the primary body (modulo the replay flag).
    for r in replays:
        rcopy = dict(r)
        pcopy = dict(primary)
        rcopy.pop("replay")
        pcopy.pop("replay")
        assert rcopy == pcopy

    # dst saw exactly 3 inserts — not 8 * 3.
    dst_total = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'dst'"
    ).fetchone()[0]
    assert dst_total == 3
    # move_log has one row.
    log_count = db._conn.execute(
        "SELECT COUNT(*) FROM move_log WHERE idempotency_key = 'race-key'"
    ).fetchone()[0]
    assert log_count == 1
    db.close()


# ── Critic finding #3: migration sweep with many sessions ─────────────


def test_hrm_t0a_migration_stamp_above_max_started_at_with_many_sessions(tmp_path):
    """Stamp must exceed max(sessions.started_at) across many pre-existing rows."""
    db = SessionDB(db_path=tmp_path / "state.db")
    # Seed 50 sessions with monotonically increasing started_at, including
    # a couple in the future to simulate clock-skew.
    base_ts = 1_700_000_000.0
    for i in range(50):
        sid = f"sess-{i:03d}"
        db.create_session(session_id=sid, source="telegram", user_id="u")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (base_ts + i, sid),
        )
    # Inject one clock-skewed future session.
    future_ts = 9_999_999_999.0
    db.create_session(session_id="skewed", source="telegram", user_id="u")
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (future_ts, "skewed"),
    )
    db._conn.commit()

    max_started = db._conn.execute(
        "SELECT MAX(started_at) FROM sessions"
    ).fetchone()[0]
    assert max_started == future_ts

    db.apply_hrm_t0a_migration()
    stamped = float(db.get_meta("hrm_t0a_applied_at"))
    # Strictly greater than the max started_at across the whole table.
    assert stamped > max_started, (
        f"stamp {stamped} must exceed max(started_at) {max_started}"
    )
    # No pre-existing session is classifiable as post-migration.
    overlaps = db._conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE started_at >= ?",
        (stamped,),
    ).fetchone()[0]
    assert overlaps == 0
    db.close()


# ── Critic finding #4: _resolve_move_range excludes tombstoned rows ───


def test_resolve_move_range_excludes_tombstoned_rows(tmp_path):
    """After a move, src tombstoned rows must not appear in a follow-up
    last:N resolution. The move primitive's soft-delete contract.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="src", source="telegram", user_id="u")
    db.create_session(session_id="dst", source="telegram", user_id="u")
    for i in range(5):
        db.append_message("src", "user", f"src-message-{i}")

    # Move the last 3 — tombstones rows in src.
    first = db.move_turns(
        src_session_id="src",
        dst_session_id="dst",
        range_spec="last:3",
        idempotency_key="k1",
    )
    moved_src = set(first["src_message_ids"])
    assert len(moved_src) == 3

    # Resolve last:5 on src now — should ONLY return the 2 active rows.
    with db._lock:
        remaining = db._resolve_move_range(db._conn, "src", "last:5")
    assert len(remaining) == 2, (
        f"expected 2 active rows after tombstoning 3, got {len(remaining)}"
    )
    assert moved_src.isdisjoint(set(remaining))

    # Likewise range:1..999 must skip tombstoned ids.
    with db._lock:
        remaining_range = db._resolve_move_range(db._conn, "src", "range:1..999")
    assert moved_src.isdisjoint(set(remaining_range))
    db.close()


def test_resolve_move_range_after_move_does_not_remove_already_moved(tmp_path):
    """A second move of last:N must NOT re-move tombstoned rows.

    Without the tombstone filter, the second move would re-pick the
    same rows and double-insert them into dst.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="src", source="telegram", user_id="u")
    db.create_session(session_id="dst", source="telegram", user_id="u")
    for i in range(4):
        db.append_message("src", "user", f"m-{i}")

    db.move_turns(
        src_session_id="src",
        dst_session_id="dst",
        range_spec="last:2",
        idempotency_key="m1",
    )
    second = db.move_turns(
        src_session_id="src",
        dst_session_id="dst",
        range_spec="last:2",
        idempotency_key="m2",
    )
    # The second move sees ONLY the remaining 2 active rows.
    assert len(second["src_message_ids"]) == 2
    # dst now has 4 total (2 from each move). If the tombstone filter
    # was missing, the second move would have grabbed already-moved
    # rows and the count would diverge.
    dst_total = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'dst'"
    ).fetchone()[0]
    assert dst_total == 4
    # src has zero active rows.
    src_active = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'src' AND active = 1"
    ).fetchone()[0]
    assert src_active == 0
    db.close()
