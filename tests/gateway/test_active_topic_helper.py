"""HRM-T0a gateway/active_topic.py helper module.

Step 3 of the implementation plan
(work/hrm-t0a-hermes-topic-implementation-plan.md):

- :class:`PlatformPrincipal` constructs from a ``SessionSource``-shaped
  object, requires ``app_id``, and is hashable / frozen.
- The per-key asyncio lock dict is bounded (TTL idle eviction + LRU
  hard cap) and never drops a held lock.
- The ``assert_registered`` adapter fails closed when no checker is
  wired, calls the wired checker when present, and re-wraps the
  checker's exception as :class:`TopicNotRegisteredError`.
- The thin read/set/clear helpers round-trip against a real SessionDB.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.active_topic import (
    PlatformPrincipal,
    PlatformPrincipalLeak,
    TopicNotRegisteredError,
    acquire_pointer_lock,
    assert_principal_match,
    assert_registered,
    clear_active_topic,
    read_active_topic,
    set_active_topic,
    set_registered_check,
)
from gateway.session import SessionSource
from gateway.config import Platform
from hermes_state import SessionDB
import gateway.active_topic as active_topic_module


# ── PlatformPrincipal envelope ────────────────────────────────────────


def test_platform_principal_from_real_session_source():
    src = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
    )
    p = PlatformPrincipal.from_source(src, app_id="hermes-agent")
    assert p.platform == "telegram"
    assert p.chat_id == "208214988"
    assert p.user_id == "208214988"
    assert p.app_id == "hermes-agent"
    assert p.key == ("telegram", "208214988", "208214988", "hermes-agent")


def test_platform_principal_from_duck_typed_object():
    src = SimpleNamespace(platform="discord", user_id="u1", chat_id="c1")
    p = PlatformPrincipal.from_source(src, app_id="my-repo")
    assert p.platform == "discord"
    assert p.app_id == "my-repo"


def test_platform_principal_requires_complete_envelope():
    src = SimpleNamespace(platform="telegram", user_id="u", chat_id="c")
    with pytest.raises(ValueError, match="app_id"):
        PlatformPrincipal.from_source(src, app_id="")
    src2 = SimpleNamespace(platform="telegram", user_id=None, chat_id="c")
    with pytest.raises(ValueError, match="user_id"):
        PlatformPrincipal.from_source(src2, app_id="x")
    src3 = SimpleNamespace(platform="telegram", user_id="u", chat_id=None)
    with pytest.raises(ValueError, match="chat_id"):
        PlatformPrincipal.from_source(src3, app_id="x")
    src4 = SimpleNamespace(platform=None, user_id="u", chat_id="c")
    with pytest.raises(ValueError, match="platform"):
        PlatformPrincipal.from_source(src4, app_id="x")


def test_platform_principal_is_frozen_and_hashable():
    p = PlatformPrincipal("telegram", "u", "c", "app")
    with pytest.raises(Exception):
        p.platform = "discord"  # frozen dataclass
    assert {p, PlatformPrincipal("telegram", "u", "c", "app")} == {p}


def test_assert_principal_match_raises_leak_on_mismatch():
    a = PlatformPrincipal("telegram", "u", "c", "app")
    b = PlatformPrincipal("telegram", "u", "c", "other-app")
    with pytest.raises(PlatformPrincipalLeak) as excinfo:
        assert_principal_match(expected=a, actual=b)
    assert excinfo.value.expected == a
    assert excinfo.value.actual == b


# ── Per-key asyncio lock primitive ────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_locks_between_tests():
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)
    yield
    active_topic_module._reset_locks_for_tests()
    set_registered_check(None)


def test_acquire_pointer_lock_is_per_key_singleton():
    async def run():
        k1 = ("telegram", "u", "c", "app")
        k2 = ("telegram", "u", "c", "other-app")
        a = await acquire_pointer_lock(k1)
        b = await acquire_pointer_lock(k1)
        c = await acquire_pointer_lock(k2)
        assert a is b
        assert a is not c

    asyncio.run(run())


def test_acquire_pointer_lock_evicts_idle_entries(monkeypatch):
    async def run():
        # Shrink the TTL so the test is fast.
        monkeypatch.setattr(active_topic_module, "_LOCK_TTL_SECONDS", 0.0)
        k1 = ("telegram", "u", "c1", "app")
        k2 = ("telegram", "u", "c2", "app")
        first = await acquire_pointer_lock(k1)
        # Touching another key should evict the idle first one because
        # TTL=0 makes every untouched entry stale immediately.
        await acquire_pointer_lock(k2)
        assert k1 not in active_topic_module._LOCKS
        # Re-acquiring creates a fresh Lock; identity differs.
        re = await acquire_pointer_lock(k1)
        assert re is not first

    asyncio.run(run())


def test_acquire_pointer_lock_never_evicts_held_lock(monkeypatch):
    async def run():
        monkeypatch.setattr(active_topic_module, "_LOCK_TTL_SECONDS", 0.0)
        held_key = ("telegram", "u", "held", "app")
        held = await acquire_pointer_lock(held_key)
        await held.acquire()
        try:
            # Trigger eviction sweep — held lock must survive.
            await acquire_pointer_lock(("telegram", "u", "other", "app"))
            assert held_key in active_topic_module._LOCKS
        finally:
            held.release()

    asyncio.run(run())


def test_acquire_pointer_lock_enforces_size_cap(monkeypatch):
    async def run():
        monkeypatch.setattr(active_topic_module, "_LOCK_MAX_ENTRIES", 3)
        # TTL infinity so the cap (not the TTL) is what drives eviction.
        monkeypatch.setattr(active_topic_module, "_LOCK_TTL_SECONDS", 1e9)
        for i in range(6):
            await acquire_pointer_lock(("telegram", "u", f"chat-{i}", "app"))
        assert len(active_topic_module._LOCKS) <= 3

    asyncio.run(run())


# ── assert_registered adapter ─────────────────────────────────────────


def test_assert_registered_fails_closed_when_no_checker_wired():
    async def run():
        with pytest.raises(TopicNotRegisteredError, match="no registry checker"):
            await assert_registered("hermes-agent", "research")

    asyncio.run(run())


def test_assert_registered_calls_wired_checker():
    seen = []

    async def checker(app_id, topic_id):
        seen.append((app_id, topic_id))
        return True

    async def run():
        set_registered_check(checker)
        await assert_registered("app", "t")

    asyncio.run(run())
    assert seen == [("app", "t")]


def test_assert_registered_raises_when_checker_returns_false():
    async def checker(app_id, topic_id):
        return False

    async def run():
        set_registered_check(checker)
        with pytest.raises(TopicNotRegisteredError, match="not registered"):
            await assert_registered("app", "t")

    asyncio.run(run())


def test_assert_registered_wraps_checker_exception():
    async def checker(app_id, topic_id):
        raise RuntimeError("registry down")

    async def run():
        set_registered_check(checker)
        with pytest.raises(TopicNotRegisteredError, match="registry check raised"):
            await assert_registered("app", "t")

    asyncio.run(run())


# ── Thin helpers wrapping SessionDB ────────────────────────────────────


def _principal() -> PlatformPrincipal:
    return PlatformPrincipal("telegram", "208214988", "208214988", "hermes-agent")


def test_read_active_topic_helper_returns_none_when_unset(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        result = asyncio.run(read_active_topic(db, _principal()))
        assert result is None
    finally:
        db.close()


def test_set_then_read_then_clear_helper(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    async def checker(app_id, topic_id):
        return True

    async def run():
        set_registered_check(checker)
        res = await set_active_topic(
            db,
            _principal(),
            topic_id="research",
            updated_by="slash:/topic switch",
        )
        assert res["prior"] is None
        assert res["current"]["topic_id"] == "research"

        snap = await read_active_topic(db, _principal())
        assert snap is not None and snap["topic_id"] == "research"

        prior = await clear_active_topic(
            db, _principal(), updated_by="slash:/topic clear"
        )
        assert prior["topic_id"] == "research"
        assert await read_active_topic(db, _principal()) is None

    try:
        asyncio.run(run())
    finally:
        db.close()


def test_set_active_topic_helper_refuses_when_unregistered(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")

    async def checker(app_id, topic_id):
        return False

    async def run():
        set_registered_check(checker)
        with pytest.raises(TopicNotRegisteredError):
            await set_active_topic(
                db,
                _principal(),
                topic_id="research",
                updated_by="slash:/topic switch",
            )
        # State is byte-equal — the SessionDB row was never written.
        assert await read_active_topic(db, _principal()) is None

    try:
        asyncio.run(run())
    finally:
        db.close()


def test_set_active_topic_helper_can_skip_registration_for_recovery(tmp_path):
    """``require_registered=False`` is the compensating-rollback path.

    When a confirmation banner fails to emit, the slash handler must
    be able to restore the prior pointer without re-running the
    registry check (which could fail again). This switch exists for
    that path only — production slash code must NEVER call with
    ``require_registered=False`` for a forward switch.
    """
    db = SessionDB(db_path=tmp_path / "state.db")

    async def run():
        # No checker wired — would normally fail closed.
        res = await set_active_topic(
            db,
            _principal(),
            topic_id="prior-topic",
            updated_by="slash:/topic switch (rollback)",
            require_registered=False,
        )
        assert res["current"]["topic_id"] == "prior-topic"

    try:
        asyncio.run(run())
    finally:
        db.close()
