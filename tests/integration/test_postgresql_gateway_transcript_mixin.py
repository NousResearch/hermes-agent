"""Real PostgreSQL contract for the SessionTranscriptMixin PG branches (L2b).

Exercises append/load/rewrite/rewind/tail-role/input-owner over a real per-test
schema with the SQLite resolvers bypassed (``_postgresql_state_store`` patched
to return the real store), and asserts zero state.db opens for the surface.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from state_store import MessageRecord
from state_store_runtime_readiness import trap_state_db_opens

pytestmark = pytest.mark.integration

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"


@pytest.fixture
def pg_mixin_home(tmp_path, monkeypatch, postgresql_test_target):
    home = tmp_path / ".hermes-dev-postgresql-transcript-mixin"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: HERMES_STATE_STORE_TEST_DSN\n"
        "    connect_timeout_seconds: 5\n    pool_max_size: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def mixin(pg_mixin_home, postgresql_test_target, monkeypatch):
    """A bare SessionStore whose PG resolver returns the real per-test store.

    The resolver is patched at the class level because the mixin methods reach
    it through ``self._postgresql_state_store()``; no config resolution or
    legacy SQLite handle is consulted on these paths.
    """
    from gateway.session import SessionStore
    from state_store import PostgreSQLStateStoreConfig
    from state_store_postgresql import PostgreSQLStateStore

    settings = PostgreSQLStateStoreConfig(
        dsn_env="HERMES_STATE_STORE_TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)
    store = PostgreSQLStateStore(settings, postgresql_test_target.dsn, schema=postgresql_test_target.schema)
    shell = object.__new__(SessionStore)
    monkeypatch.setattr(SessionStore, "_postgresql_state_store", lambda self: store)
    try:
        yield shell, store
    finally:
        store.close()


def _sid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _seed_two_user_turns(store, session_id: str) -> None:
    store.ensure_session(session_id, source="gateway")
    store.append_message_record(session_id, MessageRecord(
        role="user", content="first question", platform_message_id="plat-1", observed=True))
    store.append_message_record(session_id, MessageRecord(
        role="assistant", content="first answer", finish_reason="stop", reasoning="thinking"))
    store.append_message_record(session_id, MessageRecord(
        role="user", content="second question", platform_message_id="plat-2", observed=True))
    store.append_message_record(session_id, MessageRecord(
        role="assistant", content="second answer", finish_reason="stop"))


def test_append_load_roundtrip_repairs_alternation(mixin, pg_mixin_home):
    shell, store = mixin
    session_id = _sid("pgmix_append_load")
    with trap_state_db_opens(pg_mixin_home) as opens:
        store.ensure_session(session_id, source="gateway")
        shell.append_to_transcript(session_id, {"role": "user", "content": "hello pg",
                                                "platform_message_id": "pm-1", "observed": True})
        # Durable user;user wedge must be healed on load (live-replay invariant).
        store.append_message_record(session_id, MessageRecord(role="user", content="wedge"))
        loaded = shell.load_transcript(session_id)
    # Consecutive user rows are merged by the same repair the SQLite read applies
    # (merge consecutive user turns) — one user message survives the wedge.
    assert [m["role"] for m in loaded] == ["user"]
    assert "hello pg" in loaded[0]["content"] and "wedge" in loaded[0]["content"]
    assert loaded[0]["message_id"] == "pm-1"
    assert opens == []


def test_append_to_transcript_skip_db_writes_nothing(mixin, pg_mixin_home):
    shell, store = mixin
    session_id = _sid("pgmix_skip_db")
    with trap_state_db_opens(pg_mixin_home) as opens:
        store.ensure_session(session_id, source="gateway")
        shell.append_to_transcript(session_id, {"role": "user", "content": "skip me"}, skip_db=True)
        assert shell.load_transcript(session_id) == []
    assert opens == []


def test_rewrite_transcript_destructive(mixin, pg_mixin_home):
    shell, store = mixin
    session_id = _sid("pgmix_rewrite")
    _seed_two_user_turns(store, session_id)
    with trap_state_db_opens(pg_mixin_home) as opens:
        assert shell.rewrite_transcript(session_id, [{"role": "user", "content": "rewritten"}]) is True
        loaded = shell.load_transcript(session_id)
    assert [(m["role"], m["content"]) for m in loaded] == [("user", "rewritten")]
    assert opens == []


def test_rewind_session_shrinks_active_ids_and_returns_outcome(mixin, pg_mixin_home):
    shell, store = mixin
    session_id = _sid("pgmix_rewind")
    _seed_two_user_turns(store, session_id)
    before = set(store.get_active_message_ids(session_id))
    with trap_state_db_opens(pg_mixin_home) as opens:
        outcome = shell.rewind_session(session_id, 1)
        after = set(store.get_active_message_ids(session_id))
    assert outcome is not None
    assert outcome["turns_undone"] == 1
    assert outcome["target_text"] == "second question"
    assert after < before
    remaining = store.get_messages_as_conversation(session_id)
    assert [m["content"] for m in remaining if m["role"] == "user"] == ["first question"]
    assert opens == []


def test_has_platform_message_id_and_tail_role(mixin, pg_mixin_home):
    shell, store = mixin
    session_id = _sid("pgmix_probes")
    _seed_two_user_turns(store, session_id)
    with trap_state_db_opens(pg_mixin_home) as opens:
        assert shell.has_platform_message_id(session_id, "plat-2") is True
        assert shell.has_platform_message_id(session_id, "plat-missing") is False
        assert shell.transcript_tail_role(session_id) == "assistant"
    assert opens == []


def test_has_input_owner_finds_marker(mixin, pg_mixin_home):
    shell, store = mixin
    session_id = _sid("pgmix_owner")
    _seed_two_user_turns(store, session_id)
    store.append_message_record(session_id, MessageRecord(
        role="user", content="owned input", observed=False,
        display_metadata={"gateway_input_owner": "turn-holder-1"}))
    with trap_state_db_opens(pg_mixin_home) as opens:
        assert shell.has_input_owner(session_id, "turn-holder-1") is True
        assert shell.has_input_owner(session_id, "other-holder") is False
    assert opens == []


def test_advance_compression_session_cas_over_route(mixin, pg_mixin_home):
    shell, store = mixin
    route_store = shell._postgresql_route_store()
    key = "agent:main:telegram:pgmix_advance"
    session_id, child = _sid("pgmix_adv"), _sid("pgmix_adv_child")
    meta = {"session_key": key, "session_id": session_id,
            "created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00",
            "source": "gateway"}
    route_store.get_or_create_route(session_key=key, session_id=session_id, metadata=meta)
    with trap_state_db_opens(pg_mixin_home) as opens:
        # Stale expectation (route sits elsewhere): CAS must fail closed.
        assert shell.advance_compression_session(key, "moved-away", child) is None
        advanced = shell.advance_compression_session(key, session_id, child)
    assert advanced is not None
    assert advanced.session_id == child
    assert route_store.lookup_by_key(key).session_id == child
    # Advancing to the already-current tip is idempotent.
    assert shell.advance_compression_session(key, session_id, child).session_id == child
    assert opens == []


@pytest.mark.parametrize("tenant_schema", (None, "", "   "))
def test_route_store_rejects_missing_or_empty_resolver_tenant_schema(monkeypatch, tenant_schema):
    from gateway.session import SessionStore

    shell = object.__new__(SessionStore)
    monkeypatch.setattr(
        SessionStore, "_postgresql_state_store", lambda _self: SimpleNamespace(tenant_schema=tenant_schema),
    )

    with pytest.raises(RuntimeError, match="resolver-derived tenant namespace"):
        shell._postgresql_route_store()
