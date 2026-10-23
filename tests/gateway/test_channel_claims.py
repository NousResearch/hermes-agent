"""Channel admission claims (design/channel-claim-lock.md): the intra-host claim gate must
give one channel exactly one session — a DM thread reply joins the channel session, a peer
gateway's live claim stands down instead of double-admitting, a crashed holder's claim is
stolen after the lease, and a deleted session releases its claim."""
import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state_runtime import (
    CHANNEL_CLAIM_LEASE_SECONDS,
    RuntimeStoreError,
    acquire_channel_claim,
    get_channel_claim,
    renew_channel_claim,
    release_channel_claim,
    transfer_channel_claim,
)


def _store(tmp_path, monkeypatch, instance="gw-a"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._claim_instance_id = instance
    return store


def _dm(chat_id, thread_id=None):
    return SessionSource(
        platform=Platform.MATTERMOST, chat_id=chat_id, chat_type="dm",
        user_id="u1", chat_name="Chat", thread_id=thread_id,
    )


def _db(store, key):
    db = store._db_for_key(key)
    assert db is not None
    return db


def _claim_key(store, entry):
    origin = entry.origin
    return dict(
        platform=origin.platform.value, scope_id=origin.scope_id or "",
        chat_id=origin.chat_id, profile="main",
    )


def test_dm_thread_reply_joins_the_channel_session(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    parent = store.get_or_create_session(_dm("chat1"))
    assert list(store._entries) == [parent.session_key]

    reply = store.get_or_create_session(_dm("chat1", thread_id="t1"))
    assert reply is parent
    assert list(store._entries) == [parent.session_key]

    claim = get_channel_claim(_db(store, parent.session_key), **_claim_key(store, parent))
    assert claim["session_id"] == parent.session_id
    assert claim["thread_root_id"] == "t1"
    assert claim["lease_expire"] > 0


def test_peer_live_claim_routes_into_known_holder_no_second_session(tmp_path, monkeypatch, caplog):
    store = _store(tmp_path, monkeypatch)
    holder = store.get_or_create_session(_dm("chat2"))
    db = _db(store, holder.session_key)
    # Simulate another gateway process adopting the channel after a restart: it now owns a
    # live claim whose session this store still knows (shared state.db).
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE channel_claims SET instance_id='peer-gw'", ()))
    other_reply = store.get_or_create_session(_dm("chat2", thread_id="t9"))
    assert other_reply is holder
    assert len(store._entries) == 1
    claim = get_channel_claim(db, **_claim_key(store, holder))
    assert claim["instance_id"] == "peer-gw"


def test_expired_peer_claim_is_stolen(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    holder = store.get_or_create_session(_dm("chat3"))
    db = _db(store, holder.session_key)
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE channel_claims SET instance_id='peer-gw',"
            " lease_expire=lease_expire-%f" % (CHANNEL_CLAIM_LEASE_SECONDS + 10), ()))
    reply = store.get_or_create_session(_dm("chat3", thread_id="t2"))
    assert reply is holder
    claim = get_channel_claim(db, **_claim_key(store, holder))
    assert claim["instance_id"] == "gw-a"


def test_live_claim_with_unknown_session_refuses_to_admit(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    key_hint = store._generate_session_key(_dm("chat4"))
    db = store._db_for_key(key_hint)
    acquired, _ = acquire_channel_claim(
        db, platform="mattermost", scope_id="", chat_id="chat4", profile="main",
        instance_id="peer-gw", session_id="ghost")
    assert acquired
    with pytest.raises(RuntimeStoreError):
        store.get_or_create_session(_dm("chat4"))
    assert store._entries == {}


def test_deleted_session_releases_its_claim(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    entry = store.get_or_create_session(_dm("chat5"))
    db = _db(store, entry.session_key)
    assert get_channel_claim(db, **_claim_key(store, entry)) is not None

    assert store.remove_by_session_id(entry.session_id) == 1
    assert get_channel_claim(db, **_claim_key(store, entry)) is None


def test_reset_transition_transfers_and_end_releases(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    entry = store.get_or_create_session(_dm("chat6"))
    db = _db(store, entry.session_key)
    scope = _claim_key(store, entry)

    # /new-style transition: end + create moves the claim to the successor session.
    store._finish_channel_claim(
        entry.session_key, end_session_id=entry.session_id,
        create_kwargs={"session_id": "successor"}, origin=entry.origin)
    claim = get_channel_claim(db, **scope)
    assert claim["session_id"] == "successor"

    # Plain end (no successor) releases the channel.
    store._finish_channel_claim(
        entry.session_key, end_session_id="successor", create_kwargs=None,
        origin=entry.origin)
    assert get_channel_claim(db, **scope) is None


def test_claim_ops_acquire_renew_transfer_release(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    key_hint = store._generate_session_key(_dm("chat7"))
    db = store._db_for_key(key_hint)
    kwargs = dict(platform="mattermost", scope_id="", chat_id="chat7", profile="main")

    # Acquire-when-free, then a second acquire sees the holder as claimed=False.
    acquired, claim = acquire_channel_claim(db, instance_id="gw-a", session_id="", **kwargs)
    assert acquired and claim["session_id"] == ""
    acquired, held = acquire_channel_claim(db, instance_id="gw-b", session_id="b1", **kwargs)
    assert not acquired and held["instance_id"] == "gw-a"

    # Renew: right holder extends the lease and retags the thread root; a wrong holder no-ops.
    before = held["lease_expire"]
    assert renew_channel_claim(db, session_id="zzz", **kwargs) is False
    assert renew_channel_claim(db, session_id="", thread_root_id="t7", **kwargs) is True
    claim = get_channel_claim(db, **kwargs)
    assert claim["lease_expire"] >= before and claim["thread_root_id"] == "t7"

    # Transfer: the placeholder bind at session-create time, then a reset-style move.
    assert transfer_channel_claim(db, old_session_id="", new_session_id="s1", **kwargs) is True
    assert transfer_channel_claim(db, old_session_id="s1", new_session_id="s2", **kwargs) is True
    assert transfer_channel_claim(db, old_session_id="s1", new_session_id="s3", **kwargs) is False

    release_channel_claim(db, session_id="s2", **kwargs)
    assert get_channel_claim(db, **kwargs) is None


def test_broken_claim_store_admits_ungated(tmp_path, monkeypatch):
    """Fail-open: an unreadable claim store refuses no one — the bot is not muted."""
    import hermes_state_runtime

    def boom(*args, **kwargs):
        raise RuntimeError("no such table: channel_claims")

    monkeypatch.setattr(hermes_state_runtime, "acquire_channel_claim", boom)
    store = _store(tmp_path, monkeypatch)
    entry = store.get_or_create_session(_dm("chat8"))
    assert entry.origin is not None
    assert entry.origin.chat_id == "chat8"


def test_turn_runner_hook_renews_and_moves_the_live_claim(tmp_path, monkeypatch):
    """§7.1: each delivered turn pushes the lease back out to the full window; a
    compression split rebinds the claim to the lane's new session id."""
    import types

    from gateway.run_turn_runner import TurnRunner

    store = _store(tmp_path, monkeypatch)
    entry = store.get_or_create_session(_dm("chat9"))
    db = _db(store, entry.session_key)
    scope = _claim_key(store, entry)
    fresh = get_channel_claim(db, **scope)

    db._execute_write(lambda conn: conn.execute(
        "UPDATE channel_claims SET lease_expire = lease_expire - 60", ()))
    holder = types.SimpleNamespace(
        _ctx=types.SimpleNamespace(session_key=entry.session_key, session_id=entry.session_id),
        _runner=types.SimpleNamespace(session_store=store))
    TurnRunner._renew_channel_claim_after_run(holder, entry.session_id)
    renewed = get_channel_claim(db, **scope)
    assert renewed is not None
    assert renewed["lease_expire"] >= fresh["lease_expire"] - 1

    TurnRunner._renew_channel_claim_after_run(holder, "split-successor")
    moved = get_channel_claim(db, **scope)
    assert moved is not None
    assert moved["session_id"] == "split-successor"


def test_retired_route_heals_the_stale_self_claim_before_lease(tmp_path, monkeypatch):
    """Bulk delete (retire_live_sessions) fences the route but leaves the claim: the next
    admission must heal it onto its lane in one CAS write, long before lease expiry."""
    store = _store(tmp_path, monkeypatch)
    entry = store.get_or_create_session(_dm("chat-heal"))
    scope = _claim_key(store, entry)
    db = _db(store, entry.session_key)

    store._entries.pop(entry.session_key)  # the route vanished; the claim row stays

    fresh = store.get_or_create_session(_dm("chat-heal"))
    healed = get_channel_claim(db, **scope)
    assert healed is not None
    assert healed["session_id"] == fresh.session_id  # moved onto the live lane, not left stale


@pytest.mark.asyncio
async def test_admit_boundary_silences_claim_standdowns():
    """The boundary contract for the gate's stand-downs: silent, exactly like stale_generation —
    a chat answer here would give the channel the second voice the lock exists to prevent."""
    import types

    from gateway.session_ingress import admit_message

    for reason in ("channel_claim_inflight", "channel_claimed_elsewhere"):
        async def _raising(event, _reason=reason):
            raise RuntimeStoreError(_reason)
        assert await admit_message(types.SimpleNamespace(admit_native=_raising), None) is None
