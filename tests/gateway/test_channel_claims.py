"""Channel admission claims (design/channel-claim-lock.md): the intra-host claim gate must
give one channel exactly one session — a DM thread reply joins the channel session, a peer
gateway's live claim stands down instead of double-admitting, a crashed holder's claim is
stolen after the lease, and a deleted session releases its claim. The gateway.claim_lock
config surface (t_e4138b0a) rolls the gate out: monitor mode (the fail-open default) logs
each would-be stand-down instead of acting on it, and enabled=false turns the gate off."""
import pytest

from gateway.config import ClaimLockConfig, GatewayConfig, Platform
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


def _store(tmp_path, monkeypatch, instance="gw-a", enabled=True, mode="enforce"):
    """Enforce is the default here so the stand-down contract tests keep exercising the
    acting gate; the rollout-surface tests pass ``mode="monitor"`` / ``enabled=False``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(
        tmp_path / "sessions",
        GatewayConfig(claim_lock=ClaimLockConfig(enabled=enabled, mode=mode)))
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


# ---------------------------------------------------------------------------
# gateway.claim_lock rollout surface (t_e4138b0a): monitor (fail-open default)
# and enabled=false (kill switch). Enforce stays the default in ``_store`` so the
# stand-down contract tests above keep pinning the acting gate.


def _group(chat_id, user_id, thread_id=None):
    return SessionSource(
        platform=Platform.MATTERMOST, chat_id=chat_id, chat_type="group",
        user_id=user_id, chat_name="Chat", thread_id=thread_id,
    )


def test_monitor_admits_where_enforce_would_refuse(tmp_path, monkeypatch, caplog):
    """Monitor mode: a live claim this store cannot serve logs the would-be refusal
    (channel_claimed_elsewhere) and admits on the incoming key — the fail-open rollout
    phase, so upstream merges change no behavior. The ghost peer keeps its claim."""
    store = _store(tmp_path, monkeypatch, mode="monitor")
    key_hint = store._generate_session_key(_dm("chat-mo1"))
    db = store._db_for_key(key_hint)
    acquired, _ = acquire_channel_claim(
        db, platform="mattermost", scope_id="", chat_id="chat-mo1", profile="main",
        instance_id="peer-gw", session_id="ghost")
    assert acquired
    with caplog.at_level("WARNING"):
        entry = store.get_or_create_session(_dm("chat-mo1"))
    assert entry.origin.chat_id == "chat-mo1"
    assert "Channel claim monitor" in caplog.text
    assert "channel_claimed_elsewhere" in caplog.text
    claim = get_channel_claim(
        db, platform="mattermost", scope_id="", chat_id="chat-mo1", profile="main")
    assert claim["instance_id"] == "peer-gw"  # the claim store stays truthful


def test_monitor_admits_a_second_lane_where_enforce_would_join(tmp_path, monkeypatch, caplog):
    """Monitor mode, §6 case C: a live claim on another local lane logs the would-be
    stand-down and admits the incoming key — the route-into-holder move is deferred to
    the enforce phase, and the live lane keeps its claim untouched."""
    store = _store(tmp_path, monkeypatch, mode="monitor")
    holder = store.get_or_create_session(_group("chat-mo2", user_id="u1"))
    with caplog.at_level("WARNING"):
        other = store.get_or_create_session(_group("chat-mo2", user_id="u2"))
    assert other.session_id != holder.session_id  # monitor minted the second lane
    assert "would route this event into" in caplog.text
    claim = get_channel_claim(_db(store, holder.session_key), **_claim_key(store, holder))
    assert claim["session_id"] == holder.session_id  # the holder lane keeps the claim


def test_monitor_keeps_the_claim_store_maintained(tmp_path, monkeypatch, caplog):
    """Monitor is observation only (release §13): acquisitions still run so the
    store stays warm for enforce, but the §6.1 fold does not act — a thread reply
    admits with pre-claim keying and logs the fold enforce would have made."""
    store = _store(tmp_path, monkeypatch, mode="monitor")
    parent = store.get_or_create_session(_dm("chat-mo3"))
    with caplog.at_level("WARNING"):
        reply = store.get_or_create_session(_dm("chat-mo3", thread_id="t1"))
    assert reply is not parent  # monitor does not fold: zero routing delta
    assert reply.session_key != parent.session_key
    assert "would fold this thread reply" in caplog.text
    claim = get_channel_claim(_db(store, parent.session_key), **_claim_key(store, parent))
    assert claim is not None
    assert claim["session_id"] == parent.session_id  # the holder keeps its claim


def test_disabled_gate_is_fully_off(tmp_path, monkeypatch):
    """claim_lock.enabled=false is the kill switch: no fold, no acquire, no stand-down —
    exactly the pre-claim-lock behavior, even past a live foreign claim."""
    store = _store(tmp_path, monkeypatch, enabled=False)
    key_hint = store._generate_session_key(_dm("chat-off1"))
    db = store._db_for_key(key_hint)
    scope = dict(platform="mattermost", scope_id="", chat_id="chat-off1", profile="main")
    assert acquire_channel_claim(db, instance_id="peer-gw", session_id="ghost", **scope)[0]
    first = store.get_or_create_session(_dm("chat-off1"))
    second = store.get_or_create_session(_dm("chat-off1", thread_id="t2"))
    assert second.session_id != first.session_id  # no fold: pre-claim-lock keying
    assert len(store._entries) == 2
    assert get_channel_claim(db, **scope)["session_id"] == "ghost"  # walked past


def test_disabled_gate_skips_finish_and_renewal(tmp_path, monkeypatch):
    """enabled=false also fences the bookkeeping: _finish_channel_claim moves no relic
    claim and the TurnRunner renewal extends no lease (nothing was acquired)."""
    import types

    from gateway.run_turn_runner import TurnRunner

    store = _store(tmp_path, monkeypatch, enabled=False)
    entry = store.get_or_create_session(_dm("chat-off2"))
    scope = _claim_key(store, entry)
    db = _db(store, entry.session_key)
    assert acquire_channel_claim(db, instance_id="peer-gw", session_id="relic", **scope)[0]

    store._finish_channel_claim(
        entry.session_key, end_session_id="", create_kwargs={"session_id": "minted"},
        origin=entry.origin)
    assert get_channel_claim(db, **scope)["session_id"] == "relic"  # nothing moved

    db._execute_write(lambda conn: conn.execute(
        "UPDATE channel_claims SET lease_expire = lease_expire - 60", ()))
    before = get_channel_claim(db, **scope)["lease_expire"]
    holder = types.SimpleNamespace(
        _ctx=types.SimpleNamespace(session_key=entry.session_key, session_id=entry.session_id),
        _runner=types.SimpleNamespace(session_store=store))
    TurnRunner._renew_channel_claim_after_run(holder, entry.session_id)
    assert get_channel_claim(db, **scope)["lease_expire"] == before  # lease never moved


def test_enforce_still_stands_down_by_default_in_this_file(tmp_path, monkeypatch):
    """The rollout did not silently soften the acting gate: with the helper's enforce
    default, an unservable live claim still refuses to double-admit."""
    store = _store(tmp_path, monkeypatch)  # enforce is the helper default
    key_hint = store._generate_session_key(_dm("chat-enf1"))
    db = store._db_for_key(key_hint)
    acquired, _ = acquire_channel_claim(
        db, platform="mattermost", scope_id="", chat_id="chat-enf1", profile="main",
        instance_id="peer-gw", session_id="ghost")
    assert acquired
    with pytest.raises(RuntimeStoreError):
        store.get_or_create_session(_dm("chat-enf1"))
    assert store._entries == {}


# ── Cross-instance relay gate under gateway.claim_lock (release union) ────────


def _relay_gated_store(tmp_path, monkeypatch, instance="gw-ws", enabled=True, mode="enforce"):
    """A store whose relay slice is enrolled and pointed at a fresh routing root."""
    store = _store(tmp_path, monkeypatch, instance=instance, enabled=enabled, mode=mode)
    relay_root = tmp_path / "relay-root"
    relay_root.mkdir()
    store._routing_home = relay_root
    monkeypatch.setattr("gateway.relay.relay_instance_id", lambda: "inst-" + instance)
    return store, relay_root


def test_relay_gate_disabled_acquires_nothing(tmp_path, monkeypatch):
    """claim_lock.enabled=false: the relay slice writes no claim and blocks no one."""
    store, relay_root = _relay_gated_store(tmp_path, monkeypatch, enabled=False)
    entry = store.get_or_create_session(_dm("chat-relay1"))
    assert entry.origin.chat_id == "chat-relay1"
    claims = relay_root / "bot_relay" / "channel_claims"
    assert [p for p in claims.rglob("*") if p.is_file()] == []


def test_relay_gate_monitor_admits_behind_a_foreign_instance(tmp_path, monkeypatch, caplog):
    """Monitor mode: a live foreign relay claim logs the would-be stand-down and admits."""
    from tools.bot_relay import _normalize_profile_for_claim, acquire_relay_claim

    store, relay_root = _relay_gated_store(tmp_path, monkeypatch, mode="monitor")
    acquired, _ = acquire_relay_claim(
        relay_root, "mattermost", "", "chat-relay2",
        _normalize_profile_for_claim("main"), "inst-mac", "sid-mac")
    assert acquired
    with caplog.at_level("WARNING"):
        entry = store.get_or_create_session(_dm("chat-relay2"))
    assert entry.origin.chat_id == "chat-relay2"
    assert "Channel claim monitor" in caplog.text
    assert "inst-mac" in caplog.text


def test_relay_gate_enforce_stands_down_to_the_foreign_instance(tmp_path, monkeypatch):
    """Enforce: a live foreign relay claim refuses the double-admit (ChannelClaimBlocked)."""
    from datetime import UTC, datetime

    from gateway.session import ChannelClaimBlocked
    from tools.bot_relay import _normalize_profile_for_claim, acquire_relay_claim

    store, relay_root = _relay_gated_store(tmp_path, monkeypatch)
    acquired, _ = acquire_relay_claim(
        relay_root, "mattermost", "", "chat-relay3",
        _normalize_profile_for_claim("main"), "inst-mac", "sid-mac")
    assert acquired
    source = _dm("chat-relay3")
    with pytest.raises(ChannelClaimBlocked):
        store._check_relay_claim_gate(
            source, store._generate_session_key(source), "sid-cand",
            datetime.now(UTC))


def test_relay_standdown_releases_the_sqlite_placeholder(tmp_path, monkeypatch):
    """Pass-1 Major 1: the SQLite gate minted a claim placeholder (session_id="")
    during the admission that the relay gate then refused. The stand-down must
    unbind it, or the next admission is refused with channel_claim_inflight for
    the rest of the lease."""
    import time

    from gateway.session import ChannelClaimBlocked
    from tools.bot_relay import _normalize_profile_for_claim, acquire_relay_claim

    store, relay_root = _relay_gated_store(tmp_path, monkeypatch)
    acquired, _ = acquire_relay_claim(
        relay_root, "mattermost", "", "chat-orf1",
        _normalize_profile_for_claim("main"), "inst-mac", "sid-mac", lease_seconds=1)
    assert acquired
    with pytest.raises(ChannelClaimBlocked):
        store.get_or_create_session(_dm("chat-orf1"))
    time.sleep(1.1)  # health: allow HX013 -- lease expiry IS the behavior under test: the claim was minted with a 1 s lease and the steal must not happen early
    entry = store.get_or_create_session(_dm("chat-orf1"))
    assert entry is not None
    assert entry.origin.chat_id == "chat-orf1"
