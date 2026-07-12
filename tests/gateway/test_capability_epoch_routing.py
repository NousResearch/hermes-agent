from __future__ import annotations

import hashlib
import threading
from datetime import timedelta

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import (
    CapabilityEpochRotationBlocked,
    SessionEntry,
    SessionSource,
    SessionStore,
    _now,
    build_session_context,
    build_session_context_prompt,
)


def _store(tmp_path, monkeypatch, *, config=None, rotation_hook=None):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(
        sessions_dir=tmp_path / "routing",
        config=config or GatewayConfig(),
        before_capability_epoch_rotation_fn=rotation_hook,
    )
    # These tests exercise only the persisted routing index. Recovery from the
    # transcript database is covered separately below.
    store._db = None
    return store


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-1",
        thread_id="thread-1",
        chat_type="thread",
        user_id="owner-1",
    )


def test_gateway_restart_rotates_epoch_while_preserving_routing_continuity(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path, monkeypatch)
    entry = store.get_or_create_session(_source())
    original_epoch = entry.capability_epoch

    # Routing/transcript continuity survives restart, but mutation authority
    # intentionally does not. Otherwise a same-UID child could restore an old
    # routing snapshot, restart the gateway, and resurrect an old grant.
    entry.session_id = "compressed-child"
    store._save_entries()
    persisted = (tmp_path / "routing" / "sessions.json").read_text()

    restarted = _store(tmp_path, monkeypatch)
    recovered = restarted.get_or_create_session(_source())

    assert recovered.session_id == "compressed-child"
    assert recovered.capability_epoch != original_epoch
    assert original_epoch not in persisted
    assert "capability_epoch" not in persisted
    assert original_epoch.startswith("cap_epoch_v1_")
    assert len(original_epoch) == len("cap_epoch_v1_") + 64


def test_switch_to_db_reported_compression_tip_still_rotates_epoch(
    tmp_path,
    monkeypatch,
):
    calls = []
    store = _store(
        tmp_path,
        monkeypatch,
        rotation_hook=lambda entry, reason: calls.append((entry, reason)),
    )
    entry = store.get_or_create_session(_source())

    # state.db is same-UID writable. A model child can forge this ancestry, so
    # switch_session must not consult it to preserve process-memory authority.
    class _TamperedDB:
        compression_tip_lookups = 0

        def get_compression_tip(self, _session_id):
            self.compression_tip_lookups += 1
            return "compressed-child"

        @staticmethod
        def end_session(_session_id, _reason):
            return None

        @staticmethod
        def reopen_session(_session_id):
            return None

    tampered_db = _TamperedDB()
    store._db = tampered_db
    monkeypatch.setattr(store, "_record_gateway_session_peer", lambda *_a, **_k: None)

    continued = store.switch_session(
        entry.session_key,
        "compressed-child",
    )

    assert continued is not None
    assert continued.capability_epoch != entry.capability_epoch
    assert calls == [(entry, "explicit_switch")]
    assert tampered_db.compression_tip_lookups == 0


def test_legacy_persisted_epoch_is_ignored():
    now = _now()
    legacy = "cap_epoch_v1_" + "a" * 64
    entry = SessionEntry.from_dict(
        {
            "session_key": "agent:main:discord:thread:channel-1:thread-1",
            "session_id": "session-1",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "capability_epoch": legacy,
        }
    )

    assert entry.capability_epoch != legacy
    assert "capability_epoch" not in entry.to_dict()


def test_all_explicit_routing_boundaries_rotate_capability_epoch(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path, monkeypatch)
    first = store.get_or_create_session(_source())

    reset = store.reset_session(first.session_key)
    assert reset is not None
    assert reset.capability_epoch != first.capability_epoch

    switched = store.switch_session(reset.session_key, "resumed-session")
    assert switched is not None
    assert switched.capability_epoch != reset.capability_epoch

    forced = store.get_or_create_session(_source(), force_new=True)
    assert forced.capability_epoch != switched.capability_epoch


def test_auto_reset_rotates_capability_epoch(tmp_path, monkeypatch):
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=1)
    )
    store = _store(tmp_path, monkeypatch, config=config)
    first = store.get_or_create_session(_source())
    first.updated_at = _now() - timedelta(minutes=5)
    store._save_entries()

    reset = store.get_or_create_session(_source())

    assert reset.was_auto_reset is True
    assert reset.capability_epoch != first.capability_epoch


def test_epoch_rotation_hook_runs_unlocked_before_every_explicit_publish(
    tmp_path,
    monkeypatch,
):
    observed = []
    store = None

    def _hook(entry, reason):
        assert store is not None
        acquired = store._lock.acquire(blocking=False)
        assert acquired is True
        store._lock.release()
        assert store._entries[entry.session_key] is entry
        observed.append((reason, entry.session_id, entry.capability_epoch))

    store = _store(tmp_path, monkeypatch, rotation_hook=_hook)
    first = store.get_or_create_session(_source())
    reset = store.reset_session(first.session_key)
    assert reset is not None
    switched = store.switch_session(reset.session_key, "resumed-session")
    assert switched is not None
    forced = store.get_or_create_session(_source(), force_new=True)

    assert [item[0] for item in observed] == [
        "explicit_reset",
        "explicit_switch",
        "force_new",
    ]
    assert observed[0][2] == first.capability_epoch
    assert observed[1][2] == reset.capability_epoch
    assert observed[2][2] == switched.capability_epoch
    assert forced.capability_epoch != switched.capability_epoch


def test_writer_outage_blocks_explicit_rotation_without_changing_epoch(
    tmp_path,
    monkeypatch,
):
    def _blocked(_entry, _reason):
        raise CapabilityEpochRotationBlocked("writer unavailable")

    store = _store(tmp_path, monkeypatch, rotation_hook=_blocked)
    first = store.get_or_create_session(_source())

    with pytest.raises(CapabilityEpochRotationBlocked):
        store.reset_session(first.session_key)
    assert store._entries[first.session_key] is first

    with pytest.raises(CapabilityEpochRotationBlocked):
        store.switch_session(first.session_key, "resumed-session")
    assert store._entries[first.session_key] is first

    with pytest.raises(CapabilityEpochRotationBlocked):
        store.get_or_create_session(_source(), force_new=True)
    assert store._entries[first.session_key] is first
    assert store._entries[first.session_key].capability_epoch == first.capability_epoch


def test_writer_outage_blocks_stale_recovery_without_changing_epoch(
    tmp_path,
    monkeypatch,
):
    def _blocked(_entry, _reason):
        raise CapabilityEpochRotationBlocked("writer unavailable")

    store = _store(tmp_path, monkeypatch, rotation_hook=_blocked)
    first = store.get_or_create_session(_source())
    monkeypatch.setattr(store, "_is_session_ended_in_db", lambda _sid: True)
    monkeypatch.setattr(
        store,
        "_compression_tip_for_session_id",
        lambda session_id: session_id,
    )

    with pytest.raises(CapabilityEpochRotationBlocked):
        store.get_or_create_session(_source())

    assert store._entries[first.session_key] is first
    assert store._entries[first.session_key].capability_epoch == first.capability_epoch


def test_writer_outage_defers_automatic_reset_without_rotating(
    tmp_path,
    monkeypatch,
):
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=1)
    )

    def _blocked(_entry, _reason):
        raise CapabilityEpochRotationBlocked("writer unavailable")

    store = _store(
        tmp_path,
        monkeypatch,
        config=config,
        rotation_hook=_blocked,
    )
    first = store.get_or_create_session(_source())
    first.updated_at = _now() - timedelta(minutes=5)
    original_epoch = first.capability_epoch

    deferred = store.get_or_create_session(_source())

    assert deferred is first
    assert deferred.was_auto_reset is False
    assert deferred.capability_epoch == original_epoch


def test_writer_outage_with_concurrent_activity_keeps_unrevoked_epoch(
    tmp_path,
    monkeypatch,
):
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=1)
    )
    store = None

    def _blocked_after_live_progress(entry, _reason):
        assert store is not None
        with store._lock:
            entry.session_id = "trusted-live-child"
        raise CapabilityEpochRotationBlocked("writer unavailable")

    store = _store(
        tmp_path,
        monkeypatch,
        config=config,
        rotation_hook=_blocked_after_live_progress,
    )
    first = store.get_or_create_session(_source())
    first.updated_at = _now() - timedelta(minutes=5)
    original_epoch = first.capability_epoch

    deferred = store.get_or_create_session(_source())

    assert deferred is first
    assert deferred.session_id == "trusted-live-child"
    assert deferred.capability_epoch == original_epoch
    assert deferred.capability_rotation_deferred is True


def test_switch_to_compression_continuation_invokes_epoch_rotation_hook(
    tmp_path,
    monkeypatch,
):
    calls = []
    store = _store(
        tmp_path,
        monkeypatch,
        rotation_hook=lambda entry, reason: calls.append((entry, reason)),
    )
    first = store.get_or_create_session(_source())

    continued = store.switch_session(
        first.session_key,
        "compressed-child",
    )

    assert continued is not None
    assert continued.capability_epoch != first.capability_epoch
    assert calls == [(first, "explicit_switch")]


def test_get_or_create_treats_tampered_db_compression_lineage_as_rotation(
    tmp_path,
    monkeypatch,
):
    calls = []
    store = _store(
        tmp_path,
        monkeypatch,
        rotation_hook=lambda entry, reason: calls.append((entry, reason)),
    )
    first = store.get_or_create_session(_source())

    class _TamperedDB:
        @staticmethod
        def get_compression_tip(_session_id):
            return "attacker-selected-child"

        @staticmethod
        def get_session(_session_id):
            return None

        @staticmethod
        def save_gateway_routing_entries(_entries, *, scope):
            return None

    store._db = _TamperedDB()

    recovered = store.get_or_create_session(_source())

    assert recovered is not first
    assert recovered.session_id == "attacker-selected-child"
    assert recovered.capability_epoch != first.capability_epoch
    assert calls == [(first, "compression_tip_recovery")]


def test_compression_recovery_cas_loss_rotates_surviving_live_mapping(
    tmp_path,
    monkeypatch,
):
    store = None

    def _revoke_while_live_compression_finishes(entry, reason):
        assert store is not None
        assert reason == "compression_tip_recovery"
        # Mirrors the trusted in-turn completion path: same SessionEntry,
        # session_id updated while writer I/O is outside the routing lock.
        with store._lock:
            assert store._entries[entry.session_key] is entry
            entry.session_id = "trusted-live-child"

    store = _store(
        tmp_path,
        monkeypatch,
        rotation_hook=_revoke_while_live_compression_finishes,
    )
    first = store.get_or_create_session(_source())
    old_epoch = first.capability_epoch
    monkeypatch.setattr(
        store,
        "_compression_tip_for_session_id",
        lambda _session_id: "db-selected-child",
    )

    recovered = store.get_or_create_session(_source())

    assert recovered is not first
    assert recovered.session_id == "trusted-live-child"
    assert recovered.capability_epoch != old_epoch
    assert store._entries[first.session_key] is recovered


def test_automatic_reset_cas_loss_rotates_surviving_live_mapping(
    tmp_path,
    monkeypatch,
):
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=1)
    )
    store = None

    def _revoke_while_live_compression_finishes(entry, reason):
        assert store is not None
        assert reason == "automatic_reset:idle"
        with store._lock:
            assert store._entries[entry.session_key] is entry
            entry.session_id = "trusted-live-child"

    store = _store(
        tmp_path,
        monkeypatch,
        config=config,
        rotation_hook=_revoke_while_live_compression_finishes,
    )
    first = store.get_or_create_session(_source())
    first.updated_at = _now() - timedelta(minutes=5)
    old_epoch = first.capability_epoch

    recovered = store.get_or_create_session(_source())

    assert recovered is not first
    assert recovered.session_id == "trusted-live-child"
    assert recovered.capability_epoch != old_epoch
    assert recovered.was_auto_reset is False


def test_switch_cas_loss_preserves_live_child_with_fresh_epoch(
    tmp_path,
    monkeypatch,
):
    store = None

    def _revoke_while_live_compression_finishes(entry, reason):
        assert store is not None
        assert reason == "explicit_switch"
        with store._lock:
            entry.session_id = "trusted-live-child"

    store = _store(
        tmp_path,
        monkeypatch,
        rotation_hook=_revoke_while_live_compression_finishes,
    )
    first = store.get_or_create_session(_source())
    old_epoch = first.capability_epoch

    with pytest.raises(CapabilityEpochRotationBlocked) as blocked:
        store.switch_session(first.session_key, "requested-resume-target")

    current = store._entries[first.session_key]
    assert blocked.value.authority_rotated is True
    assert current is not first
    assert current.session_id == "trusted-live-child"
    assert current.capability_epoch != old_epoch


def test_reset_can_atomically_create_first_route_without_force_new_gap(
    tmp_path,
    monkeypatch,
):
    hook_calls = []
    store = _store(
        tmp_path,
        monkeypatch,
        rotation_hook=lambda *args: hook_calls.append(args),
    )
    source = _source()
    session_key = store._generate_session_key(source)

    created = store.reset_session(
        session_key,
        source=source,
        create_if_missing=True,
    )

    assert created is not None
    assert created.is_fresh_reset is True
    assert store._entries[session_key] is created
    assert hook_calls == []


def test_unrelated_switch_rotates_even_when_target_is_another_parents_tip(
    tmp_path,
    monkeypatch,
):
    calls = []
    store = _store(
        tmp_path,
        monkeypatch,
        rotation_hook=lambda entry, reason: calls.append((entry, reason)),
    )
    current = store.get_or_create_session(_source())

    # A caller may have resolved a stale Telegram binding from
    # bound-parent -> bound-child. That says nothing about the route's current
    # session, so SessionStore must independently prove current -> target.
    monkeypatch.setattr(
        store,
        "_compression_tip_for_session_id",
        lambda session_id: (
            "bound-child" if session_id == "bound-parent" else session_id
        ),
    )

    switched = store.switch_session(current.session_key, "bound-child")

    assert switched is not None
    assert switched.capability_epoch != current.capability_epoch
    assert calls == [(current, "explicit_switch")]


def test_paused_old_worker_cannot_consume_after_rotation_publishes(
    tmp_path,
    monkeypatch,
):
    live_epochs = set()
    live_lock = threading.Lock()
    worker_ready = threading.Event()
    allow_worker = threading.Event()
    worker_authorized = []

    def _revoke(entry, _reason):
        with live_lock:
            live_epochs.discard(entry.capability_epoch)

    store = _store(tmp_path, monkeypatch, rotation_hook=_revoke)
    first = store.get_or_create_session(_source())
    with live_lock:
        live_epochs.add(first.capability_epoch)

    def _paused_consumer():
        captured_epoch = first.capability_epoch
        worker_ready.set()
        assert allow_worker.wait(timeout=2)
        with live_lock:
            worker_authorized.append(captured_epoch in live_epochs)

    worker = threading.Thread(target=_paused_consumer)
    worker.start()
    assert worker_ready.wait(timeout=2)

    replacement = store.reset_session(first.session_key)
    assert replacement is not None
    allow_worker.set()
    worker.join(timeout=2)

    assert worker_authorized == [False]
    assert replacement.capability_epoch != first.capability_epoch


def test_recovery_without_routing_state_mints_fresh_epoch(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    row = {"id": "recovered-session", "started_at": None}

    first = store._create_entry_from_recovered_row(
        row=row,
        session_key="agent:main:discord:thread:channel-1:thread-1",
        source=_source(),
        now=_now(),
    )
    second = store._create_entry_from_recovered_row(
        row=row,
        session_key=first.session_key,
        source=_source(),
        now=_now(),
    )

    assert first.capability_epoch != second.capability_epoch


def test_stale_peer_recovery_preserves_epoch_only_for_exact_compression_tip(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path, monkeypatch)
    source = _source()
    now = _now()
    old = SessionEntry(
        session_key="route-key",
        session_id="ended-parent",
        created_at=now,
        updated_at=now,
        capability_epoch="cap_epoch_v1_" + "a" * 64,
        origin=source,
    )
    unrelated = SessionEntry(
        session_key="route-key",
        session_id="unrelated-later-peer-session",
        created_at=now,
        updated_at=now,
        capability_epoch="cap_epoch_v1_" + "b" * 64,
        origin=source,
    )

    class _DB:
        @staticmethod
        def get_session(_session_id):
            return {"end_reason": "agent_close"}

    store._db = _DB()
    store._loaded = True
    store._entries = {"route-key": old}
    monkeypatch.setattr(
        store,
        "_recover_session_from_db",
        lambda **_kwargs: unrelated,
    )
    monkeypatch.setattr(
        store,
        "_compression_tip_for_session_id",
        lambda _session_id: "different-compression-child",
    )
    monkeypatch.setattr(store, "_save", lambda: None)

    with store._lock:
        store._prune_stale_sessions_locked()

    assert store._entries["route-key"] is unrelated
    assert unrelated.capability_epoch == "cap_epoch_v1_" + "b" * 64


def test_stale_compression_tip_recovery_keeps_fresh_recovered_epoch(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path, monkeypatch)
    source = _source()
    now = _now()
    old = SessionEntry(
        session_key="route-key",
        session_id="ended-parent",
        created_at=now,
        updated_at=now,
        capability_epoch="cap_epoch_v1_" + "a" * 64,
        origin=source,
    )
    child = SessionEntry(
        session_key="route-key",
        session_id="compression-child",
        created_at=now,
        updated_at=now,
        capability_epoch="cap_epoch_v1_" + "b" * 64,
        origin=source,
    )

    class _DB:
        @staticmethod
        def get_session(_session_id):
            return {"end_reason": "compression"}

    store._db = _DB()
    store._loaded = True
    store._entries = {"route-key": old}
    monkeypatch.setattr(
        store,
        "_recover_session_from_db",
        lambda **_kwargs: child,
    )
    monkeypatch.setattr(store, "_save", lambda: None)

    with store._lock:
        store._prune_stale_sessions_locked()

    assert child.capability_epoch == "cap_epoch_v1_" + "b" * 64
    assert child.capability_epoch != old.capability_epoch


def test_writer_context_gets_only_epoch_digest_and_prompt_gets_neither():
    now = _now()
    entry = SessionEntry(
        session_key="agent:main:discord:thread:channel-1:thread-1",
        session_id="session-1",
        created_at=now,
        updated_at=now,
        capability_epoch="cap_epoch_v1_" + "a" * 64,
        origin=_source(),
        platform=Platform.DISCORD,
        chat_type="thread",
    )

    context = build_session_context(_source(), GatewayConfig(), entry)
    expected = hashlib.sha256(entry.capability_epoch.encode("ascii")).hexdigest()
    prompt = build_session_context_prompt(context)
    model_context = context.to_dict()

    assert context.capability_epoch_sha256 == expected
    assert "capability_epoch" not in model_context
    assert "capability_epoch_sha256" not in model_context
    assert entry.capability_epoch not in prompt
    assert expected not in prompt


def test_gateway_turn_binds_epoch_digest_into_trusted_writer_runtime():
    from gateway.canonical_writer_boundary import trusted_runtime_envelope
    from gateway.run import GatewayRunner

    now = _now()
    entry = SessionEntry(
        session_key="agent:main:discord:thread:channel-1:thread-1",
        session_id="session-1",
        created_at=now,
        updated_at=now,
        capability_epoch="cap_epoch_v1_" + "b" * 64,
        origin=_source(),
        platform=Platform.DISCORD,
        chat_type="thread",
    )
    context = build_session_context(_source(), GatewayConfig(), entry)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}

    tokens = runner._set_session_env(context)
    try:
        runtime = trusted_runtime_envelope()
    finally:
        runner._clear_session_env(tokens)

    assert runtime["session_id"] == "session-1"
    assert runtime["capability_epoch_sha256"] == hashlib.sha256(
        entry.capability_epoch.encode("ascii")
    ).hexdigest()
    assert entry.capability_epoch not in repr(runtime)


def test_rotation_hook_uses_isolated_exact_old_context_and_restores_caller(
    monkeypatch,
):
    from gateway.canonical_writer_boundary import trusted_runtime_envelope
    from gateway.run import GatewayRunner
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import approval

    now = _now()
    old = SessionEntry(
        session_key="agent:main:discord:thread:channel-old:thread-old",
        session_id="old-session",
        created_at=now,
        updated_at=now,
        capability_epoch="cap_epoch_v1_" + "c" * 64,
        origin=SessionSource(
            platform=Platform.DISCORD,
            chat_id="channel-old",
            thread_id="thread-old",
            chat_type="thread",
            user_id="owner-old",
            message_id="message-old",
        ),
        platform=Platform.DISCORD,
        chat_type="thread",
    )
    seen = {}

    def _revoke(session_key, *, reason):
        seen["session_key"] = session_key
        seen["reason"] = reason
        seen["runtime"] = trusted_runtime_envelope()
        return {"success": True, "scope_revoked": True}

    monkeypatch.setattr(
        approval,
        "revoke_session_capabilities_durably",
        _revoke,
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    monkeypatch.setattr(
        runner,
        "_clear_session_boundary_security_state",
        lambda session_key, strict=False, retire_capability_epoch_sha256="": seen.update({
            "cleared": session_key,
            "strict": strict,
            "retired_epoch": retire_capability_epoch_sha256,
        }),
    )

    caller_tokens = set_session_vars(
        platform="discord",
        chat_id="channel-new",
        thread_id="thread-new",
        user_id="owner-new",
        session_key="new-routing-key",
        session_id="new-session",
        capability_epoch_sha256="d" * 64,
    )
    try:
        runner._before_capability_epoch_rotation(old, "explicit_reset")
        caller_runtime = trusted_runtime_envelope()
    finally:
        clear_session_vars(caller_tokens)

    expected_epoch = hashlib.sha256(old.capability_epoch.encode("ascii")).hexdigest()
    assert seen["session_key"] == old.session_key
    assert seen["runtime"]["session_id"] == old.session_id
    assert seen["runtime"]["session_key_sha256"] == hashlib.sha256(
        old.session_key.encode()
    ).hexdigest()
    assert seen["runtime"]["capability_epoch_sha256"] == expected_epoch
    assert seen["runtime"]["user_id"] == "owner-old"
    assert seen["cleared"] == old.session_key
    assert seen["strict"] is True
    assert seen["retired_epoch"] == expected_epoch
    assert caller_runtime["session_id"] == "new-session"
    assert caller_runtime["capability_epoch_sha256"] == "d" * 64


def test_durable_session_revoke_requires_positive_exact_tombstone(monkeypatch):
    from gateway import canonical_writer_boundary as boundary
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import approval

    session_key = "agent:main:discord:thread:channel-1:thread-1"
    session_sha256 = hashlib.sha256(session_key.encode()).hexdigest()
    epoch_sha256 = "e" * 64
    calls = []

    monkeypatch.setattr(approval, "_writer_boundary_policy_required", lambda: True)
    monkeypatch.setattr(boundary, "writer_boundary_configured", lambda: True)

    def _call(operation, payload, *, idempotency_key=None):
        calls.append((operation, payload, idempotency_key))
        return {
            "success": True,
            "session_key_sha256": session_sha256,
            "capability_epoch_sha256": epoch_sha256,
            "scope_type": "session",
            "scope_revoked": False,
            "revoked": 0,
        }

    monkeypatch.setattr(boundary, "canonical_writer_call", _call)
    tokens = set_session_vars(
        platform="discord",
        session_key=session_key,
        session_id="session-1",
        capability_epoch_sha256=epoch_sha256,
    )
    try:
        with pytest.raises(RuntimeError, match="exact session-epoch tombstone"):
            approval.revoke_session_capabilities_durably(session_key)

        monkeypatch.setattr(
            boundary,
            "canonical_writer_call",
            lambda operation, payload, *, idempotency_key=None: {
                "success": True,
                "session_key_sha256": session_sha256,
                "capability_epoch_sha256": epoch_sha256,
                "scope_type": "session",
                "scope_revoked": True,
                "revoked": 0,
            },
        )
        result = approval.revoke_session_capabilities_durably(session_key)
    finally:
        clear_session_vars(tokens)

    assert result["scope_revoked"] is True
    assert calls[0][1] == {"reason": "gateway_session_boundary"}
    assert session_key not in repr(calls[0][1])
