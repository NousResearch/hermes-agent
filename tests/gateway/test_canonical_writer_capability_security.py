from __future__ import annotations

import datetime as dt
import hashlib
import time

import pytest

import gateway.canonical_writer_boundary as writer_boundary
from gateway.canonical_writer_handlers import (
    CanonicalWriterHandlers,
    CanonicalWriterTypedDispatcher,
    InMemoryCanonicalWriterBackend,
    InMemoryCanonicalWriterStore,
    RuntimeContext,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation
from gateway.canonical_writer_service import DispatchContext, PeerCredentials
from tools import approval


NOW = dt.datetime(2026, 7, 12, 10, 0, tzinfo=dt.timezone.utc)
SESSION_HASH = "a" * 64
COMMAND_HASH = "b" * 64
SOURCE_HASH = "c" * 64
CAPABILITY_EPOCH_HASH = "d" * 64


def _runtime(session_hash: str = SESSION_HASH) -> RuntimeContext:
    return RuntimeContext(
        request_id="request-capability-1",
        platform="discord",
        session_key_sha256=session_hash,
        capability_epoch_sha256=CAPABILITY_EPOCH_HASH,
        user_id="owner-1",
        message_id="message-1",
        owner_authenticated=True,
    )


def _handler_fixture():
    store = InMemoryCanonicalWriterStore()
    store.active_plans["case:1"] = {
        "plan_id": "plan:1",
        "revision": 1,
        "state": "active",
    }
    backend = InMemoryCanonicalWriterBackend(store, clock=lambda: NOW)
    return CanonicalWriterHandlers(backend), backend


def _grant_payload(*, expires_at=None, max_uses=2):
    return {
        "approval_id": "approval:1",
        "case_id": "case:1",
        "plan_id": "plan:1",
        "plan_revision": 1,
        "approval_source_sha256": SOURCE_HASH,
        "command_hashes": [COMMAND_HASH],
        "expires_at": (expires_at or NOW + dt.timedelta(hours=1)).isoformat(),
        "max_uses": max_uses,
    }


def _consume_payload(key: str):
    return {"command_sha256": COMMAND_HASH, "idempotency_key": key}


def test_successful_durable_consume_returns_exact_plan_and_scope_receipt():
    handlers, backend = _handler_fixture()
    runtime = _runtime()
    assert handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=runtime,
    )["ok"]

    consumed = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload("consume:exact:1"),
        runtime=runtime,
    )
    retry = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload("consume:exact:1"),
        runtime=runtime,
    )

    assert consumed["result"] == {
        **consumed["result"],
        "success": True,
        "authorized": True,
        "approval_id": "approval:1",
        "case_id": "case:1",
        "plan_id": "plan:1",
        "remaining_uses": 1,
    }
    assert retry["result"]["plan_id"] == "plan:1"
    assert retry["result"]["deduped"] is True
    assert backend.store.capabilities["approval:1"]["remaining_uses"][COMMAND_HASH] == 1


@pytest.mark.parametrize(
    "payload_key,transport_key",
    [("payload-key", "transport-key"), ("payload-key", None)],
)
def test_typed_dispatcher_rejects_unbound_consume_idempotency(
    payload_key,
    transport_key,
):
    handlers, backend = _handler_fixture()
    runtime = _runtime()
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=runtime,
    )
    adapter = CanonicalWriterTypedDispatcher(handlers)
    context = DispatchContext(
        request_id="request-consume-binding",
        sequence=1,
        deadline_unix_ms=9999999999999,
        idempotency_key=transport_key,
        peer=PeerCredentials(pid=101, uid=1001, gid=1001),
        runtime={
            "platform": "discord",
            "session_key_sha256": SESSION_HASH,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
            "user_id": "owner-1",
        },
    )

    result = adapter.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME,
        _consume_payload(payload_key),
        context,
    )

    assert result.status == "blocked"
    assert result.result["error_code"] == "idempotency_binding_mismatch"
    assert backend.store.capabilities["approval:1"]["remaining_uses"][COMMAND_HASH] == 2


def test_typed_dispatcher_accepts_only_matching_consume_idempotency():
    handlers, _backend = _handler_fixture()
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=_runtime(),
    )
    adapter = CanonicalWriterTypedDispatcher(handlers)
    context = DispatchContext(
        request_id="request-consume-matching",
        sequence=1,
        deadline_unix_ms=9999999999999,
        idempotency_key="consume:matching",
        peer=PeerCredentials(pid=101, uid=1001, gid=1001),
        runtime={
            "platform": "discord",
            "session_key_sha256": SESSION_HASH,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
            "user_id": "owner-1",
        },
    )

    result = adapter.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME,
        _consume_payload("consume:matching"),
        context,
    )

    assert result.status == "inserted"
    assert result.result["plan_id"] == "plan:1"


def test_typed_dispatcher_injects_authoritative_transport_idempotency():
    handlers, _backend = _handler_fixture()
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=_runtime(),
    )
    adapter = CanonicalWriterTypedDispatcher(handlers)
    context = DispatchContext(
        request_id="request-consume-injected",
        sequence=1,
        deadline_unix_ms=9999999999999,
        idempotency_key="consume:transport-only",
        peer=PeerCredentials(pid=101, uid=1001, gid=1001),
        runtime={
            "platform": "discord",
            "session_key_sha256": SESSION_HASH,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
            "user_id": "owner-1",
        },
    )

    result = adapter.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME,
        {"command_sha256": COMMAND_HASH},
        context,
    )

    assert result.status == "inserted"
    assert result.result["plan_id"] == "plan:1"


def test_grant_retry_dedupes_without_extending_expiry_or_replenishing_uses():
    handlers, backend = _handler_fixture()
    first_expiry = NOW + dt.timedelta(hours=1)
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(expires_at=first_expiry),
        runtime=_runtime(),
    )
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload("consume:before-grant-retry"),
        runtime=_runtime(),
    )

    retry = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(expires_at=NOW + dt.timedelta(hours=2)),
        runtime=_runtime(),
    )

    assert retry["ok"] is False
    assert retry["error"]["code"] == "approval_source_replay"
    stored = backend.store.capabilities["approval:1"]
    assert stored["max_uses"] == 2
    assert stored["remaining_uses"][COMMAND_HASH] == 1


def test_capability_grant_rejects_expiry_beyond_eight_hours():
    handlers, backend = _handler_fixture()

    response = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(expires_at=NOW + dt.timedelta(hours=8, seconds=1)),
        runtime=_runtime(),
    )

    assert response["error"]["code"] == "approval_expiry_out_of_bounds"
    assert backend.store.capabilities == {}


def test_revoke_is_exactly_bound_to_runtime_session():
    handlers, backend = _handler_fixture()
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=_runtime(),
    )

    wrong_session = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE.value,
        {"plan_id": "plan:1", "reason": "wrong session"},
        runtime=_runtime("d" * 64),
    )
    assert wrong_session["result"]["revoked"] == 0
    assert backend.store.capabilities["approval:1"]["state"] == "granted"

    exact_session = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE.value,
        {"plan_id": "plan:1", "reason": "plan terminal"},
        runtime=_runtime(),
    )
    assert exact_session["result"]["revoked"] == 1
    assert backend.store.capabilities["approval:1"]["state"] == "revoked"


def test_fresh_routing_epoch_cannot_consume_or_revoke_prior_capability():
    handlers, backend = _handler_fixture()
    old_runtime = _runtime()
    fresh_runtime = RuntimeContext(
        request_id="request-capability-fresh-epoch",
        platform="discord",
        session_key_sha256=SESSION_HASH,
        capability_epoch_sha256="e" * 64,
        user_id="owner-1",
        message_id="message-2",
        owner_authenticated=True,
    )
    assert handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=old_runtime,
    )["ok"]

    consumed = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload("consume:fresh-epoch"),
        runtime=fresh_runtime,
    )
    revoked = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE.value,
        {"plan_id": "plan:1", "reason": "fresh boundary"},
        runtime=fresh_runtime,
    )

    assert consumed["ok"] is False
    assert consumed["error"]["code"] == "capability_missing"
    assert revoked["result"]["revoked"] == 0
    assert backend.store.capabilities["approval:1"]["state"] == "granted"


def _writer_required_config():
    return {
        "approvals": {"plan_owner_user_ids": ["owner-1"]},
        "canonical_brain": {"writer_boundary": {"enabled": True}},
    }


def test_tools_never_fall_back_to_local_capability_when_writer_is_required(
    monkeypatch,
):
    session_key = "writer-required-no-fallback"
    plan_id = "plan:no-fallback"
    command = "git status --short"
    digest = hashlib.sha256(command.encode()).hexdigest()
    monkeypatch.setattr("hermes_cli.config.load_config", _writer_required_config)
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "owner-1")
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(approval, "_observed_session_message_id", lambda: "message-1")
    monkeypatch.setattr(approval, "_canonical_active_plan_matches", lambda **_kw: True)
    monkeypatch.setattr(
        writer_boundary, "writer_boundary_policy_required", lambda: True
    )
    monkeypatch.setattr(writer_boundary, "writer_boundary_configured", lambda: False)
    with approval._lock:
        approval._plan_capabilities[session_key] = {
            plan_id: {
                "durably_granted": True,
                "expires_at": time.time() + 3600,
                "command_uses": {digest: 1},
                "approved_by_user_id": "owner-1",
                "canonical_case_id": "case:no-fallback",
                "plan_revision": 1,
            }
        }

    with pytest.raises(RuntimeError, match="writer is required"):
        approval.grant_plan_capability(
            session_key=session_key,
            plan_id="plan:new",
            exact_commands=[command],
            approved_by_user_id="owner-1",
            canonical_case_id="case:no-fallback",
            plan_revision=1,
        )
    assert approval.consume_plan_capability(session_key, command) is None
    assert approval.revoke_plan_capability(session_key, plan_id) is False
    with approval._lock:
        capability = approval._plan_capabilities[session_key][plan_id]
        assert capability["command_uses"][digest] == 1
        approval._plan_capabilities.pop(session_key, None)


def test_tools_writer_consume_requires_success_and_returns_plan_id(monkeypatch):
    session_key = "writer-success-plan-id"
    session_hash = hashlib.sha256(session_key.encode()).hexdigest()
    calls = []
    monkeypatch.setattr("hermes_cli.config.load_config", _writer_required_config)
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "owner-1")
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(
        writer_boundary, "writer_boundary_policy_required", lambda: True
    )
    monkeypatch.setattr(writer_boundary, "writer_boundary_configured", lambda: True)
    monkeypatch.setattr(
        writer_boundary,
        "trusted_runtime_envelope",
        lambda: {
            "session_key_sha256": session_hash,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
            "user_id": "owner-1",
        },
    )

    def call(operation, payload, *, idempotency_key=None):
        calls.append((operation, dict(payload), idempotency_key))
        return {
            "success": True,
            "authorized": True,
            "plan_id": "plan:writer",
            "command_sha256": payload["command_sha256"],
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
            "approved_by_user_id": "owner-1",
            "plan_revision": 1,
        }

    monkeypatch.setattr(writer_boundary, "canonical_writer_call", call)

    assert approval.consume_plan_capability(session_key, "git status --short") == (
        "plan:writer"
    )
    assert calls[0][0] == CanonicalWriterOperation.CAPABILITY_CONSUME.value
    assert calls[0][1]["idempotency_key"] == calls[0][2]
    assert calls[0][1]["command_sha256"] == hashlib.sha256(
        b"git status --short"
    ).hexdigest()

    monkeypatch.setattr(
        writer_boundary,
        "canonical_writer_call",
        lambda *_args, **_kwargs: {"success": True, "authorized": True},
    )
    assert approval.consume_plan_capability(session_key, "git status --short") is None


def test_tools_writer_consume_rejects_another_owners_receipt(monkeypatch):
    session_key = "writer-cross-user-receipt"
    session_hash = hashlib.sha256(session_key.encode()).hexdigest()
    command = "git status --short"
    command_hash = hashlib.sha256(command.encode()).hexdigest()
    monkeypatch.setattr("hermes_cli.config.load_config", _writer_required_config)
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "owner-2")
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(
        writer_boundary, "writer_boundary_policy_required", lambda: True
    )
    monkeypatch.setattr(writer_boundary, "writer_boundary_configured", lambda: True)
    monkeypatch.setattr(
        writer_boundary,
        "trusted_runtime_envelope",
        lambda: {
            "session_key_sha256": session_hash,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
            "user_id": "owner-2",
        },
    )
    monkeypatch.setattr(
        writer_boundary,
        "canonical_writer_call",
        lambda *_args, **_kwargs: {
            "success": True,
            "authorized": True,
            "plan_id": "plan:owner-1",
            "plan_revision": 1,
            "approved_by_user_id": "owner-1",
            "command_sha256": command_hash,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
        },
    )

    assert approval.consume_plan_capability(session_key, command) is None


def test_tools_writer_consume_rejects_receipt_from_prior_routing_epoch(monkeypatch):
    session_key = "writer-stale-epoch-receipt"
    session_hash = hashlib.sha256(session_key.encode()).hexdigest()
    command = "git status --short"
    command_hash = hashlib.sha256(command.encode()).hexdigest()
    monkeypatch.setattr("hermes_cli.config.load_config", _writer_required_config)
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "owner-1")
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(
        writer_boundary, "writer_boundary_policy_required", lambda: True
    )
    monkeypatch.setattr(writer_boundary, "writer_boundary_configured", lambda: True)
    monkeypatch.setattr(
        writer_boundary,
        "trusted_runtime_envelope",
        lambda: {
            "session_key_sha256": session_hash,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
            "user_id": "owner-1",
        },
    )
    monkeypatch.setattr(
        writer_boundary,
        "canonical_writer_call",
        lambda *_args, **_kwargs: {
            "success": True,
            "authorized": True,
            "plan_id": "plan:stale",
            "command_sha256": command_hash,
            "capability_epoch_sha256": "e" * 64,
            "approved_by_user_id": "owner-1",
            "plan_revision": 1,
        },
    )

    assert approval.consume_plan_capability(session_key, command) is None


def test_tools_reject_deduped_grant_when_durable_authority_is_inactive(monkeypatch):
    session_key = "writer-inactive-grant"
    session_hash = hashlib.sha256(session_key.encode()).hexdigest()
    monkeypatch.setattr("hermes_cli.config.load_config", _writer_required_config)
    monkeypatch.setattr(approval, "_observed_session_user_id", lambda: "owner-1")
    monkeypatch.setattr(approval, "_observed_session_platform", lambda: "discord")
    monkeypatch.setattr(approval, "_observed_session_message_id", lambda: "message-1")
    monkeypatch.setattr(approval, "_canonical_active_plan_matches", lambda **_kw: True)
    monkeypatch.setattr(
        writer_boundary, "writer_boundary_policy_required", lambda: True
    )
    monkeypatch.setattr(writer_boundary, "writer_boundary_configured", lambda: True)
    monkeypatch.setattr(
        writer_boundary,
        "trusted_runtime_envelope",
        lambda: {
            "session_key_sha256": session_hash,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
        },
    )
    monkeypatch.setattr(
        writer_boundary,
        "canonical_writer_call",
        lambda *_args, **_kwargs: {
            "success": True,
            "state": "expired",
            "authority_active": False,
            "session_key_sha256": session_hash,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
            "deduped": True,
        },
    )

    with pytest.raises(RuntimeError, match="expired"):
        approval.grant_plan_capability(
            session_key=session_key,
            plan_id="plan:inactive",
            exact_commands=["git status --short"],
            approved_by_user_id="owner-1",
            canonical_case_id="case:inactive",
            plan_revision=1,
        )
