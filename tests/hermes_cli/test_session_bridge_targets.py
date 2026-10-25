from __future__ import annotations

from dataclasses import replace
import uuid

import pytest

from hermes_cli.session_bridge_targets import (
    ProviderTargetReceipt, RootsRouteAuthority, TargetUnavailable, TransportPolicy,
    receipt_binding_sha256, receipt_payload_sha256, resolve_provider_receipts,
)


def _seal(receipt: ProviderTargetReceipt) -> ProviderTargetReceipt:
    receipt = replace(receipt, binding_sha256=receipt_binding_sha256(receipt))
    return replace(receipt, payload_sha256=receipt_payload_sha256(receipt))


def _receipt(**overrides) -> ProviderTargetReceipt:
    values = {
        "receipt_id": str(uuid.uuid4()), "ticket_ref": "ESN-0407",
        "assignment_id": "assignment-7", "attempt_id": "attempt-3",
        "assignment_state": "active", "attempt_state": "active",
        "provider": "codex-app", "backend": "codex", "transport": "codex-app-thread",
        "provider_instance_id": "codex-host-local", "profile_scope": "/profiles/operator",
        "native_session_id": str(uuid.uuid4()), "delivery_handle": "thread:opaque-7",
        "agent_id": "codex-thread-agent", "team_name": None,
        "owner_identity": "operator-account", "ticket_revision": 4,
        "authority_event_sha256": "a" * 64,
        "authority_scope": "answer-only-no-changes", "identity_state": "verified",
        "transport_authorization_state": "authorized",
        "safe_delivery_state": "safe-point-queue", "reply_state": "correlated",
        "caller_cancellation_state": "stop-wait-and-invalidate-late",
        "observed_at": 100.0, "expires_at": 150.0,
        "revoked": False, "provenance": {"source": "codex-app-tools", "revision": 3},
        "binding_sha256": "", "payload_sha256": "", "provider_signature": "trusted:test",
    }
    values.update(overrides)
    return _seal(ProviderTargetReceipt(**values))


def _verify(receipt: ProviderTargetReceipt) -> bool:
    return receipt.provider_signature == "trusted:test"


def _route(**overrides) -> RootsRouteAuthority:
    values = {
        "ticket_ref": "ESN-0407", "assignment_id": "assignment-7",
        "attempt_id": "attempt-3", "owner_identity": "operator-account",
        "profile_scope": "/profiles/operator", "ticket_revision": 4,
        "authority_event_sha256": "a" * 64,
    }
    values.update(overrides)
    return RootsRouteAuthority(**values)


def _resolve(receipts, **kwargs):
    return resolve_provider_receipts(
        _route(), receipts, verify_provider_signature=_verify, now=110, **kwargs,
    )


def _enable_test_adapter(monkeypatch):
    import hermes_cli.session_bridge_targets as targets

    policies = dict(targets.TRANSPORT_POLICIES)
    policies[("codex-app", "codex-app-thread")] = TransportPolicy(
        "codex-app", "codex", "codex-app-thread", "safe-point-queue",
        "thread-read-wait", "none", True, "test provider adapter",
    )
    monkeypatch.setattr(targets, "TRANSPORT_POLICIES", policies)


def test_resolves_one_provider_issued_ticket_attempt(monkeypatch):
    _enable_test_adapter(monkeypatch)
    expected = _receipt()
    actual = _resolve([expected])
    assert actual.attempt_id == "attempt-3"
    assert actual.native_session_id != actual.delivery_handle


def test_spoofed_signature_cannot_register_a_target():
    spoof = _seal(replace(_receipt(), provider_signature="caller-asserted"))
    with pytest.raises(TargetUnavailable, match="signature"):
        _resolve([spoof])


def test_digest_tampering_is_rejected_before_eligibility():
    tampered = replace(_receipt(), delivery_handle="thread:substituted")
    with pytest.raises(TargetUnavailable, match="digest"):
        _resolve([tampered])


def test_unknown_provider_transport_pair_is_rejected():
    unknown = _receipt(provider="claude-code", backend="claude", transport="invented")
    with pytest.raises(TargetUnavailable, match="not registered"):
        _resolve([unknown])


def test_native_identity_and_delivery_handle_cannot_collapse():
    session_id = str(uuid.uuid4())
    collapsed = _receipt(native_session_id=session_id, delivery_handle=session_id)
    with pytest.raises(TargetUnavailable, match="must differ"):
        _resolve([collapsed])


@pytest.mark.parametrize("field,value", [
    ("assignment_id", "assignment-stale"),
    ("attempt_id", "attempt-stale"),
    ("owner_identity", "different-owner"),
    ("profile_scope", "/profiles/other"),
    ("ticket_revision", 3),
    ("authority_event_sha256", "b" * 64),
])
def test_receipt_must_bind_exact_current_roots_route(field: str, value):
    with pytest.raises(TargetUnavailable, match="exact current Roots"):
        resolve_provider_receipts(
            _route(), [_receipt(**{field: value})],
            verify_provider_signature=_verify, now=110,
        )


def test_newer_revocation_tombstones_older_positive_receipt():
    old = _receipt(observed_at=100, expires_at=150)
    revoked = _receipt(
        receipt_id=str(uuid.uuid4()), native_session_id=old.native_session_id,
        observed_at=105, expires_at=155, revoked=True,
    )
    with pytest.raises(TargetUnavailable, match="revoked"):
        _resolve([old, revoked])


@pytest.mark.parametrize("field", ["assignment_state", "attempt_state"])
def test_newer_inactive_state_tombstones_older_positive_receipt(field: str):
    old = _receipt(observed_at=100, expires_at=150)
    inactive = _receipt(
        receipt_id=str(uuid.uuid4()), native_session_id=old.native_session_id,
        observed_at=105, expires_at=155, **{field: "inactive"},
    )
    with pytest.raises(TargetUnavailable, match="inactive"):
        _resolve([old, inactive])


@pytest.mark.parametrize("field,value", [
    ("transport_authorization_state", "denied"),
    ("caller_cancellation_state", "unavailable"),
])
def test_newer_capability_reduction_tombstones_older_positive_receipt(field: str, value: str):
    old = _receipt(observed_at=100, expires_at=150)
    reduced = _receipt(
        receipt_id=str(uuid.uuid4()), native_session_id=old.native_session_id,
        observed_at=105, expires_at=155, **{field: value},
    )
    with pytest.raises(TargetUnavailable, match=field):
        _resolve([old, reduced])


def test_untrusted_newer_spoof_does_not_tombstone_authoritative_receipt(monkeypatch):
    _enable_test_adapter(monkeypatch)
    old = _receipt(observed_at=100, expires_at=150)
    spoof = _seal(replace(
        old, receipt_id=str(uuid.uuid4()), observed_at=105, expires_at=155,
        revoked=True, provider_signature="caller-asserted",
    ))
    assert _resolve([old, spoof]).receipt_id == old.receipt_id


def test_multiple_authoritative_active_attempts_are_ambiguous(monkeypatch):
    _enable_test_adapter(monkeypatch)
    with pytest.raises(TargetUnavailable, match="ambiguous"):
        _resolve([_receipt(), _receipt(native_session_id=str(uuid.uuid4()))])


@pytest.mark.parametrize("transport", [
    "claude-team-mailbox", "claude-peer-socket", "claude-mcp-channel",
])
def test_claude_native_transports_remain_fail_closed_for_external_hermes(transport: str):
    team = "team-a" if transport == "claude-team-mailbox" else None
    receipt = _receipt(
        provider="claude-code", backend="claude", transport=transport,
        team_name=team, delivery_handle="uds:/private/session.sock",
    )
    with pytest.raises(TargetUnavailable, match="unavailable to external Hermes"):
        _resolve([receipt])


def test_app_owned_codex_transport_is_not_callable_by_external_hermes():
    with pytest.raises(TargetUnavailable, match="owning Codex app host"):
        _resolve([_receipt()])


def test_registry_module_exposes_no_production_record_function():
    import hermes_cli.session_bridge_targets as targets

    assert not hasattr(targets, "record_provider_observation")
    assert not hasattr(targets, "ensure_schema")
