"""Tests for the Remote-Spending gate denial contract (NAS PR #481).

Behavior contracts: the HTTP→exception mapping in
``hermes_cli.nous_billing._raise_for_error`` and the
``tui_gateway.server._serialize_billing_error`` envelope the TUI branches on.
These assert the wire contract (CF-4) — error code, actor, recovery, retry —
not specific copy.
"""

import pytest

from hermes_cli.nous_billing import (
    BillingError,
    BillingRateLimited,
    BillingRemoteSpendingRevoked,
    BillingScopeRequired,
    BillingSessionRevoked,
    _raise_for_error,
)


def _raise(status, payload, headers=None):
    """Run _raise_for_error and return the exception it raises."""
    with pytest.raises(BillingError) as ei:
        _raise_for_error(status, payload, headers)
    return ei.value


# ── exception mapping (hermes_cli.nous_billing) ──────────────────────


def test_403_remote_spending_revoked_maps_to_typed_exc_with_actor():
    exc = _raise(403, {"error": "remote_spending_revoked", "recovery": "reconnect", "actor": "admin"})
    assert isinstance(exc, BillingRemoteSpendingRevoked)
    assert exc.actor == "admin"
    assert exc.recovery == "reconnect"


def test_403_revoked_absent_actor_is_none_not_crash():
    exc = _raise(403, {"error": "remote_spending_revoked"})
    assert isinstance(exc, BillingRemoteSpendingRevoked)
    assert exc.actor is None  # surface treats absent as "self"


def test_401_session_revoked_is_distinct_from_plain_401():
    revoked = _raise(401, {"error": "session_revoked", "recovery": "login"})
    assert isinstance(revoked, BillingSessionRevoked)
    assert revoked.recovery == "login"

    plain = _raise(401, {"error": "invalid_token"})
    assert not isinstance(plain, BillingSessionRevoked)


def test_403_insufficient_scope_still_maps_to_scope_required():
    exc = _raise(403, {"error": "insufficient_scope"})
    assert isinstance(exc, BillingScopeRequired)
    # NOT mistaken for a revoke.
    assert not isinstance(exc, BillingRemoteSpendingRevoked)


def test_503_is_rate_limited_not_revoked_and_carries_retry_after():
    exc = _raise(503, {"error": "temporarily_unavailable"}, {"Retry-After": "30"})
    assert isinstance(exc, BillingRateLimited)
    assert not isinstance(exc, BillingRemoteSpendingRevoked)
    assert exc.retry_after == 30


def test_403_business_denial_carries_code_and_recovery():
    exc = _raise(403, {
        "error": "cli_billing_disabled",
        "code": "remote_spending_disabled",
        "recovery": "enable_account_toggle",
        "portalUrl": "/billing",
    })
    # Generic BillingError (not a typed revoke) — the surface maps on code.
    assert type(exc) is BillingError
    assert exc.error == "cli_billing_disabled"
    assert exc.code == "remote_spending_disabled"
    assert exc.recovery == "enable_account_toggle"


def test_409_idempotency_conflict_passes_through():
    exc = _raise(409, {"error": "idempotency_conflict", "message": "same key, different amount"})
    assert exc.error == "idempotency_conflict"


# ── envelope serialization (tui_gateway.server) ──────────────────────


def _serialize(status, payload, headers=None):
    import tui_gateway.server as srv

    return srv._serialize_billing_error(_raise(status, payload, headers))


def test_envelope_threads_actor_code_recovery():
    env = _serialize(403, {"error": "remote_spending_revoked", "actor": "admin", "recovery": "reconnect"})
    assert env["error"] == "remote_spending_revoked"
    assert env["actor"] == "admin"
    assert env["recovery"] == "reconnect"
    assert env["ok"] is False


def test_envelope_session_revoked_kind():
    env = _serialize(401, {"error": "session_revoked", "recovery": "login"})
    assert env["error"] == "session_revoked"
    assert env["recovery"] == "login"


def test_envelope_503_preserves_server_code_with_retry():
    env = _serialize(503, {"error": "temporarily_unavailable"}, {"Retry-After": "30"})
    assert env["error"] == "temporarily_unavailable"
    assert env["retry_after"] == 30


def test_envelope_business_code_survives():
    env = _serialize(403, {"error": "cli_billing_disabled", "code": "remote_spending_disabled", "recovery": "enable_account_toggle"})
    assert env["error"] == "cli_billing_disabled"
    assert env["code"] == "remote_spending_disabled"
    assert env["recovery"] == "enable_account_toggle"
