"""Consent posture + org-policy enforcement tests.

Consent is a single field (``telemetry.consent_state``); the aggregate opt-in is
expressed by setting it to ``"aggregate"`` (via ``hermes config set`` or a managed-scope
pin). ``allow_aggregate`` is the hard gate.
"""

from __future__ import annotations

from agent.telemetry import policy


def _cfg(**telemetry):
    return {"telemetry": telemetry}


def test_default_posture_is_local_only():
    d = policy.resolve(_cfg(local=True, consent_state="unknown"))
    assert d.local_enabled is True
    assert d.aggregate_enabled is False
    assert d.may_upload_aggregate() is False


def test_unknown_consent_never_uploads():
    # A headless box with no choice recorded: stays unknown, never uploads.
    d = policy.resolve(_cfg(local=True, consent_state="unknown"))
    assert d.may_upload_aggregate() is False


def test_opted_in_uploads():
    d = policy.resolve(_cfg(local=True, consent_state="aggregate"))
    assert d.aggregate_enabled is True
    assert d.may_upload_aggregate() is True


def test_declined_does_not_upload():
    d = policy.resolve(_cfg(local=True, consent_state="local"))
    assert d.may_upload_aggregate() is False


def test_allow_aggregate_false_overrides_opt_in():
    # An admin pins telemetry.allow_aggregate: false via managed scope.
    cfg = _cfg(local=True, consent_state="aggregate", allow_aggregate=False)
    d = policy.resolve(cfg)
    assert d.allow_aggregate is False
    assert d.aggregate_enabled is False
    assert d.may_upload_aggregate() is False  # the hard gate wins


def test_invalid_consent_state_treated_as_unknown():
    d = policy.resolve(_cfg(local=True, consent_state="bogus"))
    assert d.consent_state == "unknown"
    assert d.may_upload_aggregate() is False


def test_install_id_minted_when_empty_and_stable_when_set():
    cfg = _cfg(install_id="")
    minted = policy.ensure_install_id(cfg)
    assert minted and len(minted) >= 32  # uuid4
    cfg2 = _cfg(install_id="fixed-id")
    assert policy.ensure_install_id(cfg2) == "fixed-id"
