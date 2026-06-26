"""Consent gate tests.

Consent is a single config field (``telemetry.consent_state``); the aggregate opt-in
is expressed by setting it to ``"aggregate"`` (via ``hermes config set`` or a
managed-scope pin). ``allow_aggregate`` is the hard gate. ``policy.may_upload_aggregate``
is the gate a future uploader must consult.
"""

from __future__ import annotations

from agent.telemetry import policy


def _cfg(**telemetry):
    return {"telemetry": telemetry}


def test_default_posture_never_uploads():
    # No consent recorded → unknown → never uploads.
    assert policy.may_upload_aggregate(_cfg(local=True, consent_state="unknown")) is False


def test_missing_telemetry_block_never_uploads():
    assert policy.may_upload_aggregate({}) is False


def test_opted_in_uploads():
    assert policy.may_upload_aggregate(_cfg(consent_state="aggregate")) is True


def test_declined_does_not_upload():
    assert policy.may_upload_aggregate(_cfg(consent_state="local")) is False


def test_allow_aggregate_false_overrides_opt_in():
    # An admin pins telemetry.allow_aggregate: false via managed scope.
    cfg = _cfg(consent_state="aggregate", allow_aggregate=False)
    assert policy.may_upload_aggregate(cfg) is False  # the hard gate wins


def test_local_off_makes_aggregate_inert():
    # Aggregate metrics derive from the local tables; with local off there is
    # nothing to aggregate, so opting in cannot upload.
    cfg = _cfg(local=False, consent_state="aggregate", allow_aggregate=True)
    assert policy.may_upload_aggregate(cfg) is False


def test_install_id_minted_when_empty_and_stable_when_set():
    cfg = _cfg(install_id="")
    minted = policy.ensure_install_id(cfg)
    assert minted and len(minted) >= 32  # uuid4
    cfg2 = _cfg(install_id="fixed-id")
    assert policy.ensure_install_id(cfg2) == "fixed-id"
