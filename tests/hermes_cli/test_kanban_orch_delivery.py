"""M5 T23-T33: Delivery protocol — obligations, ACK family, receipt, completion."""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_delivery import (
    ACK_FAMILIES,
    DeliveryError,
    DeliveryAttempt,
    DeliveryObligation,
    DeliveryReceipt,
    ManifestEntry,
    authorize_resend,
    check_delivery_satisfied,
    check_delivery_terminal_semantics,
    claim_obligation,
    create_obligations_from_manifest,
    finish_attempt,
    process_receipt,
    start_attempt,
    validate_ack_family,
)

D = "a" * 64  # 64-hex digest


def make_manifest(required_count=2, optional_count=1, origin_kind="messaging"):
    """Helper: create manifest entries."""
    entries = []
    for i in range(required_count):
        entries.append(ManifestEntry(
            manifest_entry_key=f"entry-req-{i}",
            required=True,
            route_digest=f"route-{i}",
            required_ack_family="provider",
            required_ack_strength="message_id",
        ))
    for i in range(optional_count):
        entries.append(ManifestEntry(
            manifest_entry_key=f"entry-opt-{i}",
            required=False,
            route_digest=f"route-opt-{i}",
            required_ack_family="adapter",
            required_ack_strength="adapter_acceptance",
        ))
    return entries


def test_adapter_boundary_capability():
    """T23: unauthorized send reaches adapter without capability.

    Claim must be owned; attempt must match owner/token/epoch.
    """
    entries = make_manifest(1, 0)
    obls = create_obligations_from_manifest(entries, origin_kind="messaging")
    obl = obls[0]

    # Claim on non-pending state should fail
    obl.state = "claimed"
    with pytest.raises(DeliveryError, match="claim_not_pending"):
        claim_obligation(obl, owner="worker1", token_hash="t1", ttl=30, now=100)

    # Proper claim from pending
    obl.state = "pending"
    epoch = claim_obligation(obl, owner="worker1", token_hash="t1", ttl=30, now=100)
    assert epoch == 1
    assert obl.state == "claimed"

    # Attempt with wrong owner
    with pytest.raises(DeliveryError, match="attempt_wrong_owner"):
        start_attempt(obl, attempt_id=1, send_nonce="n1", payload_digest=D,
                      result_digest=D, claim_owner="wrong", claim_token_hash="t1",
                      claim_epoch=1, lifecycle_revision=0, cancel_epoch=0)

    # Attempt with wrong token
    with pytest.raises(DeliveryError, match="attempt_wrong_token"):
        start_attempt(obl, attempt_id=1, send_nonce="n1", payload_digest=D,
                      result_digest=D, claim_owner="worker1", claim_token_hash="wrong",
                      claim_epoch=1, lifecycle_revision=0, cancel_epoch=0)

    # Attempt with wrong epoch
    with pytest.raises(DeliveryError, match="attempt_wrong_epoch"):
        start_attempt(obl, attempt_id=1, send_nonce="n1", payload_digest=D,
                      result_digest=D, claim_owner="worker1", claim_token_hash="t1",
                      claim_epoch=99, lifecycle_revision=0, cancel_epoch=0)

    # Attempt with unclaimed obligation
    obl.state = "pending"
    with pytest.raises(DeliveryError, match="attempt_not_claimed"):
        start_attempt(obl, attempt_id=1, send_nonce="n1", payload_digest=D,
                      result_digest=D, claim_owner="worker1", claim_token_hash="t1",
                      claim_epoch=1, lifecycle_revision=0, cancel_epoch=0)


def test_exact_delivery_manifest():
    """T24: manifest missing/extra required entry never completes."""
    entries = make_manifest(2, 1)
    obls = create_obligations_from_manifest(entries, origin_kind="messaging")
    assert len(obls) == 3

    # No ACKs yet → not satisfied
    assert not check_delivery_satisfied(obls, origin_kind="messaging")

    # ACK only 1 of 2 required → not satisfied
    obls[0].state = "acked"
    obls[0].acceptance_attempt_id = 1
    assert not check_delivery_satisfied(obls, origin_kind="messaging")

    # ACK both required → satisfied (optional doesn't block)
    obls[1].state = "acked"
    obls[1].acceptance_attempt_id = 2
    assert check_delivery_satisfied(obls, origin_kind="messaging")

    # Board-only with zero required → satisfied immediately
    board_entries = []
    board_obls = create_obligations_from_manifest(board_entries, origin_kind="board_only")
    assert len(board_obls) == 0
    assert check_delivery_satisfied(board_obls, origin_kind="board_only")

    # Board-only with required entries → error
    bad_entries = make_manifest(1, 0)
    with pytest.raises(DeliveryError, match="board_only_has_required_obligations"):
        create_obligations_from_manifest(bad_entries, origin_kind="board_only")


def test_receipt_exact_binding():
    """T27: receipt binds attempt/nonce/payload/result/route/family."""
    entries = make_manifest(1, 0)
    obls = create_obligations_from_manifest(entries, origin_kind="messaging")
    obl = obls[0]
    claim_obligation(obl, owner="w1", token_hash="t1", ttl=30, now=100)

    attempt = start_attempt(obl, attempt_id=1, send_nonce="n1", payload_digest=D,
                            result_digest=D, claim_owner="w1", claim_token_hash="t1",
                            claim_epoch=1, lifecycle_revision=0, cancel_epoch=0)

    # Valid receipt
    receipt = DeliveryReceipt(
        attempt_id=1, obligation_id=obl.obligation_id, send_nonce="n1",
        payload_digest=D, result_digest=D, route_digest=obl.route_digest,
        observed_ack_family="provider", observed_ack_strength="message_id",
    )
    process_receipt(obl, attempt, receipt, now=200)
    assert obl.state == "acked"
    assert obl.acceptance_attempt_id == 1

    # Reset for negative tests
    obl.state = "claimed"
    obl.acceptance_attempt_id = None
    attempt.state = "started"
    attempt.finished_at = None

    # Wrong attempt ID
    bad_receipt = DeliveryReceipt(
        attempt_id=99, obligation_id=obl.obligation_id, send_nonce="n1",
        payload_digest=D, result_digest=D, route_digest=obl.route_digest,
        observed_ack_family="provider", observed_ack_strength="message_id",
    )
    with pytest.raises(DeliveryError, match="receipt_attempt_mismatch"):
        process_receipt(obl, attempt, bad_receipt)

    # Wrong nonce
    bad_receipt = DeliveryReceipt(
        attempt_id=1, obligation_id=obl.obligation_id, send_nonce="wrong",
        payload_digest=D, result_digest=D, route_digest=obl.route_digest,
        observed_ack_family="provider", observed_ack_strength="message_id",
    )
    with pytest.raises(DeliveryError, match="receipt_nonce_mismatch"):
        process_receipt(obl, attempt, bad_receipt)

    # Cross-family ACK
    bad_receipt = DeliveryReceipt(
        attempt_id=1, obligation_id=obl.obligation_id, send_nonce="n1",
        payload_digest=D, result_digest=D, route_digest=obl.route_digest,
        observed_ack_family="adapter", observed_ack_strength="adapter_acceptance",
    )
    with pytest.raises(DeliveryError, match="cross_family_ack_satisfied"):
        process_receipt(obl, attempt, bad_receipt)

    # Wrong strength
    bad_receipt = DeliveryReceipt(
        attempt_id=1, obligation_id=obl.obligation_id, send_nonce="n1",
        payload_digest=D, result_digest=D, route_digest=obl.route_digest,
        observed_ack_family="provider", observed_ack_strength="adapter_acceptance",
    )
    with pytest.raises(DeliveryError, match="ack_strength_mismatch"):
        process_receipt(obl, attempt, bad_receipt)


def test_ack_family_exact():
    """T32: typed ACK family rejects cross-family receipt."""
    validate_ack_family("provider", "message_id")
    validate_ack_family("none", "none")

    with pytest.raises(DeliveryError, match="invalid_ack_family"):
        validate_ack_family("invalid", "none")
    with pytest.raises(DeliveryError, match="invalid_ack_strength"):
        validate_ack_family("provider", "invalid")
    with pytest.raises(DeliveryError, match="ack_family_strength_mismatch"):
        validate_ack_family("provider", "adapter_acceptance")


def test_delivery_terminal_semantics():
    """T31: required dead_letter/unknown/cancel never satisfy; optional non-blocking."""
    obls = [
        DeliveryObligation("o1", "k1", True, D, "provider", "message_id", state="acked", acceptance_attempt_id=1),
        DeliveryObligation("o2", "k2", True, D, "provider", "message_id", state="dead_letter"),
        DeliveryObligation("o3", "k3", True, D, "provider", "message_id", state="unknown"),
        DeliveryObligation("o4", "k4", True, D, "provider", "message_id", state="cancelled"),
        DeliveryObligation("o5", "k5", False, D, "adapter", "adapter_acceptance", state="dead_letter"),
        DeliveryObligation("o6", "k6", False, D, "adapter", "adapter_acceptance", state="acked"),
    ]
    result = check_delivery_terminal_semantics(obls)
    assert result["o1"] is True   # required acked
    assert result["o2"] is False  # required dead_letter
    assert result["o3"] is False  # required unknown
    assert result["o4"] is False  # required cancelled
    assert result["o5"] is True   # optional dead_letter — non-blocking
    assert result["o6"] is True   # optional acked


def test_unknown_resend_authorization():
    """T30: unknown human auth exact attempt/expiry, new generation."""
    obl = DeliveryObligation("o1", "k1", True, D, "provider", "message_id",
                             delivery_generation=1, state="unknown")
    new_obl = authorize_resend(obl, new_generation=2)
    assert new_obl.delivery_generation == 2
    assert new_obl.state == "pending"
    assert new_obl.duplicate_possible is True
    assert new_obl.obligation_id != obl.obligation_id

    # Non-unknown cannot resend
    obl2 = DeliveryObligation("o2", "k2", True, D, "provider", "message_id", state="acked")
    with pytest.raises(DeliveryError, match="resend_not_authorized"):
        authorize_resend(obl2, new_generation=2)

    # Generation must be +1
    with pytest.raises(DeliveryError, match="invalid_resend_generation"):
        authorize_resend(obl, new_generation=3)  # skip +1


def test_outcome_atomic_bundle():
    """T28: outcome attempt/event/receipt/obligation atomic transaction.

    Attempt finish + receipt processing must be atomic: if receipt fails,
    obligation must not be partially acked.
    """
    entries = make_manifest(1, 0)
    obls = create_obligations_from_manifest(entries, origin_kind="messaging")
    obl = obls[0]
    claim_obligation(obl, owner="w1", token_hash="t1", ttl=30, now=100)

    attempt = start_attempt(obl, attempt_id=1, send_nonce="n1", payload_digest=D,
                            result_digest=D, claim_owner="w1", claim_token_hash="t1",
                            claim_epoch=1, lifecycle_revision=0, cancel_epoch=0)

    # Finish attempt
    finish_attempt(obl, attempt, terminal_state="adapter_accepted", now=200)
    assert attempt.state == "adapter_accepted"

    # Process receipt — atomic with finish
    receipt = DeliveryReceipt(
        attempt_id=1, obligation_id=obl.obligation_id, send_nonce="n1",
        payload_digest=D, result_digest=D, route_digest=obl.route_digest,
        observed_ack_family="provider", observed_ack_strength="message_id",
    )
    process_receipt(obl, attempt, receipt, now=300)
    assert obl.state == "acked"

    # If receipt fails, obligation stays claimed
    obl2_state_before = "claimed"
    obl2 = DeliveryObligation("o2", "k2", True, D, "provider", "message_id", state="claimed")
    attempt2 = DeliveryAttempt(
        attempt_id=2, obligation_id="o2", send_nonce="n2", payload_digest=D,
        result_digest=D, route_digest=D, claim_owner="w1", claim_token_hash="t1",
        claim_epoch=1, lifecycle_revision=0, cancel_epoch=0,
    )
    bad_receipt = DeliveryReceipt(
        attempt_id=2, obligation_id="o2", send_nonce="n2",
        payload_digest="wrong", result_digest=D, route_digest=D,
        observed_ack_family="provider", observed_ack_strength="message_id",
    )
    with pytest.raises(DeliveryError):
        process_receipt(obl2, attempt2, bad_receipt)
    assert obl2.state == "claimed"  # not partially acked


def test_accepted_unknown_recovery():
    """T29: crash accepted recovery and unknown→acked without resend."""
    obl = DeliveryObligation("o1", "k1", True, D, "provider", "message_id",
                             state="unknown", delivery_generation=1)

    # Simulate provider-query: find valid receipt, unknown→acked
    receipt = DeliveryReceipt(
        attempt_id=1, obligation_id="o1", send_nonce="n1",
        payload_digest=D, result_digest=D, route_digest=D,
        observed_ack_family="provider", observed_ack_strength="message_id",
    )
    # unknown state can be acked (recovery path)
    process_receipt(obl, DeliveryAttempt(
        attempt_id=1, obligation_id="o1", send_nonce="n1", payload_digest=D,
        result_digest=D, route_digest=D, claim_owner="w1", claim_token_hash="t1",
        claim_epoch=1, lifecycle_revision=0, cancel_epoch=0,
    ), receipt, now=500)
    assert obl.state == "acked"
    assert obl.acceptance_attempt_id == 1
