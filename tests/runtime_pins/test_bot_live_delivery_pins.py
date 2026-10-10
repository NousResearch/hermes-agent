"""Behaviour pins for the live-owner DM mailbox (tools/bot_live_delivery.py).

Idempotency-by-id, terminal no-op, and cancel-after-claim are already pinned in
tests/tools/test_bot_live_owner_delivery.py; this file pins the remaining edge: a claim has no
lease, so no amount of elapsed time makes a claimed record claimable again.
"""

from __future__ import annotations

import json

from tools import bot_live_delivery as mailbox


def _owner(home, **over):
    return dict(dict(profile_home=str(home.resolve()), session_id="chat",
                     lease_id="lease", live_session_id="live"), **over)


def test_claimed_record_is_never_reclaimed_however_old_the_claim(tmp_path):
    """A ``claimed`` record stays claimed forever: backdating ``claimed_at``/``created_at`` to the epoch does not re-offer it."""
    owner = _owner(tmp_path)
    delivery_id = "b" * 32
    mailbox.deliver_to_live_owner(tmp_path, owner, "hi", delivery_id=delivery_id)
    claimed = mailbox.claim_pending_delivery(tmp_path, owner)
    assert claimed is not None and claimed["status"] == "claimed"

    path = mailbox._root(tmp_path) / f"{delivery_id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(claimed_at=0, created_at=0)
    path.write_text(json.dumps(record), encoding="utf-8")

    assert mailbox.claim_pending_delivery(tmp_path, owner) is None
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "claimed"
    # Cancel cannot rescue it either: the claim wins and is returned unchanged.
    assert mailbox.cancel_queued_delivery(tmp_path, delivery_id, error="x", reason="runtime_offline")["status"] == "claimed"
    # Only the holder's completion moves it on.
    assert mailbox.complete_delivery(tmp_path, delivery_id, status="settled", reply="ok")["status"] == "settled"


def test_complete_requires_claim_and_never_completes_a_queued_record(tmp_path):
    """``complete_delivery`` on a still-``queued`` record raises and leaves it claimable."""
    owner = _owner(tmp_path)
    delivery_id = "c" * 32
    mailbox.deliver_to_live_owner(tmp_path, owner, "hi", delivery_id=delivery_id)
    try:
        mailbox.complete_delivery(tmp_path, delivery_id, status="settled", reply="early")
    except ValueError:
        pass
    else:  # pragma: no cover - the pin is that this raises
        raise AssertionError("complete_delivery accepted a queued record")
    assert mailbox.read_delivery_result(tmp_path, delivery_id)["status"] == "queued"
    assert mailbox.claim_pending_delivery(tmp_path, owner)["delivery_id"] == delivery_id
