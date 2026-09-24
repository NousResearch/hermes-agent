"""End-to-end DB-level polarity for the delegation lease-defer fix (#turn-lease-fix).

Polarity BOTH directions against a real SessionDB (not mocks):
1. Lease held by a live turn  → append_delegation_delivery raises SessionTurnLeaseLostError
   ("active turn lease") — the exact class+message _is_active_turn_lease_rejection keys on.
2. Lease released            → the same append lands the delivery row (row visible, idempotent).
"""

import os
import time

from gateway.run_notifications import GatewayNotificationsMixin
from hermes_state import SessionDB
from hermes_state_errors import SessionTurnLeaseLostError


def _fence_message_polarity(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="test")
    holder = f"pid={os.getpid()}:turn=t1:platform=test"

    # 1. Live lease fences the delivery row.
    assert db.try_acquire_session_turn_lease("s1", holder, ttl_seconds=300)
    try:
        db.append_delegation_delivery(
            "s1", "result text", {"delegation_id": "deleg_1", "delivery_notice": ""}
        )
        raise AssertionError("expected SessionTurnLeaseLostError while lease held")
    except SessionTurnLeaseLostError as exc:
        assert GatewayNotificationsMixin._is_active_turn_lease_rejection(exc), (
            f"fence message not recognized as lease-busy: {exc}"
        )
    finally:
        db.release_session_turn_lease("s1", holder)

    # 2. Lease gone: the same delivery lands and is idempotent.
    first = db.append_delegation_delivery(
        "s1", "result text", {"delegation_id": "deleg_1", "delivery_notice": ""}
    )
    again = db.append_delegation_delivery(
        "s1", "result text", {"delegation_id": "deleg_1", "delivery_notice": ""}
    )
    assert first == again  # dedup by delegation_id — one row, not two
    msgs = db.get_messages("s1")
    assert any(m.get("display_kind") == "async_delegation_complete" for m in msgs)
    db.close()


def test_fence_message_polarity(tmp_path):
    _fence_message_polarity(tmp_path)
