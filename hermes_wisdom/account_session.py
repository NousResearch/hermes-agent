"""Retire profile-local Wisdom work when its Nous account signs out."""

import time

from hermes_constants import get_hermes_home

from .store import WisdomStore


def sign_out() -> bool:
    root = get_hermes_home() / "wisdom"
    if not (root / "wisdom.db").is_file():
        return False
    store = WisdomStore(root)
    now = time.time()
    with store.transaction() as db:
        changed = db.execute(
            """UPDATE installation_identity SET verified_org_id=NULL,verified_at=NULL
            WHERE verified_org_id IS NOT NULL"""
        ).rowcount
        db.execute(
            """UPDATE wisdom_assessment SET state='retired',lease_token=NULL,
            lease_until=NULL,updated_at=?
            WHERE state IN ('pending','assessing','ready','fallback')""",
            (now,),
        )
        # A send already dispatched may still succeed. Keep its immutable outbox
        # reservation so a late receipt can settle it, never schedule a resend.
        db.execute(
            """UPDATE wisdom_assessment SET state='delivery_uncertain',
            lease_token=NULL,lease_until=NULL,updated_at=? WHERE state='delivering'""",
            (now,),
        )
        db.execute(
            "UPDATE wisdom_consent SET state='stale',updated_at=? WHERE state='pending'",
            (now,),
        )
        db.execute("UPDATE wisdom_agent_session SET available=0,alive_until=0")
        db.execute("UPDATE wisdom_mute_control SET expires_at=0")
    return bool(changed)
