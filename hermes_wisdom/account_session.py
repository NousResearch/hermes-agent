"""Retire profile-local Wisdom work when its Nous account signs out."""

import time

from hermes_constants import get_hermes_home

from .store import WisdomStore, utc_now


def sign_out(store: WisdomStore | None = None) -> bool:
    if store is None:
        root = get_hermes_home() / "wisdom"
        if not (root / "wisdom.db").is_file():
            return False
        store = WisdomStore(root)
    now = time.time()
    with store.transaction() as db:
        # Keep history and provider receipts, but do not replay cached arrivals
        # or accept a response fetched under the account session being ended.
        db.execute("UPDATE feed_event SET cadence='off' WHERE cadence!='off'")
        db.execute(
            """INSERT INTO feed_state(singleton,cursor,updated_at,generation)
            VALUES(1,NULL,?,1) ON CONFLICT(singleton) DO UPDATE SET
            generation=feed_state.generation+1,updated_at=excluded.updated_at""",
            (utc_now(),),
        )
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


def reject_revoked_account(store: WisdomStore) -> None:
    from hermes_cli.auth_nous import NOUS_SESSION_TERMINAL, get_nous_session_validity

    from .client import WisdomAuthError

    # This reads the persisted quarantine marker, never refreshes credentials.
    # Network errors and ordinary token expiry are not proof of revocation.
    if get_nous_session_validity() == NOUS_SESSION_TERMINAL:
        sign_out(store)
        raise WisdomAuthError(
            "Your Nous session ended; sign in and re-verify your team with `hermes wisdom setup`.",
            code="account_session_ended",
        )
