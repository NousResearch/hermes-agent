"""Retire and resume profile-local Wisdom work at account session boundaries."""

import time
from typing import TYPE_CHECKING

from hermes_constants import get_hermes_home

from .store import WisdomStore, utc_now

if TYPE_CHECKING:
    from .client import WisdomClient


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
            """INSERT INTO feed_state(singleton,cursor,updated_at,generation,resume_required)
            VALUES(1,NULL,?,1,1) ON CONFLICT(singleton) DO UPDATE SET
            generation=feed_state.generation+1,resume_required=1,updated_at=excluded.updated_at""",
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


def resume_feed(
    store: WisdomStore,
    client: "WisdomClient",
    *,
    org_id: str,
    installation_id: str,
    generation: int,
) -> int:
    """Checkpoint signed-out arrivals silently before setup restores authority."""
    from .client import WisdomValidationError

    with store.transaction() as db:
        store.check_feed_generation(db, generation)
        row = db.execute("SELECT * FROM feed_state WHERE singleton=1").fetchone()
        same_scope = bool(
            row
            and row["organization_id"] == org_id
            and row["installation_id"] == installation_id
        )
        cursor = row["cursor"] if same_scope else None
        required = bool(row and row["resume_required"])
        if not same_scope:
            db.execute("UPDATE feed_event SET cadence='off' WHERE cadence!='off'")
        generation += 1
        db.execute(
            """INSERT INTO feed_state(singleton,cursor,updated_at,generation,
            organization_id,installation_id,resume_required) VALUES(1,?,?,?,?,?,?)
            ON CONFLICT(singleton) DO UPDATE SET cursor=excluded.cursor,
            updated_at=excluded.updated_at,generation=excluded.generation,
            organization_id=excluded.organization_id,installation_id=excluded.installation_id,
            resume_required=excluded.resume_required""",
            (cursor, utc_now(), generation, org_id, installation_id, int(required)),
        )
        db.execute(
            "UPDATE installation_identity SET verified_org_id=NULL,verified_at=NULL"
        )
    if not required:
        return generation
    for _ in range(100):
        page = client.feed(cursor, installation_id=installation_id)
        if page.has_more and page.next_cursor == cursor:
            raise WisdomValidationError(
                "Wisdom feed catch-up made no progress; retry setup"
            )
        with store.transaction() as db:
            store.check_feed_generation(db, generation)
            db.execute(
                "UPDATE feed_state SET cursor=?,updated_at=? WHERE singleton=1",
                (page.next_cursor, utc_now()),
            )
        cursor = page.next_cursor
        if not page.has_more:
            return generation
    raise WisdomValidationError(
        "Wisdom feed catch-up paused at its page limit; rerun setup to resume"
    )


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
