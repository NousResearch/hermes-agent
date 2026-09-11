"""Local, editable publisher evidence carried by the approved author description."""

from datetime import datetime, timedelta, timezone

from .qualification import _profile_timezone
from .store import WisdomStore


def publication_description(
    store: WisdomStore,
    skill_id: str,
    description: str,
    *,
    at: datetime | None = None,
) -> str:
    """Snapshot one week of local use; never send a ledger or claim verified success.

    This publication snapshot is independent of recommendation thresholds.
    Call only while preparing a new local review, never on upload or retry.
    """
    profile_tz, timezone_name = _profile_timezone()
    end = (at or datetime.now(timezone.utc)).astimezone(profile_tz).date()
    start = end - timedelta(days=6)
    with store.transaction() as db:
        count, days = db.execute(
            "SELECT COALESCE(SUM(u.use_count),0),COUNT(*) "
            "FROM usage_day u JOIN local_skill s ON s.id=u.skill_id "
            "WHERE s.id=? AND s.source_kind='local' AND s.deleted_at IS NULL "
            "AND u.timezone_name=? AND u.day_local>=? AND u.day_local<=? "
            "AND u.use_count>0",
            (skill_id, timezone_name, start.isoformat(), end.isoformat()),
        ).fetchone()
    if not count:
        return description
    return (
        f"{description}\n\nPublisher usage (client-reported)\n"
        f"Local ledger: {count} invocation{'s' if count != 1 else ''} "
        f"on {days} day{'s' if days != 1 else ''}, "
        f"{start.isoformat()} through {end.isoformat()} (profile calendar dates). "
        "Counts may span local revisions and do not verify successful outcomes."
    )
