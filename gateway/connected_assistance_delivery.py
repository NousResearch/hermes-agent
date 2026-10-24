"""Owner-private local pointers, with durable at-most-once publication receipts.

Never publish source text in a toast/replay and never fall back to messaging.
"""
from contextlib import closing
from pathlib import Path

from gateway.config import Platform
from gateway.connected_assistance_store import AssistanceStore, notification_key

POINTER = "Connected-source findings are ready. Open `hermes connectors findings` in your local terminal."


def owned_local_target(authority, sid, *, source=None):
    from gateway.session_local_recovery import owned_receipt
    live = authority.sessions.get(sid)
    if (live is None or live.source is None or live.source.platform != Platform.LOCAL
            or live.source.chat_type != "dm" or (source is not None and live.source is not source)):
        return None
    try:
        receipt, saved, route, _ = owned_receipt(authority, sid)
    except (ValueError, KeyError, OSError):
        return None
    if live.source.to_dict() != saved.to_dict() or live.route != route:
        return None
    policy = receipt["policy"]
    if policy.get("source") not in {"cli", "tui", "gui"}:
        return None
    # A login to a shared dashboard is not proof of owning this profile's connected accounts.
    natives = getattr(authority, "_native_legacy_transports", {})
    members = list(live.subscribers.values())
    if not members or not all(natives.get(m.transport_id) == m
                             and m.subject == receipt["principal_id"]
                             and m.profile_id == authority.profile_id for m in members):
        return None
    return live


def _receipts(db):
    db.execute("CREATE TABLE IF NOT EXISTS assistance_notifications "
               "(key TEXT PRIMARY KEY, finding_id TEXT NOT NULL, status TEXT NOT NULL)")


def notification_receipts(store):
    with closing(store._connect()) as db, db:
        _receipts(db)
        return [dict(r) for r in db.execute("SELECT * FROM assistance_notifications ORDER BY rowid")]


def _claim(store, row):
    key = notification_key(row)
    with closing(store._connect()) as db, db:
        _receipts(db)
        changed = db.execute("INSERT OR IGNORE INTO assistance_notifications VALUES (?, ?, 'unknown')",
                             (key, row["id"])).rowcount
    return key if changed else None


def publish_private_findings(authority, rows, *, current=lambda: True):
    if (not current() or getattr(authority, "retiring", False) or getattr(authority, "retired", False)
            or authority.runner._draining):
        return 0
    target = next(((sid, live) for sid in list(authority.sessions)
                   if (live := owned_local_target(authority, sid)) is not None), None)
    if target is None:
        return 0
    store = AssistanceStore(Path(authority.profile_id))
    shown = claimed = 0
    for row in rows:
        if claimed >= 5 or not current() or store.is_paused():
            break
        key = _claim(store, row)
        if key is None:
            continue
        claimed += 1
        sid, live = target
        # Claim BEFORE publish: an interrupted/ambiguous send is unknown, never blindly retried.
        if not current() or owned_local_target(authority, sid) is not live:
            continue
        if live.event_stream.publish(sid, {"text": POINTER, "level": "info", "kind": "sticky",
                                          "key": key, "id": key}, event_type="notification.show"):
            with closing(store._connect()) as db, db:
                db.execute("UPDATE assistance_notifications SET status='shown' WHERE key=?", (key,))
            shown += 1
    return shown
