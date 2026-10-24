"""``cron/bot_chat_pending``: Bot Chat outputs whose target owner was not running yet.

``defer`` parks one (a cross-profile target whose gateway is down or still starting), and
the retired CLI lane left records of the same shape behind. Each still-``queued`` record is handed,
under its own id, to the target profile's live owner through the canonical
``tools.bot_live_delivery.deliver_to_live_owner`` door (idempotent per id). Records are
never deleted: they keep their final status as evidence, exactly as the old drain did.
A ``claimed`` record was an uncertain CLI turn and is never resent.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from hermes_dir_acl import make_user_dir

from cron.export_writers import writer_custody_for as _writer_custody_for
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)
_warned: set[Path] = set()


def _bot_owner_home():
    return get_hermes_home()


def _pending_root() -> Path:
    return get_hermes_home().resolve() / "cron" / "bot_chat_pending"


@_writer_custody_for(_bot_owner_home)
def _persist_pending_record(record_name: str, record: dict, *, create_only: bool = False) -> None:
    if Path(record_name).name != record_name or not record_name.endswith(".json"):
        raise ValueError("invalid bot-chat pending record name")
    root = _pending_root()
    make_user_dir(root, parents=True, exist_ok=True, mode=0o700)
    path = root / record_name
    if create_only and path.exists():
        return
    atomic_json_write(path, record, fsync_dir=True, mode=0o600)


def defer(key: str, job: dict, content: str, home: Path, *, for_failure: bool) -> None:
    """Park one never-admitted output under its delivery id; a later tick hands it to the owner."""
    import time
    _persist_pending_record(
        f"{key}.json",
        dict(
            id=key, status="queued", job={k: job[k] for k in ("id", "name") if k in job}, content=content,
            home=str(home), for_failure=for_failure, sequence=time.time_ns()),
        create_only=True,
    )


def _queued(root: Path) -> list[tuple[Path, dict]]:
    found = []
    for path in root.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object, got {type(record).__name__}")
        except (OSError, ValueError) as exc:
            if path not in _warned:
                _warned.add(path)
                logger.error("Unreadable legacy Bot Chat pending record %s kept as evidence: %s", path, exc)
            continue
        if record.get("status") == "queued":
            found.append((path, record))
    return sorted(found, key=lambda item: (item[1].get("sequence", 0), item[0].name))


def _malformed(record: dict) -> str:
    """Why *record* can never be handed off (empty when its required fields are usable)."""
    missing = [key for key in ("id", "home", "content")
               if not isinstance(record.get(key), str) or not record.get(key)]
    if missing:
        return f"malformed legacy bot-chat record (missing {', '.join(missing)}); not delivered"
    if not isinstance(record.get("job") or {}, dict):
        return "malformed legacy bot-chat record (job is not an object); not delivered"
    return ""


def _dispose(path: Path, record: dict) -> bool:
    """Decide one queued record in place; False leaves it queued for a later tick."""
    from cron.scheduler_delivery import BOT_CHAT_POLICY_PLATFORM, bot_chat_message
    from gateway.warning_notifications import warning_notifications_enabled
    from hermes_cli.config_effective import load_user_config_effective
    from tools.bot_live_delivery import deliver_to_live_owner, find_canonical_live_owner

    if reason := _malformed(record):
        record.update(status="ambiguous", error=reason)
        return True
    home = Path(record["home"]).resolve()
    for_failure = bool(record.get("for_failure"))
    if not (home / "state.db").is_file():
        record.update(status="ambiguous", error=f"bot-chat delivery target no longer exists: {home}")
    elif for_failure and not warning_notifications_enabled(
            BOT_CHAT_POLICY_PLATFORM, load_user_config_effective(home / "config.yaml")):
        record.update(status="suppressed", error=None)
    else:
        try:
            owner = find_canonical_live_owner(home)
        except ValueError:
            return False  # Owner not ready: discovery uncertainty is not a disposition.
        if owner is None:
            return False
        try:
            receipt = deliver_to_live_owner(
                home, owner, bot_chat_message(record.get("job") or {}, record["content"]),
                delivery_id=record["id"], notification_category="diagnostic" if for_failure else "result")
        except Exception as exc:
            # The door is idempotent per id (a replay returns the existing receipt), so the
            # record stays queued for the next tick; log loudly once, quietly after.
            logger.log(logging.DEBUG if path in _warned else logging.WARNING,
                       "Legacy Bot Chat pending record %s not handed off yet: %s", path, exc, exc_info=True)
            _warned.add(path)
            return False
        receipt_status = receipt.get("status") if isinstance(receipt, dict) else None
        record.update(status="transferred", error=None, receipt_status=receipt_status)
    return True


def drain_legacy_pending() -> None:
    """Admit queued legacy records to their live owner; an absent owner leaves them for a later tick.

    Each record is isolated: a malformed one is kept as ``ambiguous`` evidence and an
    unexpected failure leaves only that record queued, so one bad file never stops cron.
    """
    root = _pending_root()
    if not root.is_dir():
        return
    for path, record in _queued(root):
        try:
            if not _dispose(path, record):
                continue
            _persist_pending_record(path.name, record)
        except Exception as exc:
            logger.log(logging.DEBUG if path in _warned else logging.WARNING,
                       "Legacy Bot Chat pending record %s left queued: %s", path, exc, exc_info=True)
            _warned.add(path)
            continue
        logger.info("Legacy Bot Chat pending record %s -> %s", path.name, record["status"])
