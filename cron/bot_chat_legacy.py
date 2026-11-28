"""``cron/bot_chat_pending``: Bot Chat outputs whose target owner was not running yet.

``defer`` parks one (a cross-profile target whose gateway is down or still starting), and
the retired CLI lane left records of the same shape behind. Each still-``queued`` record is handed,
under its own id, to the target profile's live owner through the canonical
``tools.bot_live_delivery.deliver_to_live_owner`` door (idempotent per id). Terminal records are
scrubbed and bounded by a crash-recoverable index; queued/claimed evidence is retained.
There is no permanent exactly-once guarantee after terminal receipt eviction.
A ``claimed`` record was an uncertain CLI turn and is never resent.
"""
from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path
from hermes_dir_acl import make_user_dir

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)
_warned: set[Path] = set()
MAX_TERMINAL_RECEIPTS = 1000
_TERMINAL = {"settled", "suppressed", "transferred", "ambiguous", "failed", "cancelled"}
_CANONICAL_KEY = re.compile(r"[0-9a-f]{64}\Z")


def _require_key(key: object) -> str:
    if not isinstance(key, str) or not _CANONICAL_KEY.fullmatch(key):
        raise ValueError("deferred Bot Chat delivery id must be 64 lowercase hexadecimal characters")
    return key


def _receipt_path(root: Path, key: object) -> Path:
    return root / f"{_require_key(key)}.json"


def _root(origin_home: Path | None = None) -> Path:
    home = origin_home if origin_home is not None else get_hermes_home()
    return Path(home).resolve() / "cron" / "bot_chat_pending"


def read_pending(key: str, *, origin_home: Path | None = None) -> dict | None:
    """Exact-key lookup. Corrupt or mismatched evidence never licenses admission."""
    key = _require_key(key)
    path = _receipt_path(_root(origin_home), key)
    try:
        record = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict) or record.get("id") != key:
        raise ValueError("malformed legacy Bot Chat owner; not overwritten")
    return record


@contextmanager
def ownership_arbitration(key: str, *, origin_home: Path | None = None):
    """Short profile-local reservation shared by SQL and JSON producers.

    Persist ownership before releasing this context; do not hold it over RPC or
    inference. A bare lookup is a snapshot, not permission to send afterwards.
    """
    from cron import delivery_queue
    from hermes_cli.active_sessions import _FileLock
    key = _require_key(key)
    root = _root(origin_home)
    make_user_dir(root, parents=True, exist_ok=True, mode=0o700)
    # Pre-upgrade producers hold this stripe across their SQL absence check
    # and JSON read/replace. Take it BEFORE SQLite, matching their lock order.
    with _FileLock(root / f".ownership-{key[:2]}.lock"):
        with delivery_queue._transaction(origin_home) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _recover_terminal_unlocked(root)
            yield conn


def arbitrate(key: str, payload_hash: str | None, *, origin_home: Path | None = None,
              conn=None, execution_id: str | None = None) -> dict:
    """Report the authoritative owner (including terminal receipts/tombstones).

    With no conn this returns a snapshot. For check-and-publish, pass the conn
    yielded by ownership_arbitration; enqueue/defer already do so internally.
    Conflicting stores fail closed rather than choosing a second sender.
    """
    from cron import delivery_queue
    key = _require_key(key)
    if conn is None:
        with ownership_arbitration(key, origin_home=origin_home) as locked:
            return arbitrate(key, payload_hash, origin_home=origin_home, conn=locked, execution_id=execution_id)
    if not conn.in_transaction:
        raise ValueError("ownership arbitration requires a write reservation")
    sql = delivery_queue._get_status_unlocked(conn, key)
    if execution_id and execution_id != key:
        canonical = delivery_queue._get_status_unlocked(conn, execution_id)
        if canonical is not None and canonical.get("payload_hash") is not None:
            if sql is not None:
                raise ValueError("conflicting SQL Bot Chat owners; do not resend")
            sql = canonical
    record = read_pending(key, origin_home=origin_home)
    if sql is not None:
        delivery_queue._validate_payload_hash(sql.get("payload_hash"), payload_hash)
    if record is not None:
        _validate_payload(record, payload_hash)
    if sql is not None and record is not None:
        raise ValueError("conflicting SQL and legacy Bot Chat owners; do not resend")
    owner = "legacy" if record is not None else "sql" if sql is not None else None
    return {"owned": owner is not None, "owner": owner,
            "record": record if record is not None else sql}


def payload_hash(job: dict, content: str, home: Path, *, for_failure: bool = False) -> str:
    from cron.delivery_queue import canonical_payload_hash
    from cron.scheduler_delivery import bot_chat_message
    return canonical_payload_hash(bot_chat_message(job, content), home,
                                  "diagnostic" if for_failure else "result")


def _validate_payload(record: dict, digest: str | None) -> None:
    from cron.delivery_queue import _validate_payload_hash
    existing = record.get("payload_hash")
    if existing is None and not _malformed(record):
        existing = payload_hash(record.get("job") or {}, record["content"], Path(record["home"]),
                                for_failure=bool(record.get("for_failure")))
    _validate_payload_hash(existing, digest)


def _read_metadata(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return default


def _read_terminal_index(root: Path) -> list[str]:
    value = _read_metadata(root / ".terminal-index", [])
    return list(dict.fromkeys(k for k in value if isinstance(k, str)
                             and _CANONICAL_KEY.fullmatch(k))) if isinstance(value, list) else []


def _write_terminal_index_unlocked(root: Path, index: list[str], key: str) -> None:
    key = _require_key(key)
    index = [k for k in index if isinstance(k, str) and _CANONICAL_KEY.fullmatch(k) and k != key]
    index.append(key)
    keep = max(0, int(MAX_TERMINAL_RECEIPTS))
    expired = index[:-keep] if keep else index
    for old in expired:
        path = _receipt_path(root, old)
        try:
            record = _read_metadata(path, None)
            if isinstance(record, dict) and record.get("id") == old and record.get("status") in _TERMINAL:
                path.unlink()
        except (OSError, ValueError):
            pass  # Corrupt evidence and active owners are never evicted.
    atomic_json_write(root / ".terminal-index", index[-keep:] if keep else [], fsync_dir=True, mode=0o600)


def _recover_terminal_unlocked(root: Path) -> None:
    """Reconcile one write-ahead receipt intent, never a directory-sized scan."""
    intent = _read_metadata(root / ".terminal-intent", None)
    if isinstance(intent, str) and _CANONICAL_KEY.fullmatch(intent):
        record = _read_metadata(_receipt_path(root, intent), None)
        if isinstance(record, dict) and record.get("id") == intent and record.get("status") in _TERMINAL:
            _write_terminal_index_unlocked(root, _read_terminal_index(root), intent)
    if intent is not None:
        atomic_json_write(root / ".terminal-intent", None, fsync_dir=True, mode=0o600)


def _terminalize_unlocked(path: Path, record: dict) -> dict:
    terminal = _terminal_record(record)
    root = path.parent
    # Malformed filenames/ids remain evidence but never enter the trusted index.
    key = record.get("id")
    if not isinstance(key, str) or not _CANONICAL_KEY.fullmatch(key) or path != _receipt_path(root, key):
        atomic_json_write(path, terminal, fsync_dir=True, mode=0o600)
        return terminal
    _recover_terminal_unlocked(root)
    atomic_json_write(root / ".terminal-intent", key, fsync_dir=True, mode=0o600)
    atomic_json_write(path, terminal, fsync_dir=True, mode=0o600)
    _write_terminal_index_unlocked(root, _read_terminal_index(root), key)
    atomic_json_write(root / ".terminal-intent", None, fsync_dir=True, mode=0o600)
    return terminal


def _next_sequence_unlocked(root: Path) -> int:
    """Allocate under the profile write reservation, without a directory scan."""
    import time
    try:
        previous = int((root / ".sequence").read_text(encoding="ascii"))
    except FileNotFoundError:
        previous = 0
    # Corrupt/unreadable floors fail closed rather than silently regressing order.
    sequence = max(previous + 1, time.time_ns())
    atomic_json_write(root / ".sequence", sequence, fsync_dir=True, mode=0o600)
    return sequence


def defer(key: str, job: dict, content: str, home: Path, *, for_failure: bool,
          origin_home: Path | None = None, conn=None, suppressed: bool = False,
          claimed: bool = False) -> dict:
    """Observe or park an owner; this API never grants permission to transport."""
    return publish(key, job, content, home, for_failure=for_failure,
                   origin_home=origin_home, conn=conn, suppressed=suppressed,
                   claimed=claimed)[0]


def publish(key: str, job: dict, content: str, home: Path, *, for_failure: bool,
            origin_home: Path | None = None, conn=None, suppressed: bool = False,
            claimed: bool = False) -> tuple[dict, bool]:
    """Return (authoritative record, created): an observed owner is never ours."""
    import time
    from utils.files import fsync_directory
    key = _require_key(key)
    digest = payload_hash(job, content, home, for_failure=for_failure)
    reservation = ownership_arbitration(key, origin_home=origin_home) if conn is None else nullcontext(conn)
    with reservation as conn:
        decision = arbitrate(key, digest, origin_home=origin_home, conn=conn, execution_id=job.get("execution_id"))
        if decision["owned"]:
            record = decision["record"]
            if suppressed and decision["owner"] == "legacy" and record.get("status") == "queued":
                record.update(status="suppressed", error=None)
                return _terminalize_unlocked(_receipt_path(_root(origin_home), key), record), False
            return record, False
        root = _root(origin_home)
        make_user_dir(root, parents=True, exist_ok=True, mode=0o700)
        path = _receipt_path(root, key)
        record = dict(
            id=key, status="claimed" if claimed else "queued",
            job={k: job[k] for k in ("id", "name", "execution_id") if k in job}, content=content,
            home=str(home.resolve()), origin_home=str(root.parent.parent),
            for_failure=for_failure, sequence=_next_sequence_unlocked(root),
            payload_hash=digest)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # Even an older/uncooperative producer cannot be overwritten. If it
            # died during publication, retain corrupt evidence and fail closed.
            return arbitrate(key, digest, origin_home=origin_home, conn=conn)["record"], False
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(root)
        if suppressed:
            record.update(status="suppressed", error=None)
            return _terminalize_unlocked(path, record), True
        return record, True


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
    return sorted(found, key=lambda item: (
        item[1].get("sequence", 0) if type(item[1].get("sequence", 0)) is int else 0,
        item[0].name,
    ))


def _malformed(record: dict) -> str:
    """Why *record* can never be handed off (empty when its required fields are usable)."""
    try:
        _require_key(record.get("id"))
    except ValueError:
        return "malformed legacy bot-chat record (noncanonical id); not delivered"
    missing = [key for key in ("id", "home", "content")
               if not isinstance(record.get(key), str) or not record.get(key)]
    if missing:
        return f"malformed legacy bot-chat record (missing {', '.join(missing)}); not delivered"
    if type(record.get("sequence", 0)) is not int:
        return "malformed legacy bot-chat record (invalid sequence); not delivered"
    if not isinstance(record.get("job") or {}, dict):
        return "malformed legacy bot-chat record (job is not an object); not delivered"
    return ""


def _dispose(path: Path, record: dict) -> bool:
    """Decide one queued record in place; False leaves it queued for a later tick."""
    from cron.scheduler_delivery import BOT_CHAT_POLICY_PLATFORM, bot_chat_message
    from gateway.warning_notifications import warning_notifications_enabled
    from hermes_cli.config_effective import load_user_config_effective
    from tools.bot_live_delivery import deliver_to_live_owner, find_canonical_live_owner

    reason = _malformed(record)
    if not reason and path.stem != record["id"]:
        reason = "malformed legacy bot-chat record (filename/id mismatch); not delivered"
    if reason:
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
            from cron.scheduler_delivery import _bot_chat_pre_admission_refusal, _redact_cron_payload
            if _bot_chat_pre_admission_refusal(exc):
                return False
            record.update(status="ambiguous", error=_redact_cron_payload(str(exc), "delivery error"))
            return True
        receipt_status = receipt.get("status") if isinstance(receipt, dict) else None
        record.update(status="transferred", error=None, receipt_status=receipt_status)
    return True


def _terminal_record(record: dict) -> dict:
    """Drop replay material, retaining only disposition and collision metadata."""
    digest = record.get("payload_hash")
    if digest is None and not _malformed(record):
        digest = payload_hash(record.get("job") or {}, record["content"], Path(record["home"]),
                              for_failure=bool(record.get("for_failure")))
    terminal = {key: record[key] for key in (
        "id", "status", "error", "home", "origin_home", "profile", "sequence", "direct", "for_failure",
        "degraded", "receipt_status",
    ) if key in record}
    terminal["payload_hash"] = digest
    return terminal


def drain_legacy_pending() -> None:
    """Admit queued legacy records to their live owner; an absent owner leaves them for a later tick.

    Each record is isolated: a malformed one is kept as ``ambiguous`` evidence and an
    unexpected failure leaves only that record queued, so one bad file never stops cron.
    """
    root = get_hermes_home().resolve() / "cron" / "bot_chat_pending"
    if not root.is_dir():
        return
    from cron import delivery_queue
    for path, snapshot in _queued(root):
        try:
            with delivery_queue._transaction() as conn:
                conn.execute("BEGIN IMMEDIATE")
                _recover_terminal_unlocked(root)
                record = _read_metadata(path, None)
                if not isinstance(record, dict) or record.get("status") != "queued":
                    continue
                reason = _malformed(record)
                if reason or path.stem != record.get("id"):
                    record.update(status="ambiguous", error=reason or "filename/id mismatch")
                    _terminalize_unlocked(path, record)
                    continue
                if record.get("origin_home", str(root.parent.parent)) != str(root.parent.parent):
                    continue  # Wrong-profile evidence is pending, never scrubbed or sent.
                digest = payload_hash(
                    record.get("job") or {}, record["content"], Path(record["home"]),
                    for_failure=bool(record.get("for_failure")))
                arbitrate(record["id"], digest, conn=conn,
                          execution_id=(record.get("job") or {}).get("execution_id"))
                record.update(status="claimed", payload_hash=digest)
                atomic_json_write(path, record, fsync_dir=True, mode=0o600)
            # A durable claim fences competing drains/producers; no lock spans transport.
            terminal = _dispose(path, record)
            with delivery_queue._transaction() as conn:
                conn.execute("BEGIN IMMEDIATE")
                if terminal:
                    _terminalize_unlocked(path, record)
                else:
                    record["status"] = "queued"
                    atomic_json_write(path, record, fsync_dir=True, mode=0o600)
        except Exception as exc:
            logger.log(logging.DEBUG if path in _warned else logging.WARNING,
                       "Legacy Bot Chat pending record %s left queued: %s", path, exc, exc_info=True)
            _warned.add(path)
            continue
        logger.info("Legacy Bot Chat pending record %s -> %s", path.name, record["status"])
