"""Defer never-started cron outputs behind unsupported Bot Chat owners.

Inspired by 686f6c61's queue proposal (#100319). Unlike retrying failed CLI
turns, only pending requests are eligible: a persisted claim never expires.
"""
from __future__ import annotations

import contextvars
import json
import logging
import threading
from pathlib import Path

from hermes_cli.active_sessions import _FileLock
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)
_warned_unreadable: set[Path] = set()
_running: set[Path] = set()
_running_lock = threading.Lock()


def _root() -> Path:
    return get_hermes_home().resolve() / "cron" / "bot_chat_pending"


def read_pending(key: str) -> dict | None:
    """Exact-id read: fails closed on anything but a JSON object, never licensing an overwrite."""
    try:
        record = json.loads((_root() / f"{key}.json").read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"deferred Bot Chat receipt {key} is not a JSON object ({type(record).__name__})")
    return record


_REQUIRED_RECEIPT_KEYS = ("id", "status", "sequence")
_QUEUED_RECEIPT_KEYS = ("home", "job", "content", "profile")


def _receipt_shape_error(record: dict) -> str | None:
    """Why a parsed Bot Chat receipt is unusable by the scans, or None when it is well-formed.

    A receipt that parses as a JSON object but lost a field (truncated rewrite,
    foreign writer, hand edit) used to raise KeyError out of the sequence scan
    and the drain — wedging admission and delivery for the whole profile the way
    corrupt JSON did before ``_records`` learned to skip it. Missing keys — and,
    for a ``queued`` receipt, required keys whose value is unusable (``home``/
    ``content`` not ``str``, ``job`` not ``dict``, both of which the drain would
    raise on) — are treated as unreadable: logged once, kept as evidence, skipped.
    """
    missing = [key for key in _REQUIRED_RECEIPT_KEYS if key not in record]
    if record.get("status") == "queued":
        missing += [key for key in _QUEUED_RECEIPT_KEYS if key not in record]
    if missing:
        return f"missing {', '.join(missing)}"
    if not isinstance(record["sequence"], int) or isinstance(record["sequence"], bool):
        return "sequence is not an integer"
    if record["status"] == "queued":
        # A present-but-unusable value is the same incident class as a missing key: the
        # drain would raise TypeError/AttributeError out of its loop and leave every peer
        # behind it ``queued`` on every tick. ``profile`` stays unchecked — it is read
        # defensively (``record['profile'] or '(own)'``) and a bad one cannot wedge peers.
        for key, expected in (("home", str), ("content", str), ("job", dict)):
            if not isinstance(record[key], expected):
                return f"{key} is not a {expected.__name__}"
    return None


def _skip_receipt(path: Path, reason: str) -> None:
    # Keep damaged or unreadable receipts as evidence; never replay them or block peers
    # (same rule as tools/bot_live_delivery.py::_scan_read — one bad file must not wedge the dir).
    # The scheduler drains every tick: ERROR once per receipt per process, DEBUG after.
    level = logging.DEBUG if path in _warned_unreadable else logging.ERROR
    _warned_unreadable.add(path)
    logger.log(level, "Unreadable deferred Bot Chat receipt %s: %s", path, reason)


def _records(root: Path) -> list[tuple[Path, dict]]:
    records = []
    for path in root.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object, got {type(record).__name__}")
        except (OSError, ValueError) as exc:  # ValueError: corrupt JSON and invalid UTF-8 alike
            _skip_receipt(path, str(exc))
            continue
        problem = _receipt_shape_error(record)
        if problem is not None:
            _skip_receipt(path, problem)
            continue
        _warned_unreadable.discard(path)
        records.append((path, record))
    return records


def defer(key: str, job: dict, content: str, profile: str, home: Path, *,
          for_failure: bool = False, suppressed: bool = False, degraded: bool = False) -> dict:
    """``degraded`` marks the short notice queued after a CLI-lane turn timed out; the record
    carries it so the consumer recognizes the marker by the record, never by its text."""
    root = _root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _FileLock(root / ".lock"):
        record = read_pending(key)
        if record is not None:
            if (record["content"] != content or record["home"] != str(home)
                    or bool(record.get("for_failure")) != for_failure):
                raise ValueError("delivery id already belongs to a different payload")
            if suppressed and record["status"] == "queued":
                record.update(status="suppressed", error=None)
                atomic_json_write(root / f"{key}.json", record, fsync_dir=True, mode=0o600)
            return record
        sequence = max((record["sequence"] for _, record in _records(root)), default=0) + 1
        record = dict(id=key, status="suppressed" if suppressed else "queued", job=job, content=content,
                      profile=profile, home=str(home), sequence=sequence)
        if for_failure:
            record["for_failure"] = True
        if degraded:
            record["degraded"] = True
        atomic_json_write(root / f"{key}.json", record, fsync_dir=True, mode=0o600)
        return record


def drain(root: Path | None = None) -> None:
    """Serialize drains across processes without holding the producer lock."""
    from hermes_cli.backend_retirement import retirement

    root = root if root is not None else _root()
    with retirement.work() as admitted:
        if admitted and root.is_dir():
            with _FileLock(root / ".drain.lock"):
                _drain(root)


def _drain(root: Path) -> None:
    """Claim before execution. Errors/interruptions never authorize another turn."""
    from cron.scheduler_delivery import _deliver_to_bot_chat
    from tools.bot_live_delivery import find_canonical_live_owner, find_canonical_owner

    with _FileLock(root / ".lock"):
        records = sorted(_records(root), key=lambda item: item[1]["sequence"])
    for path, _ in records:
        with _FileLock(root / ".lock"):
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict) or record["status"] != "queued":
                continue
            home = Path(record["home"])
            # A failure notice queued before the target profile opted out is settled as
            # suppressed at drain time; the policy is the owner's, read from its own config.
            from cron.scheduler_delivery import BOT_CHAT_POLICY_PLATFORM
            from gateway.warning_notifications import warning_notifications_enabled
            from hermes_cli.config_effective import load_user_config_effective
            if (record.get("for_failure")
                    and not warning_notifications_enabled(BOT_CHAT_POLICY_PLATFORM, load_user_config_effective(home / "config.yaml"))):
                record.update(status="suppressed", error=None)
                atomic_json_write(path, record, fsync_dir=True, mode=0o600)
                continue
            try:
                owner = find_canonical_owner(home)
                if owner is not None and find_canonical_live_owner(home) is None:
                    continue
            except Exception:
                # Discovery uncertainty is not permission to launch.
                continue
            record["status"] = "claimed"
            atomic_json_write(path, record, fsync_dir=True, mode=0o600)
        job = record["job"]
        job.pop("_bot_chat_delivery_receipts", None)
        try:
            error = _deliver_to_bot_chat(job, record["content"], record["profile"], deferred=record)
        except Exception as exc:
            # The claim survives uncertainty; one failed attempt must not stop peers.
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Deferred Bot Chat delivery %s failed", record["id"])
        receipt = job.get("_bot_chat_delivery_receipts", {}).get(
            f"bot-chat:{record['profile'] or '(own)'}")
        status = "transferred" if receipt else "ambiguous" if error else "settled"
        if job.get("_notification_all_targets_suppressed"):
            status = "suppressed"
        record.update(status=status, error=error)
        # A transferred live-owner receipt remains authoritative, including queued.
        atomic_json_write(path, record, fsync_dir=True, mode=0o600)


def drain_in_background() -> None:
    """Do not hold up unrelated cron ticks while the eventual Bot Chat turn runs."""
    home = get_hermes_home().resolve()
    root = home / "cron" / "bot_chat_pending"
    if not root.is_dir():
        return
    from hermes_cli.backend_retirement import retirement

    with _running_lock:
        if home in _running or not retirement.acquire():
            return
        _running.add(home)

    def release():
        with _running_lock:
            _running.discard(home)
        retirement.release()

    def run():
        try:
            drain(root)
        finally:
            release()

    try:
        threading.Thread(target=contextvars.copy_context().run, args=(run,), daemon=True,
                         name="cron-bot-chat-drain").start()
    except BaseException:
        release()
        raise
