"""Defer never-started cron outputs behind unsupported Bot Chat owners.

Inspired by 686f6c61's queue proposal (#100319). Unlike retrying failed CLI
turns, only pending requests are eligible: a persisted claim never expires.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path

from hermes_cli.active_sessions import _FileLock
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)
_warned_unreadable: set[Path] = set()
_running: set[Path] = set()
_running_lock = threading.Lock()
MAX_TERMINAL_RECEIPTS = 1000
_CANONICAL_KEY = re.compile(r"[0-9a-f]{64}\Z")


def _root() -> Path:
    return get_hermes_home().resolve() / "cron" / "bot_chat_pending"


def _canonical_key(value: object) -> str | None:
    return value if isinstance(value, str) and _CANONICAL_KEY.fullmatch(value) else None


def _require_key(value: object) -> str:
    key = _canonical_key(value)
    if key is None:
        raise ValueError("deferred Bot Chat delivery id must be 64 lowercase hexadecimal characters")
    return key


def _receipt_path(root: Path, key: object) -> Path:
    return root / f"{_require_key(key)}.json"


@contextlib.contextmanager
def ownership_arbitration(key: str):
    """Serialize same-id ownership across protocols using a bounded lock stripe."""
    key = _require_key(key)
    root = _root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _FileLock(root / f".ownership-{key[:2]}.lock"):
        yield


def read_pending(key: str) -> dict | None:
    """Exact-id read: fails closed on anything but a JSON object, never licensing an overwrite."""
    try:
        record = json.loads(_receipt_path(_root(), key).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"deferred Bot Chat receipt {key} is not a JSON object ({type(record).__name__})")
    return record


def _records(root: Path) -> list[tuple[Path, dict]]:
    records = []
    for path in root.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object, got {type(record).__name__}")
            _require_key(record.get("id"))
        except (OSError, ValueError) as exc:  # ValueError: corrupt JSON and invalid UTF-8 alike
            # Keep damaged or unreadable receipts as evidence; never replay them or block peers
            # (same rule as tools/bot_live_delivery.py::_scan_read — one bad file must not wedge the dir).
            # The scheduler drains every tick: ERROR once per receipt per process, DEBUG after.
            level = logging.DEBUG if path in _warned_unreadable else logging.ERROR
            _warned_unreadable.add(path)
            logger.log(level, "Unreadable deferred Bot Chat receipt %s: %s", path, exc)
            continue
        _warned_unreadable.discard(path)
        records.append((path, record))
    return records


def payload_hash(job: dict, content: str, home: Path, *, for_failure: bool = False) -> str:
    """Canonical identity for the final redacted payload, destination, and category."""
    name = str(job.get("name", job.get("id", "?")))
    message = (
        f'[Cronjob "{name}" output — scheduled job, not the user. Review it, act on '
        f"anything that needs action, and summarize for the chat.]\n\n{content}"
    )
    canonical = {
        "category": "diagnostic" if for_failure else "result",
        "destination": str(Path(home).resolve()),
        "message": message,
    }
    return hashlib.sha256(json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _minimal_job(job: dict) -> dict:
    """Persist only fields needed to render and attribute a deferred turn."""
    return {
        "id": job.get("id", "?"),
        "name": job.get("name", job.get("id", "?")),
        **({"execution_id": job["execution_id"]} if job.get("execution_id") else {}),
    }


def _validate_payload(record: dict, digest: str) -> None:
    existing = record.get("payload_hash")
    if existing is None and "job" in record and "content" in record and record.get("home"):
        existing = payload_hash(
            record["job"], record["content"], Path(record["home"]),
            for_failure=bool(record.get("for_failure")),
        )
    if existing != digest:
        raise ValueError("delivery id already belongs to a different payload")


def _next_sequence_unlocked(root: Path) -> int:
    """Allocate order without scanning an attacker- or history-sized receipt directory."""
    path = root / ".sequence"
    try:
        previous = int(path.read_text(encoding="ascii"))
    except (FileNotFoundError, OSError, ValueError):
        previous = 0
    sequence = max(previous + 1, time.time_ns())
    path.write_text(str(sequence), encoding="ascii")
    return sequence


def _read_terminal_index(root: Path) -> list[str]:
    try:
        index = json.loads((root / ".terminal-index").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return []
    return [key for item in index if (key := _canonical_key(item))] if isinstance(index, list) else []


def _write_terminal_index_unlocked(root: Path, index: list[str], key: str) -> list[str]:
    key = _require_key(key)
    index = [valid for item in index if (valid := _canonical_key(item)) and valid != key]
    index.append(key)
    keep = max(0, int(MAX_TERMINAL_RECEIPTS))
    expired = index[:-keep] if keep else index
    index = index[-keep:] if keep else []
    for old_key in expired:
        old_path = _receipt_path(root, old_key)
        try:
            old_record = json.loads(old_path.read_text(encoding="utf-8-sig"))
            if isinstance(old_record, dict) and old_record.get("status") not in {"queued", "claimed"}:
                old_path.unlink()
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            # Damaged evidence is retained; it must not license replay or wedge pruning.
            pass
    atomic_json_write(root / ".terminal-index", index, fsync_dir=True, mode=0o600)
    return index


def _terminalize_unlocked(
    root: Path, path: Path, record: dict, status: str, error: str | None,
) -> dict:
    """Scrub replay material and retain bounded collision/outcome metadata."""
    terminal = {
        key: record[key]
        for key in (
            "id", "payload_hash", "home", "profile", "sequence", "direct",
            "for_failure", "degraded",
        )
        if key in record
    }
    terminal.update(status=status, error=error)

    index = _read_terminal_index(root)
    intent_path = root / ".terminal-intent"
    try:
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        intent = None
    intent = _canonical_key(intent)
    if intent is not None and intent not in index:
        intent_receipt = _receipt_path(root, intent)
        try:
            recovered = json.loads(intent_receipt.read_text(encoding="utf-8-sig"))
            if isinstance(recovered, dict) and recovered.get("status") not in {"queued", "claimed"}:
                index = _write_terminal_index_unlocked(root, index, intent)
        except (FileNotFoundError, OSError, ValueError):
            # Missing or damaged evidence cannot be treated as a completed terminalization.
            pass

    key = _require_key(record["id"])
    atomic_json_write(intent_path, key, fsync_dir=True, mode=0o600)
    atomic_json_write(path, terminal, fsync_dir=True, mode=0o600)
    _write_terminal_index_unlocked(root, index, key)
    return terminal


def defer(key: str, job: dict, content: str, profile: str, home: Path, *,
          for_failure: bool = False, suppressed: bool = False, degraded: bool = False,
          payload_digest: str | None = None) -> dict:
    """``degraded`` marks the short notice queued after a CLI-lane turn timed out; the record
    carries it so the consumer recognizes the marker by the record, never by its text."""
    key = _require_key(key)
    root = _root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    job = _minimal_job(job)
    digest = payload_digest or payload_hash(job, content, home, for_failure=for_failure)
    with _FileLock(root / ".lock"):
        record = read_pending(key)
        if record is not None:
            _validate_payload(record, digest)
            if suppressed and record["status"] == "queued":
                record = _terminalize_unlocked(
                    root, _receipt_path(root, key), record, "suppressed", None)
            return record
        sequence = _next_sequence_unlocked(root)
        record = dict(
            id=key, status="suppressed" if suppressed else "queued", job=job,
            content=content, profile=profile, home=str(home), sequence=sequence,
            payload_hash=digest,
        )
        if for_failure:
            record["for_failure"] = True
        if degraded:
            record["degraded"] = True
        path = _receipt_path(root, key)
        atomic_json_write(path, record, fsync_dir=True, mode=0o600)
        if suppressed:
            record = _terminalize_unlocked(root, path, record, "suppressed", None)
        return record


def claim_direct(key: str, job: dict, content: str, profile: str, home: Path, *,
                 for_failure: bool = False,
                 payload_digest: str | None = None) -> tuple[dict, bool]:
    """Persist a direct-CLI owner before launch; caller holds ``ownership_arbitration``."""
    key = _require_key(key)
    root = _root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    job = _minimal_job(job)
    digest = payload_digest or payload_hash(job, content, home, for_failure=for_failure)
    with _FileLock(root / ".lock"):
        record = read_pending(key)
        if record is not None:
            _validate_payload(record, digest)
            return record, False
        sequence = _next_sequence_unlocked(root)
        record = dict(
            id=key, status="claimed", job=job, content=content, profile=profile,
            home=str(home), sequence=sequence, direct=True, payload_hash=digest,
        )
        if for_failure:
            record["for_failure"] = True
        atomic_json_write(_receipt_path(root, key), record, fsync_dir=True, mode=0o600)
        return record, True


def finish_direct(key: str, status: str, error: str | None = None) -> None:
    """Terminalize this process's durable direct-CLI claim."""
    if status not in ("settled", "ambiguous"):
        raise ValueError(f"invalid direct delivery status: {status}")
    key = _require_key(key)
    root = _root()
    with _FileLock(root / ".lock"):
        record = read_pending(key)
        if record is None or not record.get("direct") or record.get("status") != "claimed":
            return
        _terminalize_unlocked(root, _receipt_path(root, key), record, status, error)


def release_direct_refusal(key: str) -> None:
    """Make a proven pre-admission claim retryable without dropping durable ownership."""
    key = _require_key(key)
    root = _root()
    with _FileLock(root / ".lock"):
        record = read_pending(key)
        if record is not None and record.get("direct") and record.get("status") == "claimed":
            record.update(status="queued", error="SESSION_NOT_OWNED")
            atomic_json_write(_receipt_path(root, key), record, fsync_dir=True, mode=0o600)


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
    for path, snapshot in records:
        with ownership_arbitration(snapshot["id"]), _FileLock(root / ".lock"):
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
                _terminalize_unlocked(root, path, record, "suppressed", None)
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
        # A transferred live-owner receipt remains authoritative, including queued.
        with _FileLock(root / ".lock"):
            _terminalize_unlocked(root, path, record, status, error)


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
