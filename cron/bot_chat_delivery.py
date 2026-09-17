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
    try:
        return json.loads((_root() / f"{key}.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _records(root: Path) -> list[tuple[Path, dict]]:
    records = []
    for path in root.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
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


def defer(key: str, job: dict, content: str, profile: str, home: Path) -> dict:
    root = _root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _FileLock(root / ".lock"):
        record = read_pending(key)
        if record is not None:
            if record["content"] != content or record["home"] != str(home):
                raise ValueError("delivery id already belongs to a different payload")
            return record
        sequence = max((record["sequence"] for _, record in _records(root)), default=0) + 1
        record = dict(id=key, status="queued", job=job, content=content,
                      profile=profile, home=str(home), sequence=sequence)
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
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] != "queued":
                continue
            home = Path(record["home"])
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
