"""Queue ``message_agent`` DMs behind a busy local Bot Chat instead of failing them.

A local teammate's Bot Chat can be actively owned by another live surface (Desktop UI
open, another interactive session) — ``tools/bot_mode_dm.py::_run_local_turn`` detects
this as ``SESSION_NOT_OWNED`` when it tries to run the delivery turn. That used to be a
hard failure the sender saw as a completion with ``reason=target_busy``, so
agent-to-agent decisions silently fell back to other channels (#116210-class reports).

Unlike ``cron/bot_chat_delivery.py``'s queue (cron job output notices, which never
expire — a human always eventually sees the pending job), an agent-to-agent DM needs a
bound: a human's Bot Chat can stay open indefinitely, so an undelivered DM must
eventually surface as a loud log rather than waiting forever. Each queued record carries
its own TTL; the scheduler drain (see ``cron/scheduler_tick.py``) attempts redelivery on
every tick until it either succeeds (marked ``delivered``, exactly once — the record
leaves ``queued`` status so a later tick can never redeliver it) or the TTL elapses
(marked ``expired`` and logged loudly, never retried again).
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

from hermes_cli.active_sessions import _FileLock
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)
_warned_unreadable: set[Path] = set()
_running: set[Path] = set()
_running_lock = threading.Lock()

#: Long enough to outlast a typical interactive Bot Chat session left open on another
#: surface; short enough that a stuck delivery surfaces within the same working day.
DEFAULT_TTL_SECONDS = 2 * 60 * 60


def _root() -> Path:
    return get_hermes_home().resolve() / "cron" / "bot_dm_pending"


def read_pending(key: str) -> dict | None:
    """Exact-id read: fails closed on anything but a JSON object, never licensing an overwrite."""
    try:
        record = json.loads((_root() / f"{key}.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(record, dict):
        raise ValueError(f"queued Bot DM {key} is not a JSON object ({type(record).__name__})")
    return record


def _records(root: Path) -> list[tuple[Path, dict]]:
    records = []
    for path in root.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object, got {type(record).__name__}")
        except (OSError, ValueError) as exc:  # ValueError: corrupt JSON and invalid UTF-8 alike
            # Same rule as bot_chat_delivery/bot_live_delivery: one bad file never wedges the
            # dir or re-logs every tick — ERROR once per receipt per process, DEBUG after.
            level = logging.DEBUG if path in _warned_unreadable else logging.ERROR
            _warned_unreadable.add(path)
            logger.log(level, "Unreadable queued Bot DM %s: %s", path, exc)
            continue
        _warned_unreadable.discard(path)
        records.append((path, record))
    return records


def enqueue_busy_dm(*, delivery_id: str, argv: list[str], content: str, label: str,
                     author: Optional[dict] = None, profile_home: "Path | str | None" = None,
                     ttl_seconds: float = DEFAULT_TTL_SECONDS, now: float | None = None) -> dict:
    """Persist a busy DM for replay. Idempotent on ``delivery_id``: a retried dispatch of the
    SAME message (e.g. a crash-recovered runner replaying its own delivery id) returns the
    existing record instead of creating a second queue entry — at most one delivery, ever."""
    root = _root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _FileLock(root / ".lock"):
        existing = read_pending(delivery_id)
        if existing is not None:
            if existing["content"] != content or existing["label"] != label:
                raise ValueError(f"delivery id {delivery_id} already belongs to a different DM")
            return existing
        record = dict(
            id=delivery_id, status="queued", argv=list(argv), content=content, label=label,
            author=author, profile_home=str(profile_home) if profile_home else None,
            enqueued_at=now if now is not None else time.time(), ttl_seconds=ttl_seconds, attempts=0,
        )
        atomic_json_write(root / f"{delivery_id}.json", record, fsync_dir=True, mode=0o600)
        return record


def drain(root: Path | None = None, *, now: float | None = None) -> None:
    """One replay pass over every queued DM: run from the cron ticker (every profile's own
    tick — see ``cron/scheduler_tick.py``). Serialized across processes on a dedicated drain
    lock so two ticks racing on the same store never attempt the same record twice."""
    root = root if root is not None else _root()
    if not root.is_dir():
        return
    with _FileLock(root / ".drain.lock"):
        _drain(root, now=now)


def _drain(root: Path, *, now: float | None = None) -> None:
    from tools.bot_mode_dm import _delivery_lock, _run_local_turn
    from tools.bot_relay import delivery_env

    now = now if now is not None else time.time()
    with _FileLock(root / ".lock"):
        records = sorted(_records(root), key=lambda item: item[1]["enqueued_at"])
    for path, _ in records:
        # Re-read under the claim lock: a sibling tick may have already settled or expired
        # this record between the sort above and this iteration.
        with _FileLock(root / ".lock"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("status") != "queued":
                continue
            if now >= record["enqueued_at"] + record["ttl_seconds"]:
                record.update(status="expired", expired_at=now)
                atomic_json_write(path, record, fsync_dir=True, mode=0o600)
                expired = record
            else:
                record["status"] = "claimed"
                atomic_json_write(path, record, fsync_dir=True, mode=0o600)
                expired = None
        if expired is not None:
            # Loud on purpose: this is the ONLY signal anyone gets that the message never
            # arrived — the sender was already told 'queued_busy', not a hard failure, and is
            # not notified again.
            logger.error(
                "Bot DM to %s expired undelivered after %.0fs (delivery_id=%s, attempts=%d): "
                "the sender was told the message was queued and will NOT be told again — it "
                "was NEVER delivered.",
                expired["label"], expired["ttl_seconds"], expired["id"], expired.get("attempts", 0),
            )
            continue

        fd, dm_file = tempfile.mkstemp(prefix="dm-retry-", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(record["content"])
            env = delivery_env(record.get("author"), record.get("profile_home"))
            # Same per-profile turn serialization a live delivery gets — a replay must never
            # race a concurrent live delivery into the SAME target profile (#93091).
            with _delivery_lock(record["argv"], stdin_file=False):
                rc = _run_local_turn(record["argv"], dm_file, env=env)
        except Exception:
            logger.exception("Queued Bot DM replay to %s crashed (delivery_id=%s)", record["label"], record["id"])
            rc = 1
        finally:
            with suppress(OSError):
                os.unlink(dm_file)

        with _FileLock(root / ".lock"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if rc == 0:
                record.update(status="delivered", delivered_at=time.time())
            else:
                # Still busy (or a transient turn error): left for the next tick, bounded
                # only by the TTL check above — never retried past it.
                record["attempts"] = record.get("attempts", 0) + 1
                record["status"] = "queued"
            atomic_json_write(path, record, fsync_dir=True, mode=0o600)


def drain_in_background() -> None:
    """Do not hold up unrelated cron ticks while a replayed Bot Chat turn runs."""
    home = get_hermes_home().resolve()
    root = home / "cron" / "bot_dm_pending"
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
        except Exception:
            logger.exception("Background Bot DM drain failed for %s", home)
        finally:
            release()

    try:
        threading.Thread(target=contextvars.copy_context().run, args=(run,), daemon=True,
                         name="cron-bot-dm-drain").start()
    except BaseException:
        release()
        raise
