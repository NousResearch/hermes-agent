"""Defer never-started cron outputs behind unsupported Bot Chat owners.

Inspired by 686f6c61's queue proposal (#100319). Unlike retrying failed CLI
turns, only pending requests are eligible: a persisted claim never expires.
"""
from __future__ import annotations

import contextvars
import json
import threading
from pathlib import Path

from hermes_cli.active_sessions import _FileLock
from hermes_constants import get_hermes_home
from utils import atomic_json_write

_running: set[Path] = set()
_running_lock = threading.Lock()


def _root() -> Path:
    return get_hermes_home().resolve() / "cron" / "bot_chat_pending"


def read_pending(key: str) -> dict | None:
    try:
        return json.loads((_root() / f"{key}.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def defer(key: str, job: dict, content: str, profile: str, home: Path) -> dict:
    root = _root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _FileLock(root / ".lock"):
        record = read_pending(key)
        if record is not None:
            if record["content"] != content or record["home"] != str(home):
                raise ValueError("delivery id already belongs to a different payload")
            return record
        sequence = max((json.loads(p.read_text(encoding="utf-8"))["sequence"]
                        for p in root.glob("*.json")), default=0) + 1
        record = dict(id=key, status="queued", job=job, content=content,
                      profile=profile, home=str(home), sequence=sequence)
        atomic_json_write(root / f"{key}.json", record, fsync_dir=True, mode=0o600)
        return record


def drain() -> None:
    """Serialize drains across processes without holding the producer lock."""
    root = _root()
    if root.is_dir():
        with _FileLock(root / ".drain.lock"):
            _drain(root)


def _drain(root: Path) -> None:
    """Claim before execution. Errors/interruptions never authorize another turn."""
    from cron.scheduler_delivery import _deliver_to_bot_chat
    from tools.bot_live_delivery import find_canonical_live_owner, find_canonical_owner

    paths = sorted(root.glob("*.json"),
                   key=lambda p: json.loads(p.read_text(encoding="utf-8"))["sequence"])
    for path in paths:
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
        error = _deliver_to_bot_chat(record["job"], record["content"], record["profile"], deferred=True)
        record.update(status="ambiguous" if error else "settled", error=error)
        # A transferred live-owner receipt remains authoritative, including queued.
        atomic_json_write(path, record, fsync_dir=True, mode=0o600)


def drain_in_background() -> None:
    """Do not hold up unrelated cron ticks while the eventual Bot Chat turn runs."""
    home = get_hermes_home().resolve()
    if not _root().is_dir():
        return
    with _running_lock:
        if home in _running:
            return
        _running.add(home)

    def run():
        try:
            drain()
        finally:
            with _running_lock:
                _running.discard(home)

    threading.Thread(target=contextvars.copy_context().run, args=(run,), daemon=True,
                     name="cron-bot-chat-drain").start()
