"""Boot intents: a durable "run this next step after the planned restart" marker.

Contract
--------
- Written *before* a restart (agent via ``hermes boot-intent add``, or a human).
- Read once at gateway boot by ``_schedule_boot_intents`` and claimed atomically, so a
  concurrent boot / reconnect rescan never runs the same intent twice.
- A claim whose consumer died is re-claimable after :data:`CLAIM_TIMEOUT_SECONDS`.
- Expired intents (``created_at + ttl_seconds``) are marked ``result="expired"`` and never
  fire late.
- Deliberately independent of ``sessions.json`` / ``resume_pending``: the boot scheduler
  de-duplicates against those, and a bug here can never corrupt session routing.

Store: ``$HERMES_HOME/boot_intents.json`` — ``{"version": 1, "intents": [ ... ]}``.

An intent is written as *precondition + step + stop point*, never as a bare action:
a bare action re-executes work that already finished before the restart.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

INTENTS_FILENAME = "boot_intents.json"
DEFAULT_TTL_SECONDS = 24 * 3600.0
# A claim older than this is assumed abandoned (the consumer died mid-turn) and may be retaken.
CLAIM_TIMEOUT_SECONDS = 600.0
STATE_VERSION = 1


def boot_intents_path(hermes_home: Optional[Path | str] = None) -> Path:
    return Path(hermes_home or get_hermes_home()) / INTENTS_FILENAME


def _msvcrt_lock(fh, *, unlock: bool = False) -> None:
    """Windows byte-range lock. ``getattr`` keeps static checkers quiet on POSIX."""
    import msvcrt

    fh.seek(0)
    getattr(msvcrt, "locking")(fh.fileno(), getattr(msvcrt, "LK_UNLCK" if unlock else "LK_LOCK"), 1)  # noqa: B009


@contextlib.contextmanager
def _locked(path: Path):
    """Exclusive inter-process lock beside the store.

    Degrades to a no-op when the platform has no working lock primitive: a lost read-modify-write
    race can duplicate a turn, which is bad, but failing boot entirely is worse.
    """
    lock_path = path.with_suffix(".lock")
    fh = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+", encoding="utf-8")
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except ImportError:  # Windows
            _msvcrt_lock(fh)
    except Exception as exc:  # health: allow BLE001 -- lock is best-effort
        logger.debug("boot-intent lock unavailable (%s); proceeding unlocked", exc)
        if fh is not None:
            fh.close()
        fh = None
    try:
        yield
    finally:
        if fh is not None:
            try:
                try:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    _msvcrt_lock(fh, unlock=True)
            except Exception:  # health: allow BLE001 -- unlocking never raises
                pass
            fh.close()


def _empty() -> dict:
    return {"version": STATE_VERSION, "intents": []}


def _load(path: Path) -> dict:
    """Read the store; every failure mode returns an empty one instead of raising."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return _empty()
    except OSError as exc:
        logger.warning("boot intents unreadable at %s (%s); treating as empty", path, exc)
        return _empty()
    try:
        data = json.loads(raw or "{}")
    except (json.JSONDecodeError, ValueError):
        logger.warning("boot intents corrupt at %s; treating as empty (left on disk)", path)
        return _empty()
    if not isinstance(data, dict) or not isinstance(data.get("intents"), list):
        logger.warning("boot intents malformed at %s; treating as empty", path)
        return _empty()
    return {
        "version": STATE_VERSION,
        "intents": [item for item in data["intents"] if isinstance(item, dict)],
    }


def _save(path: Path, data: dict) -> None:
    atomic_json_write(path, data, mode=0o600)


def _quarantine(path: Path) -> None:
    """Move a corrupt store aside before overwriting, so evidence survives the next add."""
    try:
        if not path.exists():
            return
        raw = path.read_text(encoding="utf-8-sig")
        json.loads(raw or "{}")
        return  # readable: nothing to quarantine
    except Exception:  # health: allow BLE001 -- quarantine is best-effort
        try:
            path.replace(path.with_suffix(f".corrupt-{int(time.time())}.json"))
        except Exception as exc:  # health: allow BLE001
            logger.debug("boot-intent quarantine failed: %s", exc)


def new_intent_id(now: Optional[float] = None) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now if now is not None else time.time()))
    return f"bi_{stamp}_{uuid.uuid4().hex[:6]}"


def add_intent(
    session_key: str,
    prompt: str,
    *,
    profile: Optional[str] = None,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    hermes_home: Optional[Path | str] = None,
    now: Optional[float] = None,
    source: str = "agent",
) -> dict:
    """Append one intent and return it. ``session_key`` + ``prompt`` are mandatory."""
    session_key = (session_key or "").strip()
    prompt = (prompt or "").strip()
    if not session_key:
        raise ValueError("session_key is required")
    if not prompt:
        raise ValueError("prompt is required")
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    now = time.time() if now is None else float(now)
    path = boot_intents_path(hermes_home)
    item = {
        "id": new_intent_id(now),
        "session_key": session_key,
        "profile": profile,
        "prompt": prompt,
        "created_at": now,
        "ttl_seconds": float(ttl_seconds),
        "claimed_at": None,
        "consumed_at": None,
        "result": None,
        "source": source,
    }
    with _locked(path):
        _quarantine(path)
        data = _load(path)
        data["intents"].append(item)
        _save(path, data)
    logger.info("Boot intent registered: %s session=%s ttl=%ss", item["id"], session_key, int(ttl_seconds))
    return dict(item)


def list_intents(
    *,
    include_consumed: bool = False,
    hermes_home: Optional[Path | str] = None,
) -> list[dict]:
    data = _load(boot_intents_path(hermes_home))
    items = data["intents"]
    if not include_consumed:
        items = [item for item in items if not item.get("consumed_at")]
    return [dict(item) for item in items]


def claim_intents(
    *,
    hermes_home: Optional[Path | str] = None,
    now: Optional[float] = None,
) -> list[dict]:
    """Atomically claim every live intent; sweeps expired ones in the same pass.

    Returns the claimed copies (``claimed_at`` stamped). A claimed intent stays on disk so
    ``complete_intent`` can record the outcome; the claim timeout makes an abandoned one
    re-claimable instead of wedging it forever.
    """
    now = time.time() if now is None else float(now)
    path = boot_intents_path(hermes_home)
    claimed: list[dict] = []
    with _locked(path):
        data = _load(path)
        changed = False
        for item in data["intents"]:
            if item.get("consumed_at"):
                continue
            created = float(item.get("created_at") or 0)
            ttl = float(item.get("ttl_seconds") or DEFAULT_TTL_SECONDS)
            if created and now > created + ttl:
                item["consumed_at"] = now
                item["result"] = "expired"
                changed = True
                logger.info("Boot intent %s expired (ttl=%ss); skipped", item.get("id"), int(ttl))
                continue
            claimed_at = item.get("claimed_at")
            if claimed_at and (now - float(claimed_at)) < CLAIM_TIMEOUT_SECONDS:
                continue  # another boot or reconnect rescan still owns it
            item["claimed_at"] = now
            claimed.append(dict(item))
            changed = True
        if changed:
            _save(path, data)
    return claimed


def release_intent(
    intent_id: str,
    *,
    hermes_home: Optional[Path | str] = None,
) -> bool:
    """Drop a claim so a later boot / reconnect rescan retries it."""
    path = boot_intents_path(hermes_home)
    with _locked(path):
        data = _load(path)
        for item in data["intents"]:
            if item.get("id") == intent_id and not item.get("consumed_at"):
                item["claimed_at"] = None
                _save(path, data)
                return True
    return False


def complete_intent(
    intent_id: str,
    result: str = "done",
    *,
    hermes_home: Optional[Path | str] = None,
    now: Optional[float] = None,
) -> bool:
    now = time.time() if now is None else float(now)
    path = boot_intents_path(hermes_home)
    with _locked(path):
        data = _load(path)
        for item in data["intents"]:
            if item.get("id") == intent_id and not item.get("consumed_at"):
                item["consumed_at"] = now
                item["result"] = result
                _save(path, data)
                logger.info("Boot intent %s completed: %s", intent_id, result)
                return True
    return False


def clear_intents(
    *,
    intent_id: Optional[str] = None,
    hermes_home: Optional[Path | str] = None,
) -> int:
    """Remove intents (all, or one by id). Returns how many rows were dropped."""
    path = boot_intents_path(hermes_home)
    with _locked(path):
        data = _load(path)
        before = len(data["intents"])
        if intent_id:
            data["intents"] = [i for i in data["intents"] if i.get("id") != intent_id]
        else:
            data["intents"] = []
        if len(data["intents"]) != before:
            _save(path, data)
        return before - len(data["intents"])


def has_pending_intent(
    session_key: str,
    *,
    hermes_home: Optional[Path | str] = None,
) -> bool:
    """Cheap check used by the restart-notice path to pick its wording."""
    for item in list_intents(hermes_home=hermes_home):
        if item.get("session_key") == session_key:
            return True
    return False
