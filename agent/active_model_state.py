"""Which model actually served the last turn, for the UI to show.

The dashboard's model badge is built from the *configured* model
(``GET /api/model/info``), so a runtime fallback to a backup model is invisible:
the badge keeps naming the primary while the answers come from the fallback, and
the operator only notices from the answer quality. The agent knows the truth at
the points where the effective model flips — fallback activation, the restore
attempt at the start of the next turn, and a deliberate ``/model`` switch — so
each of those records a small snapshot here and ``/api/model/info`` reads it
back.

The snapshot lives at ``<HERMES_HOME>/active_model.json`` (profile-aware, like
the other runtime bookkeeping). A read only trusts a snapshot younger than
``max_age_seconds``, so one left behind by a dead process cannot pin the UI to a
model forever. Every function is best-effort — bookkeeping must never break a
turn — so unreadable state degrades to "unknown" instead of raising.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# A snapshot stays authoritative for an hour. Long sessions re-record it on every
# turn (see ``_should_write``), so the horizon only decides how long a snapshot
# outlives the process that wrote it — idle profile, closed chat, crashed TUI.
DEFAULT_MAX_AGE_SECONDS = 3600.0

_lock = threading.Lock()


def active_model_path(home: Path | str | None = None) -> Path:
    """``<HERMES_HOME>/active_model.json`` — *home* overrides the ambient profile."""
    return (Path(home) if home is not None else get_hermes_home()) / "active_model.json"


def record_agent_model(agent: Any, *, home: Path | str | None = None, now: float | None = None) -> None:
    """Snapshot the route *agent* is currently bound to. Never raises."""
    record_active_model(
        getattr(agent, "model", ""), getattr(agent, "provider", ""),
        fallback=bool(getattr(agent, "_provider_fallback_active", False)), home=home, now=now,
    )


def record_active_model(model: Any, provider: Any = "", *, fallback: bool = False,
                        home: Path | str | None = None, now: float | None = None) -> None:
    """Record the model now serving turns (``fallback=True`` when a backup took over)."""
    name = str(model or "").strip()
    if not name:
        return
    stamp = float(now if now is not None else time.time())
    snapshot = {"model": name, "provider": str(provider or "").strip(), "fallback": bool(fallback),
                "at": stamp}
    path = active_model_path(home)
    try:
        with _lock:
            if not _should_write(_load(path), snapshot):
                return
            atomic_json_write(path, snapshot, mode=0o600)
    except Exception:  # pragma: no cover - bookkeeping is never worth a failed turn
        logger.debug("could not record the active model in %s", path, exc_info=True)


def _should_write(current: dict, snapshot: dict) -> bool:
    """Write on a route change, and otherwise only to keep a live record fresh.

    Re-recording the same route every turn would rewrite the file for nothing, and
    concurrent sessions on one profile would fight over it; but never refreshing it
    would expire the record under a fallback that lasts longer than the horizon.
    So: same route → refresh at most once per half-horizon.
    """
    if current.get("model") != snapshot["model"] or current.get("provider") != snapshot["provider"]:
        return True
    if bool(current.get("fallback")) != snapshot["fallback"]:
        return True
    return snapshot["at"] - float(current.get("at") or 0) >= DEFAULT_MAX_AGE_SECONDS / 2


def read_active_model(*, home: Path | str | None = None, max_age_seconds: float | None = DEFAULT_MAX_AGE_SECONDS,
                      now: float | None = None) -> dict:
    """The recorded snapshot, or ``{}`` when it is missing, stale or unreadable."""
    path = active_model_path(home)
    snapshot = _load(path)
    if not snapshot:
        return {}
    model = str(snapshot.get("model") or "").strip()
    if not model:
        return {}
    at = float(snapshot.get("at") or 0)
    if max_age_seconds is not None and float(now if now is not None else time.time()) - at > max_age_seconds:
        return {}
    return {"model": model, "provider": str(snapshot.get("provider") or ""),
            "fallback": bool(snapshot.get("fallback")), "at": at}


def clear_active_model(*, home: Path | str | None = None) -> None:
    """Drop the snapshot (tests, and any future explicit "forget" path)."""
    try:
        active_model_path(home).unlink(missing_ok=True)
    except Exception:  # pragma: no cover - best-effort like every write here
        logger.debug("could not clear the active-model snapshot", exc_info=True)


def _load(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("unreadable active-model snapshot %s; treating it as unknown", path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}
