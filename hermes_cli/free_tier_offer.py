"""When the free tier's "sign in for more" offer is due: after finished tasks, backing off.

The first offer comes ``OFFER_DELAY_S`` after the first finished task. Each later offer needs another
finished task after the previous offer, comes ``OFFER_DELAY_S`` after that task, and never sooner than
``REOFFER_AFTER_S[n]`` after the previous offer, so a user who keeps saying "Not now" hears it less
often. "Finished task" is ``tui_gateway.free_tier_task_done``'s call. The record is install-wide, at
``<default root>/free_tier/sign_in_offer.json``. Signing in ends the free tier, and with it every offer.
"""

from __future__ import annotations

import json
import math
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

OFFER_DELAY_S = 180
REOFFER_AFTER_S = (60 * 60, 4 * 60 * 60, 24 * 60 * 60)  # after the 1st, 2nd, then every later offer

_LOCK = threading.local()
_clock = time.time  # tests swap this for a fake clock


def _path():
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "free_tier" / "sign_in_offer.json"


def _read() -> Dict[str, Any]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("first_task_at"), (int, float)) else {}


@contextmanager
def _transaction() -> Iterator[tuple[Dict[str, Any], Any]]:
    from hermes_cli.auth import _file_lock
    from utils import atomic_write_text
    path = _path()
    with _file_lock(path.with_suffix(".lock"), _LOCK, 5.0, f"Timed out waiting for the sign-in offer record ({path})"):
        yield _read(), lambda data: atomic_write_text(path, json.dumps(data))


def _on_free_tier() -> bool:
    from hermes_cli import anon_auth
    return anon_auth.has_guest() and anon_auth.guest_enabled()


def _due_at(state: Dict[str, Any]) -> Optional[float]:
    if not state:
        return None
    offers = int(state.get("offers") or 0)
    last_task = float(state.get("last_task_at") or state["first_task_at"])
    if offers == 0:
        return float(state["first_task_at"]) + OFFER_DELAY_S
    last_offer = float(state.get("last_offered_at") or 0)
    if last_task <= last_offer:
        return None  # nothing finished since the last offer
    return max(last_task + OFFER_DELAY_S, last_offer + REOFFER_AFTER_S[min(offers, len(REOFFER_AFTER_S)) - 1])


def _due_in(state: Dict[str, Any]) -> Optional[int]:
    due_at = _due_at(state)
    if due_at is None or not _on_free_tier():
        return None
    return max(0, math.ceil(due_at - _clock()))


def record_task_done() -> None:
    """A task finished on the free tier: the first one starts the offer clock, later ones re-arm it."""
    with _transaction() as (state, write):
        now = _clock()
        write({**state, "first_task_at": state.get("first_task_at", now), "last_task_at": now})


def offer_due_in() -> Optional[int]:
    """Seconds until the offer is due (0 = now); None when none is pending or the user is not on the
    free tier. A pure read."""
    return _due_in(_read())


def claim_offer() -> bool:
    """Compare-and-set under the lock: True for exactly one caller each time an offer comes due."""
    with _transaction() as (state, write):
        if _due_in(state) != 0:
            return False
        write({**state, "offers": int(state.get("offers") or 0) + 1, "last_offered_at": _clock()})
    return True
