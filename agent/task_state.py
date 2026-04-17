"""Task-state persistence + continuity logic for routing v2.

Goals:
- Track whether a heavy task is active across turns.
- Detect continuation/silence markers so tiny inputs like "dale", "sigue",
  "hazlo", "ok" do NOT downgrade the model mid-task.
- Track easy_streak so the router can gradually descend *only* after two
  genuinely easy turns in a row.
- Zero external deps. JSON-backed for inspection and portability.

Public API:
    load(path) -> dict
    save(path, state) -> None
    start_task(state, *, tier, model, category) -> dict
    record_turn(state, user_message, *, was_easy=False) -> dict
    is_continuation(message) -> bool
    is_silence(message) -> bool

The state dict has the following keys:
    active_task: bool
    last_tier: int
    last_model: str
    last_category: str
    turns_in_task: int
    easy_streak: int
    last_updated: float
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

_CONTINUATION_MARKERS = {
    "continúa", "continua", "sigue", "sigue con", "sigue con eso",
    "dale", "resume", "resúmelo", "mismo tema", "ok", "hazlo",
    "haz lo tuyo", "hazlo ya", "continue", "go on", "keep going",
    "proceed", "ya",
}
_WORD_BOUNDARY_MARKERS = {m for m in _CONTINUATION_MARKERS if " " not in m}
_PHRASE_MARKERS = {m for m in _CONTINUATION_MARKERS if " " in m}

_DEFAULT_STATE: Dict[str, Any] = {
    "active_task": False,
    "last_tier": 0,
    "last_model": "",
    "last_category": "",
    "turns_in_task": 0,
    "easy_streak": 0,
    "last_updated": 0.0,
}


def default_state() -> Dict[str, Any]:
    return dict(_DEFAULT_STATE)


def load(path: str | os.PathLike) -> Dict[str, Any]:
    p = Path(os.path.expanduser(str(path)))
    if not p.exists():
        return default_state()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default_state()
    # merge with defaults to tolerate older on-disk schemas
    merged = default_state()
    merged.update({k: v for k, v in (data or {}).items() if k in merged})
    return merged


def save(path: str | os.PathLike, state: Dict[str, Any]) -> None:
    p = Path(os.path.expanduser(str(path)))
    p.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["last_updated"] = time.time()
    p.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def is_silence(message: Optional[str]) -> bool:
    return not (message and message.strip())


def is_continuation(message: Optional[str]) -> bool:
    if not message:
        return False
    lowered = message.strip().lower()
    if not lowered:
        return False
    # exact short phrase match first
    if lowered in _CONTINUATION_MARKERS:
        return True
    for phrase in _PHRASE_MARKERS:
        if phrase in lowered:
            return True
    # word-boundary match for single-word markers
    for w in _WORD_BOUNDARY_MARKERS:
        if re.search(rf"\b{re.escape(w)}\b", lowered):
            return True
    return False


def start_task(state: Dict[str, Any], *, tier: int, model: str, category: str) -> Dict[str, Any]:
    s = dict(state)
    s.update({
        "active_task": True,
        "last_tier": int(tier),
        "last_model": model,
        "last_category": category,
        "turns_in_task": 1,
        "easy_streak": 0,
    })
    return s


def record_turn(state: Dict[str, Any], user_message: Optional[str], *, was_easy: bool = False) -> Dict[str, Any]:
    """Return an updated state dict based on the new turn.

    was_easy: caller's estimate (length/heuristic) that the turn is simple.
    Continuation markers keep active_task True and increment turns_in_task.
    Silence keeps the task alive too (avoids a silent downgrade).
    Easy non-continuation turns increment easy_streak; anything else resets it.
    """
    s = dict(state)
    cont = is_continuation(user_message)
    silent = is_silence(user_message)
    if cont or silent:
        s["active_task"] = True
        s["turns_in_task"] = int(s.get("turns_in_task", 0)) + 1
        # Do NOT reset easy_streak on silence/continuation — neutral event.
    else:
        if was_easy:
            s["easy_streak"] = int(s.get("easy_streak", 0)) + 1
            s["turns_in_task"] = int(s.get("turns_in_task", 0)) + 1
        else:
            s["easy_streak"] = 0
            s["turns_in_task"] = int(s.get("turns_in_task", 0)) + 1
    return s
