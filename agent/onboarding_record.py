"""Where a first-run onboarding run lives: one ``onboarding_run`` key in the session's
``sessions.model_config`` (no schema change; a compression rotation carries it, see
``hermes_state_compression._CONVERSATION_MODEL_CONFIG_KEYS``).

``advance`` is the one writer after ``start``: load, ``onboarding_flow.apply``, save, and when the
run ends write the install-level flag in the root config so the next launch does not reopen it.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from agent.onboarding_flow import (
    Applied, End, Event, Facts, Step, applies_to, apply, is_active, new_record,
)
from agent.onboarding_text import reminder

logger = logging.getLogger(__name__)

KEY = "onboarding_run"

# Told when a run stops being in progress: ``{run_id, stored_session_id, outcome}``. The gateway sends
# it to every window (tui_gateway/methods_onboarding.py::register).
_ended_listeners: list[Callable[[dict], None]] = []


def on_ended(listener: Callable[[dict], None]) -> None:
    _ended_listeners.append(listener)


def load(db, session_id: str) -> dict | None:
    record = db.get_session_model_config_value(session_id, KEY) if db is not None and session_id else None
    return record if isinstance(record, dict) else None


def _announce(run_id: str | None, session_id: str, outcome: str) -> None:
    ended = {"run_id": run_id or "", "stored_session_id": session_id, "outcome": outcome}
    for listener in _ended_listeners:
        listener(ended)


def _transition(db, session_id: str, event: Event) -> Step | None:
    """Apply *event* and write the result in one database transaction at the compression tip, so Skip
    setup pressed while a card answer is being applied is never overwritten by the older copy, and an id
    read before a rotation published still lands on the copy the chat reads. An ended run then closes the
    install flag (a Skip or Done retried after a failed config write repairs it), unless another run
    is still open: the flag belongs to that run, and an old chat must not end it."""
    holder: dict[str, Any] = {}

    def update(current: Any) -> Any:
        holder.clear()
        if not isinstance(current, dict):
            return current
        holder["was_active"] = is_active(current)
        holder["step"] = apply(current, event, Facts.from_stored(current["facts"]), now=time.time())
        return holder["step"].record

    db.update_conversation_model_config_value(session_id, KEY, update)
    step = holder.get("step")
    # Told before the flag write, which can fail: the record already says the run ended.
    if step is not None and holder["was_active"] and not is_active(step.record):
        _announce(step.record.get("run_id"), session_id, step.record["ended_by"])
    if step is not None and not is_active(step.record) and active_session_id(db) is None:
        _close_install_flag()
    return step


def _close_install_flag() -> None:
    from hermes_cli import config as config_mod
    from hermes_cli.onboarding_run import set_run
    from hermes_constants import get_default_hermes_root

    from agent.onboarding import PROFILE_BUILD_FLAG

    section = config_mod.read_user_config_raw(get_default_hermes_root() / "config.yaml").get("onboarding")
    section = section if isinstance(section, dict) else {}
    seen = section.get("seen") if isinstance(section.get("seen"), dict) else {}
    if section.get("run") is not False or not seen.get(PROFILE_BUILD_FLAG):
        set_run(False, mark_profile_offered=True)


def start(db, session_id: str, facts: Facts) -> dict:
    """Write a fresh run record (the row must exist) and mark the install as mid-onboarding, so a
    crash or a quit before the end runs onboarding again on the next launch (which starts over:
    ``onboarding.start`` with ``fresh_launch``)."""
    from hermes_cli.onboarding_run import set_run

    record = new_record(facts, now=time.time(), run_id=uuid.uuid4().hex)
    db.patch_session_model_config(session_id, {KEY: record})
    set_run(True)
    return record


def advance(db, session_id: str, event: Event) -> Step:
    """Apply one event to the stored run; LookupError when this session has no run."""
    step = _transition(db, session_id, event) if db is not None and session_id else None
    if step is None:
        raise LookupError(f"session {session_id} has no onboarding run")
    return step


def open_session_ids(db) -> list[str]:
    """The chats whose run is still in progress, newest first. Rotation leaves the active copy on the
    compression parent, so each id is read at its compression tip."""
    tips: set[str] = set()
    open_ids: list[str] = []
    for sid in db.session_ids_with_model_config_key(KEY):
        tip = db.resolve_resume_session_id(sid) or sid
        if tip not in tips:
            tips.add(tip)
            if is_active(load(db, tip)):
                open_ids.append(tip)
    return open_ids


def active_session_id(db) -> str | None:
    return next(iter(open_session_ids(db)), None)


def delete_chat(db, session_id: str, delete: Callable[[], bool]) -> bool:
    """Run *delete*; when it removed the chat holding the live run, end that run (``deleted``) on every
    compression copy and tell windows once. Only the compression tip holds the live run, so deleting an
    older copy ends nothing. The install flag closes before the delete when no other run is open: a
    failed flag write keeps the chat for a retry, and a refused delete sets the flag again."""
    from hermes_cli.onboarding_run import set_run

    record = load(db, session_id)
    live = is_active(record) and (db.resolve_resume_session_id(session_id) or session_id) == session_id
    last = live and all(sid == session_id for sid in open_session_ids(db))
    if last:
        _close_install_flag()
    deleted = False
    try:
        deleted = delete()
    finally:
        if last and not deleted:
            set_run(True)
    if not deleted:
        return False
    if live:
        # Each older copy is written on its own row: a tip-routed write would land every copy on the
        # newest one left.
        for sid in db.session_ids_with_model_config_key(KEY) if record.get("run_id") else ():
            if ((older := load(db, sid)) or {}).get("run_id") == record["run_id"]:
                db.patch_session_model_config(sid, {KEY: _end_deleted(older)})
        _announce(record.get("run_id"), session_id, "deleted")
    return True


def _end_deleted(current: Any) -> Any:
    if not is_active(current):
        return current
    return apply(current, End("deleted"), Facts.from_stored(current["facts"]), now=time.time()).record


def observe_tool_result(agent, name: str, args: Mapping[str, Any], result: Any) -> Any:
    """Settle a beat from a general tool's result (``set_theme``, ``manage_connections`` display,
    ``gui_tour``). Runs for every tool result; only an onboarding chat (one whose agent carries
    ``setup_beat``) reads its record. Returns *result*, with the next-step reminder added when it
    settled a beat."""
    if not applies_to(name) or "setup_beat" not in (getattr(agent, "valid_tool_names", None) or ()):
        return result
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", "") or ""
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed, dict) and load(db, session_id) is not None:
            step = advance(db, session_id, Applied(name, dict(args or {}), parsed))
            text = reminder(step.reminder, step.record, Facts.from_stored(step.record["facts"]))
            if text and isinstance(result, str):
                return json.dumps({**parsed, "reminder": text}, ensure_ascii=False)
    except Exception:
        logger.warning("onboarding: %s result did not update the run record", name, exc_info=True)
    return result
