"""Proactive notices: tell the user when the team published or updated a skill.

One bounded poll of the Gateway feed per profile per ``POLL_INTERVAL`` at session start, diffed
against the install ledger; the result is a short system-prompt section frozen into that
session (cache-safe: it never changes mid-conversation). Nothing is installed or downloaded
here; the agent is told what exists and points the user at ``wisdom_install`` / ``/wisdom``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

POLL_INTERVAL = 600.0
_NOTICE_KINDS = {"new": "published", "updated": "updated"}


def refresh(state, *, now: float | None = None, client=None) -> list[dict]:
    """Poll the feed if due and return the pending notices (persisted in plugin state)."""
    from plugins.wisdom.client import WisdomClient, WisdomError, entitled
    now = now or time.time()
    if state.get("muted_until", 0) > now or not entitled():
        return []
    if now - state.get("feed_polled_at", 0) < POLL_INTERVAL:
        return state.get("notices") or []
    state.set("feed_polled_at", now)
    try:
        page = (client or WisdomClient()).feed(state.get("feed_cursor"))
    except WisdomError as exc:
        logger.debug("wisdom feed poll skipped: %s", exc)
        return state.get("notices") or []
    installed = state.get("installed") or {}
    notices = {n["skill_id"]: n for n in state.get("notices") or []}
    for ev in page.get("events") or []:
        kind, sid = ev.get("kind"), ev.get("skill_id")
        if kind in ("archived", "taken_down", "uninstalled") or not sid:
            notices.pop(sid, None)
            continue
        if kind not in _NOTICE_KINDS:
            continue
        local = installed.get(sid)
        if local and (ev.get("version") or 0) <= local["version"]:
            continue
        notices[sid] = {"skill_id": sid, "version": ev.get("version"), "kind": _NOTICE_KINDS[kind],
                        "installed": local and local["version"]}
    if page.get("next_cursor"):
        state.set("feed_cursor", page["next_cursor"])
    pending = list(notices.values())[-20:]
    state.set("notices", pending)
    return pending


def dismiss(state, skill_id: str | None = None) -> None:
    """Drop one notice (after install/uninstall) or all of them."""
    keep = [] if skill_id is None else [n for n in state.get("notices") or [] if n["skill_id"] != skill_id]
    state.set("notices", keep)


def mute(state, hours: float) -> float:
    until = time.time() + hours * 3600
    state.set("muted_until", until)
    return until


def prompt_section(state) -> str:
    """Rendered once per new session by the host; empty string means no section."""
    try:
        notices = refresh(state)
    except Exception as exc:  # a broken feed must never break session start
        logger.debug("wisdom notices unavailable: %s", exc)
        return ""
    if not notices:
        return ""
    lines = [f"- {n['skill_id']} v{n['version']} ({n['kind']}" + (f", you have v{n['installed']}" if n["installed"] else "") + ")"
             for n in notices]
    return ("Your team published or updated Collective Wisdom skills since you last looked:\n" + "\n".join(lines)
            + "\nMention this once if relevant; the user can review with wisdom_browse and install with "
              "wisdom_install (native consent required). Do not install anything unprompted.")


def summary(state) -> dict[str, Any]:
    return {"notices": state.get("notices") or [], "muted_until": state.get("muted_until", 0)}
