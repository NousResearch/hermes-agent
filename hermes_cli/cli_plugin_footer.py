"""Plugin items in the classic CLI live-work footer (``ctx.register_footer_item``).

Snapshots are pulled from ``SubagentMonitor.refresh`` (about once a second), each in its own worker
thread under one shared time budget; the render path only reads the last good value. Actions run
off the UI loop as well. Display only: nothing here reaches the conversation or the system prompt.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

TONES = ("ok", "warn", "bad", "dim")
# A snapshot still running after this long is logged once; the footer keeps its last value. The
# poll itself never waits (a finished snapshot asks for the next poll through ``on_change``).
SLOW_SNAPSHOT_SECONDS = 2.0
# At most one snapshot per item per interval, however often the footer refreshes.
SNAPSHOT_INTERVAL = 1.0
_MAX_ROWS = 8
_MAX_ACTIONS = 8
_ACTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")
_LETTER = re.compile(r"[a-z]")


def session_info(cli) -> dict:
    """The session a footer item describes: the CLI's own ids, its profile and home."""
    from hermes_constants import get_hermes_home
    stored = getattr(cli, "session_id", None) or None
    agent = getattr(cli, "agent", None)
    try:
        from hermes_cli.profiles import get_active_profile_name
        profile = get_active_profile_name()
    except Exception:
        logger.debug("footer session profile lookup failed; using default", exc_info=True)
        profile = "default"
    return {"session_id": getattr(agent, "session_id", None) or stored, "stored_session_id": stored,
            "surface": "cli", "hermes_home": str(get_hermes_home()), "profile": profile}


def _text(value, limit) -> str:
    return value[:limit] if isinstance(value, str) else ""


def normalize(raw) -> Optional[dict]:
    """A snapshot reduced to the documented JSON shape; ``None`` hides the item. Malformed rows and
    actions are dropped, an unknown tone paints dim."""
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"footer snapshot must return a dict or None, not {type(raw).__name__}")
    rows, row_keys = [], set()
    for row in (raw.get("rows") or [])[:_MAX_ROWS]:
        key = row.get("key") if isinstance(row, dict) else None
        if not isinstance(key, str) or not key or key in row_keys:
            continue
        row_keys.add(key)
        rows.append({"key": key[:128], "label": _text(row.get("label"), 200) or key,
                     "detail": _text(row.get("detail"), 120)})
    actions, action_keys = [], set()
    for action in (raw.get("actions") or [])[:_MAX_ACTIONS]:
        if not isinstance(action, dict):
            continue
        ident, key = action.get("id"), action.get("key")
        if not isinstance(ident, str) or not _ACTION_ID.fullmatch(ident):
            continue
        if key != "enter" and not (isinstance(key, str) and _LETTER.fullmatch(key)) or key in action_keys:
            continue
        action_keys.add(key)
        actions.append({"id": ident, "label": _text(action.get("label"), 24) or ident, "key": key,
                         "confirm": action.get("confirm") is True})
    label = _text(raw.get("label"), 80)
    if not label and not rows:
        return None
    return {"label": label, "tone": raw.get("tone") if raw.get("tone") in TONES else "dim",
            "rows": rows, "actions": actions}


class _Slot:
    __slots__ = ("done", "session", "slow_reported", "started", "value")

    def __init__(self):
        self.done: Optional[threading.Event] = None  # set while a snapshot is in flight
        self.started = 0.0
        self.session: Optional[dict] = None
        self.value: Optional[dict] = None
        self.slow_reported = False


class FooterItemPoller:
    """Last good snapshot per registered item, refreshed off the render path: each snapshot runs
    in its own worker thread (at most one in flight per item), a failure keeps the last good
    value, and ``on_change`` fires when a finished snapshot changed what the footer shows."""

    def __init__(self, on_change=None):
        self._slots: dict[str, _Slot] = {}
        self._lock = threading.Lock()
        self._reported: set[tuple[str, str]] = set()
        self._on_change = on_change

    def poll(self, session: dict, *, wait: float = 0.0, items=None) -> list[dict]:
        """Start a snapshot for every idle item and return each item's last good value for
        ``session`` in registration order. ``wait`` (seconds, shared) lets a caller that may block
        collect snapshots started now; the footer's tick passes 0."""
        if items is None:
            from hermes_cli.plugins import get_plugin_footer_items
            items = get_plugin_footer_items()
        now, waiting = time.monotonic(), []
        with self._lock:
            live = {item["key"] for item in items}
            for key in [key for key in self._slots if key not in live]:
                del self._slots[key]
            for item in items:
                slot = self._slots.setdefault(item["key"], _Slot())
                if slot.session != session:
                    # A switched session (/new, /resume) never shows the previous one's value.
                    slot.session, slot.value, slot.started = dict(session), None, 0.0
                if slot.done is None and (slot.value is None and slot.started == 0.0
                                          or now - slot.started >= SNAPSHOT_INTERVAL):
                    slot.done, slot.started = threading.Event(), now
                    context = contextvars.copy_context()
                    threading.Thread(target=context.run, args=(self._snapshot, item, dict(session), slot),
                                     name=f"footer-item:{item['key']}", daemon=True).start()
                elif slot.done is not None and now - slot.started > SLOW_SNAPSHOT_SECONDS and not slot.slow_reported:
                    slot.slow_reported = True
                    logger.warning("Footer item %s snapshot has run for over %.0fs; showing its last value",
                                   item["key"], SLOW_SNAPSHOT_SECONDS)
                waiting.append((item, slot, slot.done))
        deadline = now + max(0.0, wait)
        for _item, _slot, done in waiting:
            if done is not None:
                done.wait(max(0.0, deadline - time.monotonic()))
        with self._lock:
            return [{**slot.value, "key": item["key"], "plugin": item["plugin"]}
                    for item, slot, _ in waiting if slot.value]

    def _snapshot(self, item: dict, session: dict, slot: _Slot) -> None:
        failure, changed = None, False
        try:
            value = normalize(item["snapshot"](dict(session)))
        except Exception as exc:  # health: allow BLE001 -- plugin callback boundary; _report logs the traceback
            value, failure = None, exc
        with self._lock:
            # A failure keeps the last good value; a result for a session that is no longer current
            # is dropped.
            if failure is None and slot.session == session:
                changed, slot.value = value != slot.value, value
            done, slot.done = slot.done, None
        if done is not None:
            done.set()
        if failure is not None:
            self._report(item["key"], "snapshot", failure)
        if changed and self._on_change is not None:
            try:
                self._on_change()
            except Exception:
                logger.debug("footer on_change failed", exc_info=True)

    def _report(self, key: str, what: str, exc: BaseException) -> None:
        marker = (key, f"{what}:{type(exc).__name__}:{exc}")
        if marker in self._reported:
            logger.debug("Footer item %s %s failed again: %s", key, what, exc)
            return
        self._reported.add(marker)
        logger.warning("Footer item %s %s failed: %s", key, what, exc, exc_info=exc)


# -- rows and actions ----------------------------------------------------------------------------

def footer_rows(items: list[dict]) -> list[dict]:
    """Footer roster rows (``kind='plugin'``) for the polled items, in item then row order."""
    return [{"kind": "plugin", "key": f"plugin:{item['key']}:{row['key']}", "item": item["key"],
             "plugin": item["plugin"], "row_key": row["key"], "label": row["label"],
             "detail": row["detail"], "tone": item["tone"], "actions": item["actions"]}
            for item in items for row in item["rows"]]


def action_for(row: Optional[dict], key: str) -> Optional[dict]:
    """The row's action for ``key``: ``'enter'`` falls back to the first action."""
    actions = (row or {}).get("actions") or []
    match = next((action for action in actions if action["key"] == key), None)
    if match is None and key == "enter" and actions:
        return actions[0]
    return match


def _short(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= 120 else text[:119] + "…"


def _repaint(cli, monitor) -> None:
    try:
        monitor.invalidate() if getattr(monitor, "app", None) is not None else cli._invalidate()
    except Exception:
        logger.debug("footer repaint failed", exc_info=True)


def run_action(cli, monitor, row: dict, action: dict) -> Optional[threading.Thread]:
    """Run ``on_action(action_id, session, row_key)`` on a worker thread and apply its result."""
    from hermes_cli.plugins import get_plugin_footer_items
    item = next((i for i in get_plugin_footer_items() if i["key"] == row["item"]), None)
    if item is None or item["on_action"] is None:
        return None
    session = session_info(cli)

    def work():
        from hermes_cli.plugins import resolve_plugin_command_result
        try:
            result = resolve_plugin_command_result(item["on_action"](action["id"], session, row["row_key"]))
            apply_result(cli, monitor, item, result)
        except Exception as exc:
            logger.warning("Footer item %s action %s failed: %s", item["key"], action["id"], exc, exc_info=exc)
            monitor.notice = f"⚠ {item['plugin']}: {_short(exc)}"
        _repaint(cli, monitor)

    thread = threading.Thread(target=contextvars.copy_context().run, args=(work,),
                              name=f"footer-action:{item['key']}", daemon=True)
    thread.start()
    return thread


def apply_result(cli, monitor, item: dict, result: Any) -> None:
    """``None`` | ``str`` notice | ``{'notice', 'open_url', 'run_in_terminal'}``."""
    if result is None:
        return
    if isinstance(result, str):
        monitor.notice = result
        return
    if not isinstance(result, dict):
        logger.debug("Footer item %s action returned %s; ignored", item["key"], type(result).__name__)
        return
    if isinstance(result.get("notice"), str):
        monitor.notice = result["notice"]
    url = result.get("open_url")
    if url is not None:
        if not isinstance(url, str) or urlsplit(url).scheme not in ("http", "https"):
            monitor.notice = f"⚠ {item['plugin']}: only http(s) URLs open from the footer"
        elif not open_url(url):
            monitor.notice = url  # no graphical browser here: show the link instead
    body = result.get("run_in_terminal")
    if body is not None:
        if not callable(body):
            monitor.notice = f"⚠ {item['plugin']}: run_in_terminal must be callable"
        elif getattr(body, "__hermes_plugin_host_ref__", None) is not None:
            # A callable from the plugin host would run in that process, which has no terminal.
            monitor.notice = f"⚠ {item['plugin']}: run_in_terminal needs plugins.isolation: in_process"
        else:
            run_in_terminal(cli, monitor, item, body)


def open_url(url: str) -> bool:
    """Open ``url`` in a graphical browser without letting the opener write over the prompt."""
    if sys.platform.startswith("linux") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        opener = shutil.which("xdg-open")
        if opener:
            subprocess.Popen([opener, url], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
            return True
    import webbrowser
    from hermes_cli.auth_device_flow import _can_open_graphical_browser
    return _can_open_graphical_browser() and bool(webbrowser.open(url))


def run_in_terminal(cli, monitor, item: dict, body) -> None:
    """Run the blocking ``body()`` with the real terminal while the prompt UI is suspended
    (``in_terminal``, as the full-screen monitor does), then return to the chat."""
    app = getattr(cli, "_app", None)
    loop = getattr(app, "loop", None)
    if loop is None or loop.is_closed():
        monitor.notice = f"⚠ {item['plugin']}: needs the interactive terminal"
        return
    context = contextvars.copy_context()

    async def run():
        from prompt_toolkit.application import in_terminal
        if monitor.opening:
            return
        monitor.opening = True
        try:
            async with in_terminal():
                result = await asyncio.get_running_loop().run_in_executor(None, context.run, body)
            if isinstance(result, str) and result:
                monitor.notice = result
        except Exception as exc:
            logger.warning("Footer item %s terminal view failed: %s", item["key"], exc, exc_info=exc)
            monitor.notice = f"⚠ {item['plugin']}: {_short(exc)}"
        finally:
            monitor.opening = False
            _repaint(cli, monitor)

    asyncio.run_coroutine_threadsafe(run(), loop)
