"""Report Hermes background work to ACP clients.

ACP turns are client-driven, so Hermes cannot start a turn when a background
process exits or a detached subagent finishes. The CLI, TUI and gateway inject
those completion-queue events as the next turn; over ACP nothing drained them,
so an agent that promised to report back when its watcher finished never did,
and clients could not tell the work was still running.

Two extension notifications close that gap without changing ACP turn ownership:

* ``_hermes/process``: a ``terminal`` call left a tracked background process
  running (``status: "running"``), and later that process exited
  (``status: "exited"`` with ``exitCode`` and ``reason``). Keyed by the
  ``toolCallId`` that started it, for every background process.
* ``_hermes/notification``: the text the CLI would inject for this session's
  completion-queue events (process completions, watch matches, heartbeats,
  async delegation results), sent only while the session is idle. A client
  that wants CLI behaviour prompts the session with ``text``.

Clients that do not know these methods ignore them, as ACP requires.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any

import acp

logger = logging.getLogger(__name__)

PROCESS_METHOD = "hermes/process"
NOTIFICATION_METHOD = "hermes/notification"


def _notify(conn: acp.Client, loop: asyncio.AbstractEventLoop, method: str, params: dict) -> bool:
    """Send an extension notification from a worker thread; False when it could not be sent."""
    from agent.async_utils import safe_schedule_threadsafe

    future = safe_schedule_threadsafe(
        conn.ext_notification(method, params), loop, logger=logger, log_message=f"Failed to send ACP {method}",
    )
    if future is None:
        return False
    try:
        future.result(timeout=5)
        return True
    except Exception:
        logger.debug("Failed to send ACP %s", method, exc_info=True)
        return False


def track_background_process(
    conn: acp.Client, session_id: str, loop: asyncio.AbstractEventLoop, tool_call_id: str, result: Any,
) -> None:
    """Report a ``terminal`` call whose result is a running background process, then its exit."""
    from tools.process_registry import process_registry

    from .tools import _json_loads_maybe

    data = _json_loads_maybe(result)
    proc_id = data.get("session_id") if isinstance(data, dict) and not data.get("error") else None
    proc = process_registry.get(proc_id) if isinstance(proc_id, str) and proc_id else None
    if proc is None:
        return
    params = {"sessionId": session_id, "toolCallId": tool_call_id, "processId": proc.id, "command": proc.command}
    _notify(conn, loop, PROCESS_METHOD, {**params, "status": "running"})

    def _report_exit() -> None:
        # Every exit path (reader EOF, kill, lost backend) sets this event.
        proc._completion_event.wait()
        _notify(conn, loop, PROCESS_METHOD, {
            **params, "status": "exited", "exitCode": proc.exit_code, "reason": proc.completion_reason,
        })

    threading.Thread(target=_report_exit, daemon=True, name=f"acp-process-{proc.id}").start()


def _process_notifications_off() -> bool:
    """``display.background_process_notifications: off`` opts out of process-driven wakes."""
    try:
        from hermes_cli.config import load_config

        raw = (load_config().get("display") or {}).get("background_process_notifications")
    except Exception:
        return False
    return raw is False or str(raw or "").strip().lower() == "off"


class BackgroundNotifier:
    """Drain the process-wide completion queue into ``_hermes/notification`` for idle sessions.

    One daemon thread per ACP server. It blocks on the queue, routes each event to
    the live session whose ``session_key`` it carries (ACP turns run with the ACP
    session id as their session key), and holds events for a busy session until
    one of its turns ends. Events no live session owns are dropped, as in the TUI.
    """

    # Safety net for a turn that ended without reaching ``_finish_turn``.
    _HELD_RETRY_SECONDS = 5.0

    def __init__(self, session_manager: Any) -> None:
        self._sessions = session_manager
        self._conn: acp.Client | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._turn_ended = threading.Event()
        self._thread: threading.Thread | None = None

    def turn_ended(self, conn: acp.Client, loop: asyncio.AbstractEventLoop) -> None:
        """Start the pump on the first finished turn, then retry held events after each one."""
        self._conn, self._loop = conn, loop
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="acp-background-notifier")
            self._thread.start()
        self._turn_ended.set()

    def _run(self) -> None:
        from tools.process_registry import process_registry

        completions = process_registry.completion_queue
        while True:
            completions.put(completions.get())  # wait for news, then route it below
            while True:
                self._turn_ended.clear()
                try:
                    held, idle_owner = self._deliver(process_registry)
                except Exception:
                    logger.warning("ACP background notification delivery failed", exc_info=True)
                    break
                if not held:
                    break
                if not idle_owner:
                    self._turn_ended.wait(self._HELD_RETRY_SECONDS)

    def _deliver(self, registry: Any) -> tuple[bool, bool]:
        """One routing pass: deliver owned events to idle sessions, drop orphans, requeue the rest.

        Returns ``(held, idle_owner)``: whether events stay queued, and whether any of
        them belongs to an idle session (it arrived mid-pass; route it again now)."""
        from tools.process_registry_notifications import group_process_notifications

        with self._sessions._lock:
            states = list(self._sessions._sessions.values())
        busy: set[str] = set()
        for state in states:
            with state.runtime_lock:
                running = state.is_running
            if running:
                busy.add(state.session_id)
                continue
            drained = registry.drain_notifications(session_key=state.session_id, skip_poll_observed=False)
            for group in group_process_notifications(drained):
                self._send(state.session_id, group, registry)

        live = {state.session_id for state in states}
        held, idle_owner = [], False
        while True:
            try:
                event = registry.completion_queue.get_nowait()
            except queue.Empty:
                break
            key = str(event.get("session_key") or "")
            is_delegation = event.get("type") == "async_delegation"
            # Mirrors drain_notifications' routing: an event with no owner at all goes to
            # any session; a delegation result or another surface's event needs its owner.
            ownerless = not (key or is_delegation or event.get("origin_ui_session_id"))
            if key in live or (ownerless and live):
                held.append(event)
                idle_owner = idle_owner or (key not in busy if key else len(busy) < len(live))
                continue
            if is_delegation:
                from tools.async_delegation import return_completion_offer

                return_completion_offer(event)
            logger.debug("Dropping %s notification no live ACP session owns", event.get("type", "completion"))
        for event in held:
            registry.completion_queue.put(event)
        return bool(held), idle_owner

    def _send(self, session_id: str, group: tuple, registry: Any) -> None:
        from tools.async_delegation import claim_event_delivery, complete_event_delivery, release_event_delivery
        from tools.process_registry_notifications import (
            ProcessNotificationBatch, async_delegation_display_text, heartbeat_display_text,
        )

        first = group[0][0]
        kind = first.get("type", "completion")
        # Drained events are consumed either way; with the opt-out, the process
        # report is the whole delivery. Subagent results are not process events.
        if kind != "async_delegation" and _process_notifications_off():
            return
        claimed = [(event, text, claim) for event, text in group
                   if (claim := claim_event_delivery(event, "acp")) is not None]
        if not claimed:
            return
        if kind == "completion":
            batch = ProcessNotificationBatch(tuple((event, text) for event, text, _claim in claimed))
            text, title = batch.render(registry), batch.display_text(registry)
        else:
            text = claimed[0][1]
            title = (async_delegation_display_text(first) if kind == "async_delegation"
                     else heartbeat_display_text(first) if kind == "heartbeat" else text)
        sent = text is None or _notify(self._conn, self._loop, NOTIFICATION_METHOD, {
            "sessionId": session_id, "kind": kind, "title": title, "text": text,
        })
        for event, _text, claim in claimed:
            (complete_event_delivery if sent else release_event_delivery)(event, claim)
