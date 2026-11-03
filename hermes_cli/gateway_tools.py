"""The background-process registry main's classic CLI reads, served by the gateway session.

On main the CLI process ran every ``terminal(background=true)`` itself, so the live-work dock,
the status bar's ``⚙ N`` segment, ``/stop`` and the dock's tail view read
``tools.process_registry`` in-process. Rewired, the processes run in the gateway: this view
answers the same calls (``list_sessions`` / ``get`` / ``count_running`` / ``kill_process`` /
``kill_all``) from the session's ``process.list`` / ``process.kill`` / ``process.stop`` verbs, so
main's dock and command code render unchanged.

Polling: the dock ticks once a second. The view only asks the gateway while it has a reason to
(a tool finished since the last poll, or a listed process is still running or recently
exited); the status bar reads the cached count and never blocks a repaint on an RPC.
"""
from __future__ import annotations

import logging
import threading
import time

from hermes_cli.gateway_client import GatewayClientError

logger = logging.getLogger(__name__)

_POLL_SECONDS = 1.0
# A finished process stays interesting this long (main's dock RETAIN_SECONDS).
_RETAIN_SECONDS = 60

_view = None


class _Proc:
    """The ``ProcessSession`` attributes main's dock reads."""

    def __init__(self, row):
        self.id = row.get("session_id") or ""
        self.command = row.get("command") or ""
        self.exit_code = row.get("exit_code")
        self.started_at = float(row.get("started_epoch") or 0.0)
        self.exited_at = float(row.get("exited_at") or 0.0)
        self.output_buffer = row.get("output_tail") or ""
        self.completion_reason = row.get("completion_reason") or "exited"
        self.exited = row.get("status") == "exited"


class GatewayProcessView:
    def __init__(self, agent):
        self.agent = agent
        self.connection = agent.connection
        self.rows: list = []
        self.dirty = True
        self.delegations = 0
        self._polled = 0.0
        self._lock = threading.Lock()
        self.subagents: list = []
        self._subagents_polled = 0.0
        self._subagents_seen = 0.0
        self._subagents_lock = threading.Lock()
        self._tail = None
        self._subagents_watch_until = 0.0

    @property
    def session_id(self):
        return self.agent.session_id  # follows /new, /resume and compression rotations

    # -- polling ---------------------------------------------------------------------------
    def mark_dirty(self):
        self.dirty = True

    def _interesting(self, now):
        return any(not p.exited or (p.exited_at and now - p.exited_at <= _RETAIN_SECONDS) for p in self.rows)

    def refresh(self, *, force=False):
        now = time.time()
        if not force and (now - self._polled < _POLL_SECONDS or not (self.dirty or self._interesting(now))):
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._polled = now
            self.dirty = False
            result = self.connection.call("process.list", session_id=self.session_id, timeout=5)
            self.rows = [_Proc(row) for row in (result or {}).get("processes") or []]
        except (GatewayClientError, RuntimeError) as exc:
            logger.debug("process.list failed: %s", exc)
        finally:
            self._lock.release()

    # -- the tools.process_registry surface main's CLI calls ---------------------------------
    def list_sessions(self):
        self.refresh()
        now = time.time()
        return [{"session_id": p.id, "command": p.command[:200],
                 "status": "exited" if p.exited else "running", "exit_code": p.exit_code,
                 "completion_reason": p.completion_reason if p.exited else None,
                 "uptime_seconds": int(now - p.started_at) if p.started_at else 0} for p in self.rows]

    def get(self, process_id):
        return next((p for p in self.rows if p.id == process_id), None)

    def count_running(self):
        """Cached (the status bar repaints off the render thread)."""
        return sum(1 for p in self.rows if not p.exited)

    def kill_process(self, process_id, **_kw):
        try:
            result = self.connection.call("process.kill", session_id=self.session_id, process_id=process_id)
        except GatewayClientError as exc:
            return {"status": "error", "error": str(exc)}
        self.refresh(force=True)
        return result

    def kill_all(self, **_kw):
        """``/stop``: the session's own processes plus its background delegations (``process.stop``)."""
        result = self.connection.call("process.stop", session_id=self.session_id) or {}
        self.delegations = int(result.get("delegations") or 0)
        self.refresh(force=True)
        return int(result.get("killed") or 0)

    def interrupt_delegations(self):
        """The delegation half of ``/stop`` (``kill_all`` already ran it when processes were live)."""
        if self.delegations:
            count, self.delegations = self.delegations, 0
            return count
        self.kill_all()
        count, self.delegations = self.delegations, 0
        return count

    def active_delegations(self):
        """Running background subagents (the gateway's ``session.usage.active_subagents``)."""
        try:
            usage = self.connection.state(self.session_id).usage or {}
        except Exception:  # health: allow BLE001 -- a missing usage cache reads as none running
            return 0
        value = usage.get("active_subagents")
        return value if isinstance(value, int) else 0

    # -- the subagent registry main's Ctrl+T monitor and dock read ---------------------------
    def watch_subagents(self, seconds=_RETAIN_SECONDS):
        """Poll ``subagent.list`` for a while even with no turn running: an off-turn spawn
        (``/review``) moves neither ``busy`` nor the usage count the gate below reads."""
        self._subagents_watch_until = time.time() + seconds
        self._subagents_polled = 0.0

    def subagent_rows(self, *, busy):
        """The session's live children (``subagent.list``). Asks at most once a second, and only
        while a turn runs (``busy``), children were listed last time, the gateway counts some, or
        an off-turn spawn asked to be watched."""
        now = time.time()
        wanted = (busy or self.subagents or now < self._subagents_watch_until
                  or self.active_delegations())
        if wanted and now - self._subagents_polled >= _POLL_SECONDS and self._subagents_lock.acquire(blocking=False):
            try:
                self._subagents_polled = now
                result = self.connection.call("subagent.list", session_id=self.session_id, timeout=5)
                self.subagents = list((result or {}).get("subagents") or [])
                self._subagents_seen = now
            except (GatewayClientError, RuntimeError) as exc:
                logger.debug("subagent.list failed: %s", exc)
                if now - self._subagents_seen > _RETAIN_SECONDS:
                    self.subagents = []  # no answer for a minute: drop the rows and stop polling
            finally:
                self._subagents_lock.release()
        return [dict(row) for row in self.subagents]

    def subagent_tail(self, subagent_id, *, fetch=False):
        """The child's live transcript tail (``subagent.tail``); None when the gateway has none.
        The detail view repaints every frame and reads only the cache; ``fetch`` (the monitor's
        tick, on the spinner thread) asks the gateway at most once a second, failures included."""
        now = time.time()
        cached = self._tail if self._tail and self._tail[0] == subagent_id else None
        if not fetch or (cached and now - cached[1] < _POLL_SECONDS):
            return cached[2] if cached else None
        try:
            result = self.connection.call("subagent.tail", session_id=self.session_id,
                                          subagent_id=subagent_id, timeout=5) or {}
            text = result.get("text") if result.get("available") else None
            if text is not None:  # filter once here, not on every repaint of the detail view
                text = "".join(c for c in text if c.isprintable() or c in "\n\t")
        except (GatewayClientError, RuntimeError) as exc:
            logger.debug("subagent.tail failed: %s", exc)
            text = None
        self._tail = (subagent_id, now, text)
        return text

    def subagent_control(self, action, subagent_id, message=None):
        """Steer (``subagent.steer``) or stop (``subagent.interrupt``) one child of this session, in
        the shape ``delegate_tool_registry._handle_control_action`` answers: ``note`` or ``error``."""
        from tools.delegate_tool_registry import _CONTROL_OUTCOMES

        status, note, failure = _CONTROL_OUTCOMES[action]
        try:
            if action == "steer":
                result = self.connection.call("subagent.steer", session_id=self.session_id,
                                              subagent_id=subagent_id, text=message, timeout=5) or {}
                ok = result.get("status") == "queued"
            else:
                result = self.connection.call("subagent.interrupt", session_id=self.session_id,
                                              subagent_id=subagent_id, timeout=5) or {}
                ok = bool(result.get("found"))
        except GatewayClientError as exc:
            return {"error": str(exc)}
        self._subagents_polled = 0.0  # the next tick re-reads the roster
        if ok:
            return {"action": action, "subagent_id": subagent_id, "status": status, "note": note}
        return {"error": failure.format(sid=subagent_id)}


class LocalSubagents:
    """This process's own subagent registry behind the view's ``subagent_*`` surface, for a CLI that
    runs its agent in-process: the dock and monitor talk to one backend shape either way."""

    def __init__(self, cli):
        self.cli = cli
        self._paths: dict = {}

    def subagent_rows(self, *, busy):
        from tools.delegate_tool_registry import _list_payload, list_active_subagents
        parent = getattr(self.cli, "agent", None)
        entries = _list_payload(parent)["subagents"] if parent is not None else []
        # The scoped control-plane snapshot supplies authority and transcript paths; the public
        # lifecycle record supplies start time and the latest observed tool (as subagent.list does).
        live = {r["subagent_id"]: r for r in list_active_subagents()} if entries else {}
        for row in entries:
            record = live.get(row["subagent_id"], {})
            row["started_at"], row["last_tool"] = record.get("started_at"), record.get("last_tool") or ""
        self._paths = {row["subagent_id"]: row.get("live_transcript") for row in entries}
        return entries

    def subagent_tail(self, subagent_id, *, fetch=False):
        from hermes_cli.cli_subagent_monitor import read_tail
        return read_tail(self._paths.get(subagent_id))

    def subagent_control(self, action, subagent_id, message=None):
        import json
        from tools.delegate_tool_registry import _handle_control_action
        return json.loads(_handle_control_action(action, subagent_id, message, getattr(self.cli, "agent", None)))


class _OpenedSession:
    """The CLI's open gateway session before any chat agent exists (``_init_agent`` builds that on the
    first turn): the view's ``connection`` and ``session_id``, following the CLI's id."""

    def __init__(self, cli, connection):
        self.cli, self.connection = cli, connection

    @property
    def session_id(self):
        return self.cli.session_id


def bind(agent):
    """Called when the CLI's gateway-backed agent is built."""
    global _view
    _view = GatewayProcessView(agent) if getattr(agent, "connection", None) is not None else None
    return _view


def bind_session(cli, connection):
    """Called when the CLI opens its gateway session, so ``/agents``, ``/stop``, the dock and the status
    bar read the session's processes before the first turn (a resumed session's work keeps running)."""
    return bind(_OpenedSession(cli, connection))


def mark_dirty():
    if _view is not None:
        _view.mark_dirty()


def process_view():
    """The gateway-backed registry, or None when this process runs its own tools (not rewired) or
    its gateway connection is final (a hang-up keeps the view; its next call redials)."""
    view = _view
    return view if view is not None and not view.connection.final else None


def process_registry():
    """``tools.process_registry.process_registry`` for main's dock, status bar and ``/stop``."""
    view = process_view()
    if view is not None:
        return view
    from tools.process_registry import process_registry as local
    return local


def background_counts():
    """The status bar's ``⚙ N`` processes and running background subagents (cached, no RPC)."""
    view = process_view()
    if view is not None:
        return {"active_background_processes": view.count_running(),
                "active_background_subagents": view.active_delegations()}
    from tools.async_delegation import active_count
    from tools.process_registry import process_registry as local
    return {"active_background_processes": local.count_running(), "active_background_subagents": active_count()}


def session_tools():
    """``/tools``: the gateway session's tool catalog (``tools.show``) as
    ``(tools, toolset_of)``: ``get_tool_definitions``-shaped rows (first description sentence) and
    a name -> toolset lookup. None when the CLI is not gateway-backed."""
    view = process_view()
    if view is None:
        return None
    result = view.connection.call("tools.show", session_id=view.session_id) or {}
    tools, toolset_of = [], {}
    for section in result.get("sections") or []:
        for row in section.get("tools") or []:
            toolset_of[row["name"]] = section.get("name")
            tools.append({"type": "function", "function": {"name": row["name"],
                                                           "description": row.get("description") or ""}})
    return tools, toolset_of.get


def reload_gateway_skills():
    """Rescan skills in the gateway session too (its skill-command map serves this session's
    ``/<skill>`` and bundle invocations); the CLI's own rescan still drives main's output."""
    view = process_view()
    if view is None:
        return None
    try:
        return (view.connection.call("skills.reload", session_id=view.session_id) or {}).get("result")
    except GatewayClientError as exc:
        logger.debug("skills.reload failed: %s", exc)
        return None


def handle_skills_slash(cmd, console, cli=None):
    """Main's ``/skills`` (hub search / install / uninstall write the local skills tree), then the
    gateway's rescan so an install reaches the skill map the session's turns use, and ``cli``'s copy of
    the gateway's skill list is read again."""
    from hermes_cli.skills_hub import handle_skills_slash as local
    local(cmd, console)
    reload_gateway_skills()
    if cli is not None:
        from hermes_cli.gateway_plugin_commands import invalidate
        invalidate(cli)
