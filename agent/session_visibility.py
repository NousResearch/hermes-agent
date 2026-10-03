"""Which past sessions this turn's reader may see, for gateways shared by many people.

A plugin registers one :class:`SessionVisibility` (``ctx.register_session_visibility``). Every
``session_search`` call asks it for a :class:`SessionJudgement`, and every session whose messages
or metadata would appear in a result goes through ``may_see`` first: discovery hits and their
lineage roots, title matches, browse entries, read and scroll targets (a scroll's rebind into a
child included), and sessions read from another profile. A visible session can also have single
tool results withheld through ``may_show_tool_result``.

It fails closed. Only a literal ``True`` shows anything; an exception, any other answer or a call
slower than ``session_search.visibility_timeout_seconds`` hides it, and a timeout hides the rest
of that call too. With ``session_search.require_visibility`` on and nothing registered, every
session is hidden. A hidden session answers exactly like a missing one.
"""

import contextvars
import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 120.0
WITHHELD_TOOL_RESULT = "[tool result withheld]"


@dataclass(frozen=True)
class SessionPlace:
    """Where a session happened; the judgement is never shown message content."""

    id: str
    source: Optional[str] = None
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    chat_type: Optional[str] = None
    thread_id: Optional[str] = None
    parent_session_id: Optional[str] = None

    @classmethod
    def of(cls, session_id: str, row: Dict[str, Any]) -> "SessionPlace":
        return cls(id=session_id, **{name: row.get(name) for name in (
            "source", "user_id", "chat_id", "chat_type", "thread_id", "parent_session_id")})


@dataclass(frozen=True)
class ToolCall:
    """The call a tool result answers: its tool name and raw arguments as stored."""

    name: str
    arguments: str


class SessionJudgement(ABC):
    """One session_search call's answers for this turn's reader."""

    @abstractmethod
    def may_see(self, place: SessionPlace) -> bool:
        """True when the reader may see this session."""

    def may_show_tool_result(self, place: SessionPlace, call: ToolCall) -> bool:
        """True when a visible session's tool result may be shown; every result by default."""
        return True

    def finish(self) -> None:
        """Called once when the session_search call ends, however it ended."""
        return None


class SessionVisibility(ABC):
    """Registered by a plugin; hands each session_search call its judgement."""

    name: str = "session-visibility"

    @abstractmethod
    def judgement(self) -> SessionJudgement:
        """The judgement for the current call, built in the caller's context."""


def visibility_settings() -> Dict[str, Any]:
    from hermes_cli.config import load_config_readonly
    section = load_config_readonly().get("session_search")
    return section if isinstance(section, dict) else {}


def _timeout(settings: Dict[str, Any]) -> float:
    try:
        value = float(settings.get("visibility_timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    return value if 0 < value <= MAX_TIMEOUT_SECONDS else DEFAULT_TIMEOUT_SECONDS


class _Tripped(Exception):
    pass


class VisibilityGate:
    """Per-call, fail-closed front for a judgement; ``None`` judgement hides everything."""

    def __init__(self, judgement: Optional[SessionJudgement], timeout_seconds: float):
        self._judgement = judgement
        self._timeout = timeout_seconds
        self._tripped = False
        self._seen: Dict[str, bool] = {}
        self._places: Dict[str, Optional[SessionPlace]] = {}
        self._tool_calls: Dict[str, Dict[str, ToolCall]] = {}
        self.redacts = judgement is not None and (
            type(judgement).may_show_tool_result is not SessionJudgement.may_show_tool_result)

    @classmethod
    def for_call(cls) -> Optional["VisibilityGate"]:
        """The gate for one session_search call, or None when nothing asks for one."""
        try:
            from hermes_cli.plugins import get_plugin_session_visibility
            provider = get_plugin_session_visibility()
            settings = visibility_settings()
        except Exception:
            logger.warning("session visibility unavailable; hiding every session", exc_info=True)
            return cls(None, DEFAULT_TIMEOUT_SECONDS)
        timeout = _timeout(settings)
        if provider is None:
            return cls(None, timeout) if settings.get("require_visibility") else None
        gate = cls(None, timeout)
        try:
            judgement = gate._bounded(provider.judgement)
        except Exception as exc:
            logger.warning("session visibility %r gave no judgement (%s); hiding every session",
                           getattr(provider, "name", "?"), type(exc).__name__)
            return gate
        if not isinstance(judgement, SessionJudgement):
            logger.warning("session visibility %r gave no judgement; hiding every session",
                           getattr(provider, "name", "?"))
            return gate
        return cls(judgement, timeout)

    def _bounded(self, fn: Callable, *args) -> Any:
        if self._tripped:
            raise _Tripped()
        context, outcome, done = contextvars.copy_context(), {}, threading.Event()

        def run():
            try:
                outcome["value"] = context.run(fn, *args)
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                done.set()

        threading.Thread(target=run, name="session-visibility", daemon=True).start()
        if not done.wait(self._timeout):
            self._tripped = True
            raise TimeoutError("session visibility judgement timed out")
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def _place(self, db, session_id: str) -> Optional[SessionPlace]:
        if session_id not in self._places:
            try:
                row = db.get_session(session_id)
            except Exception:
                row = None
            self._places[session_id] = SessionPlace.of(session_id, row) if row else None
        return self._places[session_id]

    def sees(self, db, session_id: Optional[str]) -> bool:
        if not session_id or self._judgement is None:
            return False
        if session_id not in self._seen:
            place = self._place(db, session_id)
            self._seen[session_id] = place is not None and self._ask(self._judgement.may_see, place)
        return self._seen[session_id]

    def sees_all(self, db, *session_ids: Optional[str]) -> bool:
        return all(self.sees(db, sid) for sid in session_ids)

    def _ask(self, question: Callable, *args) -> bool:
        try:
            return self._bounded(question, *args) is True
        except _Tripped:
            return False
        except Exception as exc:
            logger.warning("session visibility judgement failed (%s); hiding", type(exc).__name__)
            return False

    def _calls_in(self, db, session_id: str) -> Dict[str, ToolCall]:
        if session_id not in self._tool_calls:
            calls: Dict[str, ToolCall] = {}
            try:
                rows = db.get_messages(session_id, include_compacted=True)
            except Exception:
                rows = []
            for row in rows:
                for call in _tool_calls_of(row):
                    calls[call["id"]] = ToolCall(name=call["name"], arguments=call["arguments"])
            self._tool_calls[session_id] = calls
        return self._tool_calls[session_id]

    def shows_tool_result(self, db, session_id: str, message: Dict[str, Any]) -> bool:
        if not self.redacts:
            return True
        place = self._place(db, session_id)
        if place is None or not self.sees(db, session_id):
            return False
        call = self._calls_in(db, session_id).get(message.get("tool_call_id") or "") or ToolCall(
            name=str(message.get("tool_name") or ""), arguments="")
        return self._ask(self._judgement.may_show_tool_result, place, call)

    def withhold_tool_results(self, db, session_id: str, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Shaped messages with every tool result this reader may not see replaced."""
        if not self.redacts:
            return entries
        shown = []
        for entry in entries:
            if entry.get("role") == "tool" and not self.shows_tool_result(db, session_id, entry):
                entry = {k: v for k, v in entry.items() if k not in ("content_truncated", "original_content_chars")}
                entry.update(content=WITHHELD_TOOL_RESULT, withheld=True)
            shown.append(entry)
        return shown

    def finish(self) -> None:
        if self._judgement is None:
            return
        self._tripped = False
        try:
            self._bounded(self._judgement.finish)
        except Exception as exc:
            logger.warning("session visibility finish failed (%s)", type(exc).__name__)


def _tool_calls_of(row: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = row.get("tool_calls")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    calls = []
    for call in raw if isinstance(raw, list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict) and isinstance(call.get("id"), str):
            arguments = function.get("arguments")
            calls.append({"id": call["id"], "name": str(function.get("name") or ""),
                          "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)})
    return calls
