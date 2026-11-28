"""Hold durable gateway follow-ups until the first provider request can take them.

The gateway owns the admissions and their settlement. This object only hands the
request a text snapshot and reports whether the request was actually started.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StartupFollowup:
    ticket: str
    text: str
    origin: Any
    display_text: str


class StartupFollowups:
    """A per-turn, ticketed handoff. No entry survives into the next turn."""

    def __init__(self, on_send: Callable[[list[StartupFollowup]], None],
                 is_current: Callable[[], bool] = lambda: True):
        self._lock = threading.Lock()
        self._parts: dict[str, StartupFollowup] = {}
        self._open = True
        self._sent = False
        self._committing = False
        self._cancelled = False
        self._cancel_when_safe = False
        self._staged: list[StartupFollowup] = []
        self._row: dict[str, Any] | None = None
        self._wire: dict[str, Any] | None = None
        self._old_sidecar: Any = None
        self._had_sidecar = False
        self._old_wire: str | None = None
        self._old_wire_value: Any = None
        self._expected: str | None = None
        self.added_tokens = 0
        self._on_send = on_send
        self._is_current = is_current

    def offer(self, ticket: str, text: str, origin: Any, *, display_text: str | None = None) -> bool:
        with self._lock:
            if not self._open or not text.strip() or ticket in self._parts:
                return False
            self._parts[ticket] = StartupFollowup(ticket, text, origin, display_text or text)
            return True

    def is_open(self) -> bool:
        with self._lock:
            return self._open

    @staticmethod
    def can_persist(agent: Any) -> bool:
        """Unsupported worker stores leave the follow-up in the durable FIFO."""
        from agent.agent_init_session import agent_interface

        store = agent_interface(agent)
        writer = store.messages if store is not None else getattr(agent, "_session_db", None)
        method = getattr(writer, "set_message_api_content", None)
        if not callable(method):
            return False
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return False
        return "new_content" in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )

    def stage(self, messages: list, api_messages: list, current_turn_user_idx: Any,
              *, max_extra_tokens: int | None = None) -> bool:
        """Close admission and stage text in the as-yet-unsent current user row.

        A non-text tail, an unaddressable persisted row, or a different current
        turn leaves every admission queued for its own turn.
        """
        with self._lock:
            if not self._open:
                return False
            self._open = False
            if not self._parts or not isinstance(current_turn_user_idx, int):
                return False
            if current_turn_user_idx < 0 or current_turn_user_idx != len(messages) - 1 or not api_messages:
                return False
            row, wire = messages[current_turn_user_idx], api_messages[-1]
            if not isinstance(row, dict) or not isinstance(wire, dict):
                return False
            if row.get("role") != "user" or wire.get("role") != "user":
                return False
            if (type(row.get("_row_id")) is not int or row["_row_id"] <= 0
                    or not isinstance(row.get("content"), str)):
                return False
            wire_value = wire.get("content")
            if isinstance(wire_value, str):
                wire_text = wire_value
            elif (isinstance(wire_value, list) and len(wire_value) == 1 and isinstance(wire_value[0], dict)
                  and wire_value[0].get("type") == "text" and isinstance(wire_value[0].get("text"), str)):
                wire_text = wire_value[0]["text"]
            else:
                return False
            from agent.message_sanitization import _sanitize_surrogates
            from agent.model_metadata import estimate_tokens_rough
            current = row.get("api_content") or row["content"]
            if not isinstance(current, str) or _sanitize_surrogates(current.strip()) != wire_text:
                # Selection/adjacent-user merging made the outgoing tail larger than this row.
                # Stamping that combined tail onto the row would duplicate earlier history on replay.
                return False
            extra = _sanitize_surrogates("\n\n".join(part.text for part in self._parts.values()).strip())
            self.added_tokens = estimate_tokens_rough("\n\n" + extra)
            if max_extra_tokens is not None and self.added_tokens > max_extra_tokens:
                return False
            self._row, self._wire = row, wire
            self._had_sidecar, self._old_sidecar = "api_content" in row, row.get("api_content")
            self._old_wire = wire_text
            self._old_wire_value = deepcopy(wire_value)
            self._staged = list(self._parts.values())
            self._expected = _sanitize_surrogates(
                "\n\n".join([self._old_wire, *(part.text for part in self._staged)]).strip()
            )
            row["api_content"] = self._expected
            wire["content"] = self._expected if isinstance(wire_value, str) else [
                {**wire_value[0], "text": self._expected}
            ]
            return True

    def _restore(self) -> None:
        if self._row is not None:
            if self._had_sidecar:
                self._row["api_content"] = self._old_sidecar
            else:
                self._row.pop("api_content", None)
        if self._wire is not None:
            self._wire["content"] = self._old_wire_value

    def _request_has_staged_text(self, kwargs: dict[str, Any]) -> bool:
        request = kwargs.get("messages", kwargs.get("input"))
        if not isinstance(request, list) or not request or not isinstance(request[-1], dict):
            return False
        tail = request[-1]
        content = tail.get("content")
        if tail.get("role") != "user":
            return False
        if isinstance(content, str):
            return content == self._expected
        return (isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict)
                and content[0].get("type") in ("input_text", "text")
                and content[0].get("text") == self._expected)

    def commit_request(self, agent: Any, kwargs: dict[str, Any]) -> bool:
        """Persist exactly the staged request before the transport may send it.

        False means no startup input was staged. An exception prevents sending
        a request whose follow-up could not be persisted.
        """
        with self._lock:
            if self._cancelled:
                raise InterruptedError("startup follow-up was stopped before provider dispatch")
            if not self._staged:
                return False
            if not self._is_current():
                if not self._sent:
                    self._restore()
                raise InterruptedError("startup generation ended before provider dispatch")
            if self._sent:
                return False
            if not self._request_has_staged_text(kwargs):
                self._restore()
                raise RuntimeError("startup follow-up was changed before provider dispatch")
            row = self._row
            assert row is not None and self._expected is not None
            expected = self._expected
            entries = list(self._staged)
            self._committing = True
        from agent.agent_init_session import agent_interface
        from agent.session_persistence import _persist_lock, durable_user_row_content

        db = getattr(agent, "_session_db", None)
        store = agent_interface(agent)
        writer = store.messages if store is not None else db
        durable, old_api = durable_user_row_content(agent, row, row.get("content"), self._old_sidecar)
        if not isinstance(durable, str):
            with self._lock:
                self._committing = False
                self._restore()
            raise RuntimeError("startup follow-up requires a text transcript")
        new_content = "\n\n".join([durable, *(part.display_text for part in entries)])
        try:
            with _persist_lock(agent):
                written = writer is not None and writer.set_message_api_content(
                    agent.session_id, row["_row_id"], durable, expected, new_content=new_content
                )
                if written and not self._is_current():
                    old_api = old_api if isinstance(old_api, str) else self._old_wire
                    try:
                        rolled_back = writer.set_message_api_content(
                            agent.session_id, row["_row_id"], new_content, old_api, new_content=durable
                        )
                    except Exception:
                        with self._lock:
                            self._sent = True  # an uncertain rollback must not replay the ticket
                        raise
                    with self._lock:
                        # If Stop raced the write, or rollback is uncertain, never replay the input.
                        self._sent = self._cancel_when_safe or not rolled_back
                        self._committing = False
                        self._restore()
                    raise InterruptedError("startup generation ended before provider dispatch")
            if not written:
                raise RuntimeError("startup follow-up transcript was not persisted")
        except Exception:
            with self._lock:
                self._committing = False
                self._restore()
            raise
        with self._lock:
            row["content"] = "\n\n".join([row["content"], *(part.display_text for part in entries)])
            if getattr(agent, "_persist_user_message_override", None) is not None:
                agent._persist_user_message_override = new_content
            self._sent = True
            self._committing = False
        agent._turn_user_intervened = True
        try:
            self._on_send(entries)
        except Exception:
            logger.exception("Failed to update startup follow-up delivery metadata")
        return True

    def current_message(self, fallback: Any) -> Any:
        with self._lock:
            return self._row["content"] if self._sent and self._row is not None else fallback

    def finish(self) -> None:
        """Revert a staged input when no provider request was made."""
        with self._lock:
            self._open = False
            if not self._sent and not self._committing:
                self._restore()

    def cancel_unread(self) -> tuple[str, ...]:
        """Withdraw an unread batch before a messaging Stop clears its queue."""
        with self._lock:
            if self._sent or self._committing:
                if self._committing:
                    self._cancel_when_safe = True
                return ()  # a provider request may already be starting
            self._open = False
            self._cancelled = bool(self._staged)
            self._restore()
            self._staged = []
            tickets = tuple(self._parts)
            self._parts.clear()
            return tickets

    def withdraw(self, ticket: str) -> bool | None:
        """True for an unread ticket, False after send, None for another ticket."""
        with self._lock:
            if ticket not in self._parts:
                return None
            if not self._sent and not self._committing:
                del self._parts[ticket]
                return True
            return False
