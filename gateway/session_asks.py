"""The asks of one live session: the prompts its turn blocks on, as API objects.

An ask is a dangerous-command approval, a clarify question or a skill secret request. Each is recorded
where the session's owner already raises it (``PendingControls.register`` / ``register_clarify`` and the
``secret`` server→client request in ``gateway.session_server_requests``), under the same prompt id the
``/v1/runs`` approval/clarify routes and the WS ``pending_controls`` use. The ledger only observes: an
answer still goes through the native waiter, and the ask's status is read from how that waiter ended.
A secret value never enters it.

One ledger per live session event stream (:func:`ledger_of`), so a profile's asks never mix with
another's and they go when the live session does. Closed asks are kept up to ``_CLOSED_KEEP``.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Optional

from tools.clarify_gateway import CANCELLED, SKIPPED

KINDS = ("approval", "question", "secret")
STATUSES = ("pending", "answered", "dismissed", "declined", "interrupted", "expired")
_CLOSED_KEEP = 100
# Prompt fields that identify the ask rather than describe it.
_IDENTITY_FIELDS = frozenset({"kind", "prompt_id", "execution_generation"})

# (status, answer, closed_by) once an ask is settled.
Outcome = tuple[str, Optional[dict], str]
INTERRUPTED: Outcome = ("interrupted", None, "interrupt")


@dataclass
class Ask:
    id: str
    kind: str
    session_id: str
    execution_generation: Optional[int]
    payload: dict
    created_at: float = field(default_factory=time.time)
    status: str = "pending"
    closed_at: Optional[float] = None
    closed_by: Optional[str] = None
    answer: Optional[dict] = None
    # Reads the native waiter for an ask no settle callback reports: None while it still waits.
    probe: Optional[Callable[[], Optional[Outcome]]] = field(default=None, repr=False)
    # (status, answer) of an API answer or dismiss in flight: the native settle it causes is reported
    # as that, closed by ``api``, instead of as an answer from another client.
    claim: Optional[tuple[str, Optional[dict]]] = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {"object": "hermes.session.ask", "id": self.id, "session_id": self.session_id,
                "kind": self.kind, "status": self.status, "created_at": self.created_at,
                "closed_at": self.closed_at, "closed_by": self.closed_by,
                "execution_generation": self.execution_generation,
                "payload": deepcopy(self.payload), "answer": deepcopy(self.answer)}


def prompt_payload(prompt: dict) -> dict:
    return {k: deepcopy(v) for k, v in prompt.items() if k not in _IDENTITY_FIELDS}


def approval_outcome(reason: str, outcome: dict) -> Outcome:
    """How a native approval wait ended (``tools.approval.observe_gateway_outcome``) as an ask outcome.
    ``replied``: the deny was a message the user sent instead of answering, which declines the ask."""
    choice = outcome.get("choice")
    if choice == "deny" and outcome.get("replied"):
        return "declined", {"choice": "deny"}, "message"
    if choice is not None:
        return "answered", {"choice": choice}, "session"
    if outcome.get("expired") or reason == "timeout":
        return "expired", None, "timeout"
    return INTERRUPTED


def question_outcome(response: Optional[str]) -> Outcome:
    """A settled clarify waiter's ``response`` (``tools.clarify_gateway``) as an ask outcome."""
    if response is None:
        return "expired", None, "timeout"
    if response == CANCELLED:
        return INTERRUPTED
    if response == SKIPPED:
        return "dismissed", None, "session"
    return "answered", {"text": response}, "session"


def secret_outcome(*, settled: bool, value: str, wait_reason: str) -> Outcome:
    """A ``secret`` request's end: answered (the value itself is never recorded), skipped, or unanswered."""
    if settled:
        return ("answered", None, "session") if value else ("dismissed", None, "session")
    return INTERRUPTED if wait_reason == "interrupted" else ("expired", None, "timeout")


class AskLedger:
    def __init__(self):
        self._lock = threading.Lock()
        self._asks: OrderedDict[str, Ask] = OrderedDict()

    def open(self, ask: Ask) -> Ask:
        with self._lock:
            self._asks[ask.id] = ask
            closed = [k for k, a in self._asks.items() if a.status != "pending"]
            for key in closed[:max(0, len(closed) - _CLOSED_KEEP)]:
                del self._asks[key]
        return ask

    def settle(self, ask_id: str, status: str, answer: Optional[dict] = None, closed_by: str = "session") -> None:
        """Close *ask_id*; the first close wins. A settle caused by an API claim is reported as the claim."""
        with self._lock:
            ask = self._asks.get(ask_id)
            if ask is None or ask.status != "pending":
                return
            if ask.claim is not None and status in ("answered", "dismissed"):
                (status, answer), closed_by = ask.claim, "api"
            ask.status, ask.answer, ask.closed_by, ask.closed_at = status, answer, closed_by, time.time()
            ask.claim = ask.probe = None

    def claim(self, ask_id: str, status: str, answer: Optional[dict]) -> None:
        with self._lock:
            if (ask := self._asks.get(ask_id)) is not None:
                ask.claim = (status, answer)

    def release(self, ask_id: str) -> None:
        with self._lock:
            if (ask := self._asks.get(ask_id)) is not None:
                ask.claim = None

    def absorb(self, earlier: AskLedger) -> None:
        """Take *earlier*'s asks (an earlier incarnation of this session) in front of this ledger's own."""
        with earlier._lock, self._lock:
            merged = OrderedDict(earlier._asks)
            merged.update(self._asks)
            self._asks = merged

    def _refresh(self) -> None:
        with self._lock:
            probes = [(a.id, a.probe) for a in self._asks.values() if a.status == "pending" and a.probe]
        for ask_id, probe in probes:
            if (outcome := probe()) is not None:
                self.settle(ask_id, *outcome)

    def get(self, ask_id: str) -> Optional[Ask]:
        self._refresh()
        with self._lock:
            ask = self._asks.get(ask_id)
            return None if ask is None else Ask(**{**ask.__dict__, "probe": None, "claim": None})

    def list(self, statuses=STATUSES) -> list[dict[str, Any]]:
        """Asks with one of *statuses*, newest first."""
        self._refresh()
        with self._lock:
            return [a.as_dict() for a in reversed(self._asks.values()) if a.status in statuses]


def ledger_of(events) -> AskLedger:
    """The asks ledger of one live session event stream, attached on first use (any stream object)."""
    return vars(events).setdefault("asks", AskLedger())


def retired_of(authority) -> RetiredAsks:
    """*authority*'s ledgers of retired conversations, attached on first use."""
    return vars(authority).setdefault("retired_asks", RetiredAsks())


class RetiredAsks:
    """The ledgers of conversations retirement let go (:func:`retired_of`), newest kept."""
    _KEEP = 256

    def __init__(self):
        self._ledgers: OrderedDict[str, AskLedger] = OrderedDict()

    def keep(self, session_id: str, ledger: AskLedger) -> None:
        if (earlier := self._ledgers.pop(session_id, None)) is not None:
            ledger.absorb(earlier)  # the conversation was restored and retired again
        if ledger.list():
            self._ledgers[session_id] = ledger
            while len(self._ledgers) > self._KEEP:
                self._ledgers.popitem(last=False)

    def forget(self, session_id: str) -> None:
        self._ledgers.pop(session_id, None)

    def get(self, session_id: str) -> Optional[AskLedger]:
        return self._ledgers.get(session_id)
