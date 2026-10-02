"""The ledger: what triage said, what people decided, and what that adds up to.

Built from the tenant's audit log — ``policy.decision`` escalations, ``policy.triage``
verdicts, ``policy.approval_outcome`` answers given in a chat, ``policy.autonomous_action``
records, ``autonomy.incident`` flags — plus the answers people gave on the board, which the
runtime adapter reads from its approvals store. Nothing here calls a provider or a runtime.

**Joining an answer to a verdict.** A chat answer carries the call's ``tool_call_id``; a
board answer carries the request id the plugin derived from the call. Each triage verdict
records both, so a verdict and the person's answer to the same call meet on one key.

**The two numbers that matter.**

* *Shadow agreement* — of the calls triage called safe that a person then reviewed, the
  share they approved. This is what justifies a promotion.
* *False-safe* — calls triage called safe that a person rejected. Shown first, always.

**What cannot be observed is said, not filled in.** The runtime's gate has no
"approved after editing" (a person can only approve or refuse), so that count is ``None``.
A chat ``deny`` also covers a turn interrupted while waiting, so ``rejected`` is an upper
bound. The chat payload does not say who answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

#: Shown beside the numbers wherever they appear.
CAVEATS = (
    "Approved-after-editing is not observable: the approval gate only approves or refuses.",
    "A refusal in chat also covers a conversation interrupted while waiting, so 'rejected' is an upper bound.",
    "Chat approvals do not record who answered; board decisions do.",
)


def _key(detail: Mapping[str, Any]) -> Optional[str]:
    if detail.get("request_id"):
        return f"board:{detail['request_id']}"
    if detail.get("tool_call_id"):
        return f"chat:{detail['tool_call_id']}"
    return None


@dataclass
class Decision:
    """One escalated call that triage judged and a person (maybe) answered."""

    key: str
    agent_id: str
    action: str
    ts: str
    verdict: str
    mode: str
    model_version: str
    failed: list = field(default_factory=list)
    outcome: str = ""          # approved | rejected | timed_out | not_asked | ""
    decided_by: str = ""

    @property
    def reviewed(self) -> bool:
        return self.outcome in ("approved", "rejected")

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "action": self.action, "ts": self.ts, "verdict": self.verdict,
                "mode": self.mode, "model_version": self.model_version, "failed": list(self.failed),
                "outcome": self.outcome, "decided_by": self.decided_by}


@dataclass
class Tally:
    escalations: int = 0
    approved_unchanged: int = 0
    approved_edited: Optional[int] = None  # not observable; see CAVEATS
    rejected: int = 0
    timed_out: int = 0
    autonomous_executed: int = 0
    autonomous_failed: int = 0
    autonomous_open: int = 0
    incidents: int = 0
    shadow_reviewed: int = 0
    shadow_agreed: int = 0
    false_safe: int = 0

    @property
    def agreement(self) -> Optional[float]:
        return self.shadow_agreed / self.shadow_reviewed if self.shadow_reviewed else None

    @property
    def false_safe_rate(self) -> Optional[float]:
        return self.false_safe / self.shadow_reviewed if self.shadow_reviewed else None

    def to_dict(self) -> dict[str, Any]:
        return {"escalations": self.escalations, "approved_unchanged": self.approved_unchanged,
                "approved_edited": self.approved_edited, "rejected": self.rejected,
                "timed_out": self.timed_out, "autonomous_executed": self.autonomous_executed,
                "autonomous_failed": self.autonomous_failed, "autonomous_open": self.autonomous_open,
                "incidents": self.incidents, "shadow_reviewed": self.shadow_reviewed,
                "shadow_agreed": self.shadow_agreed, "false_safe": self.false_safe,
                "agreement": self.agreement, "false_safe_rate": self.false_safe_rate}


@dataclass
class ActionLedger:
    action: str
    all_time: Tally = field(default_factory=Tally)
    window: Tally = field(default_factory=Tally)
    #: The window's reviewed safe verdicts, newest last.
    window_decisions: list = field(default_factory=list)
    by_agent: dict = field(default_factory=dict)
    model_versions: list = field(default_factory=list)  # distinct, in order first seen
    latest_model_version: str = ""
    last_incident: Optional[dict] = None
    last_autonomous_rejection: Optional[dict] = None

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "all_time": self.all_time.to_dict(), "window": self.window.to_dict(),
                "by_agent": {a: t.to_dict() for a, t in sorted(self.by_agent.items())},
                "model_versions": list(self.model_versions), "latest_model_version": self.latest_model_version,
                "last_incident": self.last_incident}


def _count_outcome(tally: Tally, outcome: str) -> None:
    if outcome == "approved":
        tally.approved_unchanged += 1
    elif outcome == "rejected":
        tally.rejected += 1
    elif outcome == "timed_out":
        tally.timed_out += 1


def build(events: Iterable[Mapping[str, Any]], board_answers: Iterable[Mapping[str, Any]], *,
          tenant_id: str, window: int = 100) -> dict[str, Any]:
    """``{"actions": {name: ActionLedger}, "decisions": [Decision], "caveats": [...]}`` for one tenant.

    ``events`` are audit events as dicts (``AuditEvent.to_dict()`` or the JSON lines);
    anything stamped with another tenant is ignored, so one tenant's ledger can never count
    another's decisions even if their records were ever to share a file.
    """
    mine = [e for e in events if (e.get("tenant_id") or "") == tenant_id]
    answers: dict[str, dict[str, str]] = {}
    for event in mine:
        if event.get("kind") == "policy.approval_outcome":
            detail = event.get("detail") or {}
            key = _key(detail)
            if key:
                answers[key] = {"outcome": str(detail.get("outcome") or ""), "decided_by": ""}
    for record in board_answers:
        if (record.get("tenant_id") or tenant_id) != tenant_id or not record.get("request_id"):
            continue
        answers[f"board:{record['request_id']}"] = {
            "outcome": "approved" if record.get("status") == "approved" else "rejected",
            "decided_by": str(record.get("decided_by") or "")}

    actions: dict[str, ActionLedger] = {}

    def ledger(action: str) -> ActionLedger:
        return actions.setdefault(action, ActionLedger(action))

    decisions: dict[str, Decision] = {}
    escalated_keys: set[str] = set()
    autonomous: dict[str, dict[str, Any]] = {}
    for event in mine:
        kind, detail = event.get("kind"), event.get("detail") or {}
        action = str(detail.get("action") or "")
        agent = str(event.get("subject") or "")
        if kind == "policy.decision" and detail.get("effect") == "require_approval" and action:
            key = _key(detail) or event.get("event_id", "")
            if key not in escalated_keys:
                escalated_keys.add(key)
                book = ledger(action)
                book.all_time.escalations += 1
                book.by_agent.setdefault(agent, Tally()).escalations += 1
        elif kind == "policy.triage" and action:
            key = _key(detail)
            version = str(detail.get("model_version") or "")
            book = ledger(action)
            if version and version not in book.model_versions:
                book.model_versions.append(version)
            if version:
                book.latest_model_version = version
            if key and not detail.get("proceed"):
                decisions[key] = Decision(key, agent, action, str(event.get("ts") or ""),
                                          str(detail.get("verdict") or ""), str(detail.get("mode") or ""),
                                          version, list(detail.get("failed") or []))
        elif kind == "policy.autonomous_action":
            # An intent names the action; its outcome (written after the tool ran) closes it.
            entry = autonomous.setdefault(str(event.get("correlation_id") or ""),
                                          {"action": "", "agent": agent, "phase": "intent"})
            if action:
                entry["action"] = action
            if event.get("phase") in ("committed", "failed"):
                entry["phase"] = event["phase"]
        elif kind == "autonomy.incident" and action:
            book = ledger(action)
            book.all_time.incidents += 1
            book.by_agent.setdefault(agent, Tally()).incidents += 1
            book.last_incident = {"ts": event.get("ts"), "reason": detail.get("reason", ""),
                                  "actor": event.get("actor", ""), "autonomous": bool(detail.get("autonomous"))}
            if detail.get("autonomous"):
                book.last_autonomous_rejection = book.last_incident

    for entry in autonomous.values():
        if not entry["action"]:
            continue
        book = ledger(entry["action"])
        agent_tally = book.by_agent.setdefault(entry["agent"], Tally())
        if entry["phase"] == "committed":
            book.all_time.autonomous_executed += 1
            agent_tally.autonomous_executed += 1
        elif entry["phase"] == "failed":
            book.all_time.autonomous_failed += 1
        else:
            book.all_time.autonomous_open += 1

    ordered = sorted(decisions.values(), key=lambda d: d.ts)
    for decision in ordered:
        answer = answers.get(decision.key)
        if answer:
            decision.outcome, decision.decided_by = answer["outcome"], answer["decided_by"]
        book = ledger(decision.action)
        agent_tally = book.by_agent.setdefault(decision.agent_id, Tally())
        for tally in (book.all_time, agent_tally):
            _count_outcome(tally, decision.outcome)
            if decision.verdict == "auto_ok" and decision.reviewed:
                tally.shadow_reviewed += 1
                if decision.outcome == "approved":
                    tally.shadow_agreed += 1
                else:
                    tally.false_safe += 1

    for book in actions.values():
        reviewed_safe = [d for d in ordered if d.action == book.action and d.verdict == "auto_ok" and d.reviewed]
        book.window_decisions = reviewed_safe[-window:]
        for decision in book.window_decisions:
            book.window.shadow_reviewed += 1
            if decision.outcome == "approved":
                book.window.shadow_agreed += 1
            else:
                book.window.false_safe += 1

    return {"actions": actions, "decisions": ordered, "caveats": list(CAVEATS)}
