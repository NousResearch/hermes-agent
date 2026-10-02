"""Promotion and demotion: when an action has earned autonomy, and when it loses it.

**Promotion is proposed, never applied here.** :func:`assess` says whether an action
qualifies; an administrator confirms it in the Control Centre, and :func:`change_state`
records that through the bundle so ``apply`` recompiles the policy.

**Demotion is automatic.** Any configured trigger — a person rejecting a call that ran
without them, a false-safe rate above the limit, the provider answering with a different
model — sends a graduated action back to supervised, recorded the same way.

A graduation is only ever valid on the model version it was earned on. Every decision in
the promotion window must have been answered by that one version; a change of model means
the trust is earned again, from zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from nova.autonomy.spec import AutonomySpec
from nova.autonomy.state import STATE_FILE, load_states, with_change


@dataclass
class Assessment:
    action: str
    state: str
    model_version: str = ""
    eligible: bool = False
    #: The model version a promotion would be valid on.
    proposed_model_version: str = ""
    #: Why it is not (yet) eligible, in words.
    waiting_on: list = field(default_factory=list)
    #: Why a graduated action must be demoted now. Empty means it stays.
    demote_because: list = field(default_factory=list)
    progress: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "state": self.state, "model_version": self.model_version,
                "eligible": self.eligible, "proposed_model_version": self.proposed_model_version,
                "waiting_on": list(self.waiting_on), "demote_because": list(self.demote_because),
                "progress": dict(self.progress)}


def _after(ts: Optional[str], since: str) -> bool:
    return bool(ts) and (not since or str(ts) >= since)


def assess(spec: AutonomySpec, ledger: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]]) -> dict[str, Assessment]:
    """One assessment per configured action. Pure: reads the ledger, writes nothing."""
    out: dict[str, Assessment] = {}
    rules, demotion = spec.promotion, spec.demotion
    for name, action in sorted(spec.actions.items()):
        book = ledger["actions"].get(name)
        recorded = states.get(name, {})
        result = Assessment(name, action.state, action.model_version)
        window = book.window if book else None
        decisions = book.window_decisions if book else []
        reviewed = window.shadow_reviewed if window else 0
        agreement = window.agreement if window else None
        false_safe = window.false_safe if window else 0
        result.progress = {
            "reviewed": reviewed, "needed": rules.min_shadow_decisions,
            "agreement": agreement, "min_agreement": rules.min_agreement,
            "false_safe": false_safe, "max_false_safe": rules.max_false_safe, "window": rules.window,
        }

        if action.state == "graduated":
            since = str(recorded.get("changed_at") or "")
            if demotion.on_rejection_of_autonomous and book and book.last_autonomous_rejection \
                    and _after(book.last_autonomous_rejection.get("ts"), since):
                result.demote_because.append(
                    f"a call it made without a person was rejected: {book.last_autonomous_rejection.get('reason') or 'no reason given'}")
            rate = window.false_safe_rate if window else None
            if rate is not None and rate > demotion.on_false_safe_rate_above:
                result.demote_because.append(
                    f"false-safe rate {rate:.1%} is above the limit {demotion.on_false_safe_rate_above:.1%}")
            latest = book.latest_model_version if book else ""
            if demotion.on_provider_model_change and latest and latest != action.model_version:
                result.demote_because.append(
                    f"the provider now answers with model {latest!r}; this action graduated on {action.model_version!r}")
            out[name] = result
            continue

        if spec.provider in ("none", "fake"):
            result.waiting_on.append(f"provider {spec.provider!r} cannot be trusted with autonomy; use a real provider")
        if reviewed < rules.min_shadow_decisions:
            result.waiting_on.append(f"{reviewed} of {rules.min_shadow_decisions} safe verdicts reviewed by a person")
        if agreement is not None and agreement < rules.min_agreement:
            result.waiting_on.append(f"agreement {agreement:.1%} is below {rules.min_agreement:.1%}")
        if false_safe > rules.max_false_safe:
            result.waiting_on.append(f"{false_safe} false-safe verdict(s) in the window; at most {rules.max_false_safe} allowed")
        versions = {d.model_version for d in decisions}
        if len(versions) > 1:
            result.waiting_on.append("the window spans more than one model version; trust is earned on one model")
        version = next(iter(versions)) if len(versions) == 1 else ""
        if version and book and book.latest_model_version and version != book.latest_model_version:
            result.waiting_on.append(f"the provider has moved to {book.latest_model_version!r} since these decisions")
        if not result.waiting_on and version:
            result.eligible, result.proposed_model_version = True, version
        out[name] = result
    return out


def change_state(root: Path, action: str, *, state: str, model_version: str, actor: str, reason: str) -> None:
    """Record an action's new state in the bundle, validated, all or nothing."""
    from nova.spec.writer import edit

    current = load_states(root)
    document = with_change(current, action, state=state, model_version=model_version, actor=actor, reason=reason)

    def mutate(staged) -> None:
        staged.write_yaml(STATE_FILE, document)

    edit(root, mutate)


def open_proposal(events: list, action: str, model_version: str, since: str) -> Optional[Mapping[str, Any]]:
    """The newest proposal for ``action`` on ``model_version`` recorded after ``since``."""
    for event in reversed(events):
        detail = event.get("detail") or {}
        if (event.get("kind") == "autonomy.promotion_proposed" and detail.get("action") == action
                and detail.get("model_version") == model_version and _after(event.get("ts"), since)):
            return event
    return None


def review(spec: AutonomySpec, ledger: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]],
           events: list, *, record: Callable[..., Any], demote: Callable[[str, str], None]) -> dict[str, Assessment]:
    """Assess every action, record new proposals, and demote what must be demoted.

    ``record(kind, subject=..., detail=...)`` writes an audit event; ``demote(action, reason)``
    changes the bundle and re-applies. Both are injected so this stays testable and so the
    control plane owns how a write is audited and applied.
    """
    assessments = assess(spec, ledger, states)
    for name, result in assessments.items():
        if result.demote_because:
            demote(name, "; ".join(result.demote_because))
            result.state = "supervised"
            continue
        since = str(states.get(name, {}).get("changed_at") or "")
        if result.eligible and not open_proposal(events, name, result.proposed_model_version, since):
            record("autonomy.promotion_proposed", subject=name, detail={
                "action": name, "model_version": result.proposed_model_version, **result.progress})
    return assessments
