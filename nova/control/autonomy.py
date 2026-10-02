"""The Control API's autonomy routes: the ledger read, and the three acts on an action.

* ``GET  /autonomy`` — every triaged action: its state, its ledger, its progress toward
  promotion, recent verdicts, and notices the screen must show (shadow mode, content
  leaving the environment). Reading also runs the review: proposals are recorded and due
  demotions happen, so a demotion never waits for someone to remember to run a job.
* ``POST /autonomy/<action>/promote`` — admin. Confirms a proposal. Re-assessed at the
  moment of confirming: a proposal that no longer holds is refused, not applied.
* ``POST /autonomy/<action>/demote`` — admin. Back to supervised, now.
* ``POST /autonomy/<action>/incident`` — admin. Flags a problem with a call; when it names a
  call that ran without a person, the action is demoted on the spot.

Every write is ``intent`` → edit the bundle → apply → ``committed`` (or ``failed``), the
same shape as every agent write, so a crash part-way leaves a detectable open intent.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Mapping

from nova.audit import new_correlation_id
from nova.errors import NovaError

AUTONOMY_ACTIONS = ("promote", "demote", "incident")

#: Verdicts shown on the screen.
RECENT_LIMIT = 25


def _events(api) -> list[dict[str, Any]]:
    try:
        return [json.loads(json.dumps(asdict(e), default=str)) for e in api.audit.read()]
    except Exception:  # noqa: BLE001 — no log yet reads as no history
        return []


def _context(api):
    from nova.autonomy import ledger as ledger_mod
    from nova.autonomy.state import load_states

    policy = api.bundle.policy
    spec = policy.autonomy if policy is not None else None
    if spec is None:
        return None
    events = _events(api) if api.audit is not None else []
    outcomes = getattr(api.runtime, "approval_outcomes", None)
    board = outcomes() if callable(outcomes) else []
    built = ledger_mod.build(events, board, tenant_id=api.bundle.tenant_id, window=spec.promotion.window)
    return spec, built, load_states(api.bundle.root), events


def _change(api, principal, action: str, *, state: str, model_version: str, reason: str, kind: str):
    """intent → bundle edit → reload → apply → committed. Returns the applied summary."""
    from nova.autonomy.review import change_state

    correlation_id = new_correlation_id()
    audit = api.audit.with_actor(principal.name if principal else "nova-autonomy")
    detail = {"action": action, "state": state, "model_version": model_version, "reason": reason}
    audit.record(kind=kind, phase="intent", subject=action, correlation_id=correlation_id, detail=detail)
    try:
        change_state(api.bundle.root, action, state=state, model_version=model_version,
                     actor=principal.name if principal else "nova-autonomy", reason=reason)
    except Exception as exc:
        audit.record(kind=kind, phase="failed", subject=action, correlation_id=correlation_id,
                     detail={**detail, "error": str(exc)})
        raise
    api._reload_bundle()
    applied = api._apply_to_runtime(correlation_id, principal.name if principal else "nova-autonomy")
    audit.record(kind=kind, phase="committed" if applied.get("applied") else "failed", subject=action,
                 correlation_id=correlation_id, detail={**detail, "applied": applied.get("applied", False)})
    return applied


def read(api):
    from nova.autonomy.review import review
    from nova.control.api import Response

    context = _context(api)
    if context is None:
        return Response(200, {"configured": False, "actions": [], "recent": [], "notices": []})
    spec, built, states, events = context

    if api.audit is not None:
        def record(kind, *, subject, detail):
            api.audit.with_actor("nova-autonomy").record(kind=kind, subject=subject,
                                                         correlation_id=new_correlation_id(), detail=detail)

        def demote(action, reason):
            _change(api, None, action, state="supervised", model_version="", reason=reason,
                    kind="autonomy.demoted")

        assessments = review(spec, built, states, events, record=record, demote=demote)
        # A demotion edited and reloaded the bundle; show what is true now.
        from nova.autonomy.state import load_states

        spec, states, events = api.bundle.policy.autonomy, load_states(api.bundle.root), _events(api)
    else:
        from nova.autonomy.review import assess

        assessments = assess(spec, built, states)

    from nova.autonomy.review import open_proposal

    rows = []
    for name, action in sorted(spec.actions.items()):
        book = built["actions"].get(name)
        result = assessments[name]
        recorded = states.get(name, {})
        proposal = open_proposal(events, name, result.proposed_model_version, str(recorded.get("changed_at") or "")) \
            if result.eligible else None
        rows.append({
            "action": name,
            "state": "proposed" if proposal and action.state == "supervised" else action.state,
            "model_version": action.model_version,
            "proposal": {"model_version": result.proposed_model_version, "ts": proposal.get("ts")} if proposal else None,
            "assessment": result.to_dict(),
            "ledger": book.to_dict() if book else None,
            "last_change": recorded or None,
            "last_demotion": recorded.get("reason") if recorded.get("state") == "supervised" and recorded.get("demoted_from_model") else None,
        })
    recent = [d.to_dict() for d in built["decisions"][-RECENT_LIMIT:]][::-1]
    notices = []
    if spec.mode == "shadow":
        notices.append({"kind": "shadow", "text": "Shadow mode: triage records what it would have done; a person still decides every call."})
    if spec.data == "full_args":
        notices.append({"kind": "full_args", "text": "Content leaves your environment to the triage provider (data: full_args)."})
    if spec.provider == "fake":
        notices.append({"kind": "fake", "text": "The fake provider is in use: verdicts are not real and no action can graduate."})
    return Response(200, {
        "configured": True, "provider": spec.provider, "mode": spec.mode, "data": spec.data,
        "promotion": spec.promotion.to_dict(), "demotion": spec.demotion.to_dict(),
        "actions": rows, "recent": recent, "notices": notices, "caveats": built["caveats"],
    })


def write(api, tail: str, verb: str, principal, payload: Mapping[str, Any]):
    from nova.autonomy.review import assess, open_proposal
    from nova.control.api import Response, _error

    action = tail.strip("/").split("/")[1]
    context = _context(api)
    if context is None or action not in context[0].actions:
        return _error(404, f"no triaged action {action!r} in this tenant's policy")
    spec, built, states, events = context
    reason = str(payload.get("reason") or "").strip()

    try:
        if verb == "promote":
            result = assess(spec, built, states)[action]
            if spec.actions[action].state == "graduated":
                return _error(409, f"{action} is already graduated")
            since = str(states.get(action, {}).get("changed_at") or "")
            if not result.eligible or not open_proposal(events, action, result.proposed_model_version, since):
                why = "; ".join(result.waiting_on) or "no proposal has been recorded yet; open the Autonomy screen"
                return _error(409, f"{action} does not qualify for promotion now: {why}")
            progress = result.progress
            applied = _change(api, principal, action, state="graduated",
                              model_version=result.proposed_model_version,
                              reason=reason or (f"promotion confirmed ({progress['reviewed']} reviewed, "
                                                f"{(progress['agreement'] or 0):.0%} agreement)"),
                              kind="autonomy.promoted")
        elif verb == "demote":
            if spec.actions[action].state != "graduated":
                return _error(409, f"{action} is not graduated")
            applied = _change(api, principal, action, state="supervised", model_version="",
                              reason=reason or "demoted by an administrator", kind="autonomy.demoted")
        else:  # incident
            if not reason:
                return _error(400, "an incident needs a reason: what went wrong")
            autonomous = bool(payload.get("autonomous"))
            api.audit.with_actor(principal.name).record(
                kind="autonomy.incident", subject=action, correlation_id=new_correlation_id(),
                detail={"action": action, "reason": reason, "autonomous": autonomous,
                        "ref": str(payload.get("ref") or "")})
            applied = {"applied": True}
            if autonomous and spec.actions[action].state == "graduated" and spec.demotion.on_rejection_of_autonomous:
                applied = _change(api, principal, action, state="supervised", model_version="",
                                  reason=f"a call it made without a person was rejected: {reason}",
                                  kind="autonomy.demoted")
    except NovaError as exc:
        return _error(400, str(exc))
    state = api.bundle.policy.autonomy.actions[action].state
    return Response(200, {"ok": True, "actor": principal.name, "action": action, "state": state,
                          "applied": applied.get("applied", False)})
