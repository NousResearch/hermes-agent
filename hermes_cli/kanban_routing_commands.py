"""Operator policy and receipt commands for guided Kanban routing."""
from __future__ import annotations

import json

from agent import model_selection_store as store
from agent.model_selection_types import RoutingBlocked
from hermes_constants import get_hermes_home
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _render(args, value, text):
    from hermes_cli.kanban import _print_json

    if getattr(args, "json", False):
        _print_json(value)
    else:
        print(text)
    return 0


def _publish(args, home):
    with open(args.policy_json, "r", encoding="utf-8") as stream:
        policy = json.load(stream)
    record = store.publish_policy(home, policy, approval_ref=args.approval_ref)
    return _render(args, record, f"published {record['policy_id']} revision {record['revision']} "
                   f"(hash {record['content_hash'][:12]}…)")


def _attest(args, home):
    from agent.model_selection import _validate_requirements
    from agent.model_selection_classification import attest_classification

    with open(args.classification_json, encoding="utf-8") as stream:
        scope = json.load(stream)
    if not isinstance(scope, dict) or set(scope) != {"requirements", "complete", "risk_flags", "evidence"}:
        raise RoutingBlocked("schema_invalid", "classification needs requirements, complete, risk_flags, evidence")
    _validate_requirements(scope["requirements"])
    record = attest_classification(
        home, scope["requirements"], authority="operator",
        attester={"approval_ref": args.approval_ref}, complete=scope["complete"],
        risk_flags=scope["risk_flags"], evidence=scope["evidence"],
        expected_version=args.expected_version,
    )
    return _render(args, record, f"attested classification version {record['version']}")


def _activate(args, home):
    store.activate_policy(home, args.policy_id, args.revision)
    print(f"activated {args.policy_id} revision {args.revision}")
    return 0


def _admission(args, home):
    operation, label = {
        "revoke": (store.revoke_route, "revoked"),
        "readmit": (store.readmit_route, "readmitted"),
    }[args.routing_action]
    record = operation(home, args.policy_id, route_id=args.route_id,
                       reason=args.reason, approval_ref=args.approval_ref)
    scope = record["route_id"] or "ALL ROUTES"
    return _render(args, record, f"{label} {scope} of {record['policy_id']} "
                   f"(generation {record['id']}, reason={record['reason']!r})")


def _reconcile(args, home):
    record = store.authorize_replay(home, args.receipt_id,
                                    reason=args.reason, approval_ref=args.approval_ref)
    return _render(args, record, f"authorized one replay for {record['receipt_id']} "
                   f"(reason={record['reason']!r})")


def _show(args, home):
    from hermes_cli.kanban import _err

    policy = store.get_active_policy(home, args.policy_id)
    if policy is None:
        return _err(f"kanban routing: no active policy for {args.policy_id!r}")
    return _render(args, policy, f"{policy['policy_id']} revision {policy['revision']}: "
                   f"{len(policy.get('routes', []))} route(s)")


def _revisions(args, home):
    from hermes_cli.kanban import _print_json

    revisions = store.list_policy_revisions(home, args.policy_id)
    if getattr(args, "json", False):
        _print_json(revisions)
    else:
        for revision in revisions:
            marker = "*" if revision["active"] else " "
            print(f"{marker} rev {revision['revision']}  approval_ref={revision['approval_ref']!r}  "
                  f"hash={revision['content_hash'][:12]}…")
    return 0


def _observations_text(outcomes):
    observed = [event for event in outcomes if "actual_model" in event["payload"]]
    if not observed:
        return "observed: unknown (no runtime identity recorded)"
    return "\n".join(
        f"observed: {event['payload'].get('actual_provider', 'unknown')}/"
        f"{event['payload']['actual_model']} ({event['kind']}, seq={event['seq']})"
        for event in observed
    )


def _receipt(args, home):
    from hermes_cli.kanban import _err

    receipt = store.get_receipt(home, args.receipt_id)
    if receipt is None:
        return _err(f"kanban routing: no such receipt {args.receipt_id!r}")
    selected = receipt["selected"]
    outcomes = store.list_outcomes(home, args.receipt_id)
    return _render(args, {**receipt, "outcomes": outcomes},
                   f"{args.receipt_id}: role={receipt['requirements']['role']} "
                   f"-> {selected['provider']}/{selected['model']} (route {selected['route_id']}, "
                   f"policy {receipt['policy_id']}#{receipt['policy_revision']})\n"
                   f"{_observations_text(outcomes)}")


def _receipt_for_task(args, home):
    from hermes_cli.kanban import _err

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, args.task_id)
    if task is None:
        return _err(f"kanban routing: no such task {args.task_id!r}")
    receipt_id = getattr(task, "routing_receipt_id", None)
    if not receipt_id:
        return _err(f"kanban routing: task {args.task_id} has no routing receipt "
                    "(unmanaged, or not yet claimed/dispatched)")
    receipt = store.get_receipt(home, receipt_id)
    selected = (receipt or {}).get("selected", {})
    outcomes = store.list_outcomes(home, receipt_id)
    return _render(args, {"receipt_id": receipt_id, "decision": receipt, "outcomes": outcomes},
                   f"{args.task_id}: receipt={receipt_id} -> "
                   f"{selected.get('provider')}/{selected.get('model')}\n"
                   f"{_observations_text(outcomes)}")


def _diagnostic(args, home):
    from hermes_cli.routing_diagnostics import run_diagnostic

    return run_diagnostic(args)


_HANDLERS = {
    "attest": _attest,
    "validate": _diagnostic, "explain": _diagnostic,
    "publish": _publish, "activate": _activate,
    "revoke": _admission, "readmit": _admission, "reconcile": _reconcile,
    "show": _show, "revisions": _revisions,
    "receipt": _receipt, "receipt-for-task": _receipt_for_task,
}


def run_routing_command(args) -> int:
    from hermes_cli.kanban import _err

    action = getattr(args, "routing_action", None)
    handler = _HANDLERS.get(action)
    if handler is None:
        return _err(f"kanban routing: unknown action {action!r}", 2)
    try:
        return handler(args, get_hermes_home())
    except (RoutingBlocked, FileNotFoundError) as exc:
        return _err(f"kanban routing: {exc}")
