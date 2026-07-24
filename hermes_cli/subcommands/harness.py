"""Operator CLI for the model-independent Harness control plane."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from hermes_cli.control_plane.nodes import NodeRegistry

_NODE_CREDENTIAL_ENV = "HERMES_NODE_CREDENTIAL"


def _registry() -> NodeRegistry:
    return NodeRegistry()


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _capabilities(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("capabilities must be a JSON object")
    return parsed


def cmd_nodes_enroll(args) -> None:
    issuance = _registry().enroll(
        enrollment_key=args.enrollment_key,
        node_id=args.node_id,
        role=args.role,
        owner=args.owner,
        actor=args.actor,
        capabilities=args.capabilities,
    )
    _print({"node": asdict(issuance.node), "credential": issuance.credential})


def cmd_nodes_rotate_credential(args) -> None:
    issuance = _registry().rotate_credential(
        args.node_id,
        actor=args.actor,
        expected_credential_revision=args.expected_credential_revision,
    )
    _print({"node": asdict(issuance.node), "credential": issuance.credential})


def cmd_nodes_revoke_credential(args) -> None:
    node = _registry().revoke_credential(
        args.node_id,
        actor=args.actor,
        expected_credential_revision=args.expected_credential_revision,
    )
    _print(asdict(node))


def cmd_nodes_list(args) -> None:
    _print([asdict(node) for node in _registry().list(state=args.state)])


def cmd_nodes_show(args) -> None:
    node = _registry().get(args.node_id)
    if node is None:
        raise SystemExit(f"managed node not found: {args.node_id}")
    _print(asdict(node))


def cmd_nodes_transition(args) -> None:
    node = _registry().transition(
        args.node_id,
        args.state,
        actor=args.actor,
        expected_revision=args.expected_revision,
        reason=args.reason,
    )
    _print(asdict(node))


def cmd_nodes_history(args) -> None:
    _print([asdict(event) for event in _registry().history(args.node_id)])


def cmd_nodes_audit(args) -> None:
    valid = _registry().verify_audit_chain()
    _print({"valid": valid})
    if not valid:
        raise SystemExit(1)


def cmd_nodes_report(args) -> None:
    credential = os.environ.get(_NODE_CREDENTIAL_ENV)
    if not credential:
        raise SystemExit(
            f"{_NODE_CREDENTIAL_ENV} must contain the managed-node credential"
        )
    report = _registry().submit_observation(
        args.node_id,
        credential=credential,
        schema_version=args.schema_version,
        report_sequence=args.report_sequence,
        observed_at=args.observed_at,
        health_state=args.health_state,
        capabilities=args.capabilities,
    )
    _print(asdict(report))


def cmd_nodes_observation(args) -> None:
    report = _registry().latest_observation(args.node_id)
    _print(asdict(report) if report is not None else None)


def cmd_nodes_policy_set(args) -> None:
    policy = _registry().set_policy(
        args.node_id,
        actor=args.actor,
        schema_version=args.schema_version,
        desired_health_state=args.health_state,
        capabilities=args.capabilities,
        expected_revision=args.expected_revision,
    )
    _print(asdict(policy))


def cmd_nodes_policy_show(args) -> None:
    policy = _registry().get_policy(args.node_id)
    _print(asdict(policy) if policy is not None else None)


def cmd_nodes_reconcile(args) -> None:
    _print(asdict(_registry().reconcile(args.node_id)))


def build_harness_parser(subparsers) -> None:
    harness = subparsers.add_parser(
        "harness",
        help="Manage the model-independent Harness control plane",
    )
    harness_sub = harness.add_subparsers(dest="harness_command", required=True)
    nodes = harness_sub.add_parser("nodes", help="Manage enrolled nodes")
    nodes_sub = nodes.add_subparsers(dest="nodes_command", required=True)

    enroll = nodes_sub.add_parser("enroll", help="Enroll a managed node")
    enroll.add_argument("--enrollment-key", required=True)
    enroll.add_argument("--node-id")
    enroll.add_argument("--role", required=True)
    enroll.add_argument("--owner", required=True)
    enroll.add_argument("--actor", required=True)
    enroll.add_argument("--capabilities", type=_capabilities, default={})
    enroll.set_defaults(func=cmd_nodes_enroll)

    list_parser = nodes_sub.add_parser("list", help="List managed nodes")
    list_parser.add_argument(
        "--state",
        choices=["enrolled", "active", "quarantined", "recovering", "retired"],
    )
    list_parser.set_defaults(func=cmd_nodes_list)

    show = nodes_sub.add_parser("show", help="Show one managed node")
    show.add_argument("node_id")
    show.set_defaults(func=cmd_nodes_show)

    transition = nodes_sub.add_parser(
        "transition", help="Apply an explicit lifecycle transition"
    )
    transition.add_argument("node_id")
    transition.add_argument(
        "state", choices=["active", "quarantined", "recovering", "retired"]
    )
    transition.add_argument("--actor", required=True)
    transition.add_argument("--expected-revision", required=True, type=int)
    transition.add_argument("--reason", required=True)
    transition.set_defaults(func=cmd_nodes_transition)

    rotate = nodes_sub.add_parser(
        "rotate-credential", help="Rotate and return a node credential once"
    )
    rotate.add_argument("node_id")
    rotate.add_argument("--actor", required=True)
    rotate.add_argument("--expected-credential-revision", required=True, type=int)
    rotate.set_defaults(func=cmd_nodes_rotate_credential)

    revoke = nodes_sub.add_parser("revoke-credential", help="Revoke a node credential")
    revoke.add_argument("node_id")
    revoke.add_argument("--actor", required=True)
    revoke.add_argument("--expected-credential-revision", required=True, type=int)
    revoke.set_defaults(func=cmd_nodes_revoke_credential)

    history = nodes_sub.add_parser("history", help="Show node audit history")
    history.add_argument("node_id")
    history.set_defaults(func=cmd_nodes_history)

    audit = nodes_sub.add_parser("audit", help="Verify the audit hash chain")
    audit.set_defaults(func=cmd_nodes_audit)

    report = nodes_sub.add_parser(
        "report",
        help="Submit an authenticated observed-state report",
        description=(
            "Submit an observed-state report using the managed-node credential "
            f"from {_NODE_CREDENTIAL_ENV}."
        ),
    )
    report.add_argument("node_id")
    report.add_argument("--schema-version", type=int, default=1)
    report.add_argument("--report-sequence", type=int, required=True)
    report.add_argument("--observed-at", type=int, required=True)
    report.add_argument(
        "--health-state",
        choices=["healthy", "degraded", "unhealthy", "unknown"],
        required=True,
    )
    report.add_argument("--capabilities", type=_capabilities, default={})
    report.set_defaults(func=cmd_nodes_report)

    observation = nodes_sub.add_parser(
        "observation", help="Show the latest observed-state report"
    )
    observation.add_argument("node_id")
    observation.set_defaults(func=cmd_nodes_observation)

    policy = nodes_sub.add_parser("policy", help="Manage desired node policy")
    policy_sub = policy.add_subparsers(dest="policy_command", required=True)
    policy_set = policy_sub.add_parser("set", help="Create or update desired policy")
    policy_set.add_argument("node_id")
    policy_set.add_argument("--actor", required=True)
    policy_set.add_argument("--schema-version", type=int, default=1)
    policy_set.add_argument(
        "--health-state",
        choices=["healthy", "degraded", "unhealthy", "unknown"],
    )
    policy_set.add_argument("--capabilities", type=_capabilities, default={})
    policy_set.add_argument("--expected-revision", type=int, required=True)
    policy_set.set_defaults(func=cmd_nodes_policy_set)
    policy_show = policy_sub.add_parser("show", help="Show desired policy")
    policy_show.add_argument("node_id")
    policy_show.set_defaults(func=cmd_nodes_policy_show)

    reconcile = nodes_sub.add_parser(
        "reconcile", help="Show read-only desired/observed drift"
    )
    reconcile.add_argument("node_id")
    reconcile.set_defaults(func=cmd_nodes_reconcile)
