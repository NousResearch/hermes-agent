"""Operator CLI for the model-independent Harness control plane."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from hermes_cli.control_plane.nodes import NodeRegistry


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
