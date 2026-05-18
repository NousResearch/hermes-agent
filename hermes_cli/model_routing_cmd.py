"""CLI subcommand for dry-run model-routing previews.

This module intentionally does not mutate Hermes runtime configuration. It loads
local policy data, calls the pure dry-run router, and prints the recommendation
for human review.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agent.model_routing import RoutingContext, load_policy, recommend_model

_SUCCESS_EXIT = 0
_USAGE_EXIT = 2
_FAILURE_EXIT = 1


def _build_context(args: argparse.Namespace) -> RoutingContext:
    """Translate argparse inputs into a router context."""

    return RoutingContext(
        task_type=args.task_type,
        agent_role=args.agent_role,
        risk_level=args.risk_level,
        client_facing=bool(args.client_facing),
        sensitive_data=bool(args.sensitive_data),
        final_authority=bool(args.final_authority),
        complexity=args.complexity,
        current_provider=args.current_provider,
        current_model=args.current_model,
    )


def _decision_payload(context: RoutingContext, decision) -> dict:
    """Return stable JSON for scripts and tests."""

    payload = asdict(decision)
    payload["context"] = asdict(context)
    payload["runtime_config_changed"] = False
    return payload


def _print_human_preview(context: RoutingContext, payload: dict) -> None:
    """Print a concise human-readable route preview."""

    print("Routing preview (DRY RUN)")
    print("=========================")
    print(f"Task type: {context.task_type}")
    print(f"Agent role: {context.agent_role}")
    print(f"Risk level: {context.risk_level}")
    print(f"Client-facing: {'yes' if context.client_facing else 'no'}")
    print(f"Sensitive data: {'yes' if context.sensitive_data else 'no'}")
    print(f"Final authority: {'yes' if context.final_authority else 'no'}")
    print()
    print(f"Recommended provider: {payload['provider']}")
    print(f"Recommended model: {payload['model']}")
    print(f"Tier: {payload['tier']}")
    print(f"Estimated cost class: {payload['estimated_cost_class']}")
    print(f"Fallback model: {payload['fallback_model'] or '(none)'}")
    print(f"Approval required: {'yes' if payload['approval_required'] else 'no'}")
    print(f"Runtime config changed: {'yes' if payload['runtime_config_changed'] else 'no'}")
    print()
    print(f"Reason: {payload['reason']}")
    if payload.get("escalation_reason"):
        print(f"Escalation reason: {payload['escalation_reason']}")
    warnings = payload.get("policy_warnings") or []
    if warnings:
        print("Policy warnings:")
        for warning in warnings:
            print(f"  - {warning}")


def _cmd_preview(args: argparse.Namespace) -> int:
    """Run the dry-run router and print the decision."""

    try:
        policy_path = Path(args.policy) if args.policy else None
        policy = load_policy(policy_path)
        context = _build_context(args)
        decision = recommend_model(context, policy)
        payload = _decision_payload(context, decision)
    except Exception as exc:  # noqa: BLE001 - CLI should report errors cleanly
        print(f"hermes routing preview: {exc}", file=sys.stderr)
        return _FAILURE_EXIT

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human_preview(context, payload)
    return _SUCCESS_EXIT


def cmd_routing(args: argparse.Namespace) -> None:
    """Entry point wired into the top-level argparse dispatcher."""

    command = getattr(args, "routing_command", None)
    if command is None:
        print(
            "hermes routing: subcommand required (try: hermes routing preview --help)",
            file=sys.stderr,
        )
        sys.exit(_USAGE_EXIT)

    if command == "preview":
        sys.exit(_cmd_preview(args))

    print(f"hermes routing: unknown subcommand {command!r}", file=sys.stderr)
    sys.exit(_USAGE_EXIT)


def register_routing_subparser(subparsers) -> argparse.ArgumentParser:
    """Create the ``routing`` subparser and return it."""

    parser = subparsers.add_parser(
        "routing",
        help="Preview policy-driven model routing decisions (dry run).",
        description=(
            "Preview which provider/model Hermes would recommend under the local "
            "model-routing policy. This is dry-run only: it does not call a model, "
            "write config, change runtime selection, or spend credits."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    routing_subparsers = parser.add_subparsers(dest="routing_command")

    preview = routing_subparsers.add_parser(
        "preview",
        help="Preview a route decision for a task context.",
        description="Run the local dry-run router for a supplied task context.",
    )
    preview.add_argument(
        "--task-type",
        required=True,
        help=(
            "Policy task key, e.g. daily_ops, strategy, client_facing_draft, "
            "security_privacy_review, code_explanation."
        ),
    )
    preview.add_argument(
        "--agent-role",
        default="hermes",
        help="Agent/persona role requesting the route (default: hermes).",
    )
    preview.add_argument(
        "--risk-level",
        choices=["low", "medium", "high"],
        default="medium",
        help="Risk level for the task (default: medium).",
    )
    preview.add_argument(
        "--complexity",
        choices=["low", "medium", "high", "standard"],
        default="standard",
        help="Task complexity signal (default: standard).",
    )
    preview.add_argument(
        "--client-facing",
        action="store_true",
        help="Mark the task as client/prospect/public-facing.",
    )
    preview.add_argument(
        "--sensitive-data",
        action="store_true",
        help="Mark the task as involving sensitive/private data.",
    )
    preview.add_argument(
        "--final-authority",
        action="store_true",
        help="Mark the model output as intended final authority rather than draft/critique.",
    )
    preview.add_argument(
        "--current-provider",
        default=None,
        help="Optional currently configured provider for comparison metadata.",
    )
    preview.add_argument(
        "--current-model",
        default=None,
        help="Optional currently configured model for comparison metadata.",
    )
    preview.add_argument(
        "--policy",
        default=None,
        help="Optional path to a routing policy YAML file (default: bundled Caelus policy).",
    )
    preview.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit JSON instead of human-readable output.",
    )

    parser.set_defaults(func=cmd_routing)
    return parser


__all__ = ["cmd_routing", "register_routing_subparser"]
