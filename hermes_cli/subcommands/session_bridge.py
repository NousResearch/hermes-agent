"""``hermes session-bridge`` persisted voice/session ingress CLI."""

from __future__ import annotations

import argparse
import json


def _emit(value) -> int:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_session_bridge(args: argparse.Namespace) -> int:
    from hermes_cli.session_bridge import BridgeStore, capabilities, request_dict, run_once

    if args.bridge_action == "capabilities":
        return _emit(capabilities())
    store = BridgeStore()
    if args.bridge_action == "submit":
        return _emit(request_dict(store.submit(
            idempotency_key=args.idempotency_key,
            requester_principal=args.requester,
            source_session_id=args.source_session,
            authorization_ref=args.authorization_ref,
            backend=args.backend,
            kind=args.kind,
            target_session_id=args.target_session,
            prompt=args.prompt,
        )))
    if args.bridge_action == "status":
        return _emit(request_dict(store.get(
            args.correlation_id, requester_principal=args.requester,
            source_session_id=args.source_session,
        )))
    if args.bridge_action == "cancel":
        return _emit(request_dict(store.cancel(
            args.correlation_id, requester_principal=args.requester,
            source_session_id=args.source_session,
        )))
    if args.bridge_action == "run-once":
        return _emit(request_dict(run_once(
            store, args.worker_id, timeout_seconds=args.timeout,
        )))
    raise ValueError("session-bridge action is required")


def build_session_bridge_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "session-bridge", help="Correlate authorized voice investigations with Codex or Claude sessions",
    )
    actions = parser.add_subparsers(dest="bridge_action", required=True)
    actions.add_parser("capabilities", help="Report installed backend and current UI connector support")

    submit = actions.add_parser("submit", help="Persist one authorized question or investigation")
    submit.add_argument("--idempotency-key", required=True)
    submit.add_argument("--requester", required=True)
    submit.add_argument("--source-session", required=True)
    submit.add_argument("--authorization-ref", required=True)
    submit.add_argument("--backend", required=True, choices=("codex", "claude"))
    submit.add_argument("--kind", required=True, choices=("question", "investigation"))
    submit.add_argument("--target-session")
    submit.add_argument("--prompt", required=True)

    for name in ("status", "cancel"):
        action = actions.add_parser(name)
        action.add_argument("correlation_id")
        action.add_argument("--requester", required=True)
        action.add_argument("--source-session", required=True)

    worker = actions.add_parser("run-once", help="Claim and execute at most one queued request")
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--timeout", type=int, default=180)
    for action in actions.choices.values():
        action.set_defaults(func=cmd_session_bridge)
