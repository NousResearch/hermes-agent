"""Operator-facing commands for governed builder dispatches."""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CONFIG = "~/.hermes/builder-adapter/runtime.json"


def build_orchestrate_parser(subparsers, *, cmd_orchestrate):
    parser = subparsers.add_parser(
        "orchestrate", help="Run and monitor governed implementation jobs"
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("HERMES_BUILDER_ADAPTER_CONFIG", DEFAULT_CONFIG),
        help="Owner-only builder adapter runtime configuration",
    )
    parser.add_argument("--key-id", help="Authentication key when more than one is active")
    actions = parser.add_subparsers(dest="orchestrate_action", required=True)

    actions.add_parser("health", help="Check whether the local adapter is reachable")
    actions.add_parser("cycles", help="List jobs registered by the owner")

    prepare = actions.add_parser(
        "prepare", help="Create a reviewable job proposal without activating it"
    )
    prepare.add_argument("--repo", required=True, help="Clean source repository root")
    prepare.add_argument("--repository-id", required=True)
    prepare.add_argument("--cycle", required=True, dest="cycle_id")
    prepare.add_argument("--contract", required=True, dest="contract_id")
    prepare.add_argument("--goal", required=True)
    prepare.add_argument("--accept", action="append", required=True, dest="acceptance")
    prepare.add_argument("--allow", action="append", required=True, dest="allowed_paths")
    prepare.add_argument("--branch", required=True, dest="planned_branch")
    prepare.add_argument("--worktree", required=True, dest="planned_worktree")
    prepare.add_argument("--output", help="Owner-only proposal JSON path")
    prepare.add_argument("--max-runtime", type=int, default=1800)
    prepare.add_argument("--heartbeat-timeout", type=int, default=180)

    activate = actions.add_parser(
        "activate", help="Bind a reviewed proposal and create its isolated worktree"
    )
    activate.add_argument("proposal", help="Owner-only proposal JSON from prepare")

    restart = actions.add_parser(
        "restart", help="Gracefully reload the supervised adapter configuration"
    )
    restart.add_argument(
        "--label", default="ai.hermes.builder-adapter", help="macOS launch-agent label"
    )
    restart.add_argument(
        "--timeout", type=float, default=30.0, help="Seconds to wait for verified readiness"
    )

    start = actions.add_parser("start", help="Start one registered implementation job")
    start.add_argument("cycle_id")
    start.add_argument("--dispatch-id", help="Reuse a previously chosen UUID for idempotent recovery")

    for name, help_text in (
        ("status", "Show the current job state"),
        ("evidence", "Show completion evidence for a finished job"),
    ):
        action = actions.add_parser(name, help=help_text)
        action.add_argument("dispatch_id")
        action.add_argument("--cycle", required=True, dest="cycle_id")

    cancel = actions.add_parser("cancel", help="Cancel a running job and terminate its worker")
    cancel.add_argument("dispatch_id")
    cancel.add_argument("--cycle", required=True, dest="cycle_id")
    cancel.add_argument(
        "--reason",
        default="HUMAN_CANCELLED",
        choices=["HUMAN_CANCELLED", "CONTRACT_SUPERSEDED", "TIMEOUT", "GOVERNANCE_REJECTED"],
    )
    parser.set_defaults(func=cmd_orchestrate)
    return parser


def _client(args, *, authenticated: bool = True):
    from plugins.builder_adapter.client import BuilderAdapterClient, load_operator_key
    from plugins.builder_adapter.runtime import RuntimeSettings

    settings = RuntimeSettings.from_file(Path(args.config).expanduser())
    key = load_operator_key(settings, args.key_id) if authenticated else None
    return settings, BuilderAdapterClient(settings.socket_path, key)


def _print_result(result: dict, *, raw: bool = False) -> None:
    if raw:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"Status: {result.get('status', 'UNKNOWN')}")
    if result.get("dispatch_id"):
        print(f"Dispatch: {result['dispatch_id']}")
    if result.get("cycle_id"):
        print(f"Cycle: {result['cycle_id']}")
    if result.get("kanban_task_id"):
        print(f"Builder task: {result['kanban_task_id']}")
    if result.get("attempt_count") is not None:
        print(f"Attempts: {result['attempt_count']}")
    for error in result.get("errors", []):
        print(f"Error: {error.get('code')}: {error.get('message')}")


def _restart_launch_agent(args, settings, client) -> dict:
    from plugins.builder_adapter.errors import AdapterError

    if sys.platform != "darwin":
        raise AdapterError(
            "PROVIDER_UNAVAILABLE", "adapter restart is supported only on macOS"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.label):
        raise AdapterError("INVALID_REQUEST", "invalid launch-agent label")
    if args.timeout <= 0 or args.timeout > 120:
        raise AdapterError("INVALID_REQUEST", "restart timeout must be 1-120 seconds")

    target = f"gui/{os.getuid()}/{args.label}"
    previous_process_id = None
    try:
        previous_process_id = client.health().get("process_id")
    except AdapterError:
        pass
    try:
        inspected = subprocess.run(
            ["/bin/launchctl", "print", target],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if inspected.returncode != 0:
            raise AdapterError(
                "PROVIDER_UNAVAILABLE", "builder adapter launch agent is not loaded"
            )
        signalled = subprocess.run(
            ["/bin/launchctl", "kill", "SIGTERM", target],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AdapterError(
            "PROVIDER_UNAVAILABLE", "could not ask launchd to restart the adapter"
        ) from exc
    if signalled.returncode != 0:
        raise AdapterError(
            "PROVIDER_UNAVAILABLE", "launchd rejected the adapter restart request"
        )

    expected_registry = hashlib.sha256(
        json.dumps(
            settings.cycle_registry,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    deadline = time.monotonic() + args.timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            health = client.health()
            if (
                health.get("operational") is True
                and health.get("cycle_registry_sha256") == expected_registry
                and isinstance(health.get("process_id"), int)
                and health.get("process_id") != previous_process_id
            ):
                return health
        except AdapterError as error:
            last_error = error
        time.sleep(0.2)
    raise AdapterError(
        "PROVIDER_UNAVAILABLE",
        "adapter did not reload the current registered-cycle configuration",
        retryable=True,
    ) from last_error


def run_operator_command(args) -> int:
    from plugins.builder_adapter.errors import AdapterError

    try:
        if args.orchestrate_action == "health":
            _, client = _client(args, authenticated=False)
            result = client.health()
            state = "reachable" if "capability_id" in result else "invalid response"
            print(f"Adapter: {state}")
            print(f"Capability: {result.get('capability_id', 'unknown')}")
            return 0

        if args.orchestrate_action == "cycles":
            settings, _ = _client(args, authenticated=False)
            if not settings.cycle_registry:
                print("No registered jobs.")
                return 0
            for cycle_id, cycle in sorted(settings.cycle_registry.items()):
                print(
                    f"{cycle_id}  repo={cycle.get('repository_id')}  "
                    f"revision={cycle.get('revision')}  branch={cycle.get('branch')}"
                )
            return 0

        if args.orchestrate_action == "prepare":
            from plugins.builder_adapter.preparation import (
                inspect_repository,
                prepare_bundle,
                write_bundle,
            )

            settings, _ = _client(args, authenticated=False)
            repository = inspect_repository(args.repo)
            bundle = prepare_bundle(
                cycle_id=args.cycle_id,
                contract_id=args.contract_id,
                repository_id=args.repository_id,
                repository=repository,
                goal=args.goal,
                acceptance_criteria=args.acceptance,
                allowed_paths=args.allowed_paths,
                planned_branch=args.planned_branch,
                planned_worktree=args.planned_worktree,
                validation_profile_id=settings.validation_profile_id,
                max_runtime_seconds=args.max_runtime,
                heartbeat_timeout_seconds=args.heartbeat_timeout,
                registered_remote=settings.repository_allowlist.get(args.repository_id),
            )
            output = args.output or (
                f"~/.hermes/builder-jobs/pending/{args.cycle_id}.json"
            )
            destination = write_bundle(output, bundle)
            print(f"Prepared: {destination}")
            print(f"State: {bundle['activation_state']}")
            print(f"Bundle SHA-256: {bundle['bundle_sha256']}")
            print("No worktree was created and no builder was started.")
            return 0

        if args.orchestrate_action == "activate":
            from plugins.builder_adapter.activation import activate_proposal

            result = activate_proposal(Path(args.config).expanduser(), args.proposal)
            print(f"Activated cycle: {result['cycle_id']}")
            print(f"Governance commit: {result['governance_commit']}")
            print(f"Worktree: {result['worktree_path']}")
            print("Adapter restart required before start.")
            return 0

        if args.orchestrate_action == "restart":
            settings, client = _client(args, authenticated=False)
            result = _restart_launch_agent(args, settings, client)
            print("Adapter: restarted and configuration verified")
            print(f"Process: {result['process_id']}")
            print(f"Registered jobs: {result['registered_cycles']}")
            return 0

        settings, client = _client(args)
        if args.orchestrate_action == "start":
            cycle = settings.cycle_registry.get(args.cycle_id)
            if not isinstance(cycle, dict):
                raise AdapterError("CONTRACT_MISMATCH", "cycle is not registered")
            result = client.start(args.cycle_id, cycle, dispatch_id=args.dispatch_id)
            _print_result(result)
            print(
                "Next: hermes orchestrate status "
                f"{result['dispatch_id']} --cycle {args.cycle_id}"
            )
            return 0

        if args.orchestrate_action in {"status", "evidence"}:
            result = client.status(args.dispatch_id, args.cycle_id)
            if args.orchestrate_action == "evidence":
                evidence = result.get("completion_evidence")
                if evidence is None:
                    print(f"No completion evidence yet (status: {result.get('status', 'UNKNOWN')}).")
                    return 2
                print(json.dumps(evidence, indent=2, sort_keys=True))
            else:
                _print_result(result)
            return 0

        result = client.cancel(args.dispatch_id, args.cycle_id, args.reason)
        _print_result(result)
        return 0
    except AdapterError as error:
        print(f"Orchestrator error [{error.code}]: {error.safe_message}")
        return 2


def cmd_orchestrate(args):
    raise SystemExit(run_operator_command(args))
