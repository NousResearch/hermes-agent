"""``hermes fleet`` — operator controls for the federated Kanban runner pool."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _config() -> dict:
    from hermes_cli.config import load_config

    return load_config() or {}


def _secret() -> str:
    try:
        from agent.secret_scope import get_secret

        return (get_secret("HERMES_FLEET_TOKEN", "") or "").strip()
    except Exception:
        return os.environ.get("HERMES_FLEET_TOKEN", "").strip()


def _fleet_config(config: dict) -> dict:
    kanban = config.get("kanban") if isinstance(config, dict) else None
    fleet = kanban.get("federated") if isinstance(kanban, dict) else None
    return fleet if isinstance(fleet, dict) else {}


def _client(config: dict):
    from hermes_cli.fleet_client import FleetClient

    url = str(_fleet_config(config).get("coordinator_url") or "").strip()
    token = _secret()
    if not url:
        raise RuntimeError("fleet coordinator is not configured; run `hermes fleet init --url <url>`")
    if not token:
        raise RuntimeError("HERMES_FLEET_TOKEN is not configured")
    return FleetClient(url, token=token)


def _emit(args, payload: dict, lines: list[str]) -> int:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("\n".join(lines))
    return 0


def _task_dict(task) -> dict:
    data = dict(task.__dict__)
    requirement = data.get("requirement")
    if requirement is not None and hasattr(requirement, "to_dict"):
        data["requirement"] = requirement.to_dict()
    return data


def cmd_fleet(args: argparse.Namespace) -> int:
    action = getattr(args, "fleet_action", None)
    if action == "init":
        from hermes_cli.config import load_config, save_config

        config = load_config() or {}
        kanban = config.setdefault("kanban", {})
        federated = kanban.setdefault("federated", {})
        federated["coordinator_url"] = args.url.rstrip("/")
        federated["enabled"] = bool(args.enable)
        save_config(config, merge_existing=True)
        return _emit(
            args,
            {"enabled": federated["enabled"], "coordinator_url": federated["coordinator_url"]},
            [
                f"Coordinator: {federated['coordinator_url']}",
                f"Federated Kanban: {'enabled' if federated['enabled'] else 'disabled'}",
                "Token: configured through HERMES_FLEET_TOKEN (value not displayed)",
            ],
        )

    config = _config()
    if action == "status":
        fleet = _fleet_config(config)
        payload = {
            "enabled": fleet.get("enabled") is True,
            "coordinator_url": str(fleet.get("coordinator_url") or ""),
            "token_configured": bool(_secret()),
        }
        try:
            client = _client(config)
            payload["health"] = client.health()
            payload["tasks"] = [_task_dict(task) for task in client.list_tasks()]
        except Exception as exc:
            payload["health"] = {"ok": False, "error": type(exc).__name__}
        return _emit(
            args,
            payload,
            [
                f"Coordinator: {payload['coordinator_url'] or '(not configured)'}",
                f"Federated Kanban: {'enabled' if payload['enabled'] else 'disabled'}",
                f"Token: {'configured' if payload['token_configured'] else 'missing'}",
                f"Health: {'ok' if payload.get('health', {}).get('ok') else 'unavailable'}",
            ],
        )

    if action == "task":
        client = _client(config)
        if args.task_action == "list":
            tasks = client.list_tasks()
            return _emit(
                args,
                {"tasks": [_task_dict(task) for task in tasks]},
                [f"{task.task_id}  {task.status:10s}  {task.title}" for task in tasks]
                or ["(no fleet tasks)"],
            )
        ok = (
            client.retry_task(args.task_id)
            if args.task_action == "retry"
            else client.cancel_task(args.task_id)
        )
        return _emit(args, {"task_id": args.task_id, "ok": ok}, [f"{args.task_id}: {'ok' if ok else 'not changed'}"])

    if action == "runner":
        from hermes_cli.fleet_protocol import RunnerCapability
        from hermes_cli.fleet_runner import FleetRunner

        if not Path(args.hermes_executable).exists():
            raise ValueError(f"Hermes executable does not exist: {args.hermes_executable}")
        if not Path(args.liveness_file).exists():
            raise ValueError(f"liveness marker does not exist: {args.liveness_file}")
        client = _client(config)
        capabilities = [
            RunnerCapability(
                node_id=args.node_id,
                profile=profile,
                models=args.model,
                tools=args.tool,
                projects=args.project,
                platform=sys.platform,
            )
            for profile in args.profile
        ]
        runner = FleetRunner(
            args.node_id,
            client,
            args.hermes_executable,
            args.profile,
            args.project,
            capabilities=capabilities,
            liveness_check=lambda: Path(args.liveness_file).is_file(),
        )
        if args.once:
            return int(not runner.run_once())
        try:
            runner.run_forever(interval=args.interval)
        except KeyboardInterrupt:
            runner.stop()
        return 0

    print("fleet: choose init, status, runner, or task", file=sys.stderr)
    return 2


def build_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "fleet",
        help="Federated Kanban runner-pool controls",
        description="Coordinate opt-in Kanban work across live Hermes installations without sharing credentials or Git checkouts.",
    )
    fleet_sub = parser.add_subparsers(dest="fleet_action")
    init = fleet_sub.add_parser("init", help="Configure the coordinator URL and opt in or out")
    init.add_argument("--url", required=True)
    init.add_argument("--enable", action="store_true", help="Enable federated Kanban admission")
    init.add_argument("--json", action="store_true")
    init = fleet_sub.add_parser("status", help="Show coordinator and task health without secrets")
    init.add_argument("--json", action="store_true")
    runner = fleet_sub.add_parser("runner", help="Run a stock-Hermes compatibility runner")
    runner.add_argument("--node-id", required=True)
    runner.add_argument("--hermes-executable", required=True)
    runner.add_argument("--liveness-file", required=True)
    runner.add_argument("--profile", action="append", required=True)
    runner.add_argument("--project", action="append", default=[])
    runner.add_argument("--model", action="append", default=[])
    runner.add_argument("--tool", action="append", default=["terminal", "git"])
    runner.add_argument("--interval", type=float, default=5.0)
    runner.add_argument("--once", action="store_true")
    task = fleet_sub.add_parser("task", help="Inspect or control coordinator tasks")
    task_sub = task.add_subparsers(dest="task_action")
    listing = task_sub.add_parser("list", aliases=["ls"])
    listing.add_argument("--json", action="store_true")
    for name in ("retry", "cancel"):
        command = task_sub.add_parser(name)
        command.add_argument("task_id")
        command.add_argument("--json", action="store_true")
    parser.set_defaults(func=cmd_fleet)
