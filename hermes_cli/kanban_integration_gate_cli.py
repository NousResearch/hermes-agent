"""``hermes kanban integration-gate …`` — declare and inspect integration gates.

The configuration surface for the opt-in gate: it writes one declaration row
(and an audit event) and never touches card status, assignees or links, so
running it on a live board changes nothing until the gate card is completed.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_integration_gate_store as gates
from hermes_cli.kanban_integration_gate import describe_receipt
from hermes_cli.kanban_output import _err, _json_out


def _dispatch_integration_gate(args: argparse.Namespace) -> int:
    sub = getattr(args, "integration_gate_action", None) or "list"
    handler = _HANDLERS.get(sub)
    if handler is None:
        return _err(f"kanban integration-gate: unknown action {sub!r}", 2)
    try:
        return handler(args)
    except gates.IntegrationGateConfigError as exc:
        return _err(f"kanban integration-gate: {exc}")


def _cmd_configure(args: argparse.Namespace) -> int:
    with kbc.connect_closing() as conn:
        config = gates.configure_gate(
            conn, args.gate_task_id,
            implementation_task_id=args.implementation, qa_task_id=args.qa,
            repository_path=args.repo, integration_remote=args.remote,
            integration_branch=args.branch,
        )
    if _json_out(args, config.as_dict()):
        return 0
    print(f"Integration gate configured on {config.gate_task_id}.")
    for line in _config_lines(config):
        print(line)
    print(f"  The gate completes only once its PR is merged by a human into "
          f"{config.integration_remote}/{config.integration_branch} and that merge commit is in "
          f"the fetched branch. Inspect with `hermes kanban integration-gate show "
          f"{config.gate_task_id}`.")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with kbc.connect_closing() as conn:
        config = gates.get_gate(conn, args.gate_task_id)
        if config is None:
            return _err(f"kanban integration-gate show: {args.gate_task_id} has no integration "
                        f"gate (it is an ordinary card)")
        receipt = _latest_receipt(conn, args.gate_task_id)
        status = kb._task_status(conn, args.gate_task_id)
    if _json_out(args, {"config": config.as_dict(), "status": status,
                        "latest_verification": receipt}):
        return 0
    print(f"Integration gate {config.gate_task_id} ({status or 'unknown status'})")
    for line in _config_lines(config):
        print(line)
    for line in describe_receipt(receipt):
        print(line)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    with kbc.connect_closing() as conn:
        configs = gates.list_gates(conn)
        rows = [{**config.as_dict(), "status": kb._task_status(conn, config.gate_task_id),
                 "latest_verification": _latest_receipt(conn, config.gate_task_id)}
                for config in configs]
    if _json_out(args, rows):
        return 0
    if not rows:
        print("No integration gates are declared on this board.")
        return 0
    for row in rows:
        receipt = row["latest_verification"]
        verdict = "never verified" if not receipt else (
            "integrated" if receipt.get("ok") else f"blocked at {receipt.get('phase')}")
        print(f"{row['gate_task_id']}  {row['status'] or '-':8s}  "
              f"{row['integration_remote']}/{row['integration_branch']}  "
              f"impl={row['implementation_task_id']} qa={row['qa_task_id']}  [{verdict}]")
    return 0


def _cmd_rm(args: argparse.Namespace) -> int:
    with kbc.connect_closing() as conn:
        removed = gates.remove_gate(conn, args.gate_task_id)
    if not removed:
        return _err(f"kanban integration-gate rm: no gate declared on {args.gate_task_id}")
    print(f"Removed the integration gate on {args.gate_task_id}; it is an ordinary card again.")
    return 0


def _config_lines(config) -> list[str]:
    return [
        f"  implementation: {config.implementation_task_id}",
        f"  qa:             {config.qa_task_id}",
        f"  repository:     {config.repository_path}",
        f"  integration:    {config.integration_remote}/{config.integration_branch}",
        # Stated on every gate, not conditional: there is no opt-out to report.
        "  human merge:    required (always; a bot merger can never satisfy a gate)",
    ]


def _latest_receipt(conn, gate_task_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'integration_acceptance' "
        "ORDER BY id DESC LIMIT 1", (gate_task_id,),
    ).fetchone()
    if row is None or not row["payload"]:
        return None
    try:
        payload = json.loads(row["payload"])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


_HANDLERS = {
    "configure": _cmd_configure, "show": _cmd_show,
    "list": _cmd_list, "ls": _cmd_list,
    "rm": _cmd_rm, "remove": _cmd_rm,
}
