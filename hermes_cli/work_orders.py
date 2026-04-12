"""Execution work orders CLI and slash-command surface for Hermes."""

from __future__ import annotations

import io
import json
import shlex
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from hermes_cli.colors import Colors, color
from tools.execution_work_orders_tool import execution_work_orders_tool



def _work_orders_api(**kwargs) -> dict[str, Any]:
    return json.loads(execution_work_orders_tool(**kwargs))



def _fmt_ts(value: Any) -> str:
    try:
        if value in (None, ""):
            return "-"
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)



def _short(text: Any, limit: int = 120) -> str:
    rendered = str(text or "").strip()
    if not rendered:
        return "-"
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."



def _print_list(limit: int = 10, status: str | None = None, work_order_id: str | None = None):
    result = _work_orders_api(action="query" if any([status, work_order_id]) else "list", limit=limit, status=status, work_order_id=work_order_id)
    if result.get("error"):
        print(color(f"Failed to query work orders: {result['error']}", Colors.RED))
        return 1

    work_orders = result.get("work_orders", [])
    print()
    print(color("┌─────────────────────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                       Execution Work Orders                            │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────────────────────┘", Colors.CYAN))
    print()
    print(f"  Due now: {result.get('due_count', 0)}  |  Stale running: {result.get('stale_running_count', 0)}")
    if result.get("status_counts"):
        counts = ", ".join(f"{k}={v}" for k, v in sorted((result.get("status_counts") or {}).items()))
        print(f"  Status counts: {counts}")
    print()

    if not work_orders:
        print(color("No execution work orders found.", Colors.DIM))
        return 0

    for row in work_orders:
        status_text = row.get("status") or "unknown"
        if status_text == "completed":
            status_display = color(f"[{status_text}]", Colors.GREEN)
        elif status_text in {"failed", "cancelled"}:
            status_display = color(f"[{status_text}]", Colors.RED)
        elif status_text == "running":
            status_display = color(f"[{status_text}]", Colors.YELLOW)
        else:
            status_display = color(f"[{status_text}]", Colors.CYAN)
        print(f"  {color(row.get('work_order_id', '?'), Colors.YELLOW)} {status_display}")
        print(f"    Goal:       {row.get('goal') or '-'}")
        print(f"    Scheduled:  {_fmt_ts(row.get('scheduled_for'))} ({row.get('schedule_display') or '-'})")
        print(f"    Attempts:   {row.get('attempt_count', 0)}/{row.get('max_attempts', 1)}")
        if row.get("claim_expires_at"):
            print(f"    Claim:      until {_fmt_ts(row.get('claim_expires_at'))} by {row.get('claim_owner') or '-'}")
        print(f"    Command:    {_short(row.get('command'))}")
        if row.get("workdir"):
            print(f"    Workdir:    {row.get('workdir')}")
        if row.get("last_receipt_id"):
            print(f"    Receipt:    {row.get('last_receipt_id')} ({row.get('last_receipt_path') or '-'})")
        if row.get("last_exit_reason"):
            print(f"    Last exit:  {row.get('last_exit_reason')}")
        if row.get("last_error"):
            print(f"    Last error: {_short(row.get('last_error'))}")
        if row.get("worker_task_id"):
            worker_line = f"{row.get('worker_mode') or '-'} via {row.get('worker_task_id')}"
            if row.get("worker_runtime_reused"):
                worker_line += " [runtime-reused]"
            print(f"    Worker:     {worker_line}")
            if row.get("worker_runtime_id"):
                print(f"    Runtime:    {row.get('worker_runtime_kind') or '-'} {row.get('worker_runtime_id')}")
        print()
    return 0



def _print_enqueue(goal: str, command: str, context: str | None, timeout_seconds: int | None, workdir: str | None,
                   schedule: str | None, delay_seconds: float | None, max_attempts: int | None, retry_delay_seconds: float | None):
    result = _work_orders_api(
        action="enqueue",
        goal=goal,
        command=command,
        context=context,
        timeout_seconds=timeout_seconds,
        workdir=workdir,
        schedule=schedule,
        delay_seconds=delay_seconds,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    if result.get("error"):
        print(color(f"Failed to enqueue work order: {result['error']}", Colors.RED))
        return 1
    row = result.get("work_order") or {}
    print(color("Enqueued execution work order.", Colors.GREEN))
    print(f"  Work order ID: {row.get('work_order_id')}")
    print(f"  Scheduled for: {_fmt_ts(row.get('scheduled_for'))}")
    print(f"  Schedule: {row.get('schedule_display') or '-'}")
    print(f"  Attempts: {row.get('attempt_count', 0)}/{row.get('max_attempts', 1)}")
    return 0



def _print_run(limit: int | None, claim_ttl_seconds: float | None):
    result = _work_orders_api(
        action="run_due",
        limit=limit,
        claim_ttl_seconds=claim_ttl_seconds,
    )
    if result.get("error"):
        print(color(f"Failed to run due work orders: {result['error']}", Colors.RED))
        return 1
    print(color("Ran due execution work orders.", Colors.GREEN))
    print(f"  Executed: {result.get('executed_count', 0)}")
    statuses = result.get("statuses") or {}
    if statuses:
        print(f"  Statuses: {', '.join(f'{k}={v}' for k, v in sorted(statuses.items()))}")
    for row in result.get("work_orders") or []:
        print(f"    - {row.get('work_order_id')} -> {row.get('status')} ({row.get('last_receipt_id') or '-'})")
    return 0



def _print_reclaim(limit: int | None):
    result = _work_orders_api(action="reclaim_stale", limit=limit)
    if result.get("error"):
        print(color(f"Failed to reclaim stale work orders: {result['error']}", Colors.RED))
        return 1
    print(color("Reclaimed stale execution work orders.", Colors.GREEN))
    print(f"  Reclaimed: {result.get('reclaimed_count', 0)}")
    for row in result.get("work_orders") or []:
        print(f"    - {row.get('work_order_id')}")
    return 0



def _print_requeue(action: str, work_order_id: str, delay_seconds: float | None):
    result = _work_orders_api(action=action, work_order_id=work_order_id, delay_seconds=delay_seconds)
    if result.get("error"):
        print(color(f"Failed to {action} work order: {result['error']}", Colors.RED))
        return 1
    row = result.get("work_order") or {}
    print(color(f"{action.capitalize()}d execution work order.", Colors.GREEN))
    print(f"  Work order ID: {row.get('work_order_id')}")
    print(f"  Status: {row.get('status')}")
    print(f"  Scheduled for: {_fmt_ts(row.get('scheduled_for'))}")
    return 0



def _print_cancel(work_order_id: str):
    result = _work_orders_api(action="cancel", work_order_id=work_order_id)
    if result.get("error"):
        print(color(f"Failed to cancel work order: {result['error']}", Colors.RED))
        return 1
    row = result.get("work_order") or {}
    print(color("Cancelled execution work order.", Colors.GREEN))
    print(f"  Work order ID: {row.get('work_order_id')}")
    print(f"  Status: {row.get('status')}")
    return 0



def _print_runner_status():
    result = _work_orders_api(action="runner_status")
    if result.get("error"):
        print(color(f"Failed to inspect runner status: {result['error']}", Colors.RED))
        return 1
    print()
    print(color("Execution work-order runner", Colors.CYAN))
    print(color("-" * 31, Colors.CYAN))
    print(f"  Due now: {result.get('due_count', 0)}")
    print(f"  Stale running: {result.get('stale_running_count', 0)}")
    counts = result.get("status_counts") or {}
    if counts:
        print(f"  Status counts: {', '.join(f'{k}={v}' for k, v in sorted(counts.items()))}")
    print(f"  Installed jobs: {result.get('installed_count', 0)}")
    jobs = result.get("jobs") or []
    if not jobs:
        print(color("  No runner job installed.", Colors.DIM))
        return 0
    for job in jobs:
        print(f"  Job ID:      {job.get('job_id')}")
        print(f"  State:       {job.get('state')}")
        print(f"  Schedule:    {job.get('schedule')}")
        print(f"  Next run:    {job.get('next_run_at')}")
        print(f"  Last run:    {job.get('last_run_at')}")
        print(f"  Last status: {job.get('last_status')}")
        config = job.get("config") or {}
        if config:
            print(f"  Run limit:         {config.get('run_limit')}")
            print(f"  Reclaim limit:     {config.get('reclaim_limit')}")
            print(f"  Claim TTL:         {config.get('claim_ttl_seconds')}s")
        print()
    return 0



def _print_install(schedule: str | None, limit: int | None, reclaim_limit: int | None,
                   claim_ttl_seconds: float | None, model: str | None, provider: str | None, base_url: str | None):
    result = _work_orders_api(
        action="install_runner",
        schedule=schedule,
        limit=limit,
        reclaim_limit=reclaim_limit,
        claim_ttl_seconds=claim_ttl_seconds,
        model=model,
        provider=provider,
        base_url=base_url,
    )
    if result.get("error"):
        print(color(f"Failed to install runner: {result['error']}", Colors.RED))
        return 1
    verb = "Created" if result.get("created") else "Updated"
    job = result.get("job") or {}
    print(color(f"{verb} execution work-order runner job.", Colors.GREEN))
    print(f"  Job ID: {job.get('job_id')}")
    print(f"  Schedule: {job.get('schedule')}")
    print(f"  State: {job.get('state')}")
    config = job.get("config") or {}
    if config:
        print(f"  Run limit: {config.get('run_limit')}")
        print(f"  Reclaim limit: {config.get('reclaim_limit')}")
        print(f"  Claim TTL: {config.get('claim_ttl_seconds')}s")
    return 0



def _print_remove():
    result = _work_orders_api(action="remove_runner")
    if result.get("error"):
        print(color(f"Failed to remove runner: {result['error']}", Colors.RED))
        return 1
    deleted = result.get("deleted_count", 0)
    if deleted == 0:
        print(color("No execution work-order runner job was installed.", Colors.DIM))
        return 0
    print(color("Removed execution work-order runner job(s).", Colors.GREEN))
    for job_id in result.get("deleted_job_ids") or []:
        print(f"  - {job_id}")
    return 0



def work_orders_command(args):
    subcmd = getattr(args, "work_orders_command", None)
    if subcmd is None or subcmd == "list":
        return _print_list(limit=getattr(args, "limit", 10), status=getattr(args, "status", None), work_order_id=getattr(args, "work_order_id", None))
    if subcmd == "enqueue":
        return _print_enqueue(
            goal=getattr(args, "goal"),
            command=getattr(args, "command"),
            context=getattr(args, "context", None),
            timeout_seconds=getattr(args, "timeout_seconds", None),
            workdir=getattr(args, "workdir", None),
            schedule=getattr(args, "schedule", None),
            delay_seconds=getattr(args, "delay_seconds", None),
            max_attempts=getattr(args, "max_attempts", None),
            retry_delay_seconds=getattr(args, "retry_delay_seconds", None),
        )
    if subcmd == "run":
        return _print_run(limit=getattr(args, "limit", None), claim_ttl_seconds=getattr(args, "claim_ttl_seconds", None))
    if subcmd == "reclaim":
        return _print_reclaim(limit=getattr(args, "limit", None))
    if subcmd == "retry":
        return _print_requeue("retry", getattr(args, "work_order_id"), getattr(args, "delay_seconds", None))
    if subcmd == "resume":
        return _print_requeue("resume", getattr(args, "work_order_id"), getattr(args, "delay_seconds", None))
    if subcmd == "cancel":
        return _print_cancel(getattr(args, "work_order_id"))
    if subcmd == "status":
        return _print_runner_status()
    if subcmd == "install":
        return _print_install(
            schedule=getattr(args, "schedule", None),
            limit=getattr(args, "limit", None),
            reclaim_limit=getattr(args, "reclaim_limit", None),
            claim_ttl_seconds=getattr(args, "claim_ttl_seconds", None),
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            base_url=getattr(args, "base_url", None),
        )
    if subcmd in {"remove", "rm", "delete"}:
        return _print_remove()
    print(f"Unknown workorders command: {subcmd}")
    print("Usage: hermes workorders [list|enqueue|run|reclaim|retry|resume|cancel|status|install|remove]")
    return 1



def _parse_slash_flags(tokens: list[str]) -> dict[str, Any] | None:
    opts: dict[str, Any] = {
        "limit": None,
        "status": None,
        "work_order_id": None,
        "goal": None,
        "command": None,
        "context": None,
        "timeout_seconds": None,
        "workdir": None,
        "schedule": None,
        "delay_seconds": None,
        "max_attempts": None,
        "retry_delay_seconds": None,
        "claim_ttl_seconds": None,
        "reclaim_limit": None,
        "model": None,
        "provider": None,
        "base_url": None,
        "positionals": [],
    }
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            opts["positionals"].append(token)
            i += 1
            continue
        if token == "--limit" and i + 1 < len(tokens):
            opts["limit"] = int(tokens[i + 1])
            i += 2
            continue
        if token == "--status" and i + 1 < len(tokens):
            opts["status"] = tokens[i + 1]
            i += 2
            continue
        if token == "--goal" and i + 1 < len(tokens):
            opts["goal"] = tokens[i + 1]
            i += 2
            continue
        if token == "--command" and i + 1 < len(tokens):
            opts["command"] = tokens[i + 1]
            i += 2
            continue
        if token == "--context" and i + 1 < len(tokens):
            opts["context"] = tokens[i + 1]
            i += 2
            continue
        if token == "--timeout-seconds" and i + 1 < len(tokens):
            opts["timeout_seconds"] = int(tokens[i + 1])
            i += 2
            continue
        if token == "--workdir" and i + 1 < len(tokens):
            opts["workdir"] = tokens[i + 1]
            i += 2
            continue
        if token == "--schedule" and i + 1 < len(tokens):
            opts["schedule"] = tokens[i + 1]
            i += 2
            continue
        if token == "--delay-seconds" and i + 1 < len(tokens):
            opts["delay_seconds"] = float(tokens[i + 1])
            i += 2
            continue
        if token == "--max-attempts" and i + 1 < len(tokens):
            opts["max_attempts"] = int(tokens[i + 1])
            i += 2
            continue
        if token == "--retry-delay-seconds" and i + 1 < len(tokens):
            opts["retry_delay_seconds"] = float(tokens[i + 1])
            i += 2
            continue
        if token == "--claim-ttl-seconds" and i + 1 < len(tokens):
            opts["claim_ttl_seconds"] = float(tokens[i + 1])
            i += 2
            continue
        if token == "--reclaim-limit" and i + 1 < len(tokens):
            opts["reclaim_limit"] = int(tokens[i + 1])
            i += 2
            continue
        if token == "--model" and i + 1 < len(tokens):
            opts["model"] = tokens[i + 1]
            i += 2
            continue
        if token == "--provider" and i + 1 < len(tokens):
            opts["provider"] = tokens[i + 1]
            i += 2
            continue
        if token == "--base-url" and i + 1 < len(tokens):
            opts["base_url"] = tokens[i + 1]
            i += 2
            continue
        print(f"(._.) Unknown flag: {token}")
        return None
    return opts



def handle_work_orders_slash(cmd: str):
    try:
        tokens = shlex.split(cmd)
    except ValueError as exc:
        print(f"(._.) Could not parse /workorders command: {exc}")
        return
    if not tokens:
        print("(._.) Usage: /workorders [list|enqueue|run|reclaim|retry|resume|cancel|status|install|remove]")
        return
    if len(tokens) == 1:
        print()
        print("+" + "-" * 68 + "+")
        print("|" + " " * 18 + "(✦_✦) Execution Work Orders" + " " * 19 + "|")
        print("+" + "-" * 68 + "+")
        print()
        print("  Commands:")
        print("    /workorders list [--limit 10] [--status queued|running|completed|failed]")
        print("    /workorders enqueue --goal <goal> --command <command> [--schedule '30m'|--delay-seconds 30]")
        print("    /workorders run [--limit 10] [--claim-ttl-seconds 900]")
        print("    /workorders reclaim [--limit 50]")
        print("    /workorders retry <work_order_id> [--delay-seconds 0]")
        print("    /workorders resume <work_order_id> [--delay-seconds 0]")
        print("    /workorders cancel <work_order_id>")
        print("    /workorders status")
        print("    /workorders install [--schedule 'every 5m'] [--limit 10] [--reclaim-limit 50] [--claim-ttl-seconds 900]")
        print("    /workorders remove")
        print()
        _print_list(limit=10)
        print()
        _print_runner_status()
        return
    subcommand = tokens[1].lower()
    opts = _parse_slash_flags(tokens[2:])
    if opts is None:
        return

    if subcommand == "list":
        _print_list(limit=opts["limit"] or 10, status=opts["status"], work_order_id=opts["positionals"][0] if opts["positionals"] else None)
        return
    if subcommand == "enqueue":
        goal = opts["goal"] or (opts["positionals"][0] if opts["positionals"] else None)
        if not goal or not opts["command"]:
            print("(._.) Usage: /workorders enqueue --goal <goal> --command <command> [--schedule '30m'|--delay-seconds 30]")
            return
        _print_enqueue(goal, opts["command"], opts["context"], opts["timeout_seconds"], opts["workdir"], opts["schedule"], opts["delay_seconds"], opts["max_attempts"], opts["retry_delay_seconds"])
        return
    if subcommand == "run":
        _print_run(limit=opts["limit"], claim_ttl_seconds=opts["claim_ttl_seconds"])
        return
    if subcommand == "reclaim":
        _print_reclaim(limit=opts["limit"])
        return
    if subcommand in {"retry", "resume", "cancel"}:
        if not opts["positionals"]:
            print(f"(._.) Usage: /workorders {subcommand} <work_order_id>")
            return
        work_order_id = opts["positionals"][0]
        if subcommand == "cancel":
            _print_cancel(work_order_id)
        else:
            _print_requeue(subcommand, work_order_id, opts["delay_seconds"])
        return
    if subcommand == "status":
        _print_runner_status()
        return
    if subcommand == "install":
        _print_install(opts["schedule"], opts["limit"], opts["reclaim_limit"], opts["claim_ttl_seconds"], opts["model"], opts["provider"], opts["base_url"])
        return
    if subcommand in {"remove", "rm", "delete"}:
        _print_remove()
        return

    print("(._.) Unknown /workorders subcommand")
    print("  Available: list, enqueue, run, reclaim, retry, resume, cancel, status, install, remove")



def capture_work_orders_slash_output(cmd: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        handle_work_orders_slash(cmd)
    return buf.getvalue()
