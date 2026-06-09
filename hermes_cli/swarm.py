"""Profile-worker swarm CLI.

This module provides a small, privacy-conscious harness for launching several
Hermes profiles against handoff files in parallel.  It intentionally does not
embed any profile names, model names, API keys, Slack IDs, or user-specific
paths.  The controller supplies explicit ``--worker`` specs and handoff files;
this command only manages process lifecycle and durable status.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


READ_ONLY_CONSTRAINT = (
    "GLOBAL CONSTRAINT: Read-only. Do not edit files, send messages, modify "
    "calendars, or perform external side effects."
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_worker_spec(spec: str) -> dict[str, str]:
    """Parse ``profile[:handoff]`` or ``profile=handoff`` worker specs.

    ``handoff`` defaults to ``<profile>.md`` under the run directory's
    ``handoffs/`` folder.  The parser deliberately keeps the shape small so
    user-specific role/model policy stays in profiles/config, not in the
    public harness.
    """
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("empty worker spec")
    if "=" in raw:
        profile, handoff = raw.split("=", 1)
    elif ":" in raw:
        profile, handoff = raw.split(":", 1)
    else:
        profile, handoff = raw, f"{raw}.md"
    profile = profile.strip()
    handoff = handoff.strip()
    if not profile:
        raise ValueError(f"invalid worker spec {spec!r}: missing profile")
    if not handoff:
        handoff = f"{profile}.md"
    return {"profile": profile, "handoff": handoff}


def _resolve_handoff(run_dir: Path, handoff: str) -> Path:
    path = Path(handoff).expanduser()
    if not path.is_absolute():
        path = run_dir / "handoffs" / path
    return path.resolve()


def _status_path(run_dir: Path, status_file: str | None) -> Path:
    if status_file:
        path = Path(status_file).expanduser()
        if not path.is_absolute():
            path = run_dir / path
        return path.resolve()
    return run_dir / "status.json"


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _base_cmd(args: argparse.Namespace, profile: str, prompt: str, source: str) -> list[str]:
    hermes_bin = getattr(args, "hermes_bin", None) or os.environ.get("HERMES_SWARM_HERMES_BIN")
    cmd = [hermes_bin or "hermes", "-p", profile, "chat", "-Q"]
    if getattr(args, "accept_hooks", False):
        cmd.append("--accept-hooks")
    if getattr(args, "yolo", False):
        cmd.append("--yolo")
    toolsets = getattr(args, "toolsets", None)
    if toolsets:
        cmd.extend(["-t", toolsets])
    skills = getattr(args, "skills", None)
    if skills:
        cmd.extend(["-s", skills])
    max_turns = getattr(args, "max_turns", None)
    if max_turns:
        cmd.extend(["--max-turns", str(max_turns)])
    cmd.extend(["--source", source, "-q", prompt])
    return cmd


def _cmd_preview(cmd: list[str]) -> list[str]:
    """Return a command preview with the prompt body removed."""
    if "-q" not in cmd:
        return cmd[:]
    idx = cmd.index("-q")
    return cmd[: idx + 1] + ["<prompt redacted>"]


def run_swarm(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    agents_dir = run_dir / "agents"
    logs_dir = run_dir / "logs"
    handoffs_dir = run_dir / "handoffs"
    for directory in (agents_dir, logs_dir, handoffs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    workers = [parse_worker_spec(spec) for spec in (args.worker or [])]
    if not workers:
        raise SystemExit("hermes swarm run requires at least one --worker profile[:handoff]")

    status_file = _status_path(run_dir, args.status_file)
    started_at = now_iso()
    status: dict[str, Any] = {
        "run_dir": str(run_dir),
        "status_file": str(status_file),
        "started_at": started_at,
        "overall_status": "running",
        "workers": {},
    }

    procs: list[dict[str, Any]] = []
    source_prefix = args.source_prefix or f"swarm:{run_dir.name}"
    timeout_seconds = float(args.timeout_seconds) if args.timeout_seconds else None
    kill_grace = float(args.kill_grace_seconds)

    for worker in workers:
        profile = worker["profile"]
        handoff_path = _resolve_handoff(run_dir, worker["handoff"])
        out_path = agents_dir / f"{profile}.out.md"
        err_path = logs_dir / f"{profile}.err.log"
        record: dict[str, Any] = {
            "profile": profile,
            "status": "pending",
            "handoff": str(handoff_path),
            "out": str(out_path),
            "err": str(err_path),
        }
        status["workers"][profile] = record
        if not handoff_path.exists():
            record.update({"status": "skipped", "reason": "missing_handoff"})
            continue
        prompt = handoff_path.read_text(encoding="utf-8", errors="replace")
        if args.read_only:
            prompt = prompt.rstrip() + "\n\n" + READ_ONLY_CONSTRAINT + "\n"
        source = f"{source_prefix}:{profile}"
        cmd = _base_cmd(args, profile, prompt, source)
        record.update(
            {
                "status": "planned" if args.dry_run else "running",
                "started_at": now_iso() if not args.dry_run else None,
                "cmd_preview": _cmd_preview(cmd),
            }
        )
        if args.dry_run:
            continue
        out_f = out_path.open("w", encoding="utf-8")
        err_f = err_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True)
        except FileNotFoundError as exc:
            out_f.close()
            err_f.close()
            record.update({"status": "failed", "reason": f"command_not_found: {exc}"})
            continue
        record["pid"] = proc.pid
        procs.append(
            {
                "profile": profile,
                "proc": proc,
                "out_path": out_path,
                "err_path": err_path,
                "out_f": out_f,
                "err_f": err_f,
                "started_monotonic": time.monotonic(),
                "terminated_at": None,
            }
        )

    if args.dry_run:
        status["overall_status"] = "planned"
        _write_status(status_file, status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    _write_status(status_file, status)

    remaining = {item["profile"]: item for item in procs}
    while remaining:
        changed = False
        for profile, item in list(remaining.items()):
            proc: subprocess.Popen[str] = item["proc"]
            record = status["workers"][profile]
            rc = proc.poll()
            elapsed = time.monotonic() - float(item["started_monotonic"])
            if rc is None and timeout_seconds and elapsed > timeout_seconds:
                if item.get("terminated_at") is None:
                    record.update(
                        {
                            "status": "terminating",
                            "reason": f"timeout_after_{int(timeout_seconds)}s",
                            "duration_seconds": round(elapsed, 1),
                        }
                    )
                    proc.terminate()
                    item["terminated_at"] = time.monotonic()
                    changed = True
                elif time.monotonic() - float(item["terminated_at"]) > kill_grace:
                    record.update(
                        {
                            "status": "killed",
                            "reason": f"timeout_after_{int(timeout_seconds)}s",
                            "duration_seconds": round(elapsed, 1),
                        }
                    )
                    proc.kill()
                    rc = proc.wait(timeout=5)
            if rc is not None:
                for handle_name in ("out_f", "err_f"):
                    try:
                        item[handle_name].close()
                    except Exception:
                        pass
                out_path = item["out_path"]
                err_path = item["err_path"]
                final_status = "completed" if rc == 0 else "failed"
                if record.get("status") == "killed":
                    final_status = "failed"
                record.update(
                    {
                        "status": final_status,
                        "exit_code": rc,
                        "finished_at": now_iso(),
                        "duration_seconds": round(elapsed, 1),
                        "out_bytes": out_path.stat().st_size if out_path.exists() else 0,
                        "err_bytes": err_path.stat().st_size if err_path.exists() else 0,
                    }
                )
                remaining.pop(profile, None)
                changed = True
        if changed:
            _write_status(status_file, status)
        if remaining:
            time.sleep(float(args.poll_interval))

    completed_records = [
        record
        for record in status["workers"].values()
        if record.get("status") in {"completed", "failed", "skipped"}
    ]
    failures = [r for r in completed_records if r.get("status") != "completed"]
    status["finished_at"] = now_iso()
    if failures:
        status["overall_status"] = "completed_with_failures"
    else:
        status["overall_status"] = "completed"
    _write_status(status_file, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if not failures or args.allow_failures else 1


def status_swarm(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    status_file = _status_path(run_dir, args.status_file)
    if not status_file.exists():
        raise SystemExit(f"status file not found: {status_file}")
    data = json.loads(status_file.read_text(encoding="utf-8"))
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def build_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "swarm",
        help="Run multiple profile-backed Hermes workers from handoff files",
        description=(
            "Launch several isolated Hermes profiles in parallel using "
            "handoff files under <run-dir>/handoffs/. Writes per-worker stdout, "
            "stderr, and a durable status.json without storing prompt bodies in "
            "the status file."
        ),
    )
    swarm_sub = parser.add_subparsers(dest="swarm_action")

    run = swarm_sub.add_parser("run", help="Launch profile workers for a run directory")
    run.add_argument("run_dir", help="Run directory containing handoffs/")
    run.add_argument(
        "--worker",
        action="append",
        help=(
            "Worker spec profile[:handoff.md] or profile=handoff.md. "
            "May be repeated. Relative handoffs resolve under <run-dir>/handoffs/."
        ),
    )
    run.add_argument("--toolsets", help="Comma-separated toolsets for every worker")
    run.add_argument("--skills", help="Comma-separated skills to preload for every worker")
    run.add_argument("--max-turns", type=int, help="Per-worker --max-turns value")
    run.add_argument("--timeout-seconds", type=float, help="Terminate a worker after N seconds")
    run.add_argument(
        "--kill-grace-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait after terminate() before kill() on timeout (default: 10)",
    )
    run.add_argument("--poll-interval", type=float, default=1.0, help="Status poll interval")
    run.add_argument("--status-file", help="Status JSON path (default: <run-dir>/status.json)")
    run.add_argument("--source-prefix", help="Source prefix for worker sessions")
    run.add_argument("--hermes-bin", help="Hermes executable to invoke (default: hermes)")
    run.add_argument("--read-only", action="store_true", help="Append a read-only constraint")
    run.add_argument("--accept-hooks", action="store_true", help="Pass --accept-hooks to workers")
    run.add_argument("--yolo", action="store_true", help="Pass --yolo to workers")
    run.add_argument("--allow-failures", action="store_true", help="Exit 0 even if workers fail")
    run.add_argument("--dry-run", action="store_true", help="Write planned status but do not start workers")
    run.set_defaults(func=run_swarm)

    stat = swarm_sub.add_parser("status", help="Print a run directory's status JSON")
    stat.add_argument("run_dir", help="Run directory")
    stat.add_argument("--status-file", help="Status JSON path (default: <run-dir>/status.json)")
    stat.set_defaults(func=status_swarm)

    parser.set_defaults(func=lambda _args: (parser.print_help(), 0)[1])
    return parser
