"""Programmatic verifier for Agent Office evidence gates.

The verifier treats gate definitions as canonical requirements and inspects real
command/artifact evidence. Worker summaries are hints only; they never make a
gate pass by themselves.
"""

from __future__ import annotations

import glob
import hashlib
import json
import operator
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb

POLICY_VERSION = "office-verification-v1"
_VALID_STATUSES = {"pass", "fail", "partial", "blocked"}


def _now() -> int:
    return int(time.time())


def _safe_rel_path(raw: str) -> str:
    if not raw or not str(raw).strip():
        raise ValueError("artifact path is required")
    s = str(raw).strip()
    p = Path(s)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe workspace-relative path: {raw!r}")
    return s


def _resolve_workspace(task: kb.Task) -> Path:
    if task.workspace_path:
        return Path(task.workspace_path).expanduser().resolve()
    return kb.resolve_workspace(task).resolve()


def _report_dir(workspace: Path, task_id: str, run_id: int | str) -> Path:
    return workspace / ".hermes" / "verification" / task_id / str(run_id)


def _ensure_under_workspace(workspace: Path, path: Path, label: str) -> Path:
    resolved_workspace = workspace.resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(resolved_workspace)
    except ValueError:
        raise ValueError(f"{label} escapes workspace: {path}")
    return resolved


def _resolve_report_path(workspace: Path, task_id: str, run_id: int | str, report_json: str | os.PathLike[str] | None) -> Path:
    if report_json is None:
        return _report_dir(workspace, task_id, run_id) / "report.json"
    candidate = Path(report_json).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return _ensure_under_workspace(workspace, candidate, "report_json")


def _canonical_json_hash(data: dict[str, Any]) -> str:
    payload = dict(data)
    payload.pop("report_hash", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report_file_hash(path: Path) -> str:
    return _canonical_json_hash(json.loads(path.read_text(encoding="utf-8")))


def _load_gates(conn, task: kb.Task, run_id: int | None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if run_id is not None:
        run = kb.get_run(conn, int(run_id))
        if run and isinstance(run.metadata, dict):
            candidates.append(run.metadata)
    latest = kb.latest_run(conn, task.id)
    if latest and isinstance(latest.metadata, dict):
        candidates.append(latest.metadata)
    for meta in candidates:
        gates = meta.get("verification_gates")
        if isinstance(gates, list):
            return gates
    body = task.body or ""
    # Compatibility bridge: fenced JSON block named verification_gates.
    m = re.search(r"```(?:json)?\s*verification_gates\s*(\[.*?\])\s*```", body, re.S)
    if m:
        try:
            gates = json.loads(m.group(1))
            if isinstance(gates, list):
                return gates
        except Exception:
            pass
    return []


def validate_gates(raw_gates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    seen: set[str] = set()
    gates: list[dict[str, Any]] = []
    errors: list[str] = []
    if not isinstance(raw_gates, list):
        return [], ["verification_gates must be a list"]
    for idx, gate in enumerate(raw_gates):
        if not isinstance(gate, dict):
            errors.append(f"gate[{idx}] must be an object")
            continue
        gate_errors: list[str] = []
        gid = str(gate.get("id") or "").strip()
        if not gid:
            errors.append(f"gate[{idx}] missing id")
            continue
        if gid in seen:
            errors.append(f"duplicate gate id: {gid}")
            continue
        seen.add(gid)
        for artifact in gate.get("expected_artifacts") or []:
            if not isinstance(artifact, dict):
                gate_errors.append(f"gate {gid}: expected_artifacts entries must be objects")
                continue
            try:
                _safe_rel_path(str(artifact.get("path") or ""))
            except Exception as exc:
                gate_errors.append(f"gate {gid}: {exc}")
        for cmd in gate.get("commands") or []:
            if not isinstance(cmd, dict):
                gate_errors.append(f"gate {gid}: commands entries must be objects")
                continue
            argv = cmd.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(part, str) and part for part in argv):
                gate_errors.append(f"gate {gid}: commands must use argv: [executable, ...]; shell command strings are rejected")
            cwd = str(cmd.get("cwd") or ".")
            try:
                _safe_rel_path(cwd)
            except Exception as exc:
                gate_errors.append(f"gate {gid}: unsafe command cwd {cwd!r}: {exc}")
        if gate_errors:
            errors.extend(gate_errors)
            continue
        gates.append(gate)
    return gates, errors


def _write_event(conn, task_id: str, kind: str, payload: dict[str, Any] | None = None, *, run_id: int | None = None) -> None:
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, run_id, kind, json.dumps(payload, ensure_ascii=False) if payload else None, _now()),
    )


def _run_command(workspace: Path, report_dir: Path, gate_id: str, cmd_def: dict[str, Any]) -> dict[str, Any]:
    cid = str(cmd_def.get("id") or f"cmd-{int(time.time() * 1000)}")
    cmd = str(cmd_def.get("cmd") or "")
    timeout = int(cmd_def.get("timeout_seconds") or 300)
    cwd_rel = _safe_rel_path(str(cmd_def.get("cwd") or "."))
    cwd = _ensure_under_workspace(workspace, workspace / cwd_rel, "command cwd")
    logs = report_dir / "logs" / gate_id
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{cid}.stdout"
    stderr_path = logs / f"{cid}.stderr"
    started = _now()
    try:
        argv = cmd_def.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(part, str) and part for part in argv):
            raise ValueError("verifier commands must use argv: [executable, ...]; shell command strings are rejected")
        proc = subprocess.run(
            argv,
            shell=False,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        exit_code = int(proc.returncode)
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text((exc.stderr or "") + f"\nTIMEOUT after {timeout}s\n", encoding="utf-8")
        timed_out = True
    ended = _now()
    return {
        "id": cid,
        "cmd": cmd,
        "cwd": str(cwd),
        "started_at": started,
        "ended_at": ended,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _is_committed(workspace: Path, rel_path: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace), "ls-files", "--error-unmatch", rel_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _inspect_artifact(workspace: Path, artifact: dict[str, Any]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    rel = _safe_rel_path(str(artifact.get("path") or ""))
    pattern = str(workspace / rel)
    matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    evidence: list[str] = []
    must_exist = bool(artifact.get("must_exist", True))
    if must_exist and not matches:
        missing.append(rel)
        checks.append({"name": f"artifact_exists:{rel}", "expected": True, "actual": False, "status": "fail"})
        return evidence, missing, checks
    if not matches:
        return evidence, missing, checks
    min_bytes = int(artifact.get("min_bytes") or 0)
    content_regex = artifact.get("content_regex")
    must_be_committed = bool(artifact.get("must_be_committed", False))
    for path in matches:
        try:
            resolved = _ensure_under_workspace(workspace, path, f"artifact {rel}")
        except ValueError as exc:
            checks.append({"name": f"artifact_workspace:{rel}", "expected": "under workspace", "actual": str(exc), "status": "fail"})
            continue
        rel_found = str(resolved.relative_to(workspace.resolve()))
        evidence.append(rel_found)
        size = resolved.stat().st_size if resolved.exists() else 0
        if min_bytes:
            checks.append({
                "name": f"artifact_min_bytes:{rel_found}",
                "expected": min_bytes,
                "actual": size,
                "status": "pass" if size >= min_bytes else "fail",
            })
        if content_regex:
            try:
                text = resolved.read_text(encoding="utf-8", errors="replace")
                matched = re.search(str(content_regex), text, re.I | re.M) is not None
            except Exception:
                matched = False
            checks.append({
                "name": f"artifact_content_regex:{rel_found}",
                "expected": str(content_regex),
                "actual": matched,
                "status": "pass" if matched else "fail",
            })
        if must_be_committed:
            committed = _is_committed(workspace, rel_found)
            checks.append({
                "name": f"artifact_committed:{rel_found}",
                "expected": True,
                "actual": committed,
                "status": "pass" if committed else "fail",
            })
    return evidence, missing, checks


def _json_lookup(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _eval_threshold(workspace: Path, threshold: dict[str, Any]) -> dict[str, Any]:
    name = str(threshold.get("name") or "threshold")
    source = threshold.get("source")
    actual = threshold.get("actual")
    status = "fail"
    error = None
    if source:
        rel = _safe_rel_path(str(source))
        try:
            path = _ensure_under_workspace(workspace, workspace / rel, f"threshold source {rel}")
        except ValueError as exc:
            return {"name": name, "expected": threshold.get("expected"), "actual": None, "status": "fail", "error": str(exc)}
        if not path.exists():
            return {"name": name, "expected": threshold.get("expected"), "actual": None, "status": "fail", "error": f"missing source {rel}"}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            actual = _json_lookup(data, str(threshold.get("json_path") or ""))
        except Exception as exc:
            error = str(exc)
    expected = threshold.get("expected")
    op = str(threshold.get("op") or "==")
    ops = {"==": operator.eq, "!=": operator.ne, "<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}
    try:
        if op in ops and actual is not None:
            status = "pass" if ops[op](float(actual), float(expected)) else "fail"
    except Exception as exc:
        error = str(exc)
    return {"name": name, "expected": expected, "actual": actual, "op": op, "status": status, **({"error": error} if error else {})}


def _gate_verdict(workspace: Path, report_dir: Path, gate: dict[str, Any]) -> dict[str, Any]:
    gid = str(gate.get("id"))
    commands_run: list[dict[str, Any]] = []
    evidence_paths: list[str] = []
    missing_artifacts: list[str] = []
    threshold_results: list[dict[str, Any]] = []
    notes: list[str] = []
    gate_type = str(gate.get("type") or "").lower()
    artifact_defs = gate.get("expected_artifacts") or []

    blocked_reason: str | None = None
    for cmd in gate.get("commands") or []:
        try:
            result = _run_command(workspace, report_dir, gid, cmd)
            commands_run.append(result)
            allowed = [int(x) for x in cmd.get("allowed_exit_codes", [0])]
            threshold_results.append({
                "name": f"command_exit:{result['id']}",
                "expected": allowed,
                "actual": result["exit_code"],
                "status": "pass" if result["exit_code"] in allowed else "fail",
            })
        except Exception as exc:
            if gate.get("allow_blocked"):
                blocked_reason = str(exc)
            else:
                threshold_results.append({"name": "command_policy", "expected": "safe executable command", "actual": str(exc), "status": "fail"})

    for artifact in artifact_defs:
        ev, missing, checks = _inspect_artifact(workspace, artifact)
        evidence_paths.extend(ev)
        missing_artifacts.extend(missing)
        threshold_results.extend(checks)

    for threshold in gate.get("thresholds") or []:
        threshold_results.append(_eval_threshold(workspace, threshold))

    artifact_required_types = {"benchmark", "benchmark_artifact", "performance", "load_test", "k6"}
    if gate_type in artifact_required_types and not artifact_defs:
        threshold_results.append({
            "name": "artifact_policy:benchmark_requires_real_artifact",
            "expected": "at least one expected_artifacts entry for benchmark/performance evidence",
            "actual": "none",
            "status": "fail",
        })

    failed_checks = [c for c in threshold_results if c.get("status") == "fail"]
    if blocked_reason:
        status = "blocked"
    elif missing_artifacts or failed_checks:
        status = "fail"
    elif threshold_results or evidence_paths or not gate.get("required", True):
        status = "pass"
    else:
        status = "partial"
        notes.append("gate has no executable commands or inspectable artifacts")
    return {
        "gate_id": gid,
        "status": status,
        "score": 1.0 if status == "pass" else (0.5 if status == "partial" else 0.0),
        "required": bool(gate.get("required", True)),
        "type": gate.get("type"),
        "commands_run": commands_run,
        "evidence_paths": sorted(set(evidence_paths)),
        "missing_artifacts": sorted(set(missing_artifacts)),
        "threshold_results": threshold_results,
        "blocked_reason": blocked_reason,
        "notes": "; ".join(notes) if notes else None,
    }


def _aggregate(verdicts: list[dict[str, Any]]) -> str:
    required = [v for v in verdicts if v.get("required", True)]
    if any(v["status"] == "fail" for v in required):
        return "fail"
    if any(v["status"] == "blocked" for v in required):
        return "blocked"
    if any(v["status"] == "partial" for v in required):
        return "partial"
    return "pass"


def _merge_run_metadata(conn, run_id: int, report_summary: dict[str, Any]) -> None:
    row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (int(run_id),)).fetchone()
    meta: dict[str, Any] = {}
    if row and row["metadata"]:
        try:
            parsed = json.loads(row["metadata"])
            if isinstance(parsed, dict):
                meta = parsed
        except Exception:
            meta = {}
    meta["verification_report"] = report_summary
    conn.execute(
        "UPDATE task_runs SET metadata = ? WHERE id = ?",
        (json.dumps(meta, ensure_ascii=False), int(run_id)),
    )


def verify_task(
    conn,
    task_id: str,
    *,
    run_id: int | None = None,
    strict: bool = False,
    report_json: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    task = kb.get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    if run_id is None:
        latest = kb.latest_run(conn, task_id)
        run_id = latest.id if latest else 0
    workspace = _resolve_workspace(task)
    workspace.mkdir(parents=True, exist_ok=True)
    report_path = _resolve_report_path(workspace, task_id, run_id, report_json)
    rdir = report_path.parent
    rdir.mkdir(parents=True, exist_ok=True)

    started = _now()
    with kb.write_txn(conn):
        _write_event(conn, task_id, "office.verification.started", {"policy_version": POLICY_VERSION, "run_id": run_id}, run_id=run_id)

    raw_gates = _load_gates(conn, task, run_id)
    gates, errors = validate_gates(raw_gates)
    verdicts = [_gate_verdict(workspace, rdir, gate) for gate in gates]
    if errors or (strict and not gates):
        verdicts.append({
            "gate_id": "gate-definition",
            "status": "fail" if errors else "blocked",
            "score": 0.0,
            "required": True,
            "type": "schema",
            "commands_run": [],
            "evidence_paths": [],
            "missing_artifacts": [],
            "threshold_results": [],
            "blocked_reason": None if errors else "no verification_gates configured",
            "notes": "; ".join(errors) if errors else "no verification_gates configured",
        })
    overall = _aggregate(verdicts)
    ended = _now()
    counts = {s: sum(1 for v in verdicts if v.get("status") == s) for s in _VALID_STATUSES}
    report = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "task_id": task_id,
        "run_id": run_id,
        "started_at": started,
        "ended_at": ended,
        "overall_status": overall,
        "passed": counts["pass"],
        "failed": counts["fail"],
        "partial": counts["partial"],
        "blocked": counts["blocked"],
        "total": len(verdicts),
        "report_path": str(report_path),
        "gate_errors": errors,
        "gate_verdicts": verdicts,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_hash = _canonical_json_hash(report)
    report["report_hash"] = report_hash
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "overall_status": overall,
        "report_path": str(report_path),
        "report_hash": report_hash,
        "run_id": run_id,
        "passed": counts["pass"],
        "failed": counts["fail"],
        "partial": counts["partial"],
        "blocked": counts["blocked"],
        "total": len(verdicts),
    }
    with kb.write_txn(conn):
        if run_id:
            _merge_run_metadata(conn, int(run_id), summary)
        for verdict in verdicts:
            if verdict.get("required", True) and verdict.get("status") == "fail":
                _write_event(conn, task_id, "office.verification.gate_failed", {"gate_id": verdict.get("gate_id"), "missing_artifacts": verdict.get("missing_artifacts")}, run_id=run_id)
        _write_event(conn, task_id, "office.verification.completed", summary, run_id=run_id)
    return report


def latest_verification_summary(conn, task_id: str) -> dict[str, Any] | None:
    """Return latest available verification report for dashboard/CLI views."""
    for run in reversed(kb.list_runs(conn, task_id)):
        if isinstance(run.metadata, dict) and isinstance(run.metadata.get("verification_report"), dict):
            return run.metadata["verification_report"]
    events = [e for e in kb.list_events(conn, task_id) if e.kind == "office.verification.completed" and isinstance(e.payload, dict)]
    return events[-1].payload if events else None


def completion_verification_summary(conn, task_id: str) -> dict[str, Any] | None:
    """Return verification report for the latest/current run only.

    Completion must not fall back to stale evidence from an older run. Manual
    Office dashboard verification can be run-bound-less when no worker run has
    ever existed; in that no-run case, use the latest verification event as the
    current evidence instead of making legacy/ungated Office cards impossible to
    close through the staged cockpit.
    """
    latest = kb.latest_run(conn, task_id)
    if latest and isinstance(latest.metadata, dict):
        report = latest.metadata.get("verification_report")
        if isinstance(report, dict):
            return report
        return None
    events = [e for e in kb.list_events(conn, task_id) if e.kind == "office.verification.completed" and isinstance(e.payload, dict)]
    return events[-1].payload if events else None


def has_pending_scope_change(conn, task_id: str) -> bool:
    events = kb.list_events(conn, task_id)
    scope_requests = [e for e in events if e.kind == "office.scope_change_requested"]
    scope_approvals = [
        e for e in events
        if e.kind == "office.scope_change.approved"
        and isinstance(e.payload, dict)
        and e.payload.get("approved") is True
    ]
    latest_scope_approval_id = max((e.id for e in scope_approvals), default=0)
    return any(e.id > latest_scope_approval_id for e in scope_requests)


def task_requires_final_review(conn, task_id: str) -> bool:
    """True when a task/run carries Office verification gates or a report.

    Generic Kanban boards stay backward-compatible: only gate-bearing Office
    work is subject to the verifier + independent-review completion invariant.
    """
    task = kb.get_task(conn, task_id)
    if task is None:
        return False
    for run in kb.list_runs(conn, task_id):
        if isinstance(run.metadata, dict):
            if isinstance(run.metadata.get("verification_gates"), list):
                return True
            if isinstance(run.metadata.get("verification_report"), dict):
                return True
    return bool(_load_gates(conn, task, None))


def final_completion_ready(conn, task_id: str) -> tuple[bool, str]:
    """Check the Office final verifier + reviewer invariant for gate-bearing tasks."""
    report = completion_verification_summary(conn, task_id)
    if not report:
        return False, "missing verifier report"
    if report.get("overall_status") != "pass":
        return False, f"verification status is {report.get('overall_status')}"
    events = kb.list_events(conn, task_id)
    if has_pending_scope_change(conn, task_id):
        return False, "pending scope change request requires approval"
    verification_events = [
        e for e in events
        if e.kind == "office.verification.completed"
        and isinstance(e.payload, dict)
        and e.payload.get("report_path") == report.get("report_path")
        and e.payload.get("report_hash") == report.get("report_hash")
        and e.payload.get("run_id") == report.get("run_id")
    ]
    if not verification_events:
        return False, "missing matching verification completion event"
    latest_verification_event_id = verification_events[-1].id
    reviews = [e for e in events if e.kind == "office.review.completed" and isinstance(e.payload, dict) and e.id > latest_verification_event_id]
    if not reviews:
        return False, "missing independent reviewer approval"
    latest_event = reviews[-1]
    latest = latest_event.payload or {}
    required = ("approved", "reviewed_report_path", "reviewed_report_hash", "reviewed_run_id", "reviewed_diff_ref", "findings", "gate_report_overall_status")
    missing = [k for k in required if k not in latest]
    if missing:
        return False, "review metadata missing: " + ", ".join(missing)
    if latest.get("approved") is not True:
        return False, "reviewer did not approve"
    if latest.get("gate_report_overall_status") != "pass":
        return False, "reviewer did not review a passing gate report"
    reviewed_path = Path(str(latest.get("reviewed_report_path") or "")).expanduser()
    latest_report_path = Path(str(report.get("report_path") or "")).expanduser()
    if str(reviewed_path) != str(latest_report_path):
        return False, "reviewed report path does not match latest verifier report"
    if latest.get("reviewed_report_hash") != report.get("report_hash"):
        return False, "reviewed report hash does not match latest verifier report"
    if latest.get("reviewed_run_id") != report.get("run_id"):
        return False, "reviewed run id does not match latest verifier report"
    if not reviewed_path.exists():
        return False, "reviewed verifier report path does not exist"
    try:
        current_report_hash = _report_file_hash(reviewed_path.resolve())
    except Exception:
        return False, "reviewed verifier report file could not be hashed"
    if current_report_hash != report.get("report_hash"):
        return False, "reviewed verifier report file hash changed after verification"
    return True, "verified and reviewer-approved"


def cli(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="hermes office verify")
    parser.add_argument("task_id")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args(argv)
    with kb.connect() as conn:
        report = verify_task(conn, args.task_id, run_id=args.run_id, strict=args.strict, report_json=args.report_json)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["overall_status"] == "pass" else 2
