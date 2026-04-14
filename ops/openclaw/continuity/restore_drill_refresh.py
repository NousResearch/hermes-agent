#!/usr/bin/env python3
"""Deterministic restore-drill freshness refresher.

Runs a bounded dry-run rollback drill when restore evidence is missing/stale,
writes fresh evidence, and emits machine-readable status for watchdog callers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any


def _now_utc() -> dt.datetime:
    fixed = str(os.environ.get("OPENCLAW_AUTOPILOT_FIXED_NOW_TS", "")).strip()
    if fixed and fixed.lstrip("-").isdigit():
        return dt.datetime.fromtimestamp(int(fixed), tz=dt.timezone.utc)
    return dt.datetime.now(dt.timezone.utc)


def _iso(value: dt.datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(raw: Any) -> dt.datetime | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    norm = txt[:-1] + "+00:00" if txt.endswith("Z") else txt
    try:
        parsed = dt.datetime.fromisoformat(norm)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _rel(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def _evidence_age_meta(evidence_path: pathlib.Path, now_dt: dt.datetime) -> dict[str, Any]:
    if not evidence_path.exists() or not evidence_path.is_file():
        return {
            "present": False,
            "age_sec": None,
            "source": "missing",
            "status": None,
            "payload": {},
            "timestamp": None,
        }

    payload = _load_json(evidence_path)
    ts_source = "mtime"
    ref_dt: dt.datetime | None = None
    for field in ("drilled_at", "executed_at", "timestamp", "generated_at", "updated_at"):
        if field not in payload:
            continue
        parsed = _parse_iso(payload.get(field))
        if parsed is not None:
            ref_dt = parsed
            ts_source = f"payload:{field}"
            break

    if ref_dt is None:
        ref_dt = dt.datetime.fromtimestamp(evidence_path.stat().st_mtime, tz=dt.timezone.utc)

    age_sec = max(0, int((now_dt - ref_dt).total_seconds()))
    return {
        "present": True,
        "age_sec": age_sec,
        "source": ts_source,
        "status": str(payload.get("status") or "").strip().lower() or None,
        "payload": payload,
        "timestamp": _iso(ref_dt),
    }


def _checkpoint_path_from_id(root: pathlib.Path, checkpoint_id: str) -> pathlib.Path:
    return root / "state" / "continuity" / "checkpoints" / f"{checkpoint_id}.json"


def _checkpoint_id_from_payload(payload: dict[str, Any], fallback: str = "") -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    checkpoint_id = str(metadata.get("checkpoint_id") or "").strip()
    if checkpoint_id:
        return checkpoint_id
    return fallback


def _checkpoint_ready(payload: dict[str, Any]) -> bool:
    objective = payload.get("objective") if isinstance(payload.get("objective"), dict) else {}
    return str(objective.get("status") or "").strip().upper() == "READY"


def _resolve_checkpoint(
    *,
    root: pathlib.Path,
    explicit: str | None,
    latest_pointer_path: pathlib.Path,
) -> tuple[str, pathlib.Path, str]:
    explicit_txt = str(explicit or "").strip()
    if explicit_txt:
        explicit_path = pathlib.Path(explicit_txt)
        if explicit_path.exists() and explicit_path.is_file():
            payload = _load_json(explicit_path)
            checkpoint_id = _checkpoint_id_from_payload(payload, fallback=explicit_path.stem)
            return checkpoint_id or explicit_txt, explicit_path, "explicit_path"

        candidate_path = _checkpoint_path_from_id(root, explicit_txt)
        if candidate_path.exists() and candidate_path.is_file():
            return explicit_txt, candidate_path, "explicit_id"
        raise RuntimeError(f"checkpoint_not_found:{explicit_txt}")

    latest_pointer = _load_json(latest_pointer_path)
    pointer_id = str(latest_pointer.get("checkpoint_id") or "").strip()
    if pointer_id:
        pointer_path = _checkpoint_path_from_id(root, pointer_id)
        if pointer_path.exists() and pointer_path.is_file():
            payload = _load_json(pointer_path)
            if _checkpoint_ready(payload):
                return pointer_id, pointer_path, "latest_pointer_ready"

    checkpoints_dir = root / "state" / "continuity" / "checkpoints"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        for candidate in sorted(checkpoints_dir.glob("chk_*.json"), reverse=True):
            payload = _load_json(candidate)
            if not payload:
                continue
            if not _checkpoint_ready(payload):
                continue
            checkpoint_id = _checkpoint_id_from_payload(payload, fallback=candidate.stem)
            if checkpoint_id:
                return checkpoint_id, candidate, "scan_latest_ready"

    if pointer_id:
        pointer_path = _checkpoint_path_from_id(root, pointer_id)
        if pointer_path.exists() and pointer_path.is_file():
            return pointer_id, pointer_path, "latest_pointer_fallback"

    raise RuntimeError("ready_checkpoint_missing")


def _write_report(
    *,
    report_path: pathlib.Path,
    now_iso: str,
    checkpoint_id: str,
    checkpoint_source: str,
    verify_cmd: list[str],
    verify_rc: int,
    verify_stdout: str,
    verify_stderr: str,
    verify_reason: str | None,
    verify_status: str | None,
) -> None:
    status = "pass" if verify_rc == 0 else "fail"
    first_out = next((ln.strip() for ln in verify_stdout.splitlines() if ln.strip()), "")
    first_err = next((ln.strip() for ln in verify_stderr.splitlines() if ln.strip()), "")

    body = [
        "# Restore drill (automated)",
        "",
        f"Date: {now_iso}",
        "Mode: bounded dry-run (automation)",
        f"Target checkpoint: `{checkpoint_id}`",
        f"Checkpoint selection source: `{checkpoint_source}`",
        "",
        "## Procedure executed",
        "1. Resolve latest READY checkpoint (or configured override).",
        "2. Run bounded dry-run rollback verification:",
        f"   - `{' '.join(shlex.quote(token) for token in verify_cmd)}`",
        "",
        "## Result",
        f"Status: `{status}`",
        f"verify_then_resume exit_code: `{verify_rc}`",
        f"verify_last.status: `{verify_status or 'unknown'}`",
        f"verify_last.reason: `{verify_reason or 'unknown'}`",
        "",
        "## Runtime excerpts",
        f"- first_stdout_line: `{first_out or 'none'}`",
        f"- first_stderr_line: `{first_err or 'none'}`",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(body) + "\n", encoding="utf-8")


def _emit(json_mode: bool, payload: dict[str, Any]) -> int:
    status = str(payload.get("status") or "").strip().lower()
    decision = str(payload.get("decision") or "").strip()

    if json_mode:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if status != "error" else 2

    if status == "error":
        reason = str(payload.get("error") or "unknown_error").replace("\n", " ").strip()
        print(f"BLOCKER: restore_drill_auto_refresh_failed; reason={reason[:200]}")
        return 0

    if decision == "skipped_fresh":
        print(
            "READY: restore_drill_freshness_ok; "
            f"age_sec={payload.get('evidence_age_sec_before')}; "
            f"threshold_sec={payload.get('refresh_after_sec')}"
        )
        return 0

    print(
        "PROGRESS: restore_drill_refreshed; "
        f"drill_status={payload.get('drill_status')}; "
        f"checkpoint_id={payload.get('checkpoint_id')}; "
        f"report={payload.get('report_ref')}"
    )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Refresh restore-drill evidence when stale.")
    parser.add_argument("--json", action="store_true", dest="json_mode", help="Emit JSON payload.")
    parser.add_argument("--force", action="store_true", help="Run drill even when evidence is fresh.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint id or checkpoint JSON path.")
    parser.add_argument("--trigger", default="watchdog.no_nudge_continuity", help="Automation trigger label.")
    parser.add_argument(
        "--refresh-after-sec",
        type=int,
        default=int(os.environ.get("OPENCLAW_RESTORE_DRILL_REFRESH_AFTER_SEC", "518400")),
        help="Run drill when evidence age is >= this threshold (default 6 days).",
    )
    parser.add_argument(
        "--max-age-sec",
        type=int,
        default=int(os.environ.get("OPENCLAW_SLO_RESTORE_DRILL_MAX_AGE_SEC", "604800")),
        help="Declared freshness budget for metadata/reporting.",
    )
    args = parser.parse_args(argv)

    refresh_after_sec = max(0, int(args.refresh_after_sec))
    max_age_sec = max(0, int(args.max_age_sec))

    root = pathlib.Path(os.environ.get("OPENCLAW_ROOT", "/home/yeqiuqiu/clawd-architect")).resolve()
    evidence_path = pathlib.Path(
        os.environ.get(
            "OPENCLAW_RESTORE_DRILL_EVIDENCE_PATH",
            str(root / "state" / "continuity" / "latest" / "restore_drill_latest.json"),
        )
    ).resolve()
    verify_script = pathlib.Path(
        os.environ.get(
            "OPENCLAW_RESTORE_DRILL_VERIFY_SCRIPT",
            str(root / "ops" / "openclaw" / "continuity" / "verify_then_resume.sh"),
        )
    ).resolve()
    latest_pointer_path = pathlib.Path(
        os.environ.get(
            "OPENCLAW_RESTORE_DRILL_LATEST_POINTER_PATH",
            str(root / "state" / "continuity" / "latest" / "latest_pointer.json"),
        )
    ).resolve()
    reports_dir = pathlib.Path(
        os.environ.get("OPENCLAW_RESTORE_DRILL_REPORTS_DIR", str(root / "reports"))
    ).resolve()

    now_dt = _now_utc()
    now_iso = _iso(now_dt)

    payload: dict[str, Any] = {
        "schema": "clawd.restore_drill.refresh.v1",
        "generated_at": now_iso,
        "status": "ok",
        "decision": "",
        "trigger": str(args.trigger or "watchdog.no_nudge_continuity"),
        "refresh_after_sec": refresh_after_sec,
        "max_age_sec": max_age_sec,
        "evidence_path": _rel(evidence_path, root),
        "evidence_present_before": False,
        "evidence_age_sec_before": None,
        "evidence_age_source_before": None,
        "drill_status": None,
        "checkpoint_id": None,
        "checkpoint_path": None,
        "checkpoint_source": None,
        "report_ref": None,
        "verify_exit_code": None,
        "verify_reason": None,
        "verify_status": None,
        "evidence_status_before": None,
        "refresh_reason": None,
        "error": None,
    }

    try:
        if not verify_script.exists() or not verify_script.is_file():
            raise RuntimeError(f"verify_script_missing:{verify_script}")

        age_meta = _evidence_age_meta(evidence_path, now_dt)
        payload["evidence_present_before"] = bool(age_meta.get("present"))
        payload["evidence_age_sec_before"] = age_meta.get("age_sec")
        payload["evidence_age_source_before"] = age_meta.get("source")
        payload["evidence_status_before"] = age_meta.get("status")

        evidence_age = age_meta.get("age_sec")
        evidence_status = str(age_meta.get("status") or "").strip().lower() or None
        should_refresh = bool(args.force)
        refresh_reason = "forced" if should_refresh else None
        if not should_refresh:
            if evidence_status != "pass":
                should_refresh = True
                if evidence_status:
                    refresh_reason = f"status_not_pass:{evidence_status}"
                else:
                    refresh_reason = "status_missing"
            elif evidence_age is None or int(evidence_age) >= refresh_after_sec:
                should_refresh = True
                refresh_reason = "stale_or_missing"

        payload["refresh_reason"] = refresh_reason or "fresh_pass"

        if not should_refresh:
            payload["decision"] = "skipped_fresh"
            payload["drill_status"] = age_meta.get("status")
            return _emit(args.json_mode, payload)

        checkpoint_id, checkpoint_path, checkpoint_source = _resolve_checkpoint(
            root=root,
            explicit=str(args.checkpoint or "").strip() or None,
            latest_pointer_path=latest_pointer_path,
        )
        payload["checkpoint_id"] = checkpoint_id
        payload["checkpoint_path"] = _rel(checkpoint_path, root)
        payload["checkpoint_source"] = checkpoint_source

        verify_cmd = [
            "bash",
            str(verify_script),
            "--checkpoint",
            checkpoint_id,
            "--run-rollback",
            "--status-evidence-repair",
        ]
        verify_cp = subprocess.run(verify_cmd, text=True, capture_output=True, check=False)

        verify_last = _load_json(root / "state" / "continuity" / "latest" / "verify_last.json")
        verify_reason = str(verify_last.get("reason") or "").strip() or None
        verify_status = str(verify_last.get("status") or "").strip() or None

        report_stamp = now_dt.strftime("%Y%m%dT%H%M%SZ")
        report_path = reports_dir / f"restore_drill_auto_{report_stamp}.md"
        _write_report(
            report_path=report_path,
            now_iso=now_iso,
            checkpoint_id=checkpoint_id,
            checkpoint_source=checkpoint_source,
            verify_cmd=verify_cmd,
            verify_rc=int(verify_cp.returncode),
            verify_stdout=verify_cp.stdout or "",
            verify_stderr=verify_cp.stderr or "",
            verify_reason=verify_reason,
            verify_status=verify_status,
        )

        drill_status = "pass" if verify_cp.returncode == 0 else "fail"
        evidence_payload = {
            "schema": "clawd.restore_drill.evidence.v1",
            "drilled_at": now_iso,
            "status": drill_status,
            "drill_ref": _rel(report_path, root),
            "checkpoint_id": checkpoint_id,
            "verify_exit_code": int(verify_cp.returncode),
            "verify_status": verify_status,
            "verify_reason": verify_reason,
            "automation": {
                "schema": "clawd.restore_drill.automation.v1",
                "trigger": payload["trigger"],
                "refresh_after_sec": refresh_after_sec,
                "max_age_sec": max_age_sec,
            },
            "notes": (
                "Automated bounded dry-run rollback drill via verify_then_resume --run-rollback "
                "--status-evidence-repair; "
                f"status={drill_status}; checkpoint={checkpoint_id}; verify_reason={verify_reason or 'unknown'}."
            ),
        }

        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        payload["decision"] = "refreshed_pass" if drill_status == "pass" else "refreshed_fail"
        payload["drill_status"] = drill_status
        payload["report_ref"] = _rel(report_path, root)
        payload["verify_exit_code"] = int(verify_cp.returncode)
        payload["verify_reason"] = verify_reason
        payload["verify_status"] = verify_status
        return _emit(args.json_mode, payload)

    except Exception as exc:
        payload["status"] = "error"
        payload["decision"] = "error"
        payload["error"] = str(exc)
        return _emit(args.json_mode, payload)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
