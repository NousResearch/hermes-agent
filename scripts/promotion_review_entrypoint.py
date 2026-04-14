#!/usr/bin/env python3
"""Minimal promotion/review workflow entrypoint.

Composes doctrine-object lint (when relevant) with promotion_gate_runner and emits
one machine-readable fail-closed workflow decision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DOCTRINE_LINT_RUNNER = SCRIPT_PATH.parent / "doctrine_object_lint.py"
PROMOTION_GATE_RUNNER = SCRIPT_PATH.parent / "promotion_gate_runner.py"
DEFAULT_WORKFLOW_DECISION_LOG = (
    DEFAULT_REPO_ROOT / "state" / "continuity" / "promotion_review_workflow" / "decisions.jsonl"
)

DEFAULT_DOCTRINE_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "doctrine_object.schema.json"
DEFAULT_PROMOTION_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "promotion_candidate.schema.json"
DEFAULT_DOCTRINE_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "doctrine_object_lint" / "decisions.jsonl"
DEFAULT_PROMOTION_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "promotion_gate_runner" / "decisions.jsonl"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def append_decision_record(
    *,
    decision_log_path: Optional[Path],
    repo_root: Path,
    decision_row: Dict[str, Any],
) -> Dict[str, Any]:
    if decision_log_path is None:
        return {"enabled": False, "appended": False, "reason": "disabled"}

    path = decision_log_path
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()

    if not is_within(repo_root, path):
        return {
            "enabled": True,
            "appended": False,
            "reason": "unsafe_path",
            "path": str(path),
        }

    try:
        if path.exists() and not path.is_file():
            return {
                "enabled": True,
                "appended": False,
                "reason": "path_not_file",
                "path": str(path),
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(decision_row) + "\n")

        return {
            "enabled": True,
            "appended": True,
            "path": str(path),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "appended": False,
            "reason": "append_failed",
            "path": str(path),
            "error": str(exc),
        }


def _run_json_runner(command: List[str], *, cwd: Path) -> Dict[str, Any]:
    cp = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        cwd=str(cwd),
    )

    stdout = cp.stdout.strip()
    payload: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None
    if stdout:
        try:
            loaded = json.loads(stdout)
            if isinstance(loaded, dict):
                payload = loaded
            else:
                parse_error = "runner_stdout_not_object_json"
        except Exception as exc:
            parse_error = f"runner_stdout_not_json: {exc}"
    else:
        parse_error = "runner_stdout_empty"

    ok_exit = cp.returncode in {0, 2}
    decision = payload.get("decision") if isinstance(payload, dict) else None
    ok_decision = decision in {"PASS", "BLOCK"}

    return {
        "exit_code": cp.returncode,
        "stdout": cp.stdout,
        "stderr": cp.stderr,
        "payload": payload,
        "ok": bool(ok_exit and payload is not None and ok_decision),
        "error": parse_error,
    }


def _candidate_metadata(candidate_path: Path) -> Dict[str, Any]:
    try:
        candidate = load_json_file(candidate_path)
    except Exception:
        return {
            "loaded": False,
            "promotion_id": None,
            "target_surface": None,
            "target_path": None,
        }

    if not isinstance(candidate, dict):
        return {
            "loaded": False,
            "promotion_id": None,
            "target_surface": None,
            "target_path": None,
        }

    promotion_id = candidate.get("promotion_id") if isinstance(candidate.get("promotion_id"), str) else None
    target = candidate.get("target") if isinstance(candidate.get("target"), dict) else {}
    target_surface = target.get("surface") if isinstance(target.get("surface"), str) else None
    target_path = target.get("target_path") if isinstance(target.get("target_path"), str) else None
    return {
        "loaded": True,
        "promotion_id": promotion_id,
        "target_surface": target_surface,
        "target_path": target_path,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Promotion/review workflow entrypoint (doctrine lint + promotion gate)")
    ap.add_argument("--candidate", required=True, help="Path to promotion candidate JSON")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")

    ap.add_argument(
        "--doctrine-object",
        default=None,
        help=(
            "Doctrine object JSON path for pre-promotion lint. "
            "Required for doctrine-surface candidates unless target_path points at a JSON doctrine object."
        ),
    )
    ap.add_argument("--doctrine-schema-path", default=str(DEFAULT_DOCTRINE_SCHEMA_PATH), help="Doctrine object schema path")
    ap.add_argument("--doctrine-decision-log", default=str(DEFAULT_DOCTRINE_DECISION_LOG), help="Doctrine lint decision log path")
    ap.add_argument("--no-doctrine-decision-log", action="store_true", help="Disable doctrine lint decision recording")

    ap.add_argument("--promotion-schema-path", default=str(DEFAULT_PROMOTION_SCHEMA_PATH), help="Promotion candidate schema path")
    ap.add_argument("--promotion-decision-log", default=str(DEFAULT_PROMOTION_DECISION_LOG), help="Promotion gate decision log path")
    ap.add_argument("--no-promotion-decision-log", action="store_true", help="Disable promotion gate decision recording")
    ap.add_argument("--publish-note-path", default=None, help="Optional publish note path forwarded to promotion_gate_runner")

    ap.add_argument(
        "--workflow-decision-log",
        default=str(DEFAULT_WORKFLOW_DECISION_LOG),
        help="Append-only workflow decision log path",
    )
    ap.add_argument("--no-workflow-decision-log", action="store_true", help="Disable workflow decision recording")

    ap.add_argument("--json", action="store_true", help="Emit pretty JSON")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()

    candidate_meta = _candidate_metadata(candidate_path)
    target_surface = candidate_meta.get("target_surface")
    target_path = candidate_meta.get("target_path")

    doctrine_required = False
    doctrine_object_path: Optional[Path] = None

    if args.doctrine_object:
        doctrine_required = True
        doctrine_object_path = Path(args.doctrine_object).expanduser()
        if not doctrine_object_path.is_absolute():
            doctrine_object_path = (repo_root / doctrine_object_path).resolve()
        else:
            doctrine_object_path = doctrine_object_path.resolve()
    elif target_surface == "doctrine":
        doctrine_required = True
        if isinstance(target_path, str) and target_path.strip():
            inferred = resolve_repo_path(repo_root, target_path)
            if inferred.suffix.lower() == ".json":
                doctrine_object_path = inferred

    stages: List[Dict[str, Any]] = []
    doctrine_result: Optional[Dict[str, Any]] = None
    promotion_result: Optional[Dict[str, Any]] = None

    blocked = False
    block_stage: Optional[str] = None
    block_reason: Optional[str] = None

    if doctrine_required:
        if doctrine_object_path is None:
            blocked = True
            block_stage = "doctrine_lint"
            block_reason = "doctrine_object_missing"
            stages.append(
                {
                    "stage": "doctrine_lint",
                    "status": "fail",
                    "reason": block_reason,
                    "details": {
                        "target_surface": target_surface,
                        "target_path": target_path,
                    },
                }
            )
        else:
            doctrine_cmd = [
                sys.executable,
                str(DOCTRINE_LINT_RUNNER),
                "--object",
                str(doctrine_object_path),
                "--repo-root",
                str(repo_root),
                "--schema-path",
                str(Path(args.doctrine_schema_path).expanduser().resolve()),
            ]
            if args.no_doctrine_decision_log:
                doctrine_cmd.append("--no-decision-log")
            else:
                doctrine_cmd.extend(["--decision-log", str(Path(args.doctrine_decision_log).expanduser())])
            doctrine_cmd.append("--json")

            run = _run_json_runner(doctrine_cmd, cwd=DEFAULT_REPO_ROOT)
            doctrine_result = run.get("payload") if isinstance(run.get("payload"), dict) else None

            if run.get("ok"):
                doctrine_decision = doctrine_result.get("decision") if doctrine_result else None
                if doctrine_decision == "PASS":
                    stages.append({"stage": "doctrine_lint", "status": "pass", "details": {"path": str(doctrine_object_path)}})
                else:
                    blocked = True
                    block_stage = "doctrine_lint"
                    block_reason = doctrine_result.get("block_reason") if doctrine_result else "gate_unavailable"
                    stages.append(
                        {
                            "stage": "doctrine_lint",
                            "status": "fail",
                            "reason": block_reason,
                            "details": {
                                "path": str(doctrine_object_path),
                                "block_gate": doctrine_result.get("block_gate") if doctrine_result else None,
                            },
                        }
                    )
            else:
                blocked = True
                block_stage = "doctrine_lint"
                block_reason = "gate_unavailable"
                stages.append(
                    {
                        "stage": "doctrine_lint",
                        "status": "fail",
                        "reason": block_reason,
                        "details": {
                            "path": str(doctrine_object_path),
                            "exit_code": run.get("exit_code"),
                            "runner_error": run.get("error"),
                            "stderr": run.get("stderr"),
                        },
                    }
                )
    else:
        stages.append(
            {
                "stage": "doctrine_lint",
                "status": "skipped",
                "reason": "not_relevant",
            }
        )

    if blocked:
        stages.append({"stage": "promotion_gate", "status": "skipped", "reason": "blocked_by_previous_stage"})
    else:
        promotion_cmd = [
            sys.executable,
            str(PROMOTION_GATE_RUNNER),
            "--candidate",
            str(candidate_path),
            "--repo-root",
            str(repo_root),
            "--schema-path",
            str(Path(args.promotion_schema_path).expanduser().resolve()),
        ]
        if args.publish_note_path:
            promotion_cmd.extend(["--publish-note-path", str(args.publish_note_path)])
        if args.no_promotion_decision_log:
            promotion_cmd.append("--no-decision-log")
        else:
            promotion_cmd.extend(["--decision-log", str(Path(args.promotion_decision_log).expanduser())])
        promotion_cmd.append("--json")

        run = _run_json_runner(promotion_cmd, cwd=DEFAULT_REPO_ROOT)
        promotion_result = run.get("payload") if isinstance(run.get("payload"), dict) else None

        if run.get("ok"):
            promotion_decision = promotion_result.get("decision") if promotion_result else None
            if promotion_decision == "PASS":
                stages.append({"stage": "promotion_gate", "status": "pass"})
            else:
                blocked = True
                block_stage = "promotion_gate"
                block_reason = promotion_result.get("block_reason") if promotion_result else "gate_unavailable"
                stages.append(
                    {
                        "stage": "promotion_gate",
                        "status": "fail",
                        "reason": block_reason,
                        "details": {
                            "block_gate": promotion_result.get("block_gate") if promotion_result else None,
                        },
                    }
                )
        else:
            blocked = True
            block_stage = "promotion_gate"
            block_reason = "gate_unavailable"
            stages.append(
                {
                    "stage": "promotion_gate",
                    "status": "fail",
                    "reason": block_reason,
                    "details": {
                        "exit_code": run.get("exit_code"),
                        "runner_error": run.get("error"),
                        "stderr": run.get("stderr"),
                    },
                }
            )

    decision = "BLOCK" if blocked else "PASS"
    final_state = "BLOCKED" if blocked else "PROMOTED"

    workflow_result: Dict[str, Any] = {
        "schema": "clawd.promotion_review_workflow.decision.v1",
        "evaluated_at": now_iso(),
        "decision": decision,
        "final_state": final_state,
        "block_stage": block_stage,
        "block_reason": block_reason,
        "promotion_id": candidate_meta.get("promotion_id"),
        "candidate": {
            "path": str(candidate_path),
        },
        "doctrine_lint": {
            "required": doctrine_required,
            "object_path": str(doctrine_object_path) if doctrine_object_path else None,
        },
        "stages": stages,
        "stage_results": {
            "doctrine_lint": doctrine_result,
            "promotion_gate": promotion_result,
        },
    }

    workflow_decision_log_path: Optional[Path] = None
    if not args.no_workflow_decision_log:
        workflow_decision_log_path = Path(args.workflow_decision_log).expanduser()

    record = append_decision_record(
        decision_log_path=workflow_decision_log_path,
        repo_root=repo_root,
        decision_row=workflow_result,
    )
    workflow_result["decision_record"] = record

    if args.json:
        print(json.dumps(workflow_result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(workflow_result))

    return 0 if decision == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
