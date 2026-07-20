#!/usr/bin/env python3
"""Create a durable ULW-loop evidence ledger scaffold.

The helper writes only under the selected root and refuses path-traversal run
IDs. It does not inspect secrets, Hermes config, or runtime state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_run_id(run_id: str) -> str:
    value = str(run_id).strip()
    if not value or value in {".", ".."}:
        raise ValueError("run-id must be a non-empty slug")
    if "/" in value or "\\" in value:
        raise ValueError("run-id must not contain path separators")
    if not _RUN_ID_RE.fullmatch(value):
        raise ValueError("run-id must start with an alphanumeric character and contain only alnum, dot, underscore, or dash")
    return value


def _default_hermes_home() -> Path:
    if home := os.environ.get("HERMES_HOME"):
        return Path(home)
    if os.name == "nt" and (local_app_data := os.environ.get("LOCALAPPDATA")):
        return Path(local_app_data) / "hermes"
    return Path.home() / ".hermes"


def default_root() -> Path:
    return _default_hermes_home() / "reports" / "ulw-loop"


def _is_reparse_point(path: Path) -> bool:
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _ensure_safe_existing_path(path: Path, label: str, *, reject_hardlinks: bool = False) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if _is_reparse_point(path):
        raise ValueError(f"{label} must not be a reparse point or junction: {path}")
    if reject_hardlinks and path.exists() and not path.is_dir():
        if os.stat(path, follow_symlinks=False).st_nlink > 1:
            raise ValueError(f"{label} must not be a hardlink: {path}")


def _target_under_root(root: Path | str, run_id: str) -> tuple[Path, Path]:
    clean_run_id = validate_run_id(run_id)
    root_path = Path(root).expanduser().resolve()
    requested_target = root_path / clean_run_id
    _ensure_safe_existing_path(requested_target, "run directory")
    resolved_target = requested_target.resolve()
    try:
        resolved_target.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("target path must stay under the selected root") from exc
    return root_path, requested_target


def build_scaffold_plan(
    *,
    root: Path | str,
    run_id: str,
    goal: str,
    owner: str = "HQ",
) -> dict[str, Any]:
    root_path, target = _target_under_root(root, run_id)
    created_at = _utc_now()
    return {
        "run_id": validate_run_id(run_id),
        "goal": goal,
        "owner": owner,
        "status": "planned",
        "created_at": created_at,
        "root": str(root_path),
        "run_dir": str(target),
        "files": {
            "brief": str(target / "brief.md"),
            "goals": str(target / "goals.json"),
            "ledger": str(target / "ledger.jsonl"),
            "evidence_readme": str(target / "evidence" / "README.md"),
        },
    }


def _brief_markdown(plan: dict[str, Any]) -> str:
    return f"""# {plan['run_id']}

Goal: {plan['goal']}
Mode: ulw-loop
Owner: {plan['owner']}
Status: planned
Created: {plan['created_at']}

## Scope

Describe the approved scope before execution starts.

## Stop conditions

- Stop on system-enforced blocks, raw secret exposure risk, an explicit user stop, or an external action outside the approved task contract.
- For approved destructive/admin, restart, external, or paid/network work: continue within scope while recording impact, mitigation, recovery path, and evidence.
- Pause only when success criteria, scope, budget, priority, or evidence paths are materially ambiguous.

## Evidence rule

Every worker/reviewer claim must include an evidence path plus command output/readback.
"""


def _goals_json(plan: dict[str, Any]) -> str:
    payload = {
        "run_id": plan["run_id"],
        "goal": plan["goal"],
        "owner": plan["owner"],
        "status": "planned",
        "created_at": plan["created_at"],
        "criteria": [
            {
                "id": "criterion-1",
                "description": "Define concrete acceptance criteria before marking the run complete.",
                "status": "pending",
                "evidence_paths": [],
            }
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _ledger_jsonl(plan: dict[str, Any]) -> str:
    event = {
        "event": "created",
        "run_id": plan["run_id"],
        "status": "planned",
        "created_at": plan["created_at"],
        "evidence_root": str(Path(plan["run_dir"]) / "evidence"),
    }
    return json.dumps(event, sort_keys=True) + "\n"


def _evidence_readme(plan: dict[str, Any]) -> str:
    return f"""# Evidence for {plan['run_id']}

Store command outputs, file readbacks, reviewer artifacts, screenshots, QA receipts, and cleanup receipts here.

Rules:

- Do not store raw secrets or credential files.
- Prefer sanitized stdout/stderr excerpts with command, cwd, timestamp, and exit code.
- Treat subagent summaries as claims until backed by a concrete artifact or readback.
"""


def _ensure_safe_scaffold_targets(plan: dict[str, Any]) -> None:
    run_dir = Path(plan["run_dir"])
    evidence_dir = run_dir / "evidence"

    _ensure_safe_existing_path(run_dir, "run directory")
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError(f"run directory path exists but is not a directory: {run_dir}")

    _ensure_safe_existing_path(evidence_dir, "evidence directory")
    if evidence_dir.exists() and not evidence_dir.is_dir():
        raise ValueError(f"evidence path exists but is not a directory: {evidence_dir}")

    for name, raw_path in plan["files"].items():
        file_path = Path(raw_path)
        _ensure_safe_existing_path(file_path, f"scaffold file {name}", reject_hardlinks=True)
        if file_path.exists() and file_path.is_dir():
            raise ValueError(f"scaffold file path exists but is a directory: {file_path}")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_scaffold(
    *,
    root: Path | str | None = None,
    run_id: str,
    goal: str,
    owner: str = "HQ",
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    selected_root = default_root() if root is None else Path(root)
    plan = build_scaffold_plan(root=selected_root, run_id=run_id, goal=goal, owner=owner)
    plan["dry_run"] = dry_run
    plan["created"] = False

    if dry_run:
        return plan

    run_dir = Path(plan["run_dir"])
    if run_dir.exists() and not force:
        raise FileExistsError(f"run directory already exists: {run_dir}")

    _ensure_safe_scaffold_targets(plan)

    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _ensure_safe_scaffold_targets(plan)

    _atomic_write_text(Path(plan["files"]["brief"]), _brief_markdown(plan))
    _atomic_write_text(Path(plan["files"]["goals"]), _goals_json(plan))
    _atomic_write_text(Path(plan["files"]["ledger"]), _ledger_jsonl(plan))
    _atomic_write_text(Path(plan["files"]["evidence_readme"]), _evidence_readme(plan))

    plan["created"] = True
    return plan


def _format_text(result: dict[str, Any]) -> str:
    state = "DRY RUN" if result.get("dry_run") else "CREATED"
    lines = [f"{state}: {result['run_id']}", f"Run dir: {result['run_dir']}", "Files:"]
    lines.extend(f"- {name}: {path}" for name, path in result["files"].items())
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a ULW-loop evidence ledger scaffold.")
    parser.add_argument(
        "--root",
        type=Path,
        help=(
            "Root directory for run ledgers (default: HERMES_HOME/reports/ulw-loop; "
            "if unset, LOCALAPPDATA/hermes on Windows or ~/.hermes elsewhere)"
        ),
    )
    parser.add_argument("--run-id", required=True, help="Run slug, e.g. hq-task-20260629")
    parser.add_argument("--goal", required=True, help="One-sentence run goal")
    parser.add_argument("--owner", default="HQ")
    parser.add_argument("--dry-run", action="store_true", help="Print intended paths without writing files")
    parser.add_argument("--force", action="store_true", help="Overwrite scaffold files if the run directory already exists")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = create_scaffold(
            root=args.root,
            run_id=args.run_id,
            goal=args.goal,
            owner=args.owner,
            dry_run=args.dry_run,
            force=args.force,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        print(str(exc), file=os.sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(_format_text(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
