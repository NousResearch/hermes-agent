"""Workflow launcher for Codex to Claude alignment gates.

The launcher creates a small, file-backed operating shell inside a target
repository. Linear remains a coordination surface; the local `.workflow`
folder is the evidence-bearing source of truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from hermes_constants import get_hermes_home


WORKFLOW_DIRNAME = ".workflow"
EXPECTED_ARTIFACT_FILES = (
    "preview.html",
    "metadata.json",
    "artifact.md",
    "notes.md",
    "thumbnail.png",
)
SOURCE_FILES = ("source.html", "source.jsx", "source.fixed.jsx", "source.tsx")


@dataclass(frozen=True)
class ArtifactRecord:
    slug: str
    title: str
    updated_at: str
    missing: tuple[str, ...]
    source_file: str | None


@dataclass(frozen=True)
class ArtifactInventory:
    repo: Path
    total: int
    with_preview: int
    with_thumbnail: int
    missing_required_count: int
    records: tuple[ArtifactRecord, ...]


@dataclass(frozen=True)
class WorkflowWrite:
    path: Path
    action: str


def default_artifact_repository() -> Path:
    return get_hermes_home() / "artifact-repository"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _read_metadata(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _record_updated_at(artifact_dir: Path, metadata: dict) -> str:
    for key in ("updated_at", "updatedAt", "created_at", "createdAt"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    latest = 0.0
    for child in artifact_dir.iterdir():
        if child.is_file():
            latest = max(latest, child.stat().st_mtime)
    if latest:
        return datetime.fromtimestamp(latest, timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    return "Unknown"


def inventory_artifact_repository(repo: Path, limit: int = 12) -> ArtifactInventory:
    repo = repo.expanduser().resolve()
    artifacts_dir = repo / "artifacts"
    records: list[ArtifactRecord] = []

    if artifacts_dir.exists():
        for artifact_dir in sorted(artifacts_dir.iterdir()):
            if not artifact_dir.is_dir():
                continue
            metadata = _read_metadata(artifact_dir / "metadata.json")
            title = str(metadata.get("title") or metadata.get("name") or artifact_dir.name)
            missing = [
                expected
                for expected in EXPECTED_ARTIFACT_FILES
                if not (artifact_dir / expected).exists()
            ]
            source_file = next(
                (candidate for candidate in SOURCE_FILES if (artifact_dir / candidate).exists()),
                None,
            )
            if source_file is None:
                missing.append("source.html|source.jsx")
            records.append(
                ArtifactRecord(
                    slug=artifact_dir.name,
                    title=title,
                    updated_at=_record_updated_at(artifact_dir, metadata),
                    missing=tuple(missing),
                    source_file=source_file,
                )
            )

    records.sort(key=lambda item: item.updated_at, reverse=True)
    selected = tuple(records[:limit])
    with_preview = sum(1 for record in records if "preview.html" not in record.missing)
    with_thumbnail = sum(1 for record in records if "thumbnail.png" not in record.missing)
    missing_required_count = sum(1 for record in records if record.missing)
    return ArtifactInventory(
        repo=repo,
        total=len(records),
        with_preview=with_preview,
        with_thumbnail=with_thumbnail,
        missing_required_count=missing_required_count,
        records=selected,
    )


def _markdown_table(rows: Iterable[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    rendered = ["| " + " | ".join(headers) + " |"]
    rendered.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        rendered.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    return "\n".join(rendered)


def _render_inventory(inventory: ArtifactInventory) -> str:
    if not inventory.records:
        return "No artifacts were found under `artifacts/`."

    rows = []
    for record in inventory.records:
        rows.append(
            (
                f"`{record.slug}`",
                record.title,
                record.updated_at,
                record.source_file or "Missing",
                ", ".join(record.missing) if record.missing else "None",
            )
        )
    return _markdown_table(
        rows,
        ("Artifact", "Title", "Last Seen", "Source", "Missing"),
    )


def _workflow_files(
    repo: Path,
    workflow_name: str,
    inventory: ArtifactInventory,
    linear_issue: str | None,
    linear_project: str | None,
    claude_gate_note: str | None,
) -> dict[Path, str]:
    generated = _utc_now()
    linear_ref = linear_issue or "Pending"
    linear_project_ref = linear_project or "Pending"
    claude_note = claude_gate_note or (
        "Pending. Run Claude Code in read-only/plan mode before implementation."
    )

    summary = (
        f"- Target repository: `{repo}`\n"
        f"- Workflow name: `{workflow_name}`\n"
        f"- Generated: {generated}\n"
        f"- Linear issue: {linear_ref}\n"
        f"- Linear project: {linear_project_ref}\n"
        f"- Artifact count: {inventory.total}\n"
        f"- Artifacts with preview: {inventory.with_preview}\n"
        f"- Artifacts with thumbnail: {inventory.with_thumbnail}\n"
        f"- Artifacts missing required files: {inventory.missing_required_count}\n"
    )

    return {
        Path("WORKFLOW_STATE.md"): f"""# Workflow State

{summary}

## Gate Status

| Gate | Owner | Status | Evidence |
| --- | --- | --- | --- |
| 1. Scope packet | Codex | Ready | `ARCHITECT_PACK.md` |
| 2. Codex plan | Codex | Pending | `CODEX_PLAN.md` |
| 3. Claude adversarial review | Claude Code | Pending | `CLAUDE_CRITIQUE.md` |
| 4. Reconciliation | Codex | Pending | `RECONCILIATION.md` |
| 5. Alignment decision | Codex + Claude | Pending | `ALIGNMENT_DECISION.md` |
| 6. Build | Codex | Pending | repo diff |
| 7. Verification | Codex | Pending | `VERIFY.md` |
| 8. Linear update | Operator | Pending | `linear/LINEAR_ISSUE_TEMPLATE.md` |

## Operating Rule

Implementation starts only after Codex and Claude have aligned on the best path,
or after the operator explicitly accepts a documented exception.
""",
        Path("ARCHITECT_PACK.md"): f"""# Architect Pack

## Objective

Use `{workflow_name}` as the first repository to test the Codex to Claude
adversarial alignment workflow.

## Repository Snapshot

{summary}

## Artifact Inventory

{_render_inventory(inventory)}

## Source Of Truth Boundary

- Local files in this repository hold evidence, plans, review notes, decisions,
  implementation diffs, and verification logs.
- Linear tracks coordination state, acceptance criteria, blockers, and links.
- Linear should not duplicate generated evidence files or become a second audit
  log.

## First Pilot Question

Can the launcher reliably create the planning, review, reconciliation, and
verification shell before any implementation work starts?
""",
        Path("CODEX_PLAN.md"): """# Codex Plan

## Proposed Path

Pending.

## Acceptance Criteria

- The scope is specific enough for Claude Code to critique.
- The plan names files, commands, expected outputs, and non-goals.
- Risks are explicit before implementation starts.

## Non-Goals

- Pending.
""",
        Path("CLAUDE_CRITIQUE.md"): f"""# Claude Critique

## Gate Status

{claude_note}

## Review Prompt Template

Ask Claude Code to review `ARCHITECT_PACK.md` and `CODEX_PLAN.md` in read-only
or plan mode.

```bash
claude --dangerously-skip-permissions
```

Inside Claude Code, request:

```text
Read-only adversarial review. Do not edit files. Critique CODEX_PLAN.md against
ARCHITECT_PACK.md for architecture risk, missing acceptance criteria, source of
truth drift, and verification gaps. Return required changes before build.
```
""",
        Path("RECONCILIATION.md"): """# Reconciliation

## Claude Objections

Pending.

## Codex Response

Pending.

## Plan Changes

Pending.
""",
        Path("ALIGNMENT_DECISION.md"): """# Alignment Decision

## Decision

Pending.

## Required Before Build

- Codex plan is complete.
- Claude critique has been reviewed.
- Reconciliation notes name accepted and rejected changes.
- Operator exception is documented if Claude review is unavailable.

## Exception Log

None.
""",
        Path("VERIFY.md"): """# Verification

## Commands

Pending.

## Runtime Checks

Pending.

## Artifact Checks

Pending.

## Result

Pending.
""",
        Path("linear/LINEAR_ISSUE_TEMPLATE.md"): f"""# Linear Issue Template

## Title

Workflow Launcher Pilot: {workflow_name}

## Description

Use `{repo}` as the first live repository for the Codex to Claude adversarial
alignment workflow.

## Acceptance Criteria

- `.workflow/WORKFLOW_STATE.md` shows gate status.
- `.workflow/ARCHITECT_PACK.md` captures repository context.
- `.workflow/CODEX_PLAN.md` contains a build plan before code changes.
- `.workflow/CLAUDE_CRITIQUE.md` captures the read-only critique or a documented
  auth/runtime blocker.
- `.workflow/RECONCILIATION.md` records plan changes.
- `.workflow/VERIFY.md` lists executed checks and results.

## Evidence Links

- Local workflow folder: `{repo / WORKFLOW_DIRNAME}`
- Linear project: {linear_project_ref}
- Existing Linear issue: {linear_ref}
""",
    }


def init_workflow(
    repo: Path,
    workflow_name: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    linear_issue: str | None = None,
    linear_project: str | None = None,
    claude_gate_note: str | None = None,
) -> list[WorkflowWrite]:
    repo = repo.expanduser().resolve()
    workflow_name = workflow_name or repo.name
    inventory = inventory_artifact_repository(repo)
    files = _workflow_files(
        repo=repo,
        workflow_name=workflow_name,
        inventory=inventory,
        linear_issue=linear_issue,
        linear_project=linear_project,
        claude_gate_note=claude_gate_note,
    )
    workflow_dir = repo / WORKFLOW_DIRNAME
    writes: list[WorkflowWrite] = []

    for relative_path, content in files.items():
        destination = workflow_dir / relative_path
        if destination.exists() and not force:
            writes.append(WorkflowWrite(destination, "exists"))
            continue
        action = "write" if destination.exists() else "create"
        writes.append(WorkflowWrite(destination, action))
        if dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content.rstrip() + "\n", encoding="utf-8")

    return writes


def inspect_workflow(repo: Path) -> str:
    inventory = inventory_artifact_repository(repo)
    summary = [
        f"Repository: {inventory.repo}",
        f"Artifacts: {inventory.total}",
        f"With preview: {inventory.with_preview}",
        f"With thumbnail: {inventory.with_thumbnail}",
        f"Missing required files: {inventory.missing_required_count}",
        "",
        _render_inventory(inventory),
    ]
    return "\n".join(summary)


def _print_writes(writes: Iterable[WorkflowWrite], dry_run: bool) -> None:
    prefix = "would " if dry_run else ""
    for write in writes:
        if write.action == "exists":
            print(f"exists  {write.path}")
        else:
            print(f"{prefix}{write.action:<6} {write.path}")


def workflow_command(args: argparse.Namespace) -> None:
    repo = Path(args.repo) if args.repo else default_artifact_repository()
    action = getattr(args, "workflow_action", None)
    if action == "inspect":
        print(inspect_workflow(repo))
        return
    if action == "init":
        writes = init_workflow(
            repo=repo,
            workflow_name=args.name,
            dry_run=args.dry_run,
            force=args.force,
            linear_issue=args.linear_issue,
            linear_project=args.linear_project,
            claude_gate_note=args.claude_gate_note,
        )
        _print_writes(writes, args.dry_run)
        return
    raise SystemExit("Missing workflow action. Use `hermes workflow --help`.")


def register_workflow_subparser(subparsers) -> None:
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Create and inspect Codex to Claude workflow gates",
        description=(
            "Create a local .workflow shell for Codex planning, Claude Code "
            "adversarial review, reconciliation, verification, and Linear-ready "
            "coordination."
        ),
    )
    workflow_sub = workflow_parser.add_subparsers(dest="workflow_action")

    inspect_parser = workflow_sub.add_parser(
        "inspect",
        help="Inspect a repository before creating workflow files",
    )
    inspect_parser.add_argument(
        "--repo",
        default=None,
        help="Repository path. Defaults to Hermes artifact repository.",
    )

    init_parser = workflow_sub.add_parser(
        "init",
        help="Create workflow gate files in the target repository",
    )
    init_parser.add_argument(
        "--repo",
        default=None,
        help="Repository path. Defaults to Hermes artifact repository.",
    )
    init_parser.add_argument(
        "--name",
        default=None,
        help="Human-readable workflow name. Defaults to the repository name.",
    )
    init_parser.add_argument(
        "--linear-issue",
        default=None,
        help="Existing Linear issue URL or identifier to include in the brief.",
    )
    init_parser.add_argument(
        "--linear-project",
        default=None,
        help="Linear project name or URL to include in the brief.",
    )
    init_parser.add_argument(
        "--claude-gate-note",
        default=None,
        help="Current Claude gate status or blocker note.",
    )
    init_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned writes without creating files.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing workflow files.",
    )
    workflow_parser.set_defaults(func=workflow_command)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hermes workflow launcher")
    subparsers = parser.add_subparsers(dest="command")
    register_workflow_subparser(subparsers)
    parsed = parser.parse_args(argv)
    if not hasattr(parsed, "func"):
        parser.print_help()
        return
    parsed.func(parsed)


if __name__ == "__main__":
    main()
