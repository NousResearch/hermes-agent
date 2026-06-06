#!/usr/bin/env python3
"""Safe remediation helper for workspace config audit findings.

This script applies only reversible `.gitignore` protection and writes central
reports with env key names. It does not read, copy, or print secret values.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


MANAGED_START = "# hermes-managed:start workspace-config"
MANAGED_END = "# hermes-managed:end workspace-config"
MANAGED_BLOCK_LINES = [
    MANAGED_START,
    "# Keep real local secrets out of git. Edit project-specific rules outside this block.",
    ".env",
    ".env.*",
    "!.env.example",
    "!.env.sample",
    "!.env.template",
    "!.env.defaults",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "secrets/",
    MANAGED_END,
]
MANAGED_BLOCK = "\n".join(MANAGED_BLOCK_LINES) + "\n"


def _load_audit_module():
    path = Path(__file__).resolve().with_name("workspace_config_audit.py")
    spec = importlib.util.spec_from_file_location("_workspace_config_audit_runtime", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load audit module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _replace_managed_block(existing: str) -> tuple[str, bool]:
    pattern = re.compile(
        rf"{re.escape(MANAGED_START)}.*?{re.escape(MANAGED_END)}\n?",
        re.DOTALL,
    )
    if pattern.search(existing):
        updated = pattern.sub(MANAGED_BLOCK, existing)
        return updated, updated != existing

    prefix = existing.rstrip("\n")
    if prefix:
        return prefix + "\n\n" + MANAGED_BLOCK, True
    return MANAGED_BLOCK, True


def _needs_gitignore_write(project_report: dict[str, Any]) -> bool:
    return any(
        finding["code"] in {"gitignore-env-missing", "tracked-env-file"}
        for finding in project_report["findings"]
    )


def _write_reports(workspace_report: dict[str, Any], result: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    registry: dict[str, Any] = {
        "root": workspace_report["root"],
        "projects": [],
    }
    for project in workspace_report["projects"]:
        registry["projects"].append(
            {
                "project": project["project"],
                "env_keys": project["env_keys"],
                "example_keys": project["example_keys"],
                "code_refs": project["code_refs"],
                "findings": [
                    {
                        "severity": finding["severity"],
                        "code": finding["code"],
                        "evidence": finding["evidence"],
                    }
                    for finding in project["findings"]
                ],
            }
        )

    _write_text(report_dir / "summary.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    _write_text(report_dir / "env-registry.json", json.dumps(registry, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Workspace Config Remediation Report",
        "",
        f"Root: `{workspace_report['root']}`",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Projects | {workspace_report['summary']['projects']} |",
        f"| Projects with findings | {workspace_report['summary']['projects_with_findings']} |",
        f"| Total findings | {workspace_report['summary']['findings']} |",
        f"| Planned `.gitignore` writes | {result['summary']['planned_gitignore_writes']} |",
        f"| Applied `.gitignore` writes | {result['summary']['applied_gitignore_writes']} |",
        "",
        "## Project Actions",
        "",
        "| Project | Action | Findings |",
        "|---|---|---:|",
    ]
    for action in result["actions"]:
        lines.append(
            f"| {Path(action['project']).name} | {action['action']} | {action['findings']} |"
        )
    _write_text(report_dir / "remediation-report.md", "\n".join(lines) + "\n")


def remediate_workspace(
    root: str | Path,
    *,
    apply: bool,
    report_dir: str | Path | None,
    max_depth: int = 4,
) -> dict[str, Any]:
    audit = _load_audit_module()
    workspace_report = audit.inspect_workspace(root, max_depth=max_depth)
    actions: list[dict[str, Any]] = []
    planned_gitignore_writes = 0
    applied_gitignore_writes = 0

    for project in workspace_report["projects"]:
        action = "none"
        if _needs_gitignore_write(project):
            planned_gitignore_writes += 1
            action = "plan-gitignore-managed-block"
            gitignore = Path(project["project"]) / ".gitignore"
            updated, changed = _replace_managed_block(_read_text(gitignore))
            if apply and changed:
                _write_text(gitignore, updated)
                applied_gitignore_writes += 1
                action = "applied-gitignore-managed-block"
        actions.append(
            {
                "project": project["project"],
                "action": action,
                "findings": project["summary"]["findings"],
            }
        )

    result = {
        "root": workspace_report["root"],
        "apply": apply,
        "summary": {
            "projects": workspace_report["summary"]["projects"],
            "projects_with_findings": workspace_report["summary"]["projects_with_findings"],
            "findings": workspace_report["summary"]["findings"],
            "planned_gitignore_writes": planned_gitignore_writes,
            "applied_gitignore_writes": applied_gitignore_writes,
        },
        "actions": actions,
    }

    if report_dir is not None:
        _write_reports(workspace_report, result, Path(report_dir).expanduser().resolve())

    return result


def render_result(result: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(result, ensure_ascii=False, indent=2) + "\n"

    lines = [
        "Workspace config remediation",
        f"Root: {result['root']}",
        f"Mode: {'apply' if result['apply'] else 'dry-run'}",
        (
            "Summary: "
            f"{result['summary']['projects']} project(s), "
            f"{result['summary']['planned_gitignore_writes']} planned `.gitignore` write(s), "
            f"{result['summary']['applied_gitignore_writes']} applied"
        ),
        "",
    ]
    for action in result["actions"]:
        if action["action"] == "none":
            continue
        lines.append(f"- {Path(action['project']).name}: {action['action']}")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Workspace or project root.")
    parser.add_argument("--max-depth", type=int, default=4, help="Project discovery depth.")
    parser.add_argument("--apply", action="store_true", help="Write managed `.gitignore` blocks.")
    parser.add_argument("--report-dir", help="Directory for non-secret reports.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    args = parser.parse_args(argv)

    result = remediate_workspace(
        args.root,
        apply=args.apply,
        report_dir=args.report_dir,
        max_depth=args.max_depth,
    )
    print(render_result(result, args.format), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
