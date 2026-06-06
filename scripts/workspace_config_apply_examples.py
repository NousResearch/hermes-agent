#!/usr/bin/env python3
"""Apply managed `.env.example` blocks from a central workspace env schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MANAGED_START = "# hermes-managed:start env-example"
MANAGED_END = "# hermes-managed:end env-example"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _managed_block(keys: list[str]) -> str:
    if not keys:
        return ""
    lines = [
        MANAGED_START,
        "# Generated from workspace env schema. Values are intentionally blank.",
    ]
    lines.extend(f"{key}=" for key in sorted(dict.fromkeys(keys)))
    lines.append(MANAGED_END)
    return "\n".join(lines) + "\n"


def _existing_managed_keys(existing: str) -> list[str]:
    pattern = re.compile(
        rf"{re.escape(MANAGED_START)}(.*?){re.escape(MANAGED_END)}",
        re.DOTALL,
    )
    match = pattern.search(existing)
    if not match:
        return []
    keys = []
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def _replace_block(existing: str, keys: list[str]) -> tuple[str, bool]:
    if not keys:
        return existing, False
    merged_keys = sorted(dict.fromkeys([*_existing_managed_keys(existing), *keys]))
    block = _managed_block(merged_keys)
    if not merged_keys:
        return existing, False
    pattern = re.compile(
        rf"{re.escape(MANAGED_START)}.*?{re.escape(MANAGED_END)}\n?",
        re.DOTALL,
    )
    if pattern.search(existing):
        updated = pattern.sub(block, existing)
        return updated, updated != existing
    prefix = existing.rstrip("\n")
    if prefix:
        return prefix + "\n\n" + block, True
    return block, True


def _eligible_keys(project: dict[str, Any]) -> list[str]:
    keys = []
    existing = {
        entry["key"]
        for entry in project.get("entries", [])
        if entry.get("status") == "present_in_example"
    }
    for entry in project.get("entries", []):
        key = entry.get("key", "")
        if not key or key in existing:
            continue
        if entry.get("status") != "missing_from_example":
            continue
        keys.append(key)
    return sorted(dict.fromkeys(keys))


def apply_examples(
    schema_path: str | Path,
    *,
    apply: bool,
    report_dir: str | Path | None,
) -> dict[str, Any]:
    schema_file = Path(schema_path).expanduser().resolve()
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    actions: list[dict[str, Any]] = []
    planned = 0
    applied = 0
    total_keys = 0

    for project in schema.get("projects", []):
        project_path = Path(project["project"])
        keys = _eligible_keys(project)
        total_keys += len(keys)
        env_example = project_path / ".env.example"
        updated, changed = _replace_block(_read_text(env_example), keys)
        action = "none"
        if changed:
            planned += 1
            action = "plan-env-example-managed-block"
            if apply:
                _write_text(env_example, updated)
                applied += 1
                action = "applied-env-example-managed-block"
        actions.append(
            {
                "project": str(project_path),
                "env_example": str(env_example),
                "action": action,
                "keys": keys,
                "key_count": len(keys),
            }
        )

    result = {
        "schema": str(schema_file),
        "root": schema.get("root", ""),
        "apply": apply,
        "summary": {
            "projects": len(schema.get("projects", [])),
            "managed_keys": total_keys,
            "planned_env_example_writes": planned,
            "applied_env_example_writes": applied,
        },
        "actions": actions,
    }
    if report_dir is not None:
        _write_reports(result, Path(report_dir).expanduser().resolve())
    return result


def _write_reports(result: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        report_dir / "env-example-remediation.json",
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# Env Example Remediation Report",
        "",
        f"Schema: `{result['schema']}`",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Projects | {result['summary']['projects']} |",
        f"| Managed keys | {result['summary']['managed_keys']} |",
        f"| Planned writes | {result['summary']['planned_env_example_writes']} |",
        f"| Applied writes | {result['summary']['applied_env_example_writes']} |",
        "",
        "| Project | Action | Keys |",
        "|---|---|---:|",
    ]
    for action in result["actions"]:
        lines.append(f"| {Path(action['project']).name} | {action['action']} | {action['key_count']} |")
    _write_text(report_dir / "env-example-remediation.md", "\n".join(lines) + "\n")


def render_result(result: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    lines = [
        "Workspace env example remediation",
        f"Schema: {result['schema']}",
        f"Mode: {'apply' if result['apply'] else 'dry-run'}",
        (
            "Summary: "
            f"{result['summary']['projects']} project(s), "
            f"{result['summary']['managed_keys']} managed key(s), "
            f"{result['summary']['planned_env_example_writes']} planned write(s), "
            f"{result['summary']['applied_env_example_writes']} applied"
        ),
        "",
    ]
    for action in result["actions"]:
        if action["action"] == "none":
            continue
        lines.append(f"- {Path(action['project']).name}: {action['action']} ({action['key_count']} key(s))")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True, help="Path to env-schema.json.")
    parser.add_argument("--apply", action="store_true", help="Write `.env.example` managed blocks.")
    parser.add_argument("--report-dir", help="Directory for non-secret reports.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    args = parser.parse_args(argv)

    result = apply_examples(args.schema, apply=args.apply, report_dir=args.report_dir)
    print(render_result(result, args.format), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
