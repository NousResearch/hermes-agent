#!/usr/bin/env python3
"""Generate central env schema and remediation queue from workspace audit data."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SECRET_HINTS = ("SECRET", "TOKEN", "KEY", "PASSWORD", "PRIVATE", "CREDENTIAL")


def _load_audit_module():
    path = Path(__file__).resolve().with_name("workspace_config_audit.py")
    spec = importlib.util.spec_from_file_location("_workspace_config_audit_schema", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load audit module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_secret_key(key: str) -> bool:
    return any(hint in key.upper() for hint in SECRET_HINTS)


def _status_for_key(key: str, env_keys: set[str], example_keys: set[str], code_refs: set[str]) -> str:
    if key in example_keys and key in code_refs:
        return "present_in_example"
    if key in code_refs and key not in example_keys:
        return "missing_from_example"
    if key in env_keys and key not in code_refs and key not in example_keys:
        return "unused_env"
    return "review"


def _action_for_status(status: str) -> str:
    return {
        "present_in_example": "keep",
        "missing_from_example": "review_add_to_env_example",
        "unused_env": "review_remove_or_document",
        "review": "review_context",
    }[status]


def _confidence_for_status(status: str) -> str:
    return {
        "present_in_example": "high",
        "missing_from_example": "medium",
        "unused_env": "medium",
        "review": "low",
    }[status]


def _project_schema(project: dict[str, Any]) -> dict[str, Any]:
    env_keys = set(project["env_keys"])
    example_keys = set(project["example_keys"])
    code_refs = set(project["code_refs"])
    all_keys = sorted(env_keys | example_keys | code_refs)
    entries = []
    for key in all_keys:
        status = _status_for_key(key, env_keys, example_keys, code_refs)
        entries.append(
            {
                "key": key,
                "status": status,
                "action": _action_for_status(status),
                "confidence": _confidence_for_status(status),
                "secret": _is_secret_key(key),
                "sources": {
                    "env": key in env_keys,
                    "example": key in example_keys,
                    "code": key in code_refs,
                },
            }
        )
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return {
        "project": project["project"],
        "entries": entries,
        "summary": dict(sorted(counts.items())),
    }


def _write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "env-schema.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Workspace Env Schema",
        "",
        f"Root: `{result['root']}`",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Projects | {result['summary']['projects']} |",
        f"| Keys | {result['summary']['keys']} |",
        f"| Present in example | {result['summary']['present_in_example']} |",
        f"| Missing from example | {result['summary']['missing_from_example']} |",
        f"| Unused env | {result['summary']['unused_env']} |",
        f"| Review | {result['summary']['review']} |",
        "",
    ]
    for project in result["projects"]:
        lines.extend(
            [
                f"## {Path(project['project']).name}",
                "",
                "| Key | Status | Action | Secret | Confidence |",
                "|---|---|---|---|---|",
            ]
        )
        for entry in project["entries"]:
            lines.append(
                f"| `{entry['key']}` | {entry['status']} | {entry['action']} | "
                f"{'yes' if entry['secret'] else 'no'} | {entry['confidence']} |"
            )
        lines.append("")
    (output_dir / "env-schema.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    queue = [
        "# Next Env Example Remediation Queue",
        "",
        "Review these keys before editing real `.env.example` files.",
        "",
        "| Project | Key | Status | Action | Secret | Confidence |",
        "|---|---|---|---|---|---|",
    ]
    for project in result["projects"]:
        project_name = Path(project["project"]).name
        for entry in project["entries"]:
            if entry["status"] == "present_in_example":
                continue
            queue.append(
                f"| {project_name} | `{entry['key']}` | {entry['status']} | {entry['action']} | "
                f"{'yes' if entry['secret'] else 'no'} | {entry['confidence']} |"
            )
    (output_dir / "next-remediation-queue.md").write_text(
        "\n".join(queue).rstrip() + "\n",
        encoding="utf-8",
    )


def generate_workspace_schema(
    root: str | Path,
    *,
    output_dir: str | Path | None,
    max_depth: int = 4,
) -> dict[str, Any]:
    audit = _load_audit_module()
    workspace = audit.inspect_workspace(root, max_depth=max_depth)
    projects = [_project_schema(project) for project in workspace["projects"]]
    totals = {
        "projects": len(projects),
        "keys": 0,
        "present_in_example": 0,
        "missing_from_example": 0,
        "unused_env": 0,
        "review": 0,
    }
    for project in projects:
        totals["keys"] += len(project["entries"])
        for status, count in project["summary"].items():
            totals[status] = totals.get(status, 0) + count

    result = {
        "root": workspace["root"],
        "summary": totals,
        "projects": projects,
    }
    if output_dir is not None:
        _write_outputs(result, Path(output_dir).expanduser().resolve())
    return result


def render_result(result: dict[str, Any], fmt: str = "text") -> str:
    if fmt == "json":
        return json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    return (
        "Workspace env schema\n"
        f"Root: {result['root']}\n"
        f"Projects: {result['summary']['projects']}\n"
        f"Keys: {result['summary']['keys']}\n"
        f"present_in_example: {result['summary']['present_in_example']}\n"
        f"missing_from_example: {result['summary']['missing_from_example']}\n"
        f"unused_env: {result['summary']['unused_env']}\n"
        f"review: {result['summary']['review']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Workspace or project root.")
    parser.add_argument("--max-depth", type=int, default=4, help="Project discovery depth.")
    parser.add_argument("--output-dir", help="Directory to write schema artifacts.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    args = parser.parse_args(argv)

    result = generate_workspace_schema(
        args.root,
        output_dir=args.output_dir,
        max_depth=args.max_depth,
    )
    print(render_result(result, args.format), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
