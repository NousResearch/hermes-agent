#!/usr/bin/env python3
"""Scaffold a governed OpenClaw skill package (SYS-04 foundation)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_ROOT = REPO_ROOT / "memory" / "skills"
SCHEMA_PATH = REPO_ROOT / "docs" / "ops" / "schemas" / "skill_package_manifest.schema.json"


def _slugify(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^a-z0-9_\-]", "", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_manifest(
    *,
    skill_id: str,
    display_name: str,
    summary: str,
    owner: str,
    intents: list[str],
    keywords: list[str],
    tools: list[str],
    execution_mode: str,
    offload_policy: str,
    risk_class: str,
    root_dir: Path,
) -> dict[str, Any]:
    skill_doc = root_dir / "SKILL.md"
    scripts_dir = root_dir / "scripts"
    references_dir = root_dir / "references"

    benchmark_path = references_dir / "benchmark_plan.md"

    return {
        "schema_version": "openclaw.skill_package_manifest.v1",
        "skill_id": skill_id,
        "version": "0.1.0",
        "display_name": display_name,
        "summary": summary,
        "owner": owner,
        "updated_at": _now_utc(),
        "package": {
            "root_dir": _to_repo_path(root_dir),
            "skill_doc": _to_repo_path(skill_doc),
            "scripts_dir": _to_repo_path(scripts_dir),
            "references_dir": _to_repo_path(references_dir),
            "assets": [_to_repo_path(skill_doc)],
        },
        "activation": {
            "intents": intents,
            "keywords": keywords,
            "requires_explicit_invocation": False,
            "default_priority": "P2",
        },
        "runtime": {
            "execution_mode": execution_mode,
            "offload_policy": offload_policy,
            "max_runtime_seconds": 300,
            "preflight": {
                "min_ram_gb": 1,
                "min_disk_gb": 1,
                "requires_gpu": False,
            },
        },
        "interoperability": {
            "import_formats": ["openclaw-native"],
            "export_formats": ["openclaw-native"],
        },
        "quality": {
            "benchmark_suite": _to_repo_path(benchmark_path),
            "contract_tests": ["tests/test_sys_04_skill_packaging_schema_pack.py"],
        },
        "governance": {
            "risk_class": risk_class,
            "advisory_only": True,
            "approval_policy": "risk_based",
            "allowed_tools": sorted(set(tools)),
            "requires_human_approval_for": ["file_write", "shell_exec"],
        },
        "provenance": {
            "source_refs": [
                {
                    "ref_id": "SYS04_SPEC",
                    "kind": "spec",
                    "location": "docs/ops/skill_packaging_standard_contract_v1.md",
                    "confidence": 0.75,
                }
            ]
        },
    }


def _build_skill_md(manifest: dict[str, Any]) -> str:
    activation = manifest["activation"]
    runtime = manifest["runtime"]
    governance = manifest["governance"]

    frontmatter = [
        "---",
        f"skill_id: {manifest['skill_id']}",
        f"version: {manifest['version']}",
        f"display_name: {manifest['display_name']}",
        f"summary: {manifest['summary']}",
        "activation_intents:",
    ]
    frontmatter.extend([f"  - {x}" for x in activation["intents"]])
    frontmatter.append("keywords:")
    frontmatter.extend([f"  - {x}" for x in activation["keywords"]])
    frontmatter.extend(
        [
            f"execution_mode: {runtime['execution_mode']}",
            f"offload_policy: {runtime['offload_policy']}",
            f"risk_class: {governance['risk_class']}",
            f"advisory_only: {str(governance['advisory_only']).lower()}",
            f"owner: {manifest['owner']}",
            f"manifest_path: {manifest['package']['root_dir']}/skill.package.json",
            "---",
            "",
            f"# {manifest['display_name']}",
            "",
            "## When to use",
        ]
    )
    frontmatter.extend([f"- {x}" for x in activation["intents"]])
    frontmatter.extend(
        [
            "",
            "## Package layout",
            "- `SKILL.md`",
            "- `skill.package.json`",
            "- `scripts/`",
            "- `references/`",
            "",
            "## Notes",
            "- Fill in task-specific instructions and examples.",
            "- Keep runtime/governance metadata synchronized with `skill.package.json`.",
            "",
        ]
    )
    return "\n".join(frontmatter)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaffold OpenClaw governed skill package")
    parser.add_argument("--skill-id", required=True, help="skill identifier (slug)")
    parser.add_argument("--display-name", default=None, help="human-readable skill name")
    parser.add_argument("--summary", required=True, help="single-line capability summary")
    parser.add_argument("--owner", default="architect", help="owner label")
    parser.add_argument("--intent", action="append", default=[], help="activation intent (repeatable)")
    parser.add_argument("--keyword", action="append", default=[], help="activation keyword (repeatable)")
    parser.add_argument("--tool", action="append", default=[], help="allowed tool name (repeatable)")
    parser.add_argument(
        "--execution-mode",
        default="inline",
        choices=["inline", "worker_offload_preferred", "worker_offload_required"],
    )
    parser.add_argument(
        "--offload-policy",
        default="prefer_worker_for_large_batches",
        choices=["never_offload", "prefer_worker_for_large_batches", "always_offload"],
    )
    parser.add_argument(
        "--risk-class",
        default="RG1_MODERATE",
        choices=["RG0_LOW", "RG1_MODERATE", "RG2_HIGH", "RG3_CRITICAL"],
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="skill root directory")
    parser.add_argument("--force", action="store_true", help="overwrite existing skill directory")
    parser.add_argument("--json", action="store_true", help="output machine-readable result")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    skill_id = _slugify(args.skill_id)
    if not re.fullmatch(r"[a-z][a-z0-9_\-]{2,62}", skill_id or ""):
        print("ERROR: invalid --skill-id after normalization", file=sys.stderr)
        return 2

    display_name = args.display_name or skill_id.replace("_", " ").replace("-", " ").title()
    intents = list(dict.fromkeys(args.intent or [f"use {display_name}"]))
    keywords = list(dict.fromkeys(args.keyword or [skill_id.split("_")[0]]))
    tools = list(dict.fromkeys(args.tool or ["read"]))

    root_base = Path(args.root).expanduser().resolve()
    skill_dir = root_base / skill_id

    if skill_dir.exists() and not args.force:
        print(f"ERROR: skill directory already exists: {skill_dir}", file=sys.stderr)
        return 2

    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(
        skill_id=skill_id,
        display_name=display_name,
        summary=args.summary,
        owner=args.owner,
        intents=intents,
        keywords=keywords,
        tools=tools,
        execution_mode=args.execution_mode,
        offload_policy=args.offload_policy,
        risk_class=args.risk_class,
        root_dir=skill_dir,
    )

    _write_text(skill_dir / "SKILL.md", _build_skill_md(manifest))
    _write_text(skill_dir / "skill.package.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _write_text(skill_dir / "scripts" / "README.md", "# Scripts\n\nAdd skill-local helper scripts here.\n")
    _write_text(skill_dir / "references" / "README.md", "# References\n\nAdd citations, benchmark notes, and design docs here.\n")
    _write_text(
        skill_dir / "references" / "benchmark_plan.md",
        "# Benchmark Plan\n\n- Define objective quality and latency checks for this skill.\n",
    )

    # Optional local schema validation when jsonschema is available.
    schema_validation = "skipped"
    try:
        import jsonschema  # type: ignore

        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.validate(manifest, schema)
        schema_validation = "passed"
    except Exception:
        schema_validation = "skipped_or_failed"

    payload = {
        "ok": True,
        "skill_id": skill_id,
        "skill_dir": str(skill_dir),
        "manifest_path": str(skill_dir / "skill.package.json"),
        "skill_doc": str(skill_dir / "SKILL.md"),
        "schema_validation": schema_validation,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Scaffolded skill package: {payload['skill_dir']}")
        print(f"- manifest: {payload['manifest_path']}")
        print(f"- skill doc: {payload['skill_doc']}")
        print(f"- schema validation: {schema_validation}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
