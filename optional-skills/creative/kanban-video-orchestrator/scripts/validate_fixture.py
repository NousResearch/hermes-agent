#!/usr/bin/env python3
"""Validate the kanban-video-orchestrator skill against a deterministic fixture.

The validator intentionally avoids calling the Hermes CLI or creating real
profiles. It exercises the skill-local generator with a known plan, checks the
rendered setup/brief/team artifacts for safety-critical invariants, and writes a
small reproducible artifact bundle that reviewers can inspect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = SKILL_DIR / "fixtures" / "sample-plan.json"
DEFAULT_OUT_DIR = SKILL_DIR / "samples" / "fixture-run"
BOOTSTRAP = SKILL_DIR / "scripts" / "bootstrap_pipeline.py"
VALIDATOR_MARKER = ".kanban-video-orchestrator-validator"
REQUIRED_SNIPPETS = {
    "setup.sh": [
        "set -euo pipefail",
        "hermes kanban create \"Direct production of Fixture Product Teaser\"",
        "--workspace dir:\"$WORKSPACE\"",
        "--tenant \"$TENANT\"",
        "configure_profile 'director' '[\"kanban\", \"terminal\", \"file\"]' '[\"kanban-orchestrator\"]'",
        "workspace_kind=\"dir\"",
        "workspace_path=\"$WORKSPACE\"",
    ],
    "brief.md": [
        "# Video Brief — Fixture Product Teaser",
        "fixture-product-teaser",
        "ASCII title card: Fixture Product Teaser",
        "Product silhouette resolves into call to action",
    ],
    "TEAM.md": [
        "# Team & Task Graph — Fixture Product Teaser",
        "T0  director — decompose",
        "ascii-renderer — scene 1",
        "reviewer — final QA",
        "tenant=\"fixture-product-teaser\"",
    ],
}
FORBIDDEN_PATTERNS = [
    re.compile(r"approvals\s*[:=]"),
    re.compile(r"--yolo\b"),
    re.compile(r"rm\s+-rf\s+[/~$]"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_generator(plan: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "setup.sh": out_dir / "setup.sh",
        "brief.md": out_dir / "brief.md",
        "TEAM.md": out_dir / "TEAM.md",
    }
    cmd = [
        sys.executable,
        str(BOOTSTRAP),
        str(plan),
        "--out",
        str(outputs["setup.sh"]),
        "--brief-out",
        str(outputs["brief.md"]),
        "--team-out",
        str(outputs["TEAM.md"]),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "generator failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return outputs


def validate_outputs(outputs: dict[str, Path]) -> list[str]:
    failures: list[str] = []
    for name, snippets in REQUIRED_SNIPPETS.items():
        path = outputs[name]
        text = path.read_text()
        for snippet in snippets:
            if snippet not in text:
                failures.append(f"{name}: missing required snippet {snippet!r}")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                failures.append(f"{name}: forbidden pattern matched {pattern.pattern!r}")
        if "{{" in text or "}}" in text:
            failures.append(f"{name}: unresolved template marker remains")
    setup_mode = outputs["setup.sh"].stat().st_mode & 0o777
    if setup_mode != 0o755:
        failures.append(f"setup.sh: expected mode 0755, got {setup_mode:o}")
    return failures


def build_manifest(plan: Path, outputs: dict[str, Path], validation_log: Path) -> dict:
    artifacts = {
        name: {
            "path": str(path.relative_to(validation_log.parent)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in sorted(outputs.items())
    }
    artifacts["validation.log"] = {
        "path": validation_log.name,
        "bytes": validation_log.stat().st_size,
        "sha256": sha256(validation_log),
    }
    try:
        plan_ref = str(plan.relative_to(SKILL_DIR))
    except ValueError:
        plan_ref = str(plan)
    return {
        "validator": "kanban-video-orchestrator fixture validator",
        "plan": plan_ref,
        "plan_sha256": sha256(plan),
        "artifacts": artifacts,
    }


def prepare_out_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    if out_dir == DEFAULT_OUT_DIR.resolve() or (out_dir / VALIDATOR_MARKER).is_file():
        shutil.rmtree(out_dir)
        return
    if out_dir.is_dir() and not any(out_dir.iterdir()):
        out_dir.rmdir()
        return
    raise RuntimeError(
        "refusing to replace non-validator-owned output directory: "
        f"{out_dir}\n"
        "Use an absent/empty directory, the default sample directory, or rerun "
        "against a directory containing the validator marker."
    )


def write_marker(out_dir: Path) -> None:
    if out_dir == DEFAULT_OUT_DIR.resolve():
        return
    (out_dir / VALIDATOR_MARKER).write_text(
        "owned by kanban-video-orchestrator fixture validator\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN), help="Fixture plan JSON")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Artifact output directory")
    parser.add_argument("--check-determinism", action="store_true", help="Generate twice and compare hashes")
    args = parser.parse_args()

    plan = Path(args.plan).resolve()
    out_dir = Path(args.out_dir).resolve()
    try:
        prepare_out_dir(out_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    outputs = run_generator(plan, out_dir)
    write_marker(out_dir)
    failures = validate_outputs(outputs)

    if args.check_determinism:
        with tempfile.TemporaryDirectory() as tmp:
            second = run_generator(plan, Path(tmp))
            for name, first_path in outputs.items():
                second_hash = sha256(second[name])
                first_hash = sha256(first_path)
                if first_hash != second_hash:
                    failures.append(f"{name}: nondeterministic hash {first_hash} != {second_hash}")

    log_path = out_dir / "validation.log"
    if failures:
        log_path.write_text("FAIL\n" + "\n".join(f"- {f}" for f in failures) + "\n")
        print(log_path.read_text(), end="", file=sys.stderr)
        return 1

    log_path.write_text(
        "PASS kanban-video-orchestrator fixture validation\n"
        f"plan={plan}\n"
        f"setup_sha256={sha256(outputs['setup.sh'])}\n"
        f"brief_sha256={sha256(outputs['brief.md'])}\n"
        f"team_sha256={sha256(outputs['TEAM.md'])}\n"
    )
    manifest = build_manifest(plan, outputs, log_path)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(log_path.read_text(), end="")
    print(f"manifest={out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
