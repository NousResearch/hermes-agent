#!/usr/bin/env python3
"""Validate the jcode-side native tool registration patch."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JCODE = ROOT / ".codex-research" / "jcode"
DEFAULT_PATCH = ROOT / "patches" / "jcode" / "register-external-toolset.patch"


def _check(name: str, ok: bool, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "ok": bool(ok)}
    result.update(details)
    return result


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def check_registration_patch(jcode_path: Path, patch_path: Path) -> dict[str, Any]:
    jcode_path = jcode_path.expanduser().resolve()
    patch_path = patch_path.expanduser().resolve()

    checks: list[dict[str, Any]] = [
        _check("jcode_checkout:exists", jcode_path.is_dir(), path=str(jcode_path)),
        _check("patch:exists", patch_path.is_file(), path=str(patch_path)),
    ]
    if not jcode_path.is_dir() or not patch_path.is_file():
        return {
            "success": False,
            "checks": checks,
            "jcode_path": str(jcode_path),
            "patch_path": str(patch_path),
        }

    patch_text = patch_path.read_text(encoding="utf-8")
    registry_source = jcode_path / "src" / "tool" / "mod.rs"
    tests_source = jcode_path / "src" / "tool" / "tests.rs"
    registry_text = (
        registry_source.read_text(encoding="utf-8")
        if registry_source.exists()
        else ""
    )
    tests_text = tests_source.read_text(encoding="utf-8") if tests_source.exists() else ""

    checks.extend([
        _check(
            "jcode_registry:has_dynamic_register",
            "pub async fn register(&self, name: String, tool: Arc<dyn Tool>)" in registry_text,
            path=str(registry_source),
        ),
        _check(
            "patch:adds_register_toolset",
            "pub async fn register_toolset" in patch_text
            and "IntoIterator<Item = (String, Arc<dyn Tool>)>" in patch_text,
        ),
        _check(
            "patch:adds_namespace_test",
            "register_toolset_namespaces_external_tools" in patch_text,
        ),
        _check(
            "jcode_tests:expected_anchor_present",
            "tool_definitions_do_not_auto_inject_intent" in tests_text,
            path=str(tests_source),
        ),
    ])

    completed = _run(["git", "apply", "--check", str(patch_path)], cwd=jcode_path)
    checks.append(_check(
        "patch:git_apply_check",
        completed.returncode == 0,
        returncode=completed.returncode,
        stdout=completed.stdout[-4000:],
        stderr=completed.stderr[-4000:],
    ))

    return {
        "success": all(item["ok"] for item in checks),
        "checks": checks,
        "jcode_path": str(jcode_path),
        "patch_path": str(patch_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jcode", default=str(DEFAULT_JCODE), help="Path to a jcode checkout.")
    parser.add_argument("--patch", default=str(DEFAULT_PATCH), help="Patch file to validate.")
    ns = parser.parse_args(argv)

    report = check_registration_patch(Path(ns.jcode), Path(ns.patch))
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
