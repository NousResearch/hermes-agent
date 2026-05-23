#!/usr/bin/env python3
"""Validate the native jcode Tool scaffold for Hermes-backed capabilities."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NATIVE_TOOL_DIR = ROOT / "bridges" / "jcode-native-hermes-tool"


def _check(name: str, ok: bool, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "ok": bool(ok)}
    result.update(details)
    return result


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _default_jcode_path() -> Path:
    in_scaffold = ROOT / "upstreams" / "jcode"
    if in_scaffold.exists():
        return in_scaffold
    return ROOT / ".codex-research" / "jcode"


def _copy_native_tool(destination: Path) -> None:
    shutil.copytree(
        NATIVE_TOOL_DIR,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("target", "__pycache__", "*.pyc"),
    )


def _prepare_workspace(jcode_path: Path) -> tuple[tempfile.TemporaryDirectory[str] | None, Path]:
    in_place_jcode = ROOT / "upstreams" / "jcode"
    if in_place_jcode.exists():
        return None, ROOT

    temp = tempfile.TemporaryDirectory(prefix="jcode-native-hermes-tool-")
    workspace = Path(temp.name)
    bridge_dir = workspace / "bridges" / "jcode-native-hermes-tool"
    upstreams_dir = workspace / "upstreams"
    upstreams_dir.mkdir(parents=True, exist_ok=True)
    _copy_native_tool(bridge_dir)
    (upstreams_dir / "jcode").symlink_to(jcode_path.resolve(), target_is_directory=True)
    return temp, workspace


def _source_checks(native_dir: Path) -> list[dict[str, Any]]:
    cargo_toml = native_dir / "Cargo.toml"
    lib_rs = native_dir / "src" / "lib.rs"
    cargo_text = cargo_toml.read_text(encoding="utf-8") if cargo_toml.exists() else ""
    lib_text = lib_rs.read_text(encoding="utf-8") if lib_rs.exists() else ""
    return [
        _check("native_tool:cargo_toml_exists", cargo_toml.exists(), path=str(cargo_toml)),
        _check("native_tool:lib_rs_exists", lib_rs.exists(), path=str(lib_rs)),
        _check(
            "native_tool:uses_jcode_tool_core",
            "jcode-tool-core" in cargo_text and "jcode_tool_core" in lib_text,
        ),
        _check(
            "native_tool:implements_tool_trait",
            "impl Tool for HermesNativeTool" in lib_text,
        ),
        _check(
            "native_tool:defines_hermes_research_tools",
            "hermes_web_search" in lib_text and "hermes_web_extract" in lib_text,
        ),
        _check(
            "native_tool:defines_hermes_state_tools",
            "hermes_session_search" in lib_text and "hermes_memory" in lib_text,
        ),
        _check(
            "native_tool:exports_default_toolset",
            "default_hermes_toolset" in lib_text,
        ),
    ]


def check_native_tool(jcode_path: Path, *, cargo: bool, target_dir: Path | None) -> dict[str, Any]:
    jcode_path = jcode_path.expanduser().resolve()
    checks = _source_checks(NATIVE_TOOL_DIR)
    checks.append(_check("jcode_checkout:exists", jcode_path.is_dir(), path=str(jcode_path)))
    if not jcode_path.is_dir():
        return {
            "success": False,
            "checks": checks,
            "native_tool_dir": str(NATIVE_TOOL_DIR),
            "jcode_path": str(jcode_path),
        }

    temp: tempfile.TemporaryDirectory[str] | None = None
    try:
        temp, workspace = _prepare_workspace(jcode_path)
        manifest = workspace / "bridges" / "jcode-native-hermes-tool" / "Cargo.toml"
        checks.append(_check("workspace:manifest_exists", manifest.exists(), path=str(manifest)))

        if cargo:
            env = os.environ.copy()
            if target_dir is None:
                target_dir = Path(tempfile.gettempdir()) / "jcode-native-hermes-tool-check-target"
            env["CARGO_TARGET_DIR"] = str(target_dir)
            completed = _run(
                ["cargo", "check", "--manifest-path", str(manifest)],
                cwd=workspace,
                env=env,
            )
            checks.append(_check(
                "cargo:check",
                completed.returncode == 0,
                returncode=completed.returncode,
                stdout=completed.stdout[-4000:],
                stderr=completed.stderr[-4000:],
                target_dir=str(target_dir),
            ))
    finally:
        if temp is not None:
            temp.cleanup()

    return {
        "success": all(item.get("ok") for item in checks),
        "checks": checks,
        "native_tool_dir": str(NATIVE_TOOL_DIR),
        "jcode_path": str(jcode_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jcode", default=str(_default_jcode_path()), help="Path to a jcode checkout.")
    parser.add_argument("--skip-cargo", action="store_true", help="Only run source/layout checks.")
    parser.add_argument("--target-dir", help="Cargo target directory for native-tool checks.")
    ns = parser.parse_args(argv)

    report = check_native_tool(
        Path(ns.jcode),
        cargo=not ns.skip_cargo,
        target_dir=Path(ns.target_dir).expanduser().resolve() if ns.target_dir else None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
