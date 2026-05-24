#!/usr/bin/env python3
"""MemoryMunch Hermes plugin watchdog.

Default mode is read-only: compare this vendored repo copy against the deployed
runtime plugin under $HERMES_HOME/plugins/memorymunch. Use --repair to copy the
vendored files into the runtime plugin directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import shutil
import sys
from pathlib import Path
from typing import Any

PLUGIN_FILES = ("__init__.py", "original_bridge.py", "readonly_recall.py", "plugin.yaml")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_repo_dir() -> Path:
    return Path(__file__).resolve().parent


def default_runtime_dir() -> Path:
    hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    return hermes_home / "plugins" / "memorymunch"


def compile_check(path: Path) -> dict[str, Any]:
    if path.suffix != ".py":
        return {"ok": True, "skipped": True}
    try:
        py_compile.compile(str(path), doraise=True)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def inspect(repo_dir: Path, runtime_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    missing_repo: list[str] = []
    missing_runtime: list[str] = []
    drift: list[str] = []
    compile_errors: list[str] = []
    for name in PLUGIN_FILES:
        repo_path = repo_dir / name
        runtime_path = runtime_dir / name
        row: dict[str, Any] = {
            "file": name,
            "repo_path": str(repo_path),
            "runtime_path": str(runtime_path),
            "repo_exists": repo_path.exists(),
            "runtime_exists": runtime_path.exists(),
        }
        if not repo_path.exists():
            missing_repo.append(name)
        else:
            row["repo_sha256"] = sha256(repo_path)
            c = compile_check(repo_path)
            row["repo_compile"] = c
            if not c.get("ok"):
                compile_errors.append(f"repo:{name}:{c.get('error')}")
        if not runtime_path.exists():
            missing_runtime.append(name)
        else:
            row["runtime_sha256"] = sha256(runtime_path)
            c = compile_check(runtime_path)
            row["runtime_compile"] = c
            if not c.get("ok"):
                compile_errors.append(f"runtime:{name}:{c.get('error')}")
        if row.get("repo_sha256") and row.get("runtime_sha256") and row["repo_sha256"] != row["runtime_sha256"]:
            drift.append(name)
        files.append(row)
    status = "PASS" if not (missing_repo or missing_runtime or drift or compile_errors) else "FAIL"
    return {
        "status": status,
        "repo_dir": str(repo_dir),
        "runtime_dir": str(runtime_dir),
        "files": files,
        "missing_repo": missing_repo,
        "missing_runtime": missing_runtime,
        "drift": drift,
        "compile_errors": compile_errors,
        "live_db_write": False,
        "live_vault_write": False,
    }


def repair(repo_dir: Path, runtime_dir: Path) -> list[str]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in PLUGIN_FILES:
        src = repo_dir / name
        if not src.exists():
            raise FileNotFoundError(f"repo plugin file missing: {src}")
        dst = runtime_dir / name
        shutil.copy2(src, dst)
        copied.append(name)
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check or repair deployed MemoryMunch Hermes plugin drift.")
    ap.add_argument("--repo-plugin-dir", default=str(default_repo_dir()))
    ap.add_argument("--runtime-plugin-dir", default=str(default_runtime_dir()))
    ap.add_argument("--repair", action="store_true", help="Copy vendored plugin files to runtime plugin dir.")
    ap.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = ap.parse_args(argv)
    repo_dir = Path(args.repo_plugin_dir).expanduser().resolve()
    runtime_dir = Path(args.runtime_plugin_dir).expanduser()
    before = inspect(repo_dir, runtime_dir)
    copied: list[str] = []
    after = before
    if args.repair:
        copied = repair(repo_dir, runtime_dir)
        after = inspect(repo_dir, runtime_dir)
    payload = {"before": before, "repair_copied": copied, "after": after}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"before={before['status']} drift={before['drift']} missing_runtime={before['missing_runtime']}")
        if copied:
            print(f"repair_copied={','.join(copied)}")
            print(f"after={after['status']} drift={after['drift']} missing_runtime={after['missing_runtime']}")
    return 0 if after["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
