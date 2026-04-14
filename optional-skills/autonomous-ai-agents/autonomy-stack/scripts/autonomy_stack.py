#!/usr/bin/env python3
"""Bootstrap a small autonomy plugin stack for Hermes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

EVEY_REPO_URL = "https://github.com/42-evey/hermes-plugins.git"
DEFAULT_PLUGINS = [
    "evey-autonomy",
    "evey-telemetry",
    "evey-status",
    "evey-reflect",
    "evey-learner",
    "evey-goals",
]
ALL_PLUGIN_PREFIX = "evey-"


class AutonomyStackError(RuntimeError):
    """Domain-specific autonomy stack failure."""


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def _plugins_dir() -> Path:
    return _hermes_home() / "plugins"


def _cache_dir() -> Path:
    return _hermes_home() / ".integrations" / "autonomy-stack" / "hermes-plugins"


def _which(binary: str) -> str | None:
    return shutil.which(binary)


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
    )


def _ensure_repo(*, update: bool = False) -> Path:
    repo_dir = _cache_dir()
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists() and (repo_dir / ".git").exists():
        if update:
            _run(["git", "-C", str(repo_dir), "pull", "--ff-only"])
        return repo_dir
    if not _which("git"):
        raise AutonomyStackError("Git is required to clone the autonomy plugin repo")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    _run(["git", "clone", "--depth", "1", EVEY_REPO_URL, str(repo_dir)])
    return repo_dir


def _recommended_plugins(plugin_names: list[str] | None = None) -> list[str]:
    values = plugin_names or DEFAULT_PLUGINS
    cleaned: list[str] = []
    for name in values:
        stripped = name.strip()
        if not stripped:
            continue
        if not stripped.startswith(ALL_PLUGIN_PREFIX):
            raise AutonomyStackError(f"Unsupported plugin name: {name}")
        cleaned.append(stripped)
    if not cleaned:
        raise AutonomyStackError("No plugin names provided")
    return cleaned


def detect_plugin_status(plugin_names: list[str] | None = None) -> dict[str, Any]:
    plugins = _recommended_plugins(plugin_names) if plugin_names else list(DEFAULT_PLUGINS)
    plugins_dir = _plugins_dir()
    installed = [name for name in plugins if (plugins_dir / name).is_dir()]
    missing = [name for name in plugins if name not in installed]
    return {
        "success": True,
        "plugins_dir": str(plugins_dir),
        "installed": installed,
        "missing": missing,
        "evey_utils": (plugins_dir / "evey_utils.py").exists(),
        "hermes_skill_loop": {
            "built_in": True,
            "note": "Hermes already creates and improves skills from experience; no extra skill-factory dependency is required for this stack.",
        },
    }


def install_plugins(plugin_names: list[str] | None = None, *, update_repo: bool = True) -> dict[str, Any]:
    plugins = _recommended_plugins(plugin_names) if plugin_names else list(DEFAULT_PLUGINS)
    repo_dir = _ensure_repo(update=update_repo)
    plugins_dir = _plugins_dir()
    plugins_dir.mkdir(parents=True, exist_ok=True)

    installed: list[str] = []
    for name in plugins:
        src = repo_dir / name
        if not src.is_dir():
            raise AutonomyStackError(f"Plugin not found in repo cache: {name}")
        shutil.copytree(src, plugins_dir / name, dirs_exist_ok=True)
        installed.append(name)

    utils_src = repo_dir / "evey_utils.py"
    if utils_src.exists():
        shutil.copy2(utils_src, plugins_dir / "evey_utils.py")

    status = detect_plugin_status(plugins)
    status.update(
        {
            "installed_now": installed,
            "repo_cache": str(repo_dir),
            "repo_url": EVEY_REPO_URL,
        }
    )
    return status


def doctor(plugin_names: list[str] | None = None) -> dict[str, Any]:
    return {
        "success": True,
        "tools": {
            "git": bool(_which("git")),
        },
        "repo_cache": str(_cache_dir()),
        "plugin_status": detect_plugin_status(plugin_names),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes autonomy stack helper")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("doctor", "install", "update"):
        sub_parser = sub.add_parser(name)
        sub_parser.add_argument(
            "--plugins",
            default="",
            help="Comma-separated list of evey-* plugin directories",
        )
    return parser


def _parse_plugins(raw: str) -> list[str] | None:
    if not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    plugins = _parse_plugins(args.plugins)
    try:
        if args.command == "doctor":
            payload = doctor(plugins)
        elif args.command == "install":
            payload = install_plugins(plugins, update_repo=True)
        elif args.command == "update":
            payload = install_plugins(plugins, update_repo=True)
        else:
            raise AutonomyStackError(f"Unknown command: {args.command}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except AutonomyStackError as exc:
        print(json.dumps({"success": False, "error": str(exc)}))
        return 1
    except subprocess.CalledProcessError as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Command failed: {' '.join(exc.cmd)}",
                    "stdout": exc.stdout,
                    "stderr": exc.stderr,
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
