"""Hermes Self-Evolution (HSE) operating-copy bridge.

This module intentionally keeps the optimizer implementation outside the Hermes
core checkout.  It resolves the standalone ``hermes-agent-self-evolution`` repo
and exposes small, explicit subprocess entrypoints for ``evolution.*`` modules.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:  # normal runtime path
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - defensive for partial import contexts
    def get_hermes_home() -> str:  # type: ignore[misc]
        return str(Path.home() / ".hermes")

_EVOLUTION_MODULE_RE = re.compile(r"^evolution(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_hse_repo() -> Path:
    env = os.environ.get("HERMES_SELF_EVOLUTION_REPO")
    if env:
        return Path(env).expanduser()
    return Path(get_hermes_home()).expanduser() / "evolution" / "repos" / "hermes-agent-self-evolution"


def _default_active_repo() -> Path:
    env = os.environ.get("HERMES_AGENT_REPO")
    if env:
        return Path(env).expanduser()
    return _repo_root()


def _resolve_hse_python(hse_repo: Path) -> Path | None:
    candidates = [
        hse_repo / ".venv" / "bin" / "python",
        hse_repo / "venv" / "bin" / "python",
        Path(get_hermes_home()).expanduser() / "evolution" / "venvs" / "self-evolution" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _git_head(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _git_status_count(repo: Path) -> int | None:
    if not (repo / ".git").exists():
        return None
    proc = subprocess.run(["git", "status", "--porcelain=v1"], cwd=repo, text=True, capture_output=True)
    if proc.returncode != 0:
        return None
    return 0 if not proc.stdout else len(proc.stdout.splitlines())


def build_status_payload(*, hse_repo: Path | None = None, active_repo: Path | None = None) -> dict[str, Any]:
    """Return a read-only HSE bridge status payload."""

    hse = (hse_repo or _default_hse_repo()).expanduser().resolve()
    active = (active_repo or _default_active_repo()).expanduser().resolve()
    hse_python = _resolve_hse_python(hse)
    return {
        "schema_version": "hermes-hse-bridge-status-v1",
        "status": "ready" if hse.exists() and hse_python is not None else "not_ready",
        "hse_repo": {
            "path": str(hse),
            "exists": hse.exists(),
            "git_head": _git_head(hse),
            "git_status_count": _git_status_count(hse),
        },
        "active_hermes_repo": {
            "path": str(active),
            "exists": active.exists(),
            "git_head": _git_head(active),
            "git_status_count": _git_status_count(active),
        },
        "hse_python": str(hse_python) if hse_python is not None else None,
        "boundary": {
            "default_read_only": True,
            "active_apply_performed": False,
            "github_write_performed": False,
            "cron_or_gateway_mutation_performed": False,
            "provider_or_model_spend_performed": False,
            "deploy_or_publication_performed": False,
            "arbitrary_module_execution_allowed": False,
        },
    }


def _module_env(hse_repo: Path, active_repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HERMES_SELF_EVOLUTION_REPO"] = str(hse_repo)
    env["HERMES_AGENT_REPO"] = str(active_repo)
    existing = env.get("PYTHONPATH", "")
    parts = [str(hse_repo)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _normalized_module_args(module_args: list[str]) -> list[str]:
    if module_args and module_args[0] == "--":
        return module_args[1:]
    return module_args


def run_evolution_module(args: Any) -> int:
    module = str(getattr(args, "module", ""))
    if not _EVOLUTION_MODULE_RE.match(module):
        print(f"Refusing non-evolution module: {module}", file=sys.stderr)
        return 2

    hse_repo = Path(getattr(args, "hse_repo", None) or _default_hse_repo()).expanduser().resolve()
    active_repo = Path(getattr(args, "active_hermes_repo", None) or _default_active_repo()).expanduser().resolve()
    hse_python = _resolve_hse_python(hse_repo)
    if hse_python is None:
        print(f"HSE Python not found under {hse_repo}", file=sys.stderr)
        return 2
    if not hse_repo.exists():
        print(f"HSE repo not found: {hse_repo}", file=sys.stderr)
        return 2

    command = [str(hse_python), "-m", module, *_normalized_module_args(list(getattr(args, "module_args", []) or []))]
    proc = subprocess.run(command, cwd=str(hse_repo), env=_module_env(hse_repo, active_repo), shell=False)
    return int(proc.returncode)


def hse_command(args: Any) -> int:
    command = getattr(args, "hse_command", None)
    if command in (None, "status"):
        payload = build_status_payload(
            hse_repo=Path(args.hse_repo).expanduser() if getattr(args, "hse_repo", None) else None,
            active_repo=Path(args.active_hermes_repo).expanduser() if getattr(args, "active_hermes_repo", None) else None,
        )
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"HSE bridge status: {payload['status']}")
            print(f"HSE repo: {payload['hse_repo']['path']}")
            print(f"Active Hermes repo: {payload['active_hermes_repo']['path']}")
            print(f"HSE Python: {payload['hse_python']}")
        return 0 if payload["status"] == "ready" else 1
    if command == "module":
        return run_evolution_module(args)
    print(f"Unknown HSE subcommand: {command}", file=sys.stderr)
    return 2
