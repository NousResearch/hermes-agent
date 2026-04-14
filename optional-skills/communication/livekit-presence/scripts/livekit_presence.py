#!/usr/bin/env python3
"""Bootstrap a LiveKit companion project for Hermes presence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

STARTER_REPO = "https://github.com/livekit-examples/agent-starter-python.git"
STARTER_TEMPLATE = "agent-starter-python"
REQUIRED_ENV_VARS = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
OPTIONAL_ENV_VARS = ("OPENAI_API_KEY",)


class LiveKitPresenceError(RuntimeError):
    """Domain-specific error for the LiveKit presence helper."""


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def _persona_path() -> Path:
    return _hermes_home() / "SOUL.md"


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            continue
        key, _, value = raw_line.partition("=")
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        values[key] = value
    return values


def _quote_env(value: str) -> str:
    if value and all(ch.isalnum() or ch in "._-/:+@" for ch in value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _upsert_env(path: Path, updates: dict[str, str], *, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    output: list[str] = []

    for raw_line in existing_lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            output.append(raw_line)
            continue
        key, _, current = raw_line.partition("=")
        key = key.strip()
        if key in updates and overwrite:
            output.append(f"{key}={_quote_env(str(updates[key]))}")
            seen.add(key)
            continue
        if key in updates:
            output.append(raw_line)
            seen.add(key)
            continue
        if current:
            output.append(raw_line)

    if output and output[-1].strip():
        output.append("")

    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={_quote_env(str(value))}")

    path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    return path


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


def _detect_livekit_env(project_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    env_file_values: dict[str, str] = {}
    if project_dir:
        env_file_values = _load_dotenv(project_dir / ".env.local")

    result: dict[str, dict[str, Any]] = {}
    for key in REQUIRED_ENV_VARS + OPTIONAL_ENV_VARS:
        value = os.environ.get(key) or env_file_values.get(key, "")
        result[key] = {
            "configured": bool(value),
            "source": "environment" if os.environ.get(key) else ("project .env.local" if env_file_values.get(key) else "missing"),
            "preview": value[:4] + "..." if value else "",
        }
    return result


def _bootstrap_notes() -> str:
    return (
        "# Hermes LiveKit Bootstrap Notes\n\n"
        "- Upstream starter: https://github.com/livekit-examples/agent-starter-python\n"
        "- Exported persona: docs/hermes-persona.md\n"
        "- Local env file: .env.local\n"
        "- Recommended next steps:\n"
        "  1. `uv sync`\n"
        "  2. `uv run python src/agent.py download-files`\n"
        "  3. `uv run python src/agent.py console`\n"
        "  4. Review `src/agent.py` and merge the Hermes persona snapshot into the agent instructions if desired.\n"
    )


def export_persona(output_path: str | Path, *, source_path: str | Path | None = None) -> dict[str, Any]:
    source = Path(source_path) if source_path else _persona_path()
    target = Path(output_path)
    if not source.exists():
        raise LiveKitPresenceError(f"Hermes persona file not found: {source}")

    content = source.read_text(encoding="utf-8").strip()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Hermes Persona Snapshot\n\n"
        f"Source: `{source}`\n\n"
        "```md\n"
        f"{content}\n"
        "```\n",
        encoding="utf-8",
    )
    return {
        "success": True,
        "source": str(source),
        "output": str(target),
        "chars": len(content),
    }


def write_env_local(project_dir: str | Path, *, overwrite: bool = False, values: dict[str, str] | None = None) -> dict[str, Any]:
    project = Path(project_dir)
    env_path = project / ".env.local"
    current_values = dict(values or {})
    for key in REQUIRED_ENV_VARS + OPTIONAL_ENV_VARS:
        if key not in current_values and os.environ.get(key):
            current_values[key] = os.environ[key]

    placeholder_updates: dict[str, str] = {}
    for key in REQUIRED_ENV_VARS:
        placeholder_updates[key] = current_values.get(key, "")
    for key in OPTIONAL_ENV_VARS:
        if key in current_values:
            placeholder_updates[key] = current_values[key]

    _upsert_env(env_path, placeholder_updates, overwrite=overwrite)
    return {
        "success": True,
        "project_dir": str(project),
        "env_path": str(env_path),
        "configured": [key for key, value in placeholder_updates.items() if value],
        "missing": [key for key in REQUIRED_ENV_VARS if not placeholder_updates.get(key)],
    }


def doctor(project_dir: str | Path | None = None) -> dict[str, Any]:
    project = Path(project_dir).expanduser() if project_dir else None
    env_status = _detect_livekit_env(project)
    persona = _persona_path()
    return {
        "success": True,
        "bootstrap_method": "lk" if _which("lk") else ("git" if _which("git") else "missing"),
        "tools": {
            "git": bool(_which("git")),
            "uv": bool(_which("uv")),
            "lk": bool(_which("lk")),
        },
        "persona": {
            "path": str(persona),
            "exists": persona.exists(),
            "chars": len(persona.read_text(encoding="utf-8")) if persona.exists() else 0,
        },
        "env": env_status,
        "project_dir": str(project) if project else "",
    }


def bootstrap_project(target_dir: str | Path, *, prefer: str = "auto") -> dict[str, Any]:
    target = Path(target_dir).expanduser()
    if target.exists() and any(target.iterdir()):
        raise LiveKitPresenceError(f"Target directory is not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    method = prefer
    if prefer == "auto":
        method = "lk" if _which("lk") else "git"
    if method == "lk":
        if not _which("lk"):
            raise LiveKitPresenceError("LiveKit CLI `lk` is not installed")
        _run(["lk", "agent", "init", str(target), "--template", STARTER_TEMPLATE])
    elif method == "git":
        if not _which("git"):
            raise LiveKitPresenceError("Git is required for bootstrap when `lk` is unavailable")
        _run(["git", "clone", "--depth", "1", STARTER_REPO, str(target)])
    else:
        raise LiveKitPresenceError(f"Unsupported bootstrap method: {prefer}")

    docs_dir = target / "docs"
    export_result = export_persona(docs_dir / "hermes-persona.md")
    (docs_dir / "hermes-bootstrap.md").write_text(_bootstrap_notes(), encoding="utf-8")
    env_result = write_env_local(target, overwrite=False)

    return {
        "success": True,
        "target_dir": str(target),
        "method": method,
        "starter_repo": STARTER_REPO,
        "persona": export_result,
        "env": env_result,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes LiveKit presence helper")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_p = sub.add_parser("doctor", help="Show bootstrap requirements and env status")
    doctor_p.add_argument("--project", help="Existing LiveKit project directory", default="")

    bootstrap_p = sub.add_parser("bootstrap", help="Clone or initialize the upstream LiveKit starter")
    bootstrap_p.add_argument("--target", required=True, help="Target directory for the LiveKit project")
    bootstrap_p.add_argument("--prefer", choices=("auto", "lk", "git"), default="auto")

    env_p = sub.add_parser("write-env", help="Write .env.local from the current environment")
    env_p.add_argument("--project", required=True, help="LiveKit project directory")
    env_p.add_argument("--overwrite", action="store_true", help="Overwrite existing keys in .env.local")

    persona_p = sub.add_parser("export-persona", help="Export Hermes persona into the project docs")
    persona_p.add_argument("--project", required=True, help="LiveKit project directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            payload = doctor(args.project or None)
        elif args.command == "bootstrap":
            payload = bootstrap_project(args.target, prefer=args.prefer)
        elif args.command == "write-env":
            payload = write_env_local(args.project, overwrite=args.overwrite)
        elif args.command == "export-persona":
            payload = export_persona(Path(args.project).expanduser() / "docs" / "hermes-persona.md")
        else:
            raise LiveKitPresenceError(f"Unknown command: {args.command}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except LiveKitPresenceError as exc:
        print(json.dumps({"success": False, "error": str(exc)}), flush=True)
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
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
