"""Bridge to zapabob/notebooklm-mcp-cli (``nlm`` / ``notebooklm-mcp``)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_REPO = "https://github.com/zapabob/notebooklm-mcp-cli"
DEFAULT_REF = "b9cf0e2"
ENV_CLI_REF = "NOTEBOOKLM_MCP_CLI_REF"
ENV_NLM_PROFILE = "NOTEBOOKLM_NLM_PROFILE"
ENV_MCP_NOTEBOOK_ID = "NOTEBOOKLM_MCP_NOTEBOOK_ID"


def _env(name: str, default: str = "") -> str:
    try:
        from hermes_cli.config import get_env_value

        value = get_env_value(name)
        if value is not None:
            return str(value).strip()
    except Exception:
        pass
    return os.environ.get(name, default).strip()


def cli_ref() -> str:
    return _env(ENV_CLI_REF, DEFAULT_REF) or DEFAULT_REF


def nlm_profile() -> str:
    return _env(ENV_NLM_PROFILE, "")


def mcp_notebook_id() -> str:
    return _env(ENV_MCP_NOTEBOOK_ID, "")


def _uvx_from_spec() -> str:
    ref = cli_ref()
    if ref.startswith("git+"):
        return ref
    if ref.startswith("http://") or ref.startswith("https://"):
        return f"git+{ref}@{DEFAULT_REF}" if "@" not in ref else f"git+{ref}"
    if "/" in ref or ref.startswith("."):
        return f"git+{ref}"
    return f"git+{DEFAULT_REPO}@{ref}"


def resolve_nlm_command() -> tuple[str, list[str]] | None:
    """Return (executable, prefix args) for ``nlm`` CLI invocations."""
    if shutil.which("nlm"):
        return ("nlm", [])
    if shutil.which("uvx"):
        return ("uvx", ["--from", _uvx_from_spec(), "nlm"])
    if shutil.which("uv"):
        return ("uv", ["tool", "run", "--from", _uvx_from_spec(), "nlm"])
    return None


def resolve_mcp_command() -> tuple[str, list[str]] | None:
    """Return (executable, prefix args) for ``notebooklm-mcp`` stdio server."""
    if shutil.which("notebooklm-mcp"):
        return ("notebooklm-mcp", [])
    if shutil.which("uvx"):
        return ("uvx", ["--from", _uvx_from_spec(), "notebooklm-mcp"])
    return None


def cli_available() -> bool:
    return resolve_nlm_command() is not None


def mcp_binary_available() -> bool:
    return resolve_mcp_command() is not None


def run_nlm(
    args: list[str],
    *,
    timeout: int = 180,
    profile: str | None = None,
) -> dict[str, Any]:
    """Run ``nlm`` with JSON-friendly error wrapping."""
    resolved = resolve_nlm_command()
    if resolved is None:
        return {
            "ok": False,
            "error": "nlm CLI is not available (install via `hermes notebooklm setup-mcp`).",
            "missing": ["nlm or uvx"],
        }
    executable, prefix = resolved
    cmd = [executable, *prefix]
    effective_profile = (profile if profile is not None else nlm_profile()).strip()
    if effective_profile:
        cmd.extend(["--profile", effective_profile])
    cmd.extend(args)
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": str(exc), "command": cmd}
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    parsed: Any | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None
    ok = result.returncode == 0
    payload: dict[str, Any] = {
        "ok": ok,
        "returncode": result.returncode,
        "command": cmd,
        "stdout": stdout,
        "stderr": stderr,
    }
    if parsed is not None:
        payload["data"] = parsed
    if not ok and not payload.get("error"):
        payload["error"] = stderr or stdout or f"nlm exited {result.returncode}"
    return payload


def auth_status(*, profile: str | None = None) -> dict[str, Any]:
    result = run_nlm(["login", "--check"], profile=profile, timeout=60)
    authenticated = False
    if result.get("ok"):
        data = result.get("data")
        stdout = (result.get("stdout") or "").lower()
        if isinstance(data, dict):
            authenticated = bool(data.get("authenticated", data.get("valid", True)))
        elif "authentication valid" in stdout or "✓ authentication valid" in stdout:
            authenticated = True
        elif "authenticated" in stdout:
            authenticated = True
        elif result.get("returncode") == 0 and "authentication failed" not in stdout:
            authenticated = "valid" in stdout or "notebooks found" in stdout
    return {
        "ok": bool(result.get("ok")),
        "authenticated": authenticated,
        "cli": result,
        "profile": profile or nlm_profile() or "default",
    }


def list_notebooks(*, profile: str | None = None) -> dict[str, Any]:
    return run_nlm(["notebook", "list", "--json"], profile=profile, timeout=120)


def create_notebook(title: str, *, profile: str | None = None) -> dict[str, Any]:
    clean = (title or "").strip() or "Hermes NotebookLM"
    return run_nlm(
        ["notebook", "create", clean, "--json"],
        profile=profile,
        timeout=120,
    )


def add_file_source(
    notebook_id: str,
    file_path: str | Path,
    *,
    title: str = "",
    wait: bool = False,
    profile: str | None = None,
) -> dict[str, Any]:
    path = Path(file_path).expanduser()
    if not path.is_file():
        return {"ok": False, "error": f"source file not found: {path}"}
    args = ["source", "add", notebook_id, "--file", str(path)]
    if title.strip():
        args.extend(["--title", title.strip()])
    if wait:
        args.append("--wait")
    return run_nlm(args, profile=profile, timeout=900 if wait else 300)


def add_text_source(
    notebook_id: str,
    text: str,
    *,
    title: str = "",
    wait: bool = False,
    profile: str | None = None,
) -> dict[str, Any]:
    content = (text or "").strip()
    if not content:
        return {"ok": False, "error": "empty text source"}
    args = ["source", "add", notebook_id, "--text", content]
    if title.strip():
        args.extend(["--title", title.strip()])
    if wait:
        args.append("--wait")
    return run_nlm(args, profile=profile, timeout=900 if wait else 300)


def doctor(*, profile: str | None = None) -> dict[str, Any]:
    return run_nlm(["doctor"], profile=profile, timeout=120)


def install_skill_hermes(*, profile: str | None = None) -> dict[str, Any]:
    return run_nlm(["skill", "install", "hermes"], profile=profile, timeout=120)


def bridge_status() -> dict[str, Any]:
    nlm_cmd = resolve_nlm_command()
    mcp_cmd = resolve_mcp_command()
    return {
        "repo": DEFAULT_REPO,
        "ref": cli_ref(),
        "nlm_resolved": list(nlm_cmd) if nlm_cmd else None,
        "mcp_resolved": list(mcp_cmd) if mcp_cmd else None,
        "cli_available": cli_available(),
        "mcp_binary_available": mcp_binary_available(),
        "configured_notebook_id": mcp_notebook_id(),
        "profile": nlm_profile() or "default",
    }
