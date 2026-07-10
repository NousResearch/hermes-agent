"""Windows prerequisite installs for the HyperFrames plugin (winget + UAC)."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from typing import Any


def is_windows() -> bool:
    return os.name == "nt"


def is_admin() -> bool:
    if not is_windows():
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _run(cmd: list[str], *, timeout: float = 900.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
            "command": cmd,
        }
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "command": cmd}


def _powershell_quote(script: str) -> str:
    return script.replace("'", "''")


def run_elevated_powershell(script: str, *, timeout: float = 900.0) -> dict[str, Any]:
    """Run a PowerShell script elevated via UAC (operator must approve)."""
    if not is_windows():
        return {"ok": False, "error": "UAC elevation is Windows-only."}
    if is_admin():
        return _run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            timeout=timeout,
        )
    params = subprocess.list2cmdline(
        [
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ]
    )
    try:
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            "powershell.exe",
            params,
            None,
            1,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elevated": True}
    if result <= 32:
        return {
            "ok": False,
            "elevated": True,
            "error": f"UAC prompt was not approved (ShellExecuteW={result})",
        }
    return {
        "ok": True,
        "elevated": True,
        "action": "uac_handoff",
        "note": "Elevated install launched — approve the UAC prompt, then re-run hyperframes status.",
    }


def winget_install(package_id: str, *, elevated: bool = True) -> dict[str, Any]:
    """Install a winget package, optionally requesting UAC."""
    base = (
        f"winget install --id {package_id} -e --accept-source-agreements "
        f"--accept-package-agreements"
    )
    if elevated and not is_admin():
        return run_elevated_powershell(base)
    return _run(["winget", "install", "--id", package_id, "-e", "--accept-source-agreements", "--accept-package-agreements"])


def ensure_node(*, min_major: int = 22) -> dict[str, Any]:
    """Ensure Node.js >= min_major is available (winget on Windows)."""
    import shutil

    node = shutil.which("node")
    if node:
        proc = _run([node, "--version"])
        if proc.get("ok"):
            version = (proc.get("stdout") or "").strip().lstrip("v")
            try:
                major = int(version.split(".", 1)[0])
                if major >= min_major:
                    return {"ok": True, "action": "present", "version": version, "path": node}
            except ValueError:
                pass
    if not is_windows():
        return {
            "ok": False,
            "error": f"Node.js >= {min_major} is required. Install from https://nodejs.org/",
        }
    # Node 22+ from winget (OpenJS.NodeJS tracks current major).
    result = winget_install("OpenJS.NodeJS")
    result["package"] = "OpenJS.NodeJS"
    return result


def ensure_ffmpeg() -> dict[str, Any]:
    import shutil

    if shutil.which("ffmpeg"):
        return {"ok": True, "action": "present"}
    if not is_windows():
        return {"ok": False, "error": "FFmpeg is required. Install via your package manager."}
    result = winget_install("Gyan.FFmpeg")
    result["package"] = "Gyan.FFmpeg"
    return result


def ensure_npm_global(package: str, *, elevated: bool = True) -> dict[str, Any]:
    """Install an npm package globally; retry elevated on Windows permission errors."""
    import shutil

    npm = shutil.which("npm")
    if not npm:
        return {"ok": False, "error": "npm not found after Node install."}
    attempt = _run([npm, "install", "-g", package], timeout=600.0)
    if attempt.get("ok"):
        return attempt
    stderr = (attempt.get("stderr") or "").lower()
    needs_admin = any(
        token in stderr
        for token in ("eacces", "eperm", "access is denied", "permission", "administrator")
    )
    if not (is_windows() and elevated and needs_admin and not is_admin()):
        return attempt
    script = f"& '{_powershell_quote(npm)}' install -g {_powershell_quote(package)}"
    elevated_result = run_elevated_powershell(script, timeout=600.0)
    elevated_result["fallback_from"] = attempt
    return elevated_result
