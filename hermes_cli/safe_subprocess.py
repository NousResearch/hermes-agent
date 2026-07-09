"""safe_subprocess — Windows PATH-resolved subprocess wrapper.

Solves the Windows quirks where:
- shutil.which + subprocess.run use the CURRENT env's PATH, not the registry
- User-level PATH additions (added via [Environment]::SetEnvironmentVariable)
  are visible to NEW processes but not to the current Python sandbox
- .CMD/.BAT files aren't auto-resolved by subprocess.run (only by cmd.exe)
- .PS1 files (like scoop) need explicit powershell invocation

Usage:
    from hermes_cli.safe_subprocess import run
    result = run(['cargo', '--version'])
    print(result.stdout)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import List, Optional, Tuple

_PATHEXT_FALLBACKS = (".EXE", ".CMD", ".BAT", ".COM", ".PS1")
_POWERSHELL_EXE = "powershell.exe"


def _read_registry_path(scope: str) -> str:
    """Read PATH from Windows registry (User or Machine scope)."""
    if scope == "user":
        reg_path = r"HKCU\Environment"
    else:
        reg_path = r"HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    try:
        r = subprocess.run(
            ["reg", "query", reg_path, "/v", "Path"],
            capture_output=True, text=True, timeout=10,
        )
        # Format: \n\n<key>\n    Path    REG_SZ    <value>\n
        for line in r.stdout.splitlines():
            if line.strip().startswith("Path"):
                return line.split(None, 2)[-1].strip()
    except Exception:
        pass
    return ""


def _build_merged_path() -> str:
    """Build a merged PATH: system registry + user registry + current env.
    
    Preserves order (system first, user second, env overrides).
    De-duplicates by case-insensitive path comparison.
    """
    sys_path = _read_registry_path("machine")
    user_path = _read_registry_path("user")
    env_path = os.environ.get("PATH", "")
    
    seen = set()
    merged = []
    for raw in [sys_path, user_path, env_path]:
        for p in raw.split(";"):
            if p and p.lower() not in seen:
                seen.add(p.lower())
                merged.append(p)
    return ";".join(merged)


def _resolve(cmd: str, merged_path: str) -> Optional[str]:
    """Resolve a command name to a full path using the merged PATH.
    
    Tries multiple strategies:
    1. shutil.which on current env (fast path)
    2. shutil.which on merged PATH (catches user-level additions)
    3. PATHEXT fallback for known extension-sensitive tools
    """
    # 1. Current env (fast path)
    found = shutil.which(cmd)
    if found:
        return found
    # 2. Try with merged PATH temporarily
    old = os.environ.get("PATH", "")
    try:
        os.environ["PATH"] = merged_path
        found = shutil.which(cmd)
        if found:
            return found
    finally:
        os.environ["PATH"] = old
    # 3. PATHEXT fallback (for tools not in PATH but in a known location)
    for ext in _PATHEXT_FALLBACKS:
        if cmd.lower().endswith(ext.lower()):
            return cmd  # already has the extension
    return None


def _is_cmd_or_bat(path: str) -> bool:
    return path.lower().endswith((".cmd", ".bat"))


def _is_powershell(path: str) -> bool:
    return path.lower().endswith(".ps1")


def run(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    timeout: int = 60,
    env: Optional[dict] = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """subprocess.run with Windows quirks handled.
    
    - Resolves cmd[0] against the full registry PATH
    - For .CMD/.BAT: uses shell=True (subprocess.run doesn't resolve these)
    - For .PS1: invokes via powershell -NoProfile -ExecutionPolicy Bypass -File
    - Merges the caller's env with the registry PATH
    
    Returns subprocess.CompletedProcess. The result has .returncode,
    .stdout, .stderr.
    """
    if not cmd:
        raise ValueError("cmd must be a non-empty list")
    
    merged_path = _build_merged_path()
    resolved = _resolve(cmd[0], merged_path)
    
    if resolved is None:
        raise FileNotFoundError(f"Command not found in PATH: {cmd[0]}")
    
    # Build effective command
    if _is_powershell(resolved):
        # Invoke .ps1 via powershell
        effective = [_POWERSHELL_EXE, "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", resolved] + cmd[1:]
    elif _is_cmd_or_bat(resolved):
        # .CMD/.BAT files: subprocess.run doesn't auto-resolve them.
        # Use shell=True OR explicit cmd.exe invocation. We use shell=True
        # so PATHEXT is honored and the cmd.exe lookup finds the .CMD/.BAT.
        # But shell=True needs the full command as a string.
        cmd_str = subprocess.list2cmdline([resolved] + cmd[1:])
        if env is None:
            env = os.environ.copy()
        env["PATH"] = merged_path
        return subprocess.run(
            cmd_str, shell=True, cwd=cwd, timeout=timeout,
            env=env, check=check,
            capture_output=True, text=True,
        )
    else:
        # .EXE or other: use the resolved path directly
        effective = [resolved] + cmd[1:]
    
    # Build effective env
    if env is None:
        env = os.environ.copy()
    else:
        env = env.copy()
    env["PATH"] = merged_path
    
    return subprocess.run(
        effective, cwd=cwd, timeout=timeout,
        env=env, check=check,
        capture_output=True, text=True,
    )
