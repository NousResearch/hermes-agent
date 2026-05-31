"""Handlers for the eikon plugin."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data)


def _herm_bin() -> str | None:
    override = os.getenv("HERM_EIKON_HERM_BIN", "").strip()
    if override:
        return override
    return shutil.which("herm")


def check_herm_available() -> bool:
    binary = _herm_bin()
    if not binary:
        return False
    if os.path.sep in binary:
        return Path(binary).is_file() and os.access(binary, os.X_OK)
    return shutil.which(binary) is not None


def _handle_eikon_install(args: dict[str, Any], **_: Any) -> str:
    source = str(args.get("source") or "").strip()
    if not source:
        return _json({"ok": False, "error": "source is required"})

    binary = _herm_bin()
    if not binary:
        return _json({"ok": False, "error": "herm executable not found on PATH"})

    cmd = [binary, "eikon", "install", source, "--json"]
    name = str(args.get("name") or "").strip()
    if name:
        cmd.extend(["--name", name])
    if args.get("media") is False or args.get("no_source") is True:
        cmd.append("--no-source")
    if args.get("set_active") is False:
        cmd.append("--no-use")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except FileNotFoundError:
        return _json({"ok": False, "error": f"herm executable not found: {binary}"})
    except subprocess.TimeoutExpired:
        return _json({"ok": False, "error": "herm eikon install timed out"})

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = stderr or stdout or f"exit {proc.returncode}"
        return _json({"ok": False, "error": f"herm eikon install failed: {detail}"})

    line = next((ln for ln in reversed(stdout.splitlines()) if ln.strip()), "")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return _json({"ok": False, "error": "herm eikon install returned non-JSON output", "output": stdout})
    return _json(payload)
