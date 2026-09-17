"""Fresh-interpreter preflight and source rollback for post-swap updates."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


_REQUIRED_IMPORTS = (
    "utils", "hermes_cli.main", "gateway.run", "gateway.status",
    "plugins.platforms.slack.adapter",
)


def run_gateway_startup_preflight(root: Path) -> dict:
    """Import boot-critical modules and parse gateway config in a fresh interpreter."""
    probe = (
        "import importlib,json,sys\n"
        f"mods={_REQUIRED_IMPORTS!r}\n"
        "errors=[]\n"
        "for name in mods:\n"
        "  try: importlib.import_module(name)\n"
        "  except BaseException as exc: errors.append([name,type(exc).__name__,str(exc)])\n"
        "try:\n"
        "  from utils import file_signature\n"
        "  from gateway.config import load_gateway_config\n"
        "  load_gateway_config()\n"
        "except BaseException as exc: errors.append(['gateway-startup-preflight',type(exc).__name__,str(exc)])\n"
        "print(json.dumps({'ok':not errors,'errors':errors}))\n"
        "sys.exit(0 if not errors else 1)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=root, env=env,
        capture_output=True, text=True, timeout=90, check=False,
    )
    try:
        payload = json.loads((result.stdout or "").splitlines()[-1])
    except (ValueError, IndexError, TypeError):
        payload = {"ok": False, "errors": [["probe", "InvalidOutput", (result.stderr or result.stdout)[-2000:]]]}
    payload["returncode"] = result.returncode
    return payload


def rollback_source_after_preflight_failure(root: Path, git_cmd: list[str], target_sha: str | None) -> dict:
    """Restore the pre-update checkout and prove its startup imports before returning."""
    if not target_sha:
        return {"ok": False, "reason": "missing-pre-update-sha"}
    reset = subprocess.run(
        [*git_cmd, "reset", "--hard", target_sha], cwd=root,
        capture_output=True, text=True, timeout=120, check=False,
    )
    if reset.returncode != 0:
        return {"ok": False, "reason": "git-reset-failed", "stderr": reset.stderr[-2000:]}
    for cache in root.rglob("__pycache__"):
        for pyc in cache.glob("*.pyc"):
            try:
                pyc.unlink()
            except OSError:
                pass
    verified = run_gateway_startup_preflight(root)
    return {"ok": bool(verified.get("ok")), "reason": "rolled-back", "preflight": verified}
