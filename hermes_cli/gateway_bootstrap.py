"""Pre-import gateway startup checks shared by every gateway entry point."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hermes_cli.config import get_hermes_home

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def purge_stale_gateway_pycache_before_import() -> bool:
    """Remove stale project bytecode before importing ``gateway.run``."""
    try:
        previous = (get_hermes_home() / ".gateway_boot_fingerprint").read_text(encoding="utf-8").strip()
        current = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    if not previous or not current or previous.rsplit(":", 1)[-1] == current:
        return False

    purged = False
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if ".venv" in pycache.parts or "node_modules" in pycache.parts:
            continue
        for pyc in pycache.glob("*.pyc"):
            try:
                pyc.unlink()
                purged = True
            except OSError:
                continue
    return purged
