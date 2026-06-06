import sys
import os
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PRODUCERS_DIR = Path("/home/ameobius/projects/security-workstation/.hermes/profiles/producers")
if not PRODUCERS_DIR.is_dir():
    PRODUCERS_DIR = Path.home() / ".hermes" / "profiles" / "producers"

RUNNER_SCRIPT = PRODUCERS_DIR / "scripts" / "brev_generation_queue_runner.py"

def run_brev_generation(request_id: str) -> dict[str, Any]:
    """Invokes the local queue runner script inside a python subprocess."""
    env = os.environ.copy()
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"

    cmd = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--request-id", request_id,
        "--issue-id", "producers-music",
        "--poll-seconds", "120",
        "--poll-interval", "5.0",
    ]

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
            env=env
        )
        try:
            manifest = json.loads(res.stdout)
        except Exception:
            return {"ok": False, "error": f"Failed to parse runner output: {res.stdout}"}
        final_status = manifest.get("final_status")
        if manifest.get("ok") is True and final_status == "completed" and manifest.get("asset_urls"):
            return manifest
        return {
            "ok": False,
            "final_status": final_status,
            "error": manifest.get("reason") or final_status or "generation_not_completed",
            "manifest_path": manifest.get("manifest_path"),
            "notification_path": manifest.get("notification_path"),
            "asset_urls": manifest.get("asset_urls") or [],
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Brev generation subprocess failed with code {e.returncode}: {e.stderr or e.stdout}")
        return {"ok": False, "error": f"Subprocess exit {e.returncode}: {e.stderr or e.stdout}"}
    except subprocess.TimeoutExpired:
        logger.error("Brev generation subprocess timed out after 180s")
        return {"ok": False, "error": "Subprocess execution timed out"}
