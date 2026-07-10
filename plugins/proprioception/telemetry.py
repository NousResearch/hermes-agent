"""Native host telemetry collection.

Gathers system state directly via CLI tools instead of relying on the
Dashboard HTTP endpoint. Writes to ~/.hermes/telemetry.json for low-latency
reads by the collector and optional Dashboard consumption.

Sensors:
- GPU: nvidia-smi (temp, VRAM used/total, name)
- Disk: df (C: drive free/total)
- Network: tailscale status (peers online)
- Gateway: gateway_state.json (file read, no HTTP)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TELEMETRY_PATH = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser() / "telemetry.json"

# Known fallback locations when PATH lookup fails
_FALLBACKS = {
    "nvidia-smi": [r"C:\Windows\System32\nvidia-smi"],
    "df": ["/usr/bin/df"],
    "tailscale": [r"C:\Program Files\Tailscale\tailscale"],
}


def _resolve(cmd_name: str) -> str:
    """Resolve a command name: prefer PATH, fall back to known defaults."""
    found = shutil.which(cmd_name)
    if found:
        return found
    for candidate in _FALLBACKS.get(cmd_name, []):
        if os.path.isfile(candidate):
            return candidate
    return cmd_name  # last resort: hope it's in PATH


def _run(cmd: List[str], timeout: float = 5.0) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    cmd[0] = _resolve(cmd[0])
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return res.stdout.strip() if res.returncode == 0 else None
    except Exception as exc:
        logger.debug("telemetry: %s failed: %s", cmd[0], exc)
        return None


def collect_gpu() -> Dict[str, Any]:
    """Collect GPU metrics via nvidia-smi."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=temperature.gpu,memory.used,memory.total,name",
        "--format=csv,noheader"
    ])
    if not out:
        return {"error": "nvidia-smi unavailable"}

    gpus: List[Dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                gpus.append({
                    "name": parts[3],
                    "temp_c": int(parts[0].replace("C", "").strip()),
                    "vram_used_mb": int(parts[1].replace("MiB", "").strip()),
                    "vram_total_mb": int(parts[2].replace("MiB", "").strip()),
                })
            except ValueError:
                continue
    return gpus or {"error": "no GPUs parsed"}


def collect_disk() -> Dict[str, Any]:
    """Collect C: drive space via df (MSYS2/Git Bash)."""
    out = _run(["df", "-h", "/c/"])
    if not out:
        return {"error": "df unavailable"}

    try:
        parts = out.split("\n")[1].split()
        return {
            "total_gb": parts[1],
            "used_gb": parts[2],
            "free_gb": parts[3],
        }
    except (IndexError, ValueError):
        return {"error": "df output parse failed"}


def collect_network() -> Dict[str, Any]:
    """Collect Tailscale peer count."""
    out = _run(["tailscale", "status", "--json"])
    if not out:
        return {"error": "tailscale unavailable"}

    try:
        ts = json.loads(out)
        peers = ts.get("Peers", [])
        return {
            "tailscale_peers_online": len([p for p in peers if p.get("Online")]),
            "total_peers": len(peers),
        }
    except json.JSONDecodeError:
        return {"error": "tailscale JSON parse failed"}


def collect_gateway() -> Optional[Dict[str, Any]]:
    """Read gateway runtime status directly from disk."""
    try:
        from gateway.status import read_runtime_status
        return read_runtime_status()
    except Exception as exc:
        logger.debug("telemetry: gateway read failed: %s", exc)
        return None


def collect_logprobs_proxy() -> Dict[str, Any]:
    """Placeholder for logprobs proxy sensor.

    Reads from a local cache file written by the vLLM provider hook.
    Returns empty dict until the hook is wired.
    """
    cache = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser() / "logprobs_cache.json"
    if cache.exists():
        try:
            with open(cache, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"status": "inactive", "note": "awaiting vLLM sampling hook"}


def snapshot() -> Dict[str, Any]:
    """Return a full native telemetry snapshot."""
    return {
        "collected_at": time.time(),
        "gpu": collect_gpu(),
        "disk": collect_disk(),
        "network": collect_network(),
        "gateway": collect_gateway(),
        "logprobs_proxy": collect_logprobs_proxy(),
    }


def persist(data: Dict[str, Any]) -> None:
    """Write snapshot to telemetry.json atomically."""
    tmp = TELEMETRY_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(TELEMETRY_PATH)
    except Exception as exc:
        logger.warning("telemetry: persist failed: %s", exc)


def load() -> Optional[Dict[str, Any]]:
    """Load last persisted snapshot, or None."""
    if not TELEMETRY_PATH.exists():
        return None
    try:
        with open(TELEMETRY_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None