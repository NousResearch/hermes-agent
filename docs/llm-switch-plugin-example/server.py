"""Server lifecycle management for local LLM servers.

Pure Python — no shell or OS-specific dependencies beyond subprocess and /proc.
Works with any OpenAI-compatible server (llama.cpp, vLLM, etc.).
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Map from YAML sampling key names to llama-server CLI flags
_SAMPLING_FLAGS = {
    "temp": "--temp",
    "top_p": "--top-p",
    "top_k": "--top-k",
    "min_p": "--min-p",
    "presence_penalty": "--presence-penalty",
    "repeat_penalty": "--repeat-penalty",
    "frequency_penalty": "--frequency-penalty",
}


def kill_server(binary: str = "llama-server", timeout: float = 5.0) -> None:
    """Kill any running server process and wait for it to exit."""
    subprocess.run(["pkill", "-f", binary], capture_output=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(["pgrep", "-f", binary], capture_output=True)
        if result.returncode != 0:
            return  # process gone
        time.sleep(0.2)
    # Force kill if still alive
    subprocess.run(["pkill", "-9", "-f", binary], capture_output=True)


def build_command(model_key: str, config: Dict[str, Any]) -> List[str]:
    """Build the server command line from a models.yaml config."""
    server_cfg = config.get("server", {})
    model_cfg = config["models"][model_key]
    models_dir = Path(server_cfg.get("models_dir", "~/llama-models")).expanduser()

    cmd = [
        server_cfg.get("binary", "llama-server"),
        "-m", str(models_dir / model_cfg["gguf"]),
        "-c", str(model_cfg.get("context", 8192)),
        "-ngl", str(server_cfg.get("gpu_layers", 99)),
        "-np", str(server_cfg.get("parallel", 1)),
        "--host", server_cfg.get("host", "0.0.0.0"),
        "--port", str(server_cfg.get("port", 8080)),
    ]

    if server_cfg.get("flash_attention"):
        cmd += ["-fa", "on"]
    if server_cfg.get("jinja"):
        cmd.append("--jinja")

    # KV cache quantization
    kv = model_cfg.get("kv_cache", {})
    if kv.get("key"):
        cmd += ["-ctk", str(kv["key"])]
    if kv.get("value"):
        cmd += ["-ctv", str(kv["value"])]

    # Sampling defaults
    sampling = model_cfg.get("sampling", {})
    for param, flag in _SAMPLING_FLAGS.items():
        if param in sampling:
            cmd += [flag, str(sampling[param])]

    # Model alias (reported by /v1/models)
    alias = model_cfg.get("alias", model_key)
    cmd += ["--alias", alias]

    return cmd


def start_server(
    model_key: str,
    config: Dict[str, Any],
    timeout: int = 60,
) -> bool:
    """Kill existing server, start a new one, and wait for it to be healthy.

    Returns True if the server is ready, False if it timed out.
    """
    binary = config.get("server", {}).get("binary", "llama-server")
    kill_server(binary=binary)

    cmd = build_command(model_key, config)
    logger.info("Starting server: %s", " ".join(cmd))

    log_path = Path("/tmp/llama-server.log")
    log_file = log_path.open("w")
    subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

    port = config.get("server", {}).get("port", 8080)
    health_url = f"http://localhost:{port}/health"

    for _ in range(timeout):
        try:
            import urllib.request
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)

    return False


def get_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check if the server is running and which model is loaded."""
    binary = config.get("server", {}).get("binary", "llama-server")
    result = subprocess.run(
        ["pgrep", "-f", binary], capture_output=True, text=True
    )
    if result.returncode != 0:
        return {"running": False}

    pid_str = result.stdout.strip().split("\n")[0]
    try:
        pid = int(pid_str)
    except ValueError:
        return {"running": False}

    # Extract --alias from /proc/<pid>/cmdline
    alias = None
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        parts = cmdline.decode(errors="replace").split("\0")
        for i, part in enumerate(parts):
            if part == "--alias" and i + 1 < len(parts):
                alias = parts[i + 1]
                break
    except Exception:
        pass

    port = config.get("server", {}).get("port", 8080)
    return {
        "running": True,
        "pid": pid,
        "alias": alias,
        "model": alias,  # convenience alias
        "endpoint": f"http://localhost:{port}/v1",
    }
