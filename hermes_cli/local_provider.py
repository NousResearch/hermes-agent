from __future__ import annotations

import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from hermes_cli.config import get_hermes_home, load_config

DEFAULT_PORT = 8899

MODEL_RECOMMENDATIONS: list[dict[str, str | float]] = [
    {
        "id": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "name": "Qwen 3.5 4B (4-bit)",
        "description": "Fast, lightweight everyday model",
        "min_ram_gb": 8,
    },
    {
        "id": "mlx-community/Hermes-3-Llama-3.1-8B-4bit",
        "name": "Hermes 3 8B (4-bit)",
        "description": "Stable Nous baseline",
        "min_ram_gb": 8,
    },
    {
        "id": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "name": "Qwen 3.5 9B (4-bit)",
        "description": "Strong quality/speed balance",
        "min_ram_gb": 16,
    },
    {
        "id": "mlx-community/Hermes-4-14B-4bit",
        "name": "Hermes 4 14B (4-bit)",
        "description": "Newer Nous model, high quality",
        "min_ram_gb": 16,
    },
    {
        "id": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        "name": "Qwen3 Coder 30B MoE (4-bit)",
        "description": "Excellent coding, MoE architecture",
        "min_ram_gb": 20,
    },
    {
        "id": "mlx-community/Qwen3.5-27B-4bit",
        "name": "Qwen 3.5 27B (4-bit)",
        "description": "High-quality general model",
        "min_ram_gb": 32,
    },
    {
        "id": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "name": "Qwen 3.5 35B-A3B (4-bit)",
        "description": "Large MoE model for quality-sensitive tasks",
        "min_ram_gb": 40,
    },
    {
        "id": "mlx-community/Hermes-4-70B-4bit",
        "name": "Hermes 4 70B (4-bit)",
        "description": "Nous flagship local model",
        "min_ram_gb": 48,
    },
    {
        "id": "mlx-community/Qwen3.5-122B-A10B-4bit",
        "name": "Qwen 3.5 122B-A10B (4-bit)",
        "description": "Very large model for high-end Apple Silicon",
        "min_ram_gb": 96,
    },
]


def detect_hardware() -> Optional[dict[str, object]]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        ram_bytes = int(result.stdout.strip())
        ram_gb = ram_bytes / (1024**3)
    except Exception:
        return None

    chip = "Apple Silicon"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            chip = result.stdout.strip()
    except Exception:
        pass

    return {"chip": chip, "ram_gb": round(ram_gb, 1), "apple_silicon": True}


def recommend_models(ram_gb: float) -> list[dict[str, str | float]]:
    usable = ram_gb * 0.8
    compatible = [m for m in MODEL_RECOMMENDATIONS if m["min_ram_gb"] <= usable]
    return sorted(compatible, key=lambda m: m["min_ram_gb"], reverse=True)


def _hf_hub_cache_dir() -> Path:
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_model_installed(model_id: str) -> bool:
    """Best-effort check whether a model appears in the local HuggingFace cache."""
    cache_key = "models--" + model_id.replace("/", "--")
    model_dir = _hf_hub_cache_dir() / cache_key
    snapshots = model_dir / "snapshots"
    try:
        return model_dir.exists() and snapshots.exists() and any(snapshots.iterdir())
    except Exception:
        return False


def installed_model_ids(model_ids: list[str]) -> set[str]:
    return {mid for mid in model_ids if is_model_installed(mid)}


def list_cached_model_ids(limit: int = 200) -> list[str]:
    """Best-effort discovery of all cached HF model IDs with snapshots.

    Returns model IDs decoded from hub directory names like:
      models--org--model  -> org/model
    """
    hub_dir = _hf_hub_cache_dir()
    if not hub_dir.exists():
        return []

    found: list[str] = []
    try:
        for entry in sorted(hub_dir.glob("models--*")):
            snapshots = entry / "snapshots"
            if not snapshots.exists():
                continue
            try:
                if not any(snapshots.iterdir()):
                    continue
            except OSError:
                continue

            model_id = entry.name[len("models--") :].replace("--", "/")
            if model_id:
                found.append(model_id)
    except OSError:
        return []

    unique_sorted = sorted(set(found))
    if limit > 0:
        return unique_sorted[:limit]
    return unique_sorted


def _pid_file_path() -> Path:
    return get_hermes_home() / "local_server.pid"


def _log_file_path() -> Path:
    return get_hermes_home() / "local_server.log"


def _read_pid_file() -> Optional[dict]:
    path = _pid_file_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        pid = data.get("pid")
        if pid is None:
            return None
        os.kill(pid, 0)
        return data
    except (json.JSONDecodeError, OSError, TypeError):
        _remove_pid_file()
        return None


def _write_pid_file(pid: int, model_id: str, port: int) -> None:
    path = _pid_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": pid, "model_id": model_id, "port": port}))


def _remove_pid_file() -> None:
    try:
        _pid_file_path().unlink(missing_ok=True)
    except OSError:
        pass


def check_mlx_lm_installed() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


def install_mlx_lm() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "mlx-lm"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0
    except Exception:
        return False


def is_server_running(port: int = DEFAULT_PORT) -> bool:
    try:
        req = urllib.request.Request(f"http://localhost:{port}/v1/models")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def get_server_status(port: int = DEFAULT_PORT) -> Optional[dict]:
    pid_data = _read_pid_file()
    if pid_data is None:
        return None
    running = is_server_running(pid_data.get("port", port))
    return {
        "pid": pid_data["pid"],
        "model_id": pid_data.get("model_id", "unknown"),
        "port": pid_data.get("port", port),
        "running": running,
    }


def start_server(
    model_id: str,
    port: int = DEFAULT_PORT,
    timeout: int = 120,
) -> dict:
    if is_server_running(port):
        status = _read_pid_file()
        if status and status.get("model_id") == model_id:
            return {
                "pid": status["pid"],
                "model_id": model_id,
                "port": port,
                "running": True,
                "reused": True,
            }
        stop_server(port)

    log_file = _log_file_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_file, "w")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            model_id,
            "--port",
            str(port),
        ],
        stdout=log_handle,
        stderr=log_handle,
    )

    _write_pid_file(proc.pid, model_id, port)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            _remove_pid_file()
            raise RuntimeError(
                f"mlx_lm.server exited with code {proc.returncode}. "
                f"Check {log_file} for details."
            )
        if is_server_running(port):
            return {
                "pid": proc.pid,
                "model_id": model_id,
                "port": port,
                "running": True,
                "reused": False,
            }
        time.sleep(2)

    proc.terminate()
    _remove_pid_file()
    raise TimeoutError(
        f"mlx_lm.server did not become ready within {timeout}s. "
        f"Check {log_file} for details."
    )


def stop_server(port: int = DEFAULT_PORT) -> bool:
    pid_data = _read_pid_file()
    if pid_data is None:
        return False

    pid = pid_data["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        _remove_pid_file()
        return True

    for _ in range(50):
        try:
            os.kill(pid, 0)
        except OSError:
            _remove_pid_file()
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

    _remove_pid_file()
    return True


def ensure_server(
    model_id: str,
    port: int = DEFAULT_PORT,
    timeout: int = 120,
) -> dict:
    status = get_server_status(port)
    if status and status["running"] and status["model_id"] == model_id:
        return {
            "pid": status["pid"],
            "model_id": model_id,
            "port": port,
            "running": True,
            "reused": True,
        }
    return start_server(model_id, port, timeout)


def _safe_print(msg: str = "") -> None:
    try:
        print(msg)
    except OSError:
        pass


def _cmd_status() -> None:
    hw = detect_hardware()
    if hw:
        _safe_print(f"Hardware: {hw['chip']} / {hw['ram_gb']} GB")
    else:
        _safe_print("Hardware: not Apple Silicon (or detection failed)")
    _safe_print()

    status = get_server_status()
    if status is None:
        _safe_print("Server: not running")
        return

    if status["running"]:
        _safe_print(f"Server:  running (PID {status['pid']})")
        _safe_print(f"Model:   {status['model_id']}")
        _safe_print(f"Port:    {status['port']}")
        _safe_print(f"URL:     http://localhost:{status['port']}/v1")
    else:
        _safe_print(f"Server:  stopped (stale PID {status['pid']})")
        _safe_print(f"Model:   {status['model_id']}")


def _cmd_start() -> None:
    local_cfg = load_config().get("local", {})
    model_id = local_cfg.get("model_id")
    port = local_cfg.get("port", DEFAULT_PORT)

    if not model_id:
        _safe_print("No local model configured.")
        _safe_print("Run 'hermes model' and select Local (Apple Silicon) first.")
        sys.exit(1)

    if not check_mlx_lm_installed():
        _safe_print("mlx-lm is not installed.")
        _safe_print("Install it with: pip install mlx-lm")
        sys.exit(1)

    if is_server_running(port):
        status = get_server_status(port)
        if status and status["model_id"] == model_id:
            _safe_print(f"Server already running ({model_id}) on port {port}")
            return
        _safe_print("Stopping existing server...")
        stop_server(port)

    _safe_print(f"Starting {model_id} on port {port}...")
    try:
        result = start_server(model_id, port)
        _safe_print(f"Server ready (PID {result['pid']})")
    except (RuntimeError, TimeoutError) as e:
        _safe_print(f"Failed to start server: {e}")
        sys.exit(1)


def _cmd_stop() -> None:
    stopped = stop_server()
    if stopped:
        _safe_print("Server stopped.")
    else:
        _safe_print("No server running.")


def _cmd_models() -> None:
    hw = detect_hardware()
    if hw is None:
        _safe_print(
            "Apple Silicon not detected. Local models require an Apple Silicon Mac."
        )
        sys.exit(1)

    ram_gb = hw["ram_gb"]
    _safe_print(f"Hardware: {hw['chip']} / {ram_gb} GB")
    _safe_print()

    models = recommend_models(ram_gb)
    if not models:
        _safe_print("No compatible models found for your hardware.")
        return

    _safe_print("Recommended models:")
    _safe_print()
    installed = installed_model_ids([m["id"] for m in models])
    for idx, m in enumerate(models):
        tags = []
        if idx == 0:
            tags.append("recommended")
        if m["id"] in installed:
            tags.append("installed")
        tag_text = f" [{' | '.join(tags)}]" if tags else ""
        _safe_print(f"  {m['name']:<40s} {m['description']}{tag_text}")
        _safe_print(f"  {m['id']}")
        _safe_print(f"  Requires: {m['min_ram_gb']}+ GB RAM")
        _safe_print()


def local_command(args: object) -> None:
    subcmd = getattr(args, "local_command", None)

    if subcmd is None or subcmd == "status":
        _cmd_status()
    elif subcmd == "start":
        _cmd_start()
    elif subcmd == "stop":
        _cmd_stop()
    elif subcmd == "models":
        _cmd_models()
    else:
        _cmd_status()
