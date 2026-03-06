"""Shared runtime provider resolution for CLI, gateway, cron, and helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from hermes_cli.auth import (
    AuthError,
    format_auth_error,
    resolve_provider,
    resolve_nous_runtime_credentials,
    resolve_codex_runtime_credentials,
)
from hermes_cli.config import load_config
from hermes_constants import OPENROUTER_BASE_URL


def _get_model_config() -> Dict[str, Any]:
    config = load_config()
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return dict(model_cfg)
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def resolve_requested_provider(requested: Optional[str] = None) -> str:
    """Resolve provider request from explicit arg, env, then config.

    Also infers provider from model prefix (e.g. ``local/qwen3.5-9b`` → ``local``).
    """
    if requested and requested.strip():
        return requested.strip().lower()

    env_provider = os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
    if env_provider:
        return env_provider

    model_cfg = _get_model_config()
    cfg_provider = model_cfg.get("provider")
    if isinstance(cfg_provider, str) and cfg_provider.strip():
        return cfg_provider.strip().lower()

    # Infer provider from model prefix when no explicit provider is set
    model_id = model_cfg.get("default", "")
    if isinstance(model_id, str) and model_id.startswith("local/"):
        return "local"

    return "auto"


def _resolve_openrouter_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    model_cfg = _get_model_config()
    cfg_base_url = model_cfg.get("base_url") if isinstance(model_cfg.get("base_url"), str) else ""
    cfg_provider = model_cfg.get("provider") if isinstance(model_cfg.get("provider"), str) else ""
    requested_norm = (requested_provider or "").strip().lower()
    cfg_provider = cfg_provider.strip().lower()

    env_openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    env_openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "").strip()

    use_config_base_url = False
    if requested_norm == "auto":
        if cfg_base_url.strip() and not explicit_base_url and not env_openai_base_url:
            if not cfg_provider or cfg_provider == "auto":
                use_config_base_url = True

    base_url = (
        (explicit_base_url or "").strip()
        or env_openai_base_url
        or (cfg_base_url.strip() if use_config_base_url else "")
        or env_openrouter_base_url
        or OPENROUTER_BASE_URL
    ).rstrip("/")

    api_key = (
        explicit_api_key
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )

    source = "explicit" if (explicit_api_key or explicit_base_url) else "env/config"

    return {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": base_url,
        "api_key": api_key,
        "source": source,
    }


LOCAL_MODEL_PORTS = {
    "local/qwen3.5-9b": 8800,
}


def _local_server_alive(port: int, timeout: float = 1.0) -> bool:
    """Check if a local model server is responding on the given port."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False


# Map model IDs to serve.py aliases
_LOCAL_MODEL_ALIASES = {
    "local/qwen3.5-9b": "qwen",
}


_managed_server_proc = None


def _auto_start_local_server(model_id: str, port: int) -> bool:
    """Spawn the local model server as a child process tied to this hermes session.

    The server is killed automatically when hermes exits.
    Returns True if the server is alive (already running or successfully started).
    """
    global _managed_server_proc

    if _local_server_alive(port):
        return True

    alias = _LOCAL_MODEL_ALIASES.get(model_id)
    if not alias:
        return False

    import atexit
    import signal
    import subprocess
    import sys
    import time

    cmd = [sys.executable, "-m", "local_models.serve", alias]
    print(f"Starting local model server on port {port}...")
    print(f"  $ {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _managed_server_proc = proc
    except Exception as e:
        print(f"  Failed to start server: {e}")
        return False

    # Kill server when hermes exits
    def _cleanup_server():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_cleanup_server)
    # Also handle SIGTERM (best-effort — fails in non-main threads)
    try:
        _prev_sigterm = signal.getsignal(signal.SIGTERM)
        def _sigterm_handler(signum, frame):
            _cleanup_server()
            if callable(_prev_sigterm) and _prev_sigterm not in (signal.SIG_DFL, signal.SIG_IGN):
                _prev_sigterm(signum, frame)
            else:
                raise SystemExit(1)
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except ValueError:
        pass  # Not in main thread; atexit cleanup is sufficient

    # Wait up to 60s for the server to come alive (model loading takes time)
    print(f"  Waiting for model to load...", end="", flush=True)
    for i in range(120):
        ret = proc.poll()
        if ret is not None:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            print(f"\n  Server exited with code {ret}")
            if stderr:
                for line in stderr.strip().splitlines()[-5:]:
                    print(f"    {line}")
            return False
        if _local_server_alive(port):
            print(f" ready! (took ~{i // 2}s)")
            return True
        if i % 10 == 0 and i > 0:
            print(".", end="", flush=True)
        time.sleep(0.5)

    print(f"\n  Timed out waiting for server on port {port}")
    return False


def _resolve_local_runtime(*, requested_provider: str = "local") -> Dict[str, Any]:
    """Resolve local model provider — reads model from config to pick the right port.

    If the server isn't running, auto-starts it in the background.
    """
    model_cfg = _get_model_config()
    model_id = model_cfg.get("default", "local/qwen3.5-9b")
    if isinstance(model_id, str) and model_id.startswith("local/"):
        port = LOCAL_MODEL_PORTS.get(model_id, 8800)
    else:
        port = 8800

    base_url = model_cfg.get("base_url", "").strip()
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base_url:
        base_url = f"http://127.0.0.1:{port}/v1"

    # Auto-start the server if it's not running
    _auto_start_local_server(model_id, port)

    return {
        "provider": "local",
        "api_mode": "chat_completions",
        "base_url": base_url.rstrip("/"),
        "api_key": "local",
        "source": "local",
        "requested_provider": requested_provider,
    }


def resolve_runtime_provider(
    *,
    requested: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve runtime provider credentials for agent execution."""
    requested_provider = resolve_requested_provider(requested)

    provider = resolve_provider(
        requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )

    if provider == "nous":
        creds = resolve_nous_runtime_credentials(
            min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
            timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
        )
        return {
            "provider": "nous",
            "api_mode": "chat_completions",
            "base_url": creds.get("base_url", "").rstrip("/"),
            "api_key": creds.get("api_key", ""),
            "source": creds.get("source", "portal"),
            "expires_at": creds.get("expires_at"),
            "requested_provider": requested_provider,
        }

    if provider == "openai-codex":
        creds = resolve_codex_runtime_credentials()
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": creds.get("base_url", "").rstrip("/"),
            "api_key": creds.get("api_key", ""),
            "source": creds.get("source", "hermes-auth-store"),
            "last_refresh": creds.get("last_refresh"),
            "requested_provider": requested_provider,
        }

    if provider == "anthropic" or requested_provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return {
            "provider": "anthropic",
            "api_mode": "chat_completions",
            "base_url": "",  # litellm handles routing
            "api_key": api_key,
            "source": "env",
            "requested_provider": "anthropic",
        }

    if provider == "local" or requested_provider == "local":
        return _resolve_local_runtime(requested_provider=requested_provider)

    if provider == "webgpu" or requested_provider == "webgpu":
        return _resolve_webgpu_runtime(requested_provider=requested_provider)

    runtime = _resolve_openrouter_runtime(
        requested_provider=requested_provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=explicit_base_url,
    )
    runtime["requested_provider"] = requested_provider
    return runtime


WEBGPU_BRIDGE_PORT = 8801


def _resolve_webgpu_runtime(*, requested_provider: str = "webgpu") -> Dict[str, Any]:
    """Resolve WebGPU provider — browser-side inference via the bridge server."""
    port = int(os.getenv("HERMES_WEBGPU_PORT", str(WEBGPU_BRIDGE_PORT)))
    base_url = f"http://127.0.0.1:{port}/v1"

    if not _local_server_alive(port):
        # Auto-start the bridge server
        import subprocess
        import sys
        import time

        cmd = [sys.executable, "-m", "web_client.bridge", "--port", str(port)]
        print(f"WebGPU bridge not running. Starting it...")
        print(f"  $ {' '.join(cmd)}")

        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as e:
            print(f"  Failed to start bridge: {e}")

        print(f"  Waiting for bridge on port {port}...", end="", flush=True)
        for i in range(30):
            if _local_server_alive(port):
                print(f" ready!")
                break
            time.sleep(0.5)
            if i % 4 == 0 and i > 0:
                print(".", end="", flush=True)
        else:
            print(f"\n  Bridge started. Open http://127.0.0.1:{port}/ to load a model.")

    return {
        "provider": "webgpu",
        "api_mode": "chat_completions",
        "base_url": base_url,
        "api_key": "webgpu-local",
        "source": "webgpu-bridge",
        "requested_provider": requested_provider,
    }


def format_runtime_provider_error(error: Exception) -> str:
    if isinstance(error, AuthError):
        return format_auth_error(error)
    return str(error)
