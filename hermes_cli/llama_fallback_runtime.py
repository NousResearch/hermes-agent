"""Ensure the local llama.cpp fallback server is running before Hermes starts.

Research-backed defaults for RTX 3080 + TurboQuant + Qwen3.5 Q8_0:
- KV: K=f16 / V=turbo4 (asymmetric; K precision dominates Qwen routing)
- Speculative: ngram-mod (draftless; safe without MTP GGUF weights)
- Context: 49152 on 10 GB RTX 3080, 65536 on 12 GB RTX 3060
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)

def _default_server_exe() -> str:
    override = os.getenv("HERMES_LLAMA_SERVER_EXE", "").strip()
    if override:
        return override
    local_app = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app:
        return str(
            Path(local_app)
            / "Programs"
            / "llama-turboquant"
            / "bin"
            / "llama-server.exe"
        )
    return "llama-server.exe"


DEFAULT_SERVER_EXE = _default_server_exe()
DEFAULT_MODEL_PATH = ""
DEFAULT_PORT = 8080
DEFAULT_HOST = "127.0.0.1"

_KV_PROFILES = {
    "f16v_turbo4": ("f16", "turbo4"),
    "f16v_q4_0": ("f16", "q4_0"),
    "turbo4": ("turbo4", "turbo4"),
    "q4_0": ("q4_0", "q4_0"),
}

_GPU_CONTEXT_DEFAULTS = {
    "rtx3080": 49152,
    "rtx3060": 65536,
}


@dataclass(frozen=True)
class LlamaFallbackSettings:
    enabled: bool
    server_exe: str
    model_path: str
    host: str
    port: int
    gpu_profile: str
    context_size: int
    kv_profile: str
    spec_type: str
    wait_seconds: int


def _env_flag(name: str, default: str = "auto") -> str:
    raw = os.getenv(name, default).strip().lower()
    return raw or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _llama_cpp_configured(cfg: dict[str, Any]) -> bool:
    providers = cfg.get("providers") or {}
    if isinstance(providers, dict) and "llama-cpp" in providers:
        return True

    for entry in cfg.get("fallback_providers") or []:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip().lower()
        if provider in {"llama-cpp", "llama_cpp", "llama"}:
            return True
    return False


def _resolve_model_from_config(cfg: dict[str, Any]) -> str | None:
    providers = cfg.get("providers") or {}
    if isinstance(providers, dict):
        llama = providers.get("llama-cpp") or providers.get("llama_cpp")
        if isinstance(llama, dict):
            for key in ("default_model", "model", "default"):
                value = llama.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    for entry in cfg.get("fallback_providers") or []:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip().lower()
        if provider not in {"llama-cpp", "llama_cpp", "llama"}:
            continue
        model = entry.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return None


def _resolve_port_from_config(cfg: dict[str, Any]) -> int | None:
    providers = cfg.get("providers") or {}
    if not isinstance(providers, dict):
        return None
    llama = providers.get("llama-cpp") or providers.get("llama_cpp")
    if not isinstance(llama, dict):
        return None
    api = str(llama.get("api") or llama.get("base_url") or "").strip()
    if ":8080" in api:
        return 8080
    return None


def _resolve_model_path(cfg: dict[str, Any]) -> str:
    configured = (
        os.getenv("HERMES_LLAMA_MODEL_PATH", "").strip()
        or _resolve_model_from_config(cfg)
        or DEFAULT_MODEL_PATH
    ).strip()
    if not configured:
        return ""

    candidate = Path(configured)
    if candidate.is_file():
        return str(candidate)

    if DEFAULT_MODEL_PATH:
        default = Path(DEFAULT_MODEL_PATH)
        if default.is_file() and candidate.name == default.name:
            return str(default)

        sibling = default.parent / candidate.name
        if sibling.is_file():
            return str(sibling)

    return configured


def resolve_llama_fallback_settings(cfg: dict[str, Any] | None = None) -> LlamaFallbackSettings:
    cfg = cfg or _load_config()
    autostart = _env_flag("HERMES_LLAMA_FALLBACK_AUTOSTART", "auto")
    configured = _llama_cpp_configured(cfg)
    enabled = autostart in {"1", "true", "yes", "on"} or (
        autostart == "auto" and configured
    )

    gpu_profile = os.getenv("HERMES_LLAMA_GPU_PROFILE", "rtx3080").strip().lower() or "rtx3080"
    if gpu_profile not in _GPU_CONTEXT_DEFAULTS:
        gpu_profile = "rtx3080"

    kv_profile = os.getenv("HERMES_LLAMA_KV_PROFILE", "f16v_turbo4").strip().lower() or "f16v_turbo4"
    if kv_profile not in _KV_PROFILES:
        kv_profile = "f16v_turbo4"

    spec_type = os.getenv("HERMES_LLAMA_SPEC_TYPE", "ngram-mod").strip().lower() or "ngram-mod"
    if spec_type not in {"ngram-mod", "mtp", "none"}:
        spec_type = "ngram-mod"

    server_exe = _default_server_exe()
    model_path = _resolve_model_path(cfg)
    host = os.getenv("HERMES_LLAMA_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    port = _env_int("HERMES_LLAMA_PORT", _resolve_port_from_config(cfg) or DEFAULT_PORT)
    context_size = _env_int(
        "HERMES_LLAMA_CONTEXT_SIZE",
        _GPU_CONTEXT_DEFAULTS.get(gpu_profile, 49152),
    )
    wait_seconds = _env_int("HERMES_LLAMA_WAIT_SECONDS", 180)

    return LlamaFallbackSettings(
        enabled=enabled,
        server_exe=server_exe,
        model_path=model_path,
        host=host,
        port=port,
        gpu_profile=gpu_profile,
        context_size=context_size,
        kv_profile=kv_profile,
        spec_type=spec_type,
        wait_seconds=wait_seconds,
    )


def _models_url(settings: LlamaFallbackSettings) -> str:
    return f"http://{settings.host}:{settings.port}/v1/models"


def is_llama_fallback_ready(settings: LlamaFallbackSettings | None = None, timeout: float = 3.0) -> bool:
    settings = settings or resolve_llama_fallback_settings()
    try:
        with urlopen(_models_url(settings), timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except (URLError, OSError, TimeoutError, ValueError):
        return False


def _build_server_args(settings: LlamaFallbackSettings) -> list[str]:
    cache_k, cache_v = _KV_PROFILES[settings.kv_profile]
    args = [
        "--model",
        settings.model_path,
        "--host",
        settings.host,
        "--port",
        str(settings.port),
        "--ctx-size",
        str(settings.context_size),
        "--n-gpu-layers",
        "99",
        "--flash-attn",
        "on",
        "--cache-type-k",
        cache_k,
        "--cache-type-v",
        cache_v,
        "--parallel",
        "1",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--jinja",
        "--cont-batching",
    ]

    if settings.spec_type == "ngram-mod":
        args.extend(
            [
                "--spec-type",
                "ngram-mod",
                "--spec-ngram-mod-n-match",
                "24",
                "--spec-ngram-mod-n-min",
                "48",
                "--spec-ngram-mod-n-max",
                "64",
            ]
        )
    elif settings.spec_type == "mtp":
        args.extend(
            [
                "--spec-type",
                "mtp",
                "--spec-draft-n-max",
                os.getenv("HERMES_LLAMA_SPEC_DRAFT_N_MAX", "3"),
                "--spec-draft-p-min",
                os.getenv("HERMES_LLAMA_SPEC_DRAFT_P_MIN", "0.75"),
            ]
        )
    return args


def _start_via_powershell(settings: LlamaFallbackSettings) -> None:
    script_name = (
        "start-hermes-llama-fallback-rtx3080.ps1"
        if settings.gpu_profile == "rtx3080"
        else "start-hermes-llama-fallback-rtx3060.ps1"
    )
    script_path = _project_root() / "scripts" / "windows" / script_name
    shared_path = _project_root() / "scripts" / "windows" / "start-hermes-llama-fallback.ps1"
    if not script_path.exists() and shared_path.exists():
        script_path = shared_path

    if not script_path.exists():
        raise FileNotFoundError(f"llama fallback launcher not found: {script_path}")

    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-ServerExe",
        settings.server_exe,
        "-ModelPath",
        settings.model_path,
        "-Port",
        str(settings.port),
        "-ContextSize",
        str(settings.context_size),
        "-KvProfile",
        settings.kv_profile,
        "-SpecType",
        settings.spec_type,
        "-WaitSeconds",
        str(settings.wait_seconds),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or f"powershell launcher exited {completed.returncode}")


def _start_direct(settings: LlamaFallbackSettings) -> None:
    log_dir = Path.home() / ".hermes" / "logs" / "llama-fallback"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stdout_path = log_dir / f"llama-fallback-{stamp}.out.log"
    stderr_path = log_dir / f"llama-fallback-{stamp}.err.log"

    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_f:
        subprocess.Popen(
            [settings.server_exe, *_build_server_args(settings)],
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
        )


def ensure_llama_fallback_server(
    cfg: dict[str, Any] | None = None,
    *,
    quiet: bool = False,
) -> bool:
    """Start the local llama.cpp server when configured and not already listening."""
    settings = resolve_llama_fallback_settings(cfg)
    if not settings.enabled:
        return True

    if not Path(settings.server_exe).exists():
        msg = f"llama-server binary missing: {settings.server_exe}"
        if quiet:
            logger.warning(msg)
        else:
            print(f"WARNING llama_fallback: {msg}")
        return False

    if not Path(settings.model_path).exists():
        msg = f"fallback GGUF missing: {settings.model_path}"
        if quiet:
            logger.warning(msg)
        else:
            print(f"WARNING llama_fallback: {msg}")
        return False

    if is_llama_fallback_ready(settings):
        if not quiet:
            print(
                f"llama.cpp fallback already ready at http://{settings.host}:{settings.port}/v1"
            )
        return True

    if not quiet:
        print(
            "Starting llama.cpp fallback server "
            f"({settings.gpu_profile}, kv={settings.kv_profile}, spec={settings.spec_type})..."
        )

    if sys.platform == "win32":
        _start_via_powershell(settings)
    else:
        _start_direct(settings)

    deadline = time.time() + settings.wait_seconds
    while time.time() < deadline:
        if is_llama_fallback_ready(settings, timeout=3.0):
            if not quiet:
                print(
                    f"llama.cpp fallback ready at http://{settings.host}:{settings.port}/v1"
                )
            return True
        time.sleep(2)

    msg = (
        f"llama.cpp fallback did not become ready within {settings.wait_seconds}s "
        f"({settings.host}:{settings.port})"
    )
    if quiet:
        logger.warning(msg)
    else:
        print(f"WARNING llama_fallback: {msg}")
    return False
