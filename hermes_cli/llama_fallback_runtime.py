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
from urllib.parse import urlparse
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
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_LOCAL_SECRETARY_PORTS = {8080, 8081, 8082}
_EXTERNAL_LOCAL_SECRETARY_LAUNCHERS = {"external", "none", "disabled", "ollama"}

_KV_PROFILES = {
    "f16v_turbo4": ("f16", "turbo4"),
    "f16v_q4_0": ("f16", "q4_0"),
    "bf16v_q4_0": ("bf16", "q4_0"),
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
    launcher: str = "gguf"
    model_id: str = ""


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


def _as_url(value: Any):
    text = str(value or "").strip()
    if not text:
        return None
    if "://" not in text:
        text = f"http://{text}"
    try:
        return urlparse(text)
    except ValueError:
        return None


def _is_loopback_base_url(value: Any) -> bool:
    parsed = _as_url(value)
    if parsed is None:
        return False
    host = (parsed.hostname or "").strip().lower()
    return host in _LOOPBACK_HOSTS


def _port_from_base_url(value: Any) -> int | None:
    parsed = _as_url(value)
    if parsed is None:
        return None
    try:
        return parsed.port
    except ValueError:
        return None


def _is_llama_provider(provider: Any) -> bool:
    value = str(provider or "").strip().lower()
    return value in {"llama-cpp", "llama_cpp", "llama"}


def _is_custom_provider(provider: Any) -> bool:
    value = str(provider or "").strip().lower()
    return value == "custom" or value.startswith("custom:")


def _iter_custom_providers(cfg: dict[str, Any]):
    providers = cfg.get("custom_providers") or []
    if isinstance(providers, dict):
        for name, entry in providers.items():
            if isinstance(entry, dict):
                merged = dict(entry)
                merged.setdefault("name", name)
                yield merged
        return
    if isinstance(providers, list):
        for entry in providers:
            if isinstance(entry, dict):
                yield entry


def _entry_is_local_custom(entry: dict[str, Any]) -> bool:
    return _is_custom_provider(entry.get("provider")) and _is_loopback_base_url(
        entry.get("base_url")
    )


def _entry_is_local_secretary_endpoint(entry: dict[str, Any]) -> bool:
    if not _entry_is_local_custom(entry):
        return False
    port = _port_from_base_url(entry.get("base_url"))
    return port in _LOCAL_SECRETARY_PORTS


def _local_secretary_uses_external_launcher(cfg: dict[str, Any]) -> bool:
    local_secretary = cfg.get("local_secretary") or {}
    if not isinstance(local_secretary, dict):
        return False
    launcher = str(local_secretary.get("launcher") or "").strip().lower()
    return launcher in _EXTERNAL_LOCAL_SECRETARY_LAUNCHERS


def _local_secretary_configured(cfg: dict[str, Any]) -> bool:
    if _local_secretary_uses_external_launcher(cfg):
        return False

    if isinstance(cfg.get("local_secretary"), dict):
        return True

    model = cfg.get("model") or {}
    if isinstance(model, dict) and _entry_is_local_secretary_endpoint(model):
        return True

    for entry in cfg.get("fallback_providers") or []:
        if isinstance(entry, dict) and _entry_is_local_secretary_endpoint(entry):
            return True

    return any(
        _entry_is_local_secretary_endpoint(entry)
        for entry in _iter_custom_providers(cfg)
    )


def _default_gpu_profile(cfg: dict[str, Any]) -> str:
    local_secretary = cfg.get("local_secretary") or {}
    if isinstance(local_secretary, dict):
        profile = str(local_secretary.get("profile") or "").strip().lower()
        if profile in _GPU_CONTEXT_DEFAULTS:
            return profile
    return "rtx3080"


def _llama_cpp_configured(cfg: dict[str, Any]) -> bool:
    providers = cfg.get("providers") or {}
    if isinstance(providers, dict) and "llama-cpp" in providers:
        return True

    for entry in cfg.get("fallback_providers") or []:
        if not isinstance(entry, dict):
            continue
        if _is_llama_provider(entry.get("provider")):
            return True
    return _local_secretary_configured(cfg)


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
        if not _is_llama_provider(entry.get("provider")):
            continue
        model = entry.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()

    model = cfg.get("model") or {}
    if isinstance(model, dict) and _entry_is_local_secretary_endpoint(model):
        for key in ("default", "model"):
            value = model.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for entry in cfg.get("fallback_providers") or []:
        if isinstance(entry, dict) and _entry_is_local_secretary_endpoint(entry):
            model = entry.get("model")
            if isinstance(model, str) and model.strip():
                return model.strip()

    for entry in _iter_custom_providers(cfg):
        if _entry_is_local_secretary_endpoint(entry):
            for key in ("model", "default", "name"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _resolve_model_path_from_config(cfg: dict[str, Any]) -> str | None:
    local_secretary = cfg.get("local_secretary") or {}
    if isinstance(local_secretary, dict):
        for key in ("model_path", "gguf_path"):
            value = local_secretary.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for entry in _iter_custom_providers(cfg):
        value = entry.get("model_path") or entry.get("gguf_path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_port_from_config(cfg: dict[str, Any]) -> int | None:
    providers = cfg.get("providers") or {}
    if isinstance(providers, dict):
        llama = providers.get("llama-cpp") or providers.get("llama_cpp")
        if isinstance(llama, dict):
            port = _port_from_base_url(llama.get("api") or llama.get("base_url"))
            if port:
                return port

    model = cfg.get("model") or {}
    if isinstance(model, dict) and _entry_is_local_secretary_endpoint(model):
        port = _port_from_base_url(model.get("base_url"))
        if port:
            return port

    for entry in cfg.get("fallback_providers") or []:
        if not isinstance(entry, dict):
            continue
        if _is_llama_provider(entry.get("provider")) or _entry_is_local_secretary_endpoint(entry):
            port = _port_from_base_url(entry.get("api") or entry.get("base_url"))
            if port:
                return port

    for entry in _iter_custom_providers(cfg):
        if _entry_is_local_secretary_endpoint(entry):
            port = _port_from_base_url(entry.get("base_url"))
            if port:
                return port
    return None


def _resolve_model_path(cfg: dict[str, Any]) -> str:
    configured = (
        os.getenv("HERMES_LLAMA_MODEL_PATH", "").strip()
        or _resolve_model_path_from_config(cfg)
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
    autostart = _env_flag("HERMES_LLAMA_FALLBACK_AUTOSTART", "false")
    configured = _llama_cpp_configured(cfg)
    enabled = autostart in {"1", "true", "yes", "on"} or (
        autostart == "auto" and configured
    )

    gpu_profile = (
        os.getenv("HERMES_LLAMA_GPU_PROFILE", _default_gpu_profile(cfg)).strip().lower()
        or "rtx3080"
    )
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
    launcher = (
        "local_secretary"
        if _local_secretary_configured(cfg) and not Path(model_path).is_file()
        else "gguf"
    )
    model_id = _resolve_model_from_config(cfg) or ""
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
        launcher=launcher,
        model_id=model_id,
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
    if settings.launcher == "local_secretary":
        script_name = (
            "start-llama-secretary-fallback.ps1"
            if settings.port == 8081
            else "start-llama-secretary.ps1"
        )
        script_path = _project_root() / "scripts" / "windows" / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"local secretary launcher not found: {script_path}")

        env = os.environ.copy()
        env.setdefault("HERMES_LLAMA_SERVER_EXE", settings.server_exe)
        if settings.port == 8081:
            env.setdefault("HERMES_LLAMA_FALLBACK_PORT", str(settings.port))
            if settings.model_id:
                env.setdefault("HERMES_LLAMA_FALLBACK_ALIAS", settings.model_id)
        else:
            env.setdefault("HERMES_LLAMA_PORT", str(settings.port))
            if settings.model_id:
                env.setdefault("HERMES_LLAMA_ALIAS", settings.model_id)

        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-WaitSeconds",
            str(settings.wait_seconds),
        ]
        completed = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=env
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(detail or f"powershell launcher exited {completed.returncode}")
        return

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

    if settings.launcher != "local_secretary" and not Path(settings.model_path).exists():
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

    if settings.launcher == "local_secretary" and sys.platform != "win32":
        msg = "local secretary autostart currently requires the Windows launcher"
        if quiet:
            logger.warning(msg)
        else:
            print(f"WARNING llama_fallback: {msg}")
        return False

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
