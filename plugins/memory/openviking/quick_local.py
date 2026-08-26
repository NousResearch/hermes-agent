"""Reusable, non-interactive OpenViking Quick Local provisioning.

User interfaces own prompts and rendering.  This module owns the bounded
installation, Hermes-profile-scoped configuration, and temporary validation
needed to produce a ready-to-link OpenViking CLI profile.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from utils import atomic_json_write

DEPLOYMENT = "quick_local"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_DIMENSION = 1024
OPENVIKING_REQUIREMENT = "openviking>=0.4.16,<0.6"

_OPENVIKING_VERSION_SPECIFIER = SpecifierSet(
    OPENVIKING_REQUIREMENT.removeprefix("openviking")
)
_ROOT_DIRNAME = "openviking"
_SERVER_CONFIG_FILENAME = "ov.conf"
_OVCLI_CONFIG_FILENAME = "ovcli.conf"
_WORKSPACE_DIRNAME = "data"
_DEFAULT_PORT = 1933
_PORT_ATTEMPTS = 20
_MODEL_DOWNLOAD_SIZE = "approximately 639 MB"
_HEALTH_TIMEOUT_SECONDS = 60.0
_HEALTH_POLL_INTERVAL_SECONDS = 0.5
_PROCESS_STOP_TIMEOUT_SECONDS = 10.0


class QuickLocalStage(str, Enum):
    PREFLIGHT = "preflight"
    INSTALL_OPENVIKING = "install_openviking"
    INSTALL_OLLAMA = "install_ollama"
    START_OLLAMA = "start_ollama"
    DOWNLOAD_MODEL = "download_model"
    VALIDATE = "validate"
    WRITE_CONFIG = "write_config"
    COMPLETE = "complete"


@dataclass(frozen=True)
class QuickLocalProgress:
    stage: QuickLocalStage
    message: str


@dataclass(frozen=True)
class QuickLocalPaths:
    root: Path
    runtime: Path
    server_config: Path
    ovcli_config: Path
    workspace: Path

    @property
    def runtime_python(self) -> Path:
        scripts = "Scripts" if os.name == "nt" else "bin"
        executable = "python.exe" if os.name == "nt" else "python"
        return self.runtime / scripts / executable

    @property
    def server_command(self) -> Path:
        scripts = "Scripts" if os.name == "nt" else "bin"
        executable = "openviking-server.exe" if os.name == "nt" else "openviking-server"
        return self.runtime / scripts / executable


@dataclass(frozen=True)
class QuickLocalPreflight:
    paths: QuickLocalPaths
    reusable_endpoint: Optional[str]
    ollama_install_required: bool


@dataclass(frozen=True)
class QuickLocalSetupResult:
    paths: QuickLocalPaths
    endpoint: str
    reused: bool


class QuickLocalSetupError(RuntimeError):
    """Quick Local could not finish without partially activating it."""


class OllamaInstallRequired(QuickLocalSetupError):
    """Quick Local needs explicit permission to install Ollama."""


ProgressReporter = Callable[[QuickLocalProgress], None]
HealthCheck = Callable[[str], tuple[bool, str]]


def managed_paths(hermes_home: Path) -> QuickLocalPaths:
    root = Path(hermes_home).expanduser() / _ROOT_DIRNAME
    return QuickLocalPaths(
        root=root,
        runtime=root / "runtime",
        server_config=root / _SERVER_CONFIG_FILENAME,
        ovcli_config=root / _OVCLI_CONFIG_FILENAME,
        workspace=root / _WORKSPACE_DIRNAME,
    )


def managed_server_config_path(provider_config: Mapping[str, Any]) -> Optional[Path]:
    if provider_config.get("deployment") != DEPLOYMENT:
        return None
    value = _clean_value(provider_config.get("server_config_path"))
    return Path(value).expanduser() if value else None


def managed_server_command_path(provider_config: Mapping[str, Any]) -> Optional[Path]:
    if provider_config.get("deployment") != DEPLOYMENT:
        return None
    value = _clean_value(provider_config.get("server_command_path"))
    return Path(value).expanduser() if value else None


def clear_managed_settings(provider_config: dict[str, Any]) -> None:
    provider_config.pop("deployment", None)
    provider_config.pop("server_config_path", None)
    provider_config.pop("server_command_path", None)


def build_server_config(
    paths: QuickLocalPaths,
    vlm: Mapping[str, Any],
    *,
    port: int = _DEFAULT_PORT,
) -> dict[str, Any]:
    return {
        "server": {"host": "127.0.0.1", "port": port},
        "storage": {"workspace": str(paths.workspace)},
        "embedding": {
            "dense": {
                "provider": "ollama",
                "model": EMBEDDING_MODEL,
                "api_base": "http://localhost:11434/v1",
                "dimension": EMBEDDING_DIMENSION,
                "input": "text",
            }
        },
        "vlm": dict(vlm),
    }


def resolve_hermes_vlm_config() -> dict[str, Any]:
    """Translate the active persisted Hermes LLM into OpenViking VLM config."""

    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config = load_config()
    model_config = config.get("model", {}) if isinstance(config, Mapping) else {}
    if isinstance(model_config, str):
        model_config = {"default": model_config}
    if not isinstance(model_config, Mapping):
        model_config = {}

    default_model = model_config.get("default")
    requested_provider = _clean_value(model_config.get("provider")) or None
    if isinstance(default_model, Mapping):
        from hermes_cli.config import split_model_config_default

        nested_model, nested_provider = split_model_config_default(default_model)
        default_model = nested_model
        requested_provider = requested_provider or nested_provider or None
    model = _clean_value(
        default_model or model_config.get("model") or model_config.get("name")
    )
    if not model:
        raise QuickLocalSetupError("Hermes has no default LLM model configured.")

    runtime = resolve_runtime_provider(
        requested=requested_provider,
        target_model=model,
    )
    runtime_model = _clean_value(runtime.get("model")) or model
    provider = _clean_value(runtime.get("provider")).lower()
    api_mode = _clean_value(runtime.get("api_mode")).lower()
    source = _clean_value(runtime.get("source")).lower()
    api_base = _clean_value(runtime.get("base_url"))
    api_key = _clean_value(runtime.get("api_key"))

    if _uses_noncopyable_credentials(provider, source, api_key):
        raise QuickLocalSetupError(
            "Hermes is using refreshed OAuth, cloud-native, or external-process "
            "credentials that cannot be copied safely into OpenViking. Configure "
            "a static API-key LLM for Hermes, or connect to an OpenViking server "
            "configured separately."
        )
    if api_mode not in {"chat_completions", "anthropic_messages"}:
        raise QuickLocalSetupError(
            f"Hermes' {api_mode or 'unknown'} LLM transport is not supported by "
            "Quick Local. Use an OpenAI-compatible or Anthropic-compatible "
            "API-key provider, or connect to an OpenViking server configured "
            "separately."
        )
    if not api_base:
        raise QuickLocalSetupError(
            "Hermes' LLM provider did not resolve an API base URL."
        )
    if not api_key:
        raise QuickLocalSetupError(
            "Hermes' LLM provider did not resolve reusable static credentials."
        )

    vlm: dict[str, Any]
    if api_mode == "anthropic_messages":
        if not runtime_model.startswith("anthropic/"):
            runtime_model = f"anthropic/{runtime_model}"
        vlm = {
            "provider": "litellm",
            "model": runtime_model,
            "api_key": api_key,
            "api_base": api_base,
        }
    else:
        vlm = {
            "provider": "openai",
            "model": runtime_model,
            "api_key": api_key,
            "api_base": api_base,
        }

    extra_headers = runtime.get("extra_headers")
    if isinstance(extra_headers, dict) and extra_headers:
        vlm["extra_headers"] = dict(extra_headers)
    request_overrides = runtime.get("request_overrides")
    if isinstance(request_overrides, dict):
        extra_body = request_overrides.get("extra_body")
        if isinstance(extra_body, dict) and extra_body:
            vlm["extra_request_body"] = dict(extra_body)
    vlm.update({"temperature": 0.0, "max_retries": 2})
    return vlm


class QuickLocalSetup:
    """Provision one Hermes-home-scoped Quick Local configuration."""

    def __init__(
        self,
        *,
        health_check: HealthCheck,
        progress: Optional[ProgressReporter] = None,
    ) -> None:
        self._health_check = health_check
        self._progress = progress or (lambda _event: None)

    def preflight(self, hermes_home: Path) -> QuickLocalPreflight:
        self._emit(QuickLocalStage.PREFLIGHT, "Checking local requirements...")
        paths = managed_paths(hermes_home)
        reusable_endpoint = find_reusable_endpoint(paths, self._health_check)
        return QuickLocalPreflight(
            paths=paths,
            reusable_endpoint=reusable_endpoint,
            ollama_install_required=(
                reusable_endpoint is None and not ollama_command_available()
            ),
        )

    def provision(
        self,
        *,
        hermes_home: Path,
        allow_ollama_install: bool,
        preflight: Optional[QuickLocalPreflight] = None,
    ) -> QuickLocalSetupResult:
        try:
            return self._provision(
                hermes_home=Path(hermes_home),
                allow_ollama_install=allow_ollama_install,
                preflight=preflight,
            )
        except QuickLocalSetupError:
            raise
        except Exception as exc:
            raise QuickLocalSetupError(f"Quick Local setup failed: {exc}") from exc

    def _provision(
        self,
        *,
        hermes_home: Path,
        allow_ollama_install: bool,
        preflight: Optional[QuickLocalPreflight],
    ) -> QuickLocalSetupResult:
        preflight = preflight or self.preflight(hermes_home)
        if preflight.paths != managed_paths(hermes_home):
            raise QuickLocalSetupError(
                "Quick Local preflight belongs to a different Hermes profile."
            )
        reusable_endpoint = find_reusable_endpoint(
            preflight.paths,
            self._health_check,
        )
        if reusable_endpoint:
            self._emit(
                QuickLocalStage.COMPLETE,
                "Existing Quick Local server is reachable; reusing it.",
            )
            return QuickLocalSetupResult(
                paths=preflight.paths,
                endpoint=reusable_endpoint,
                reused=True,
            )

        vlm = resolve_hermes_vlm_config()
        self._ensure_openviking_installed(preflight.paths)
        self._ensure_ollama(
            paths=preflight.paths,
            allow_install=allow_ollama_install,
        )

        port = find_available_port(
            preferred_endpoint=_configured_endpoint(preflight.paths)
        )
        if port is None:
            last_port = _DEFAULT_PORT + _PORT_ATTEMPTS - 1
            raise QuickLocalSetupError(
                "No available local port was found for OpenViking "
                f"(checked {_DEFAULT_PORT}-{last_port})."
            )
        endpoint = f"http://127.0.0.1:{port}"
        server_config = build_server_config(preflight.paths, vlm, port=port)

        self._validate_generated_config(
            paths=preflight.paths,
            endpoint=endpoint,
            server_config=server_config,
        )

        _prepare_private_directory(preflight.paths.root)
        preflight.paths.workspace.mkdir(parents=True, exist_ok=True)
        atomic_json_write(preflight.paths.server_config, server_config, mode=0o600)
        _write_ovcli_profile(preflight.paths.ovcli_config, endpoint)
        self._emit(
            QuickLocalStage.WRITE_CONFIG,
            f"Saved Quick Local configuration to {preflight.paths.server_config}.",
        )
        self._emit(
            QuickLocalStage.COMPLETE,
            f"Quick Local is configured with {EMBEDDING_MODEL}.",
        )
        return QuickLocalSetupResult(
            paths=preflight.paths,
            endpoint=endpoint,
            reused=False,
        )

    def _ensure_openviking_installed(self, paths: QuickLocalPaths) -> None:
        if openviking_install_satisfies_requirement(paths):
            return
        self._emit(
            QuickLocalStage.INSTALL_OPENVIKING,
            f"Installing {OPENVIKING_REQUIREMENT}...",
        )
        try:
            from hermes_cli.managed_uv import ensure_uv

            uv = ensure_uv()
            if not uv:
                raise QuickLocalSetupError("uv is required to install OpenViking.")
            _prepare_private_directory(paths.root)
            install_env = os.environ.copy()
            install_env["UV_NATIVE_TLS"] = "true"
            install_env["UV_SYSTEM_CERTS"] = "true"
            if not paths.runtime_python.is_file():
                venv_result = subprocess.run(
                    [uv, "venv", str(paths.runtime), "--python", sys.executable],
                    cwd=paths.root,
                    env=install_env,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    timeout=120,
                )
                if venv_result.returncode != 0:
                    raise QuickLocalSetupError(
                        "Could not create the private OpenViking environment."
                    )
            result = subprocess.run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(paths.runtime_python),
                    OPENVIKING_REQUIREMENT,
                ],
                cwd=paths.root,
                env=install_env,
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=600,
            )
        except Exception as exc:
            if isinstance(exc, QuickLocalSetupError):
                raise
            raise QuickLocalSetupError(f"Could not install OpenViking: {exc}") from exc
        if result.returncode != 0 or not openviking_install_satisfies_requirement(
            paths
        ):
            raise QuickLocalSetupError(
                "Could not install a compatible OpenViking version. Review the "
                "installer output above."
            )

    def _ensure_ollama(
        self,
        *,
        paths: QuickLocalPaths,
        allow_install: bool,
    ) -> None:
        _add_windows_ollama_to_path()
        if not ollama_command_available():
            if not allow_install:
                raise OllamaInstallRequired("Ollama is required for Quick Local.")
            self._emit(QuickLocalStage.INSTALL_OLLAMA, "Installing Ollama...")
            if not _install_ollama() or not ollama_command_available():
                raise QuickLocalSetupError(
                    "Ollama installation failed or the ollama command is not yet "
                    "available. Install it manually, restart the terminal, and retry."
                )

        if not _ollama_running():
            self._emit(QuickLocalStage.START_OLLAMA, "Starting Ollama...")
            started, detail = _start_ollama(paths)
            if not started:
                raise QuickLocalSetupError(f"Ollama could not be started: {detail}")

        if not _ollama_model_available(EMBEDDING_MODEL, _ollama_models()):
            self._emit(
                QuickLocalStage.DOWNLOAD_MODEL,
                f"Downloading {EMBEDDING_MODEL} ({_MODEL_DOWNLOAD_SIZE})...",
            )
            if not _pull_ollama_model(EMBEDDING_MODEL):
                raise QuickLocalSetupError(f"Could not download {EMBEDDING_MODEL}.")

    def _validate_generated_config(
        self,
        *,
        paths: QuickLocalPaths,
        endpoint: str,
        server_config: dict[str, Any],
    ) -> None:
        _prepare_private_directory(paths.root)
        with tempfile.TemporaryDirectory(
            prefix="setup-validation-", dir=paths.root
        ) as root:
            validation_root = Path(root)
            validation_config = json.loads(json.dumps(server_config))
            validation_config["storage"]["workspace"] = str(
                validation_root / _WORKSPACE_DIRNAME
            )
            config_path = validation_root / _SERVER_CONFIG_FILENAME
            atomic_json_write(config_path, validation_config, mode=0o600)

            self._emit(
                QuickLocalStage.VALIDATE,
                "Validating OpenViking with a temporary local server...",
            )
            process = _start_validation_server(
                endpoint,
                config_path,
                paths.root.parent,
                paths.server_command,
            )
            try:
                if not _wait_for_health(endpoint, self._health_check):
                    raise QuickLocalSetupError(
                        "OpenViking did not become reachable. Review the server "
                        f"log at {_server_log_path(paths.root.parent)} and retry."
                    )
            finally:
                if not _stop_process(process):
                    raise QuickLocalSetupError(
                        "The temporary OpenViking validation server could not be stopped."
                    )

    def _emit(self, stage: QuickLocalStage, message: str) -> None:
        self._progress(QuickLocalProgress(stage=stage, message=message))


def openviking_install_satisfies_requirement(paths: QuickLocalPaths) -> bool:
    if not paths.runtime_python.is_file() or not paths.server_command.is_file():
        return False
    try:
        result = subprocess.run(
            [
                str(paths.runtime_python),
                "-c",
                "import importlib.metadata; print(importlib.metadata.version('openviking'))",
            ],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=15,
        )
        if result.returncode != 0:
            return False
        version = Version(result.stdout.strip())
    except (InvalidVersion, OSError, subprocess.SubprocessError):
        return False
    return version in _OPENVIKING_VERSION_SPECIFIER


def ollama_command_available() -> bool:
    _add_windows_ollama_to_path()
    return shutil.which("ollama") is not None


def find_available_port(
    *,
    preferred_endpoint: Optional[str] = None,
    first_port: int = _DEFAULT_PORT,
    attempts: int = _PORT_ATTEMPTS,
) -> Optional[int]:
    preferred_port = _endpoint_port(preferred_endpoint)
    candidates = range(first_port, first_port + attempts)
    if preferred_port is not None and preferred_port in candidates:
        candidates = [
            preferred_port,
            *(port for port in candidates if port != preferred_port),
        ]
    for port in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
                candidate.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    return None


def find_reusable_endpoint(
    paths: QuickLocalPaths,
    health_check: HealthCheck,
) -> Optional[str]:
    endpoint = _configured_endpoint(paths)
    if endpoint is None:
        return None
    healthy, _message = health_check(endpoint)
    return endpoint if healthy else None


def _configured_endpoint(paths: QuickLocalPaths) -> Optional[str]:
    if not paths.server_config.is_file() or not paths.ovcli_config.is_file():
        return None
    try:
        server_config = json.loads(paths.server_config.read_text(encoding="utf-8"))
        storage = (
            server_config.get("storage", {}) if isinstance(server_config, dict) else {}
        )
        if not isinstance(storage, dict) or not _paths_equivalent(
            storage.get("workspace"), paths.workspace
        ):
            return None
        profile = json.loads(paths.ovcli_config.read_text(encoding="utf-8"))
        return _normalize_local_endpoint(
            profile.get("url") if isinstance(profile, dict) else ""
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _write_ovcli_profile(path: Path, endpoint: str) -> None:
    atomic_json_write(
        path,
        {"url": endpoint, "actor_peer_id": "hermes"},
        mode=0o600,
    )


def _start_validation_server(
    endpoint: str,
    config_path: Path,
    hermes_home: Path,
    server_command: Path,
) -> subprocess.Popen:
    if not server_command.is_file():
        raise QuickLocalSetupError(
            "openviking-server was not found after installation."
        )
    command = str(server_command)
    host, port = _endpoint_bind(endpoint)
    log_path = _server_log_path(hermes_home)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

    popen_kwargs: dict[str, Any] = windows_detach_popen_kwargs()
    command_args = [
        command,
        "--config",
        str(config_path),
        "--host",
        host,
        "--port",
        str(port),
    ]
    try:
        with log_path.open("ab") as log_file:
            common_kwargs: dict[str, Any] = {
                "stdout": log_file,
                "stderr": log_file,
                "env": child_env,
            }
            try:
                return subprocess.Popen(
                    command_args,
                    **common_kwargs,
                    **popen_kwargs,
                    stdin=subprocess.DEVNULL,
                )
            except OSError:
                if os.name != "nt":
                    raise
                from hermes_cli._subprocess_compat import (
                    windows_detach_flags_without_breakaway,
                )

                return subprocess.Popen(
                    command_args,
                    **common_kwargs,
                    creationflags=windows_detach_flags_without_breakaway(),
                    stdin=subprocess.DEVNULL,
                )
    except Exception as exc:
        raise QuickLocalSetupError(
            f"Could not start the OpenViking validation server: {exc}"
        ) from exc


def _wait_for_health(
    endpoint: str,
    health_check: HealthCheck,
    *,
    timeout_seconds: float = _HEALTH_TIMEOUT_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        healthy, _message = health_check(endpoint)
        if healthy:
            return True
        time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)
    return False


def _stop_process(process: subprocess.Popen) -> bool:
    try:
        if process.poll() is not None:
            return True
        process.terminate()
        process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
        return True
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return False
    try:
        process.kill()
        process.wait(timeout=_PROCESS_STOP_TIMEOUT_SECONDS)
        return True
    except Exception:
        return False


def _install_ollama() -> bool:
    system = platform.system()
    if system == "Darwin":
        brew = shutil.which("brew")
        if brew:
            result = subprocess.run(
                [brew, "install", "ollama"],
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return True
        return _run_ollama_shell_installer()
    if system == "Linux":
        return _run_ollama_shell_installer()
    if system != "Windows":
        return False
    powershell = next(
        (
            executable
            for name in ("powershell.exe", "powershell", "pwsh.exe", "pwsh")
            if (executable := shutil.which(name))
        ),
        None,
    )
    if not powershell:
        return False
    result = subprocess.run(
        _windows_ollama_install_command(powershell),
        check=False,
        stdin=subprocess.DEVNULL,
    )
    _add_windows_ollama_to_path()
    return result.returncode == 0


def _windows_ollama_install_command(powershell: str) -> list[str]:
    return [
        powershell,
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "Invoke-RestMethod https://ollama.com/install.ps1 | Invoke-Expression",
    ]


def _run_ollama_shell_installer() -> bool:
    result = subprocess.run(
        ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
        check=False,
        stdin=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _ollama_running() -> bool:
    try:
        request = Request("http://127.0.0.1:11434/api/tags", method="GET")
        with build_opener(ProxyHandler({})).open(request, timeout=3):
            return True
    except (OSError, TimeoutError):
        return False


def _ollama_models() -> list[str]:
    try:
        request = Request("http://127.0.0.1:11434/api/tags", method="GET")
        with build_opener(ProxyHandler({})).open(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [
            str(model.get("name"))
            for model in payload.get("models", [])
            if isinstance(model, dict) and model.get("name")
        ]
    except (OSError, TimeoutError, json.JSONDecodeError, AttributeError):
        return []


def _ollama_model_available(model_name: str, available: list[str]) -> bool:
    return any(
        installed == model_name
        or installed.startswith(f"{model_name}-")
        or (":" not in model_name and installed.split(":", 1)[0] == model_name)
        for installed in available
    )


def _pull_ollama_model(model_name: str) -> bool:
    command = shutil.which("ollama")
    if not command:
        return False
    try:
        result = subprocess.run(
            [command, "pull", model_name],
            check=False,
            stdin=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except OSError:
        return False


def _start_ollama(paths: QuickLocalPaths) -> tuple[bool, str]:
    command = shutil.which("ollama")
    if not command:
        return False, "ollama command not found"
    log_path = paths.root.parent / "logs" / "ollama.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

    try:
        with log_path.open("ab") as log_file:
            common_kwargs: dict[str, Any] = {
                "stdout": log_file,
                "stderr": log_file,
            }
            try:
                process = subprocess.Popen(
                    [command, "serve"],
                    **common_kwargs,
                    **windows_detach_popen_kwargs(),
                    stdin=subprocess.DEVNULL,
                )
            except OSError:
                if os.name != "nt":
                    raise
                from hermes_cli._subprocess_compat import (
                    windows_detach_flags_without_breakaway,
                )

                process = subprocess.Popen(
                    [command, "serve"],
                    **common_kwargs,
                    creationflags=windows_detach_flags_without_breakaway(),
                    stdin=subprocess.DEVNULL,
                )
    except OSError as exc:
        return False, str(exc)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if _ollama_running():
            return True, "started"
        if process.poll() is not None:
            return False, f"ollama serve exited with status {process.returncode}"
        time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)
    _stop_process(process)
    return False, f"timed out; review {log_path}"


def _add_windows_ollama_to_path() -> None:
    if platform.system() != "Windows":
        return
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if not local_app_data:
        return
    install_dir = Path(local_app_data) / "Programs" / "Ollama"
    if not (install_dir / "ollama.exe").is_file():
        return
    entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    if os.path.normcase(str(install_dir)) not in {
        os.path.normcase(entry) for entry in entries
    }:
        os.environ["PATH"] = os.pathsep.join([str(install_dir), *entries])


def _uses_noncopyable_credentials(provider: str, source: str, api_key: str) -> bool:
    return (
        "oauth" in source
        or "key_cmd" in source
        or provider
        in {
            "bedrock",
            "copilot-acp",
            "minimax-oauth",
            "nous",
            "openai-codex",
            "qwen-oauth",
            "vertex",
            "xai-oauth",
        }
        or api_key.startswith("sk-ant-oat")
        or api_key == "aws-sdk"
    )


def _prepare_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _server_log_path(hermes_home: Path) -> Path:
    return hermes_home / "logs" / "openviking-server.log"


def _normalize_local_endpoint(value: Any) -> Optional[str]:
    endpoint = _clean_value(value).rstrip("/")
    if not endpoint:
        return None
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"
    parsed = urlparse(endpoint)
    if parsed.scheme.lower() != "http":
        return None
    if (parsed.hostname or "").lower() not in {"localhost", "127.0.0.1", "::1"}:
        return None
    host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
    return f"http://{host}:{parsed.port or _DEFAULT_PORT}"


def _endpoint_bind(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint)
    return parsed.hostname or "127.0.0.1", parsed.port or _DEFAULT_PORT


def _endpoint_port(endpoint: Optional[str]) -> Optional[int]:
    if not endpoint:
        return None
    try:
        return urlparse(endpoint).port or _DEFAULT_PORT
    except ValueError:
        return None


def _paths_equivalent(left: Any, right: Path) -> bool:
    if not isinstance(left, (str, os.PathLike)):
        return False
    try:
        return Path(left).expanduser().resolve() == right.expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _clean_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
