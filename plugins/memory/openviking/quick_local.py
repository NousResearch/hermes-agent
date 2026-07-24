"""Reusable, interface-neutral OpenViking Quick Local provisioning.

The CLI and future graphical surfaces supply decisions and render progress.
This module owns the deterministic work: dependency preflight, installation,
configuration, validation, and the private ovcli profile written for one
Hermes home.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from utils import atomic_json_write

from . import local_server

__all__ = [
    "DEFAULT_ACTOR_PEER_ID",
    "DEPLOYMENT",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_MODEL",
    "OllamaInstallRequired",
    "QuickLocalPaths",
    "QuickLocalPreflight",
    "QuickLocalProgress",
    "QuickLocalSetup",
    "QuickLocalSetupCancelled",
    "QuickLocalSetupError",
    "QuickLocalSetupResult",
    "QuickLocalStage",
    "build_server_config",
    "clear_managed_settings",
    "find_available_port",
    "find_reusable_endpoint",
    "managed_paths",
    "managed_server_config_path",
    "repair_server_address",
    "resolve_hermes_vlm_config",
]

DEPLOYMENT = "quick_local"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_DIMENSION = 1024
DEFAULT_ACTOR_PEER_ID = "hermes"

_ROOT_DIRNAME = "openviking"
_SERVER_CONFIG_FILENAME = "ov.conf"
_OVCLI_CONFIG_FILENAME = "ovcli.conf"
_WORKSPACE_DIRNAME = "data"
_FIRST_SERVER_PORT = 1933
_SERVER_PORT_ATTEMPTS = 20
_MODEL_DOWNLOAD_SIZE = "approximately 639 MB"
_VALIDATION_DRAIN_TIMEOUT_SECONDS = 60.0
_VALIDATION_REQUEST_SETTLE_SECONDS = 0.5


class QuickLocalStage(str, Enum):
    PREFLIGHT = "preflight"
    INSTALL_OPENVIKING = "install_openviking"
    INSTALL_OLLAMA = "install_ollama"
    START_OLLAMA = "start_ollama"
    DOWNLOAD_MODEL = "download_model"
    WRITE_CONFIG = "write_config"
    START_VALIDATION = "start_validation"
    WAIT_FOR_HEALTH = "wait_for_health"
    DRAIN_VALIDATION = "drain_validation"
    COMPLETE = "complete"


@dataclass(frozen=True)
class QuickLocalProgress:
    stage: QuickLocalStage
    message: str


@dataclass(frozen=True)
class QuickLocalPaths:
    root: Path
    server_config: Path
    ovcli_config: Path
    workspace: Path

    @property
    def config(self) -> Path:
        """Backward-compatible name used by existing provider call sites."""

        return self.server_config


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
    """Quick Local could not complete without leaving partial activation."""


class QuickLocalSetupCancelled(QuickLocalSetupError):
    """The caller cancelled Quick Local provisioning."""


class OllamaInstallRequired(QuickLocalSetupError):
    """Provisioning requires explicit permission to install Ollama."""


ProgressReporter = Callable[[QuickLocalProgress], None]
HealthCheck = Callable[[str], tuple[bool, str]]


def managed_paths(hermes_home: Path) -> QuickLocalPaths:
    root = hermes_home / _ROOT_DIRNAME
    return QuickLocalPaths(
        root=root,
        server_config=root / _SERVER_CONFIG_FILENAME,
        ovcli_config=root / _OVCLI_CONFIG_FILENAME,
        workspace=root / _WORKSPACE_DIRNAME,
    )


def managed_server_config_path(provider_config: Mapping[str, Any]) -> Optional[Path]:
    if provider_config.get("deployment") != DEPLOYMENT:
        return None
    raw_path = _clean_value(provider_config.get("server_config_path"))
    return Path(raw_path).expanduser() if raw_path else None


def clear_managed_settings(provider_config: dict[str, Any]) -> None:
    provider_config.pop("deployment", None)
    provider_config.pop("server_config_path", None)


def build_server_config(
    paths: QuickLocalPaths,
    vlm: Mapping[str, Any],
    *,
    port: int = _FIRST_SERVER_PORT,
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


def resolve_hermes_vlm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate Hermes' effective static LLM into OpenViking VLM config."""

    model_config = config.get("model", {}) if isinstance(config, Mapping) else {}
    if isinstance(model_config, str):
        model_config = {"default": model_config}
    if not isinstance(model_config, Mapping):
        model_config = {}
    model = _clean_value(
        model_config.get("default")
        or model_config.get("model")
        or model_config.get("name")
    )
    requested_provider = _clean_value(model_config.get("provider")) or None
    if not model:
        raise QuickLocalSetupError("Hermes has no default LLM model configured.")

    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(
        requested=requested_provider,
        target_model=model,
    )
    runtime_model = _clean_value(runtime.get("model")) or model
    provider = _clean_value(runtime.get("provider")).lower()
    api_mode = _clean_value(runtime.get("api_mode")).lower()
    source = _clean_value(runtime.get("source")).lower()
    api_key = _clean_value(runtime.get("api_key"))
    api_base = _clean_value(runtime.get("base_url"))

    ephemeral_auth = (
        "oauth" in source
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
    if ephemeral_auth:
        raise QuickLocalSetupError(
            "Hermes is using refreshed OAuth, cloud-native, or external-process "
            "credentials that cannot be copied safely into OpenViking. Configure "
            "a static API-key LLM for Hermes, or connect to an OpenViking server "
            "configured separately."
        )
    if api_mode not in {"chat_completions", "anthropic_messages"}:
        raise QuickLocalSetupError(
            f"Hermes' {api_mode or 'unknown'} LLM transport is not supported by "
            "quick local setup. Use an OpenAI-compatible or Anthropic-compatible "
            "API-key provider, or connect to an OpenViking server configured "
            "separately."
        )
    if not api_base:
        raise QuickLocalSetupError(
            "Hermes' LLM provider did not resolve an API base URL."
        )
    if not api_key:
        raise QuickLocalSetupError(
            "Hermes' LLM provider did not resolve usable credentials."
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
    """Provision one Hermes-home-scoped Quick Local deployment."""

    def __init__(
        self,
        *,
        health_check: HealthCheck,
        progress: Optional[ProgressReporter] = None,
    ) -> None:
        self._health_check = health_check
        self._progress = progress or (lambda _event: None)

    def preflight(self, hermes_home: Path) -> QuickLocalPreflight:
        """Inspect reusable state without mutating the selected Hermes home."""

        try:
            paths = managed_paths(hermes_home)
            self._emit(
                QuickLocalStage.PREFLIGHT,
                "Checking local OpenViking setup...",
            )
            reusable_endpoint = find_reusable_endpoint(
                paths,
                self._health_check,
            )
            return QuickLocalPreflight(
                paths=paths,
                reusable_endpoint=reusable_endpoint,
                ollama_install_required=(
                    reusable_endpoint is None and not _ollama_command_available()
                ),
            )
        except QuickLocalSetupError:
            raise
        except Exception as exc:
            raise QuickLocalSetupError(f"Quick Local preflight failed: {exc}") from exc

    def provision(
        self,
        *,
        hermes_home: Path,
        hermes_config: Mapping[str, Any],
        allow_ollama_install: bool,
        cancel_event: Optional[threading.Event] = None,
        preflight: Optional[QuickLocalPreflight] = None,
    ) -> QuickLocalSetupResult:
        """Provision Quick Local and expose only stable domain errors."""

        try:
            return self._provision(
                hermes_home=hermes_home,
                hermes_config=hermes_config,
                allow_ollama_install=allow_ollama_install,
                cancel_event=cancel_event,
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
        hermes_config: Mapping[str, Any],
        allow_ollama_install: bool,
        cancel_event: Optional[threading.Event],
        preflight: Optional[QuickLocalPreflight],
    ) -> QuickLocalSetupResult:
        self._check_cancelled(cancel_event)
        preflight = preflight or self.preflight(hermes_home)
        expected_paths = managed_paths(hermes_home)
        if preflight.paths != expected_paths:
            raise QuickLocalSetupError(
                "Quick Local preflight belongs to a different Hermes home."
            )
        reusable_endpoint = find_reusable_endpoint(
            preflight.paths,
            self._health_check,
        )
        if reusable_endpoint is not None:
            repair_server_address(
                preflight.paths,
                reusable_endpoint,
            )
            self._emit(
                QuickLocalStage.COMPLETE,
                "Existing quick local server is reachable; reusing it.",
            )
            return QuickLocalSetupResult(
                paths=preflight.paths,
                endpoint=reusable_endpoint,
                reused=True,
            )

        self._check_cancelled(cancel_event)
        vlm = resolve_hermes_vlm_config(hermes_config)
        preflight.paths.workspace.mkdir(parents=True, exist_ok=True)
        self._ensure_openviking_installed(cancel_event)
        self._ensure_ollama(
            allow_install=allow_ollama_install,
            cancel_event=cancel_event,
        )

        self._check_cancelled(cancel_event)
        port = find_available_port()
        if port is None:
            last_port = _FIRST_SERVER_PORT + _SERVER_PORT_ATTEMPTS - 1
            raise QuickLocalSetupError(
                "No available local port was found for OpenViking "
                f"(checked {_FIRST_SERVER_PORT}-{last_port})."
            )

        server_config = build_server_config(preflight.paths, vlm, port=port)
        atomic_json_write(preflight.paths.server_config, server_config, mode=0o600)
        endpoint = f"http://127.0.0.1:{port}"
        self._emit(
            QuickLocalStage.WRITE_CONFIG,
            f"Wrote Quick Local configuration to {preflight.paths.server_config}.",
        )

        self._validate_generated_config(
            paths=preflight.paths,
            endpoint=endpoint,
            server_config=server_config,
            cancel_event=cancel_event,
        )
        _write_managed_ovcli_profile(preflight.paths.ovcli_config, endpoint)
        self._emit(
            QuickLocalStage.COMPLETE,
            f"Quick local server configured with {EMBEDDING_MODEL}.",
        )
        return QuickLocalSetupResult(
            paths=preflight.paths,
            endpoint=endpoint,
            reused=False,
        )

    def _ensure_openviking_installed(
        self,
        cancel_event: Optional[threading.Event],
    ) -> None:
        if local_server.server_command():
            return
        self._check_cancelled(cancel_event)
        self._emit(
            QuickLocalStage.INSTALL_OPENVIKING,
            "Installing OpenViking...",
        )
        try:
            from hermes_cli.tools_config import _pip_install

            previous_native_tls = os.environ.get("UV_NATIVE_TLS")
            os.environ["UV_NATIVE_TLS"] = "true"
            try:
                result = _pip_install(
                    ["openviking"],
                    timeout=600,
                    capture_output=False,
                )
            finally:
                if previous_native_tls is None:
                    os.environ.pop("UV_NATIVE_TLS", None)
                else:
                    os.environ["UV_NATIVE_TLS"] = previous_native_tls
        except Exception as exc:
            raise QuickLocalSetupError(f"Could not install OpenViking: {exc}") from exc
        self._check_cancelled(cancel_event)
        if result.returncode != 0 or not local_server.server_command():
            raise QuickLocalSetupError(
                "Could not install OpenViking. Review the installer output above."
            )

    def _ensure_ollama(
        self,
        *,
        allow_install: bool,
        cancel_event: Optional[threading.Event],
    ) -> None:
        self._check_cancelled(cancel_event)
        try:
            ollama = importlib.import_module("openviking_cli.utils.ollama")
        except Exception as exc:
            raise QuickLocalSetupError(
                f"OpenViking's Ollama support could not be loaded: {exc}"
            ) from exc

        _add_windows_ollama_to_path()
        if not ollama.is_ollama_installed():
            if not allow_install:
                raise OllamaInstallRequired("Ollama is required for Quick Local setup.")
            self._emit(QuickLocalStage.INSTALL_OLLAMA, "Installing Ollama...")
            if not _install_ollama(ollama):
                raise QuickLocalSetupError(
                    "Ollama installation failed. Install it manually, then retry."
                )
            if not ollama.is_ollama_installed():
                raise QuickLocalSetupError(
                    "Ollama installation completed, but the ollama command is not "
                    "available. Restart the terminal, then retry."
                )

        self._check_cancelled(cancel_event)
        if not ollama.check_ollama_running():
            self._emit(QuickLocalStage.START_OLLAMA, "Starting Ollama...")
            start_result = ollama.start_ollama()
            if not start_result.success:
                detail = start_result.stderr_output.strip() or start_result.message
                raise QuickLocalSetupError(f"Ollama could not be started: {detail}")

        self._check_cancelled(cancel_event)
        if not ollama.is_model_available(
            EMBEDDING_MODEL,
            ollama.get_ollama_models(),
        ):
            self._emit(
                QuickLocalStage.DOWNLOAD_MODEL,
                f"Downloading {EMBEDDING_MODEL} ({_MODEL_DOWNLOAD_SIZE})...",
            )
            if not ollama.ollama_pull_model(EMBEDDING_MODEL):
                raise QuickLocalSetupError(f"Could not download {EMBEDDING_MODEL}.")
        self._check_cancelled(cancel_event)

    def _validate_generated_config(
        self,
        *,
        paths: QuickLocalPaths,
        endpoint: str,
        server_config: dict[str, Any],
        cancel_event: Optional[threading.Event],
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="setup-validation-",
            dir=paths.root,
        ) as validation_root:
            validation_root_path = Path(validation_root)
            validation_config = json.loads(json.dumps(server_config))
            validation_config["storage"]["workspace"] = str(
                validation_root_path / _WORKSPACE_DIRNAME
            )
            validation_config_path = validation_root_path / _SERVER_CONFIG_FILENAME
            atomic_json_write(
                validation_config_path,
                validation_config,
                mode=0o600,
            )

            self._check_cancelled(cancel_event)
            self._emit(
                QuickLocalStage.START_VALIDATION,
                "Starting a temporary OpenViking validation server...",
            )
            start_result = local_server.start_local_server(
                endpoint,
                hermes_home=paths.root.parent,
                config_path=validation_config_path,
            )
            if start_result.process is None:
                raise QuickLocalSetupError(start_result.message)

            validation_succeeded = False
            validation_drained = False
            stop_succeeded = False
            try:
                self._emit(
                    QuickLocalStage.WAIT_FOR_HEALTH,
                    "Waiting for OpenViking server to become reachable...",
                )
                validation_succeeded = local_server.wait_for_health(
                    endpoint,
                    self._health_check,
                    timeout_seconds=local_server.AUTOSTART_TIMEOUT_SECONDS,
                    cancel_event=cancel_event,
                )
                if validation_succeeded:
                    self._check_cancelled(cancel_event)
                    self._emit(
                        QuickLocalStage.DRAIN_VALIDATION,
                        "Finishing temporary OpenViking validation work...",
                    )
                    _wait_for_processing(
                        endpoint,
                        timeout_seconds=_VALIDATION_DRAIN_TIMEOUT_SECONDS,
                    )
                    # OpenViking can finish sending the response just before
                    # its ASGI middleware exits. Avoid delivering SIGTERM
                    # during that narrow teardown window.
                    if cancel_event is None:
                        time.sleep(_VALIDATION_REQUEST_SETTLE_SECONDS)
                    elif cancel_event.wait(_VALIDATION_REQUEST_SETTLE_SECONDS):
                        raise QuickLocalSetupCancelled(
                            "Quick Local setup was cancelled."
                        )
                    validation_drained = True
            finally:
                stop_succeeded = local_server.stop_owned_process(start_result.process)

            self._check_cancelled(cancel_event)
            if not stop_succeeded:
                raise QuickLocalSetupError(
                    "The temporary OpenViking validation server could not be "
                    "stopped; check the server log before retrying setup."
                )
            if not validation_succeeded:
                raise QuickLocalSetupError(
                    "OpenViking server did not become reachable. Check the "
                    "server log and retry."
                )
            if not validation_drained:
                raise QuickLocalSetupError(
                    "OpenViking validation work did not finish before shutdown."
                )

    def _emit(self, stage: QuickLocalStage, message: str) -> None:
        self._progress(QuickLocalProgress(stage=stage, message=message))

    @staticmethod
    def _check_cancelled(
        cancel_event: Optional[threading.Event],
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise QuickLocalSetupCancelled("Quick Local setup was cancelled.")


def find_available_port(
    *,
    first_port: int = _FIRST_SERVER_PORT,
    attempts: int = _SERVER_PORT_ATTEMPTS,
) -> Optional[int]:
    for port in range(first_port, first_port + attempts):
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
    """Return a healthy endpoint proven to use this managed workspace."""

    if not paths.server_config.is_file() or not paths.ovcli_config.is_file():
        return None

    try:
        server_config = json.loads(paths.server_config.read_text(encoding="utf-8"))
        storage = (
            server_config.get("storage", {}) if isinstance(server_config, dict) else {}
        )
        if not isinstance(storage, dict) or not _paths_equivalent(
            storage.get("workspace"),
            paths.workspace,
        ):
            return None

        profile = json.loads(paths.ovcli_config.read_text(encoding="utf-8"))
        endpoint = _normalize_local_endpoint(
            profile.get("url") if isinstance(profile, dict) else ""
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None

    if endpoint is None:
        return None
    healthy, _message = health_check(endpoint)
    return endpoint if healthy else None


def repair_server_address(paths: QuickLocalPaths, endpoint: str) -> None:
    """Keep persisted startup config aligned with a reusable server."""

    server_config = json.loads(paths.server_config.read_text(encoding="utf-8"))
    host, port = local_server.endpoint_bind(endpoint)
    server_config["server"] = {"host": host, "port": port}
    atomic_json_write(paths.server_config, server_config, mode=0o600)


def _write_managed_ovcli_profile(path: Path, endpoint: str) -> None:
    atomic_json_write(
        path,
        {
            "url": endpoint,
            "actor_peer_id": DEFAULT_ACTOR_PEER_ID,
        },
        mode=0o600,
    )


def _wait_for_processing(
    endpoint: str,
    *,
    timeout_seconds: float,
) -> None:
    """Drain bootstrap work before stopping a temporary validation server."""

    request = Request(
        f"{endpoint.rstrip('/')}/api/v1/system/wait",
        data=json.dumps({"timeout": timeout_seconds}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = build_opener(ProxyHandler({})).open(
        request,
        timeout=timeout_seconds + 1.0,
    )
    with response:
        payload = json.loads(response.read())
    result = payload.get("result") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "ok"
        or not isinstance(result, dict)
    ):
        raise QuickLocalSetupError(
            "OpenViking validation queue returned an invalid response."
        )

    failed_queues: list[tuple[str, int]] = []
    for queue_name, queue_status in result.items():
        if not isinstance(queue_status, dict):
            raise QuickLocalSetupError(
                "OpenViking validation queue returned an invalid response."
            )
        error_count = queue_status.get("error_count", 0)
        errors = queue_status.get("errors", [])
        if (
            not isinstance(error_count, int)
            or isinstance(error_count, bool)
            or error_count < 0
            or not isinstance(errors, list)
        ):
            raise QuickLocalSetupError(
                "OpenViking validation queue returned an invalid response."
            )
        if error_count or errors:
            failed_queues.append((str(queue_name), max(error_count, len(errors))))

    if failed_queues:
        summary = ", ".join(
            f"{queue_name}: {error_count}" for queue_name, error_count in failed_queues
        )
        raise QuickLocalSetupError(
            "OpenViking validation reported processing errors "
            f"({summary}). Check the OpenViking server log and Hermes LLM "
            "configuration, then retry."
        )


def _ollama_command_available() -> bool:
    _add_windows_ollama_to_path()
    return shutil.which("ollama") is not None


def _add_windows_ollama_to_path() -> None:
    if not _is_windows():
        return
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if not local_app_data:
        return
    install_dir = Path(local_app_data) / "Programs" / "Ollama"
    if not (install_dir / "ollama.exe").is_file():
        return
    entries = os.environ.get("PATH", "").split(os.pathsep)
    normalized = {os.path.normcase(entry) for entry in entries if entry}
    if os.path.normcase(str(install_dir)) not in normalized:
        os.environ["PATH"] = os.pathsep.join([
            str(install_dir),
            *(entry for entry in entries if entry),
        ])


def _install_ollama(ollama) -> bool:
    if not _is_windows():
        return bool(ollama.install_ollama())

    powershell = next(
        (
            executable
            for name in (
                "powershell.exe",
                "powershell",
                "pwsh.exe",
                "pwsh",
            )
            if (executable := shutil.which(name))
        ),
        None,
    )
    if not powershell:
        return False
    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Invoke-RestMethod https://ollama.com/install.ps1 | Invoke-Expression",
        ],
        check=False,
    )
    _add_windows_ollama_to_path()
    return result.returncode == 0


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
    return f"http://{host}:{parsed.port or local_server.DEFAULT_PORT}"


def _paths_equivalent(left: Any, right: Path) -> bool:
    if not isinstance(left, (str, os.PathLike)):
        return False
    try:
        return Path(left).expanduser().resolve() == right.expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _clean_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_windows() -> bool:
    return os.name == "nt"
