"""Reusable, non-interactive OpenViking Quick Local provisioning.

User interfaces own prompts and rendering.  This module owns the bounded
installation, Hermes-profile-scoped configuration, and temporary validation
needed to produce a ready-to-link OpenViking CLI profile.
"""

from __future__ import annotations

import json
import os
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

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

from utils import atomic_json_write

DEPLOYMENT = "quick_local"
EMBEDDING_MODEL = "bge-small-zh-v1.5-f16"
EMBEDDING_DIMENSION = 512
OPENVIKING_REQUIREMENT = "openviking[local-embed]>=0.4.16,<0.6"

_OPENVIKING_REQUIREMENT = Requirement(OPENVIKING_REQUIREMENT)
_OPENVIKING_VERSION_SPECIFIER = _OPENVIKING_REQUIREMENT.specifier
_ROOT_DIRNAME = "openviking"
_SERVER_CONFIG_FILENAME = "ov.conf"
_OVCLI_CONFIG_FILENAME = "ovcli.conf"
_WORKSPACE_DIRNAME = "data"
_MODEL_CACHE_DIRNAME = "models"
_DEFAULT_PORT = 1933
_PORT_ATTEMPTS = 20
_MODEL_DOWNLOAD_SIZE = "approximately 46 MiB"
# OpenViking downloads the built-in model while its server lifespan starts.
# Give a slow first download a bounded ten minutes, then leave another minute
# for model loading and service initialization.
_MODEL_PREPARATION_TIMEOUT_SECONDS = 600.0
_HEALTH_TIMEOUT_SECONDS = _MODEL_PREPARATION_TIMEOUT_SECONDS + 60.0
_HEALTH_POLL_INTERVAL_SECONDS = 0.5
_PROCESS_STOP_TIMEOUT_SECONDS = 10.0


class QuickLocalStage(str, Enum):
    PREFLIGHT = "preflight"
    INSTALL_OPENVIKING = "install_openviking"
    PREPARE_EMBEDDING = "prepare_embedding"
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
    model_cache: Path

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


@dataclass(frozen=True)
class QuickLocalSetupResult:
    paths: QuickLocalPaths
    endpoint: str
    reused: bool
    server_restart_required: bool = False


class QuickLocalSetupError(RuntimeError):
    """Quick Local could not finish without partially activating it."""


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
        model_cache=root / _MODEL_CACHE_DIRNAME,
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
                "provider": "local",
                "model": EMBEDDING_MODEL,
                "dimension": EMBEDDING_DIMENSION,
                "cache_dir": str(paths.model_cache),
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
    raw_api_base = runtime.get("base_url")
    raw_api_key = runtime.get("api_key")
    api_base = _clean_value(raw_api_base)
    api_key = _clean_value(raw_api_key)

    if raw_api_base is not None and not isinstance(raw_api_base, str):
        raise QuickLocalSetupError(
            "Hermes' LLM provider API base URL must be a string."
        )
    if not api_base:
        raise QuickLocalSetupError(
            "Hermes' LLM provider did not resolve an API base URL."
        )
    if raw_api_key is None or (isinstance(raw_api_key, str) and not api_key):
        raise QuickLocalSetupError(
            "Hermes' LLM provider did not resolve reusable static credentials."
        )
    if not _has_copyable_static_credentials(provider, source, api_key):
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
        )

    def provision(
        self,
        *,
        hermes_home: Path,
        preflight: Optional[QuickLocalPreflight] = None,
    ) -> QuickLocalSetupResult:
        try:
            return self._provision(
                hermes_home=Path(hermes_home),
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
        preflight: Optional[QuickLocalPreflight],
    ) -> QuickLocalSetupResult:
        preflight = preflight or self.preflight(hermes_home)
        if preflight.paths != managed_paths(hermes_home):
            raise QuickLocalSetupError(
                "Quick Local preflight belongs to a different Hermes profile."
            )
        vlm = resolve_hermes_vlm_config()
        runtime_changed = self._ensure_openviking_installed(preflight.paths)

        reusable_endpoint = find_reusable_endpoint(preflight.paths, self._health_check)
        if reusable_endpoint:
            port = _endpoint_port(reusable_endpoint)
            if port is None:
                raise QuickLocalSetupError(
                    "Quick Local's saved endpoint does not contain a valid port."
                )
            server_config = build_server_config(preflight.paths, vlm, port=port)
            config_changed = not _stored_server_config_matches(
                preflight.paths, server_config
            )
            if config_changed:
                _prepare_private_directory(preflight.paths.root)
                atomic_json_write(
                    preflight.paths.server_config,
                    server_config,
                    mode=0o600,
                )
                _write_ovcli_profile(preflight.paths.ovcli_config, reusable_endpoint)
                self._emit(
                    QuickLocalStage.WRITE_CONFIG,
                    "Updated Quick Local's saved Hermes LLM settings; the running "
                    "server must be restarted before it can use them.",
                )
            self._emit(
                QuickLocalStage.COMPLETE,
                "Existing Quick Local server is reachable; reusing it.",
            )
            return QuickLocalSetupResult(
                paths=preflight.paths,
                endpoint=reusable_endpoint,
                reused=True,
                server_restart_required=runtime_changed or config_changed,
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

    def _ensure_openviking_installed(self, paths: QuickLocalPaths) -> bool:
        """Ensure a compatible runtime, returning whether installation was needed."""
        if openviking_install_satisfies_requirement(paths):
            return False
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
        return True

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

            _prepare_private_directory(paths.model_cache)
            self._emit(
                QuickLocalStage.PREPARE_EMBEDDING,
                f"Preparing {EMBEDDING_MODEL}; its {_MODEL_DOWNLOAD_SIZE} model "
                "is downloaded once if needed...",
            )
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
            primary_error: BaseException | None = None
            try:
                if not _wait_for_health(
                    endpoint,
                    self._health_check,
                    process=process,
                ):
                    returncode = process.poll()
                    if returncode is not None:
                        raise QuickLocalSetupError(
                            "OpenViking exited before becoming reachable "
                            f"(status {returncode}). Review the server log at "
                            f"{_server_log_path(paths.root.parent)} and retry."
                        )
                    raise QuickLocalSetupError(
                        "OpenViking did not become reachable before the local model "
                        "preparation timeout. Review the server log at "
                        f"{_server_log_path(paths.root.parent)} and retry."
                    )
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                if not _stop_process(process):
                    message = "The temporary OpenViking validation server could not be stopped."
                    if primary_error is None:
                        raise QuickLocalSetupError(message)
                    primary_error.add_note(message)

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
                "import importlib.metadata; import llama_cpp; "
                "print(importlib.metadata.version('openviking'))",
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
        if _can_bind_local_port("127.0.0.1", port):
            return port
    return None


def _can_bind_local_port(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as candidate:
            candidate.bind((host, port))
        return True
    except OSError:
        return False


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
    if not _can_bind_local_port(host, port):
        raise QuickLocalSetupError(
            f"Local port {host}:{port} became unavailable before OpenViking "
            "could start. Retry Quick Local setup."
        )
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
    process: Optional[subprocess.Popen] = None,
    timeout_seconds: float = _HEALTH_TIMEOUT_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        healthy, _message = health_check(endpoint)
        if healthy:
            return True
        if process is not None and process.poll() is not None:
            return False
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


def _has_copyable_static_credentials(
    provider: str,
    source: str,
    api_key: str,
) -> bool:
    """Return whether Hermes explicitly classifies this credential as static."""

    if not api_key or api_key.startswith("sk-ant-oat") or api_key == "aws-sdk":
        return False

    if provider == "custom":
        return source in {"direct-alias", "env/config"} or source.startswith((
            "custom_provider:",
            "pool:",
        ))
    if provider == "openrouter":
        return source == "env/config" or source.startswith((
            "credential_pool:",
            "env:",
            "manual:",
            "pool:",
        ))

    from hermes_cli.auth import PROVIDER_REGISTRY

    provider_config = PROVIDER_REGISTRY.get(provider)
    if (
        provider == "copilot"
        or provider_config is None
        or provider_config.auth_type != "api_key"
    ):
        return False

    if source in {"config", "default", "env", "local-offline"}:
        return True
    api_key_sources = {value.lower() for value in provider_config.api_key_env_vars}
    if source in api_key_sources:
        return True
    if source.startswith("env:"):
        return source.removeprefix("env:") in api_key_sources
    if source.startswith("credential_pool:"):
        return source.removeprefix("credential_pool:") == provider
    return source.startswith("manual:")


def _stored_server_config_matches(
    paths: QuickLocalPaths,
    expected: Mapping[str, Any],
) -> bool:
    try:
        saved = json.loads(paths.server_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return saved == expected


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
