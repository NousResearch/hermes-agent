"""Configuration, validation, and runtime helpers for OpenViking memory."""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from utils import atomic_json_write

from .client import _OpenVikingHTTPError, _VikingClient, _format_openviking_exception
from .constants import (
    _AGENT_PROMPT_LABEL,
    _DEFAULT_AGENT,
    _DEFAULT_ENDPOINT,
    _LOCAL_OPENVIKING_AUTOSTART_TIMEOUT,
    _LOCAL_OPENVIKING_HOSTS,
    _OPENVIKING_ENV_KEYS,
    _OPENVIKING_RESPONDED_FAILURE_PREFIX,
    _OPENVIKING_SERVER_LOG_RELATIVE_PATH,
    _OPENVIKING_SERVICE_ENDPOINT,
    _OVCLI_CONFIG_ENV,
    _OVCLI_DEFAULT_RELATIVE_PATH,
    _OVCLI_SAVED_PREFIX,
    _SETUP_CANCELLED,
)

logger = logging.getLogger(__name__)


def _facade_attr(name: str, default: Any) -> Any:
    facade = sys.modules.get(__package__)
    return getattr(facade, name, default) if facade is not None else default


def _viking_client_cls():
    return _facade_attr("_VikingClient", _VikingClient)


@dataclass(frozen=True)
class _OvcliProfile:
    source: str
    name: str
    path: Path
    data: dict
    values: dict
    is_active: bool = False




def _clean_config_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _default_ovcli_config_path() -> Path:
    return Path.home() / _OVCLI_DEFAULT_RELATIVE_PATH


def _resolve_ovcli_config_path(config_path: str = "") -> Path:
    env_path = os.environ.get(_OVCLI_CONFIG_ENV, "").strip()
    if env_path:
        return Path(env_path).expanduser()
    if config_path:
        return Path(config_path).expanduser()
    return _default_ovcli_config_path()


def _ovcli_config_dir() -> Path:
    return _default_ovcli_config_path().parent


def _load_ovcli_config(path: Optional[Path] = None) -> dict:
    config_path = path or _resolve_ovcli_config_path()
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"OpenViking CLI config must be a JSON object: {config_path}")
    return data


def _connection_values_from_ovcli(data: dict) -> dict:
    api_key = _clean_config_value(data.get("api_key")) or _clean_config_value(data.get("root_api_key"))
    root_api_key = _clean_config_value(data.get("root_api_key"))
    send_identity = not api_key or api_key == root_api_key
    account = _clean_config_value(data.get("account") or data.get("account_id"))
    user = _clean_config_value(data.get("user") or data.get("user_id"))
    return {
        "endpoint": _normalize_openviking_url(data.get("url")),
        "api_key": api_key,
        "root_api_key": root_api_key,
        "account": account if send_identity else "",
        "user": user if send_identity else "",
        "agent": _clean_config_value(data.get("actor_peer_id") or data.get("agent_id")),
    }


def _is_valid_ovcli_profile_name(name: str) -> bool:
    if not name or name.strip() != name or name.startswith("."):
        return False
    if "/" in name or "\\" in name:
        return False
    return all(ch.isascii() and (ch.isalnum() or ch in {"-", "_"}) for ch in name)


def _validate_openviking_identity_value(value: str, *, field: str) -> tuple[bool, str, str]:
    label = "Account ID" if field == "account" else "User ID"
    identifier = "account_id" if field == "account" else "user_id"
    trimmed = value.strip()
    if not trimmed:
        return False, f"{label} cannot be empty.", ""
    if trimmed != value:
        return False, f"{label} cannot start or end with whitespace.", ""
    if field == "account" and trimmed.startswith("_"):
        return False, "Account ID cannot start with '_'.", ""
    if not all(ch.isascii() and (ch.isalnum() or ch in {"_", "-", ".", "@"}) for ch in trimmed):
        return False, f"{label} can only contain letters, numbers, '_', '-', '.', and '@'.", ""
    if trimmed.count("@") > 1:
        return False, f"{identifier} must have at most one '@'.", ""
    return True, "", trimmed


def _normalize_openviking_url(url: str) -> str:
    trimmed = _clean_config_value(url).rstrip("/")
    if not trimmed:
        return _DEFAULT_ENDPOINT
    lower = trimmed.lower()
    if lower in {"::1", "[::1]"}:
        return "http://[::1]:1933"
    if lower.startswith("[::1]:"):
        return f"http://[::1]:{trimmed.rsplit(':', 1)[1]}"
    if lower.startswith("::1:"):
        return f"http://[::1]:{trimmed.rsplit(':', 1)[1]}"
    if "://" in trimmed:
        return trimmed
    host, _sep, port = trimmed.partition(":")
    if host.lower() in {"localhost", "127.0.0.1"}:
        return f"http://{host}:{port or '1933'}"
    return trimmed


def _load_profile(path: Path, *, source: str, name: str) -> Optional[_OvcliProfile]:
    try:
        data = _load_ovcli_config(path)
    except Exception as e:
        logger.debug("Skipping invalid OpenViking CLI config %s: %s", path, e)
        return None
    return _OvcliProfile(
        source=source,
        name=name,
        path=path,
        data=data,
        values=_connection_values_from_ovcli(data),
    )


def _profile_identity(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except OSError:
        return str(path.expanduser())


def _profiles_equivalent(left: _OvcliProfile, right: _OvcliProfile) -> bool:
    return left.values == right.values


def _discover_ovcli_profiles() -> list[_OvcliProfile]:
    profiles: list[_OvcliProfile] = []
    seen_paths: set[str] = set()

    def add(path: Path, *, source: str, name: str) -> None:
        if not path.exists() or not path.is_file():
            return
        identity = _profile_identity(path)
        if identity in seen_paths:
            return
        profile = _load_profile(path, source=source, name=name)
        if profile is None:
            return
        seen_paths.add(identity)
        profiles.append(profile)

    env_path = os.environ.get(_OVCLI_CONFIG_ENV, "").strip()
    if env_path:
        add(Path(env_path).expanduser(), source="env", name=_OVCLI_CONFIG_ENV)

    active_path = _default_ovcli_config_path()
    active_profile = _load_profile(active_path, source="active", name="active") if active_path.exists() else None

    config_dir = _ovcli_config_dir()
    saved_start = len(profiles)
    if config_dir.exists():
        for path in sorted(config_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file():
                continue
            name = path.name.removeprefix(_OVCLI_SAVED_PREFIX)
            if name == path.name or name == "bak" or not _is_valid_ovcli_profile_name(name):
                continue
            add(path, source="saved", name=name)

    if active_profile is not None:
        marked_active = False
        for idx in range(saved_start, len(profiles)):
            if profiles[idx].source == "saved" and _profiles_equivalent(profiles[idx], active_profile):
                profiles[idx] = replace(profiles[idx], is_active=True)
                marked_active = True
                break
        has_env_profile = any(profile.source == "env" for profile in profiles)
        has_saved_profile = any(profile.source == "saved" for profile in profiles)
        active_identity = _profile_identity(active_profile.path)
        if not marked_active and not has_env_profile and not has_saved_profile and active_identity not in seen_paths:
            profiles.append(active_profile)

    return profiles


def _is_local_openviking_url(value: str) -> bool:
    candidate = _normalize_openviking_url(value)
    if not candidate:
        return False
    if "://" not in candidate:
        candidate = f"//{candidate}"
    parsed = urlparse(candidate)
    scheme = (parsed.scheme or "http").lower()
    return scheme == "http" and (parsed.hostname or "").lower() in _LOCAL_OPENVIKING_HOSTS


def _load_hermes_openviking_config() -> dict:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        memory_config = config.get("memory", {}) if isinstance(config, dict) else {}
        provider_config = memory_config.get("openviking", {}) if isinstance(memory_config, dict) else {}
        return dict(provider_config) if isinstance(provider_config, dict) else {}
    except Exception:
        return {}


def _env_value(name: str) -> Optional[str]:
    return os.environ[name].strip() if name in os.environ else None


def _first_nonempty(*values: Optional[str], default: str = "") -> str:
    for value in values:
        if value:
            return value
    return default


def _resolve_connection_settings(provider_config: Optional[dict] = None) -> dict:
    provider_config = dict(provider_config or {})
    ovcli_values: dict = {}
    if provider_config.get("use_ovcli_config"):
        ovcli_path = _resolve_ovcli_config_path(str(provider_config.get("ovcli_config_path") or ""))
        ovcli_values = _connection_values_from_ovcli(_load_ovcli_config(ovcli_path))

    endpoint_env = _env_value("OPENVIKING_ENDPOINT")
    api_key_env = _env_value("OPENVIKING_API_KEY")
    account_env = _env_value("OPENVIKING_ACCOUNT")
    user_env = _env_value("OPENVIKING_USER")
    agent_env = _env_value("OPENVIKING_AGENT")

    return {
        "endpoint": _first_nonempty(endpoint_env, ovcli_values.get("endpoint"), default=_DEFAULT_ENDPOINT),
        "api_key": api_key_env if api_key_env is not None else ovcli_values.get("api_key", ""),
        "account": account_env if account_env is not None else ovcli_values.get("account", ""),
        "user": user_env if user_env is not None else ovcli_values.get("user", ""),
        "agent": _first_nonempty(agent_env, ovcli_values.get("agent"), default=_DEFAULT_AGENT),
    }


def _env_writes_from_connection_values(values: dict) -> dict:
    writes = {}
    mapping = {
        "OPENVIKING_ENDPOINT": "endpoint",
        "OPENVIKING_API_KEY": "api_key",
        "OPENVIKING_ACCOUNT": "account",
        "OPENVIKING_USER": "user",
        "OPENVIKING_AGENT": "agent",
    }
    for env_key, value_key in mapping.items():
        value = _clean_config_value(values.get(value_key))
        if value:
            writes[env_key] = value
    return writes


def _restrict_secret_file_permissions(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError as e:
        logger.debug("Could not restrict permissions on %s: %s", path, e)


def _precreate_secret_file(path: Path) -> None:
    """Create (or tighten) a secret-bearing file with 0600 BEFORE writing.

    Writing the file first and chmod-ing afterwards leaves a window where a
    freshly-created file is world-readable under the default umask (e.g. 0644),
    briefly exposing the api_key/root_api_key. Pre-creating with 0600 closes
    that window; an existing file is tightened to 0600 here too.
    """
    try:
        if not path.exists():
            os.close(os.open(str(path), os.O_CREAT | os.O_WRONLY, 0o600))
        _restrict_secret_file_permissions(path)
    except OSError as e:
        logger.debug("Could not pre-create secret file %s: %s", path, e)


def _write_env_vars(env_path: Path, env_writes: dict, remove_keys: tuple[str, ...] = ()) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    remove_set = set(remove_keys) - set(env_writes)
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated_keys = set()
    new_lines = []
    for line in existing_lines:
        key_match = line.split("=", 1)[0].strip() if "=" in line else ""
        if key_match in remove_set:
            continue
        if key_match in env_writes:
            new_lines.append(f"{key_match}={env_writes[key_match]}")
            updated_keys.add(key_match)
        else:
            new_lines.append(line)
    for key, val in env_writes.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")
    # Pre-create with 0600 so secrets are never briefly world-readable.
    _precreate_secret_file(env_path)
    env_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    _restrict_secret_file_permissions(env_path)


def _remember_ovcli_path(provider_config: dict, ovcli_path: Path) -> None:
    default_path = _default_ovcli_config_path().expanduser()
    if os.environ.get(_OVCLI_CONFIG_ENV, "").strip() or ovcli_path.expanduser() != default_path:
        provider_config["ovcli_config_path"] = str(ovcli_path)
    else:
        provider_config.pop("ovcli_config_path", None)


def _ovcli_data_from_connection_values(values: dict) -> dict:
    data = {"url": _normalize_openviking_url(_clean_config_value(values.get("endpoint")) or _DEFAULT_ENDPOINT)}
    api_key = _clean_config_value(values.get("api_key"))
    root_api_key = _clean_config_value(values.get("root_api_key"))
    account = _clean_config_value(values.get("account"))
    user = _clean_config_value(values.get("user"))
    agent = _clean_config_value(values.get("agent")) or _DEFAULT_AGENT
    if api_key:
        data["api_key"] = api_key
    if root_api_key:
        data["root_api_key"] = root_api_key
    if account:
        data["account"] = account
    if user:
        data["user"] = user
    if agent:
        data["actor_peer_id"] = agent
    return data


def _write_ovcli_config(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic_json_write creates the temp file with mode 0o600 and os.replace()s
    # it into place — no half-written config on crash and no chmod-after-write
    # TOCTOU window for the api_key/root_api_key it carries.
    atomic_json_write(path, _ovcli_data_from_connection_values(values), mode=0o600)


def _validate_openviking_reachability(endpoint: str) -> tuple[bool, str]:
    endpoint = _normalize_openviking_url(endpoint)
    try:
        client = _viking_client_cls()(endpoint)
        if hasattr(client, "health_payload"):
            payload = client.health_payload()
            if payload.get("healthy") is False:
                return False, "OpenViking server responded but reported unhealthy status."
            if payload:
                return True, ""
        elif client.health():
            return True, ""
    except Exception as e:
        if _status_code_from_error(e) is not None:
            return False, f"OpenViking server responded with {_format_openviking_exception(e)}."
        return False, f"OpenViking server is not reachable at {endpoint}: {_format_openviking_exception(e)}"
    return False, f"OpenViking server is not reachable at {endpoint}."


def _validate_openviking_auth(values: dict) -> tuple[bool, str]:
    endpoint = _normalize_openviking_url(values.get("endpoint"))
    try:
        client = _viking_client_cls()(
            endpoint,
            _clean_config_value(values.get("api_key")),
            account=_clean_config_value(values.get("account")),
            user=_clean_config_value(values.get("user")),
            agent=_clean_config_value(values.get("agent")) or _DEFAULT_AGENT,
        )
        client.validate_auth()
    except Exception as e:
        return False, f"OpenViking authentication validation failed: {_format_openviking_exception(e)}"
    return True, ""


def _validate_openviking_root_access(values: dict) -> tuple[bool, str]:
    endpoint = _normalize_openviking_url(values.get("endpoint"))
    try:
        client = _viking_client_cls()(
            endpoint,
            _clean_config_value(values.get("api_key")),
            agent=_clean_config_value(values.get("agent")) or _DEFAULT_AGENT,
        )
        client.validate_root_access()
    except Exception as e:
        return False, f"OpenViking root API key validation failed: {_format_openviking_exception(e)}"
    return True, ""


def _validate_openviking_user_key_scope(values: dict) -> tuple[bool, str]:
    root_ok, _message = _validate_openviking_root_access(values)
    if not root_ok:
        return True, ""
    return (
        False,
        "That key has ROOT access. Choose Root API key and provide account/user, "
        "or enter a user API key.",
    )


def _status_code_from_error(error: Exception) -> Optional[int]:
    if isinstance(error, _OpenVikingHTTPError):
        return error.status_code
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def _admin_probe_means_regular_key(error: Exception) -> bool:
    return _status_code_from_error(error) in {401, 403, 404}


def _should_probe_openviking_auth(health: dict, *, require_api_key: bool, has_api_key: bool) -> bool:
    if require_api_key or has_api_key:
        return True
    auth_mode = health.get("auth_mode")
    if auth_mode == "dev":
        return False
    if auth_mode in {"api_key", "trusted", None}:
        return True
    return False


def _validate_openviking_setup_values(
    values: dict,
    *,
    require_api_key: bool = False,
) -> tuple[bool, str, Optional[str]]:
    endpoint = _normalize_openviking_url(values.get("endpoint"))
    api_key = _clean_config_value(values.get("api_key"))
    if require_api_key and not api_key:
        return False, "Remote OpenViking configs require an API key.", None

    try:
        client = _viking_client_cls()(
            endpoint,
            api_key,
            account=_clean_config_value(values.get("account")),
            user=_clean_config_value(values.get("user")),
            agent=_clean_config_value(values.get("agent")) or _DEFAULT_AGENT,
        )
        health = client.health_payload()
        if health.get("healthy") is False:
            return False, "OpenViking server responded but reported unhealthy status.", None
        if _should_probe_openviking_auth(
            health,
            require_api_key=require_api_key,
            has_api_key=bool(api_key),
        ):
            client.validate_auth()
        if not api_key:
            return True, "", None
        try:
            client.validate_root_access()
            return True, "", "root"
        except Exception as e:
            if _admin_probe_means_regular_key(e):
                return True, "", "user"
            raise
    except Exception as e:
        return False, f"OpenViking validation failed: {_format_openviking_exception(e)}", None


def _retry_or_cancel_manual_setup(select, title: str, message: str, cancelled):
    print(f"  {message}")
    choice = select(
        title,
        [
            ("Retry", "try this step again"),
            ("Cancel setup", "no changes saved"),
        ],
        default=0,
        cancel_returns=cancelled,
    )
    if choice == 0:
        return True
    return _SETUP_CANCELLED


def _print_validation_progress(message: str) -> None:
    print(f"  {message}", flush=True)


def _local_openviking_bind(endpoint: str) -> tuple[str, int]:
    normalized = _normalize_openviking_url(endpoint)
    parsed = urlparse(normalized)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 1933
    return host, port


def _openviking_server_log_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    except Exception:
        home = Path(os.environ.get("HERMES_HOME", "")).expanduser() if os.environ.get("HERMES_HOME") else Path.home() / ".hermes"
    return home / _OPENVIKING_SERVER_LOG_RELATIVE_PATH


def _start_local_openviking_server(endpoint: str) -> tuple[bool, str]:
    server_cmd = shutil.which("openviking-server")
    if not server_cmd:
        return False, "openviking-server was not found on PATH. Start it manually, then retry."
    try:
        host, port = _local_openviking_bind(endpoint)
    except ValueError as e:
        return False, f"Could not parse local OpenViking URL: {e}"
    log_path = _openviking_server_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_file:
            subprocess.Popen(
                [server_cmd, "--host", host, "--port", str(port)],
                stdout=log_file,
                stderr=log_file,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
    except Exception as e:
        return False, f"Could not start openviking-server: {e}"
    return True, f"Started openviking-server on {host}:{port} in the background. Logs: {log_path}"


def _wait_for_openviking_health(endpoint: str, *, timeout_seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        ok, _message = _validate_openviking_reachability(endpoint)
        if ok:
            return True
        time.sleep(0.5)
    return False


def _reachability_failure_allows_local_autostart(message: str) -> bool:
    return not (message or "").startswith(_OPENVIKING_RESPONDED_FAILURE_PREFIX)


def _handle_unreachable_endpoint(
    endpoint: str,
    message: str,
    select,
    cancelled,
    *,
    allow_local_autostart: bool = True,
):
    if _is_local_openviking_url(endpoint) and allow_local_autostart:
        print(f"  {message}")
        choice = select(
            "  Local OpenViking server is down",
            [
                ("Start local OpenViking", "run openviking-server and retry"),
                ("Retry URL", "enter the server URL again"),
                ("Cancel setup", "no changes saved"),
            ],
            default=0,
            cancel_returns=cancelled,
        )
        if choice == 0:
            started, start_message = _facade_attr(
                "_start_local_openviking_server",
                _start_local_openviking_server,
            )(endpoint)
            print(f"  {start_message}")
            if not started:
                return False
            print("  Waiting for OpenViking server to become reachable...", flush=True)
            if _facade_attr(
                "_wait_for_openviking_health",
                _wait_for_openviking_health,
            )(
                endpoint,
                timeout_seconds=_LOCAL_OPENVIKING_AUTOSTART_TIMEOUT,
            ):
                print("  OpenViking server is reachable.")
                return True
            print("  OpenViking server did not become reachable.")
            return False
        if choice == 1:
            return False
        return _SETUP_CANCELLED

    return _retry_or_cancel_manual_setup(
        select,
        "  OpenViking server unhealthy" if _is_local_openviking_url(endpoint) else "  OpenViking server unreachable",
        message,
        cancelled,
    )


def _emit_runtime_warning(message: str, warning_callback=None) -> None:
    logger.warning("%s", message)
    if warning_callback:
        try:
            warning_callback(message)
        except Exception:
            logger.debug("OpenViking runtime warning callback failed", exc_info=True)


def _emit_runtime_status(message: str, status_callback=None) -> None:
    logger.info("%s", message)
    if status_callback:
        try:
            status_callback(message)
        except Exception:
            logger.debug("OpenViking runtime status callback failed", exc_info=True)


def _runtime_openviking_timeout_message(endpoint: str) -> str:
    return (
        f"Local OpenViking server at {endpoint} is not reachable. "
        "Tried to start openviking-server, but it did not become reachable "
        f"within {_LOCAL_OPENVIKING_AUTOSTART_TIMEOUT:.0f} seconds. "
        "OpenViking memory disabled for this Hermes run."
    )


def _classify_runtime_openviking_health(client: _VikingClient, endpoint: str) -> tuple[str, str]:
    """Classify runtime health without treating every false result as server absence."""
    try:
        if hasattr(client, "health_payload"):
            payload = client.health_payload()
            if payload.get("healthy") is False:
                return (
                    "responded",
                    f"OpenViking server at {endpoint} responded but reported unhealthy status.",
                )
            return "healthy", ""
        if client.health():
            return "healthy", ""
    except _OpenVikingHTTPError as e:
        return (
            "responded",
            f"OpenViking server at {endpoint} responded with {_format_openviking_exception(e)}.",
        )
    except Exception:
        return "unreachable", ""
    return "unreachable", ""
