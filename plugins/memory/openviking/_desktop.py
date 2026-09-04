"""Non-interactive OpenViking configuration adapter for Hermes Desktop."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Optional

from agent.memory_provider import MemoryProviderConfigConflictError


class OpenVikingProfileNotFoundError(ValueError):
    """Hermes is linked to an OpenViking profile that no longer exists."""


def _ov():
    """Resolve the plugin lazily so tests and runtime reloads share one state."""
    return sys.modules[__package__]


def _profile_description(profile) -> str:
    ov = _ov()
    endpoint = (
        ov._clean_config_value(profile.values.get("endpoint")) or ov._DEFAULT_ENDPOINT
    )
    return f"{endpoint} ({profile.path})"


def _profile_payload(profile, *, active_path: Optional[Path]) -> dict:
    ov = _ov()
    is_active = bool(
        active_path
        and ov._profile_identity(profile.path) == ov._profile_identity(active_path)
    )
    label = ov._profile_display_name(profile)
    return {
        "value": str(profile.path),
        "label": f"{label} (Active)" if is_active else label,
        "description": _profile_description(profile),
    }


def _profiles(
    active_path: Optional[Path],
    *,
    env_values: Mapping[str, str],
) -> list:
    ov = _ov()
    profiles = ov._discover_ovcli_profiles(env_values=env_values)
    if active_path and active_path.is_file():
        active_identity = ov._profile_identity(active_path)
        if not any(
            ov._profile_identity(profile.path) == active_identity
            for profile in profiles
        ):
            profile = ov._load_profile(active_path, source="active", name="active")
            if profile is not None:
                profiles.append(profile)
    return profiles


def _connection_type(endpoint: str) -> str:
    ov = _ov()
    return (
        "OpenViking Service"
        if ov._is_openviking_service_endpoint(endpoint)
        else "Custom"
    )


def _health(state: str, label: str, message: str) -> dict:
    return {"state": state, "label": label, "message": message}


def _health_for_settings(settings: dict) -> dict:
    ov = _ov()
    ok, message, _role = ov._validate_openviking_setup_values(
        settings,
        require_api_key=not ov._is_local_openviking_url(settings.get("endpoint", "")),
    )
    if ok:
        return _health("healthy", "Healthy", "OpenViking is ready to use.")

    lowered = (message or "").lower()
    unreachable = any(
        marker in lowered
        for marker in (
            "not reachable",
            "connection refused",
            "connection reset",
            "timed out",
            "timeout",
        )
    )
    return _health(
        "unreachable" if unreachable else "unhealthy",
        "Unreachable" if unreachable else "Unhealthy",
        message,
    )


def _is_running_hermes_home(hermes_home: str | Path) -> bool:
    from hermes_constants import get_process_hermes_home

    return (
        Path(hermes_home).expanduser().resolve()
        == get_process_hermes_home().expanduser().resolve()
    )


def snapshot(*, hermes_home: str | Path, probe_health: bool) -> dict:
    """Return a redacted profile-scoped snapshot for the shared settings form."""
    from hermes_cli.config import load_env

    ov = _ov()
    provider_config = ov._load_hermes_openviking_config()
    env_values = os.environ if _is_running_hermes_home(hermes_home) else load_env()
    active_path: Optional[Path] = None
    missing_profile = False
    invalid_profile = False

    if provider_config.get("use_ovcli_config"):
        active_path = ov._resolve_ovcli_config_path(
            str(provider_config.get("ovcli_config_path") or ""),
            env_values=env_values,
        )
        missing_profile = not active_path.is_file()

    if missing_profile:
        settings = {
            "endpoint": ov._DEFAULT_ENDPOINT,
            "api_key": "",
            "account": "",
            "user": "",
            "agent": ov._DEFAULT_AGENT,
        }
    else:
        try:
            settings = ov._resolve_connection_settings(
                provider_config,
                env_values=env_values,
            )
        except (OSError, UnicodeError, ValueError):
            if not provider_config.get("use_ovcli_config"):
                raise
            invalid_profile = True
            settings = {
                "endpoint": ov._DEFAULT_ENDPOINT,
                "api_key": "",
                "account": "",
                "user": "",
                "agent": ov._DEFAULT_AGENT,
            }

    profiles = _profiles(active_path, env_values=env_values)
    active_profile = next(
        (
            profile
            for profile in profiles
            if active_path
            and ov._profile_identity(profile.path) == ov._profile_identity(active_path)
        ),
        None,
    )
    source_type = _connection_type(settings.get("endpoint") or ov._DEFAULT_ENDPOINT)
    if active_profile is not None:
        active_label = f"{ov._profile_display_name(active_profile)} ({source_type})"
    elif provider_config.get("use_ovcli_config"):
        active_label = "Missing profile"
    else:
        active_label = f"Environment variables ({source_type})"

    if missing_profile:
        health = _health(
            "unreachable",
            "Profile missing",
            "The linked OpenViking profile no longer exists. Choose another profile or recreate it.",
        )
    elif invalid_profile:
        health = _health(
            "unreachable",
            "Profile invalid",
            "The linked OpenViking profile could not be read. Fix it or choose another profile.",
        )
    elif probe_health:
        health = _health_for_settings(settings)
    else:
        health = _health(
            "checking",
            "Checking",
            "Checking OpenViking connection status.",
        )

    current_path = str(active_path) if active_path and active_path.is_file() else ""
    setup_type = (
        "profile"
        if current_path
        else "service"
        if source_type == "OpenViking Service"
        else "custom"
    )
    credential = "none"
    if settings.get("api_key"):
        credential = (
            "root" if settings.get("account") or settings.get("user") else "user"
        )

    return {
        "values": {
            "setup_type": setup_type,
            "profile_path": current_path,
            "profile_name": "openviking",
            "url": settings.get("endpoint") or ov._DEFAULT_ENDPOINT,
            "credential": credential,
            "api_key_service": "",
            "api_key": "",
            "account": settings.get("account") or "",
            "user": settings.get("user") or "",
            "actor_peer_id": settings.get("agent") or ov._DEFAULT_AGENT,
        },
        "options": {
            "profile_path": [
                _profile_payload(profile, active_path=active_path)
                for profile in profiles
            ]
        },
        "summary": {
            "items": [
                {"label": "Active profile", "value": active_label},
                {
                    "label": "OpenViking URL",
                    "value": settings.get("endpoint") or ov._DEFAULT_ENDPOINT,
                },
            ],
            "status": health,
        },
    }


def connection_values(values: dict) -> tuple[dict, str]:
    """Normalize submitted form values and discard fields hidden by its mode."""
    ov = _ov()
    setup_type = ov._clean_config_value(values.get("setup_type"))
    if setup_type not in {"service", "custom"}:
        raise ValueError(
            "Choose OpenViking Service, Existing Profiles, or Custom Server."
        )

    endpoint = (
        ov._OPENVIKING_SERVICE_ENDPOINT
        if setup_type == "service"
        else ov._clean_config_value(values.get("url"))
    )
    credential = (
        "user"
        if setup_type == "service"
        else ov._clean_config_value(values.get("credential"))
    )
    if credential not in {"none", "user", "root"}:
        raise ValueError("Choose No API key, User API key, or Root API key.")
    if credential == "none" and not ov._is_local_openviking_url(endpoint):
        raise ValueError("Remote OpenViking servers require an API key.")

    submitted_key = ov._clean_config_value(
        values.get("api_key_service")
        if setup_type == "service"
        else values.get("api_key")
    )
    api_key = "" if credential == "none" else submitted_key
    if credential != "none" and not api_key:
        raise ValueError("Enter an OpenViking API key.")

    account = (
        ov._clean_config_value(values.get("account")) if credential == "root" else ""
    )
    user = ov._clean_config_value(values.get("user")) if credential == "root" else ""
    if credential == "root" and (not account or not user):
        raise ValueError("Account and User are required for a Root API key.")

    actor_peer_id = ov._clean_config_value(values.get("actor_peer_id"))
    if "/" in actor_peer_id or "\\" in actor_peer_id:
        raise ValueError("Agent ID cannot contain '/' or '\\'.")

    return {
        "endpoint": ov._normalize_openviking_url(endpoint),
        "api_key": api_key,
        "root_api_key": api_key if credential == "root" else "",
        "account": account,
        "user": user,
        "agent": actor_peer_id,
        "api_key_type": credential,
    }, credential


def validate_values(values: dict) -> tuple[dict, Optional[str]]:
    ov = _ov()
    connection, credential = connection_values(values)
    ok, message, role = ov._validate_openviking_setup_values(
        connection,
        require_api_key=not ov._is_local_openviking_url(connection["endpoint"]),
    )
    if not ok:
        raise ValueError(message)
    if credential == "user" and role == "root":
        raise ValueError(
            "This key has root access. Select Root API key and provide Account and User, "
            "or enter a User API key."
        )
    if credential == "root" and role == "user":
        raise ValueError(
            "This is a User API key. Select User API key, or enter a Root API key."
        )
    return connection, role


def save(*, values: dict, hermes_home: str | Path, overwrite: bool) -> dict:
    """Validate, persist, and link one complete Desktop setup transaction."""
    from hermes_cli.config import load_config, reload_env, save_config

    ov = _ov()
    config = load_config()
    if not isinstance(config.get("memory"), dict):
        config["memory"] = {}
    provider_config = config["memory"].get("openviking", {})
    if not isinstance(provider_config, dict):
        provider_config = {}

    setup_type = ov._clean_config_value(values.get("setup_type"))
    if setup_type == "profile":
        path = Path(ov._clean_config_value(values.get("profile_path"))).expanduser()
        if not path.is_file():
            raise OpenVikingProfileNotFoundError(
                f"OpenViking profile file was not found: {path}"
            )
        profile = ov._load_profile(path, source="saved", name=path.name)
        if profile is None:
            raise ValueError(
                "The selected OpenViking profile could not be read. "
                "Refresh profiles or choose another profile."
            )
        require_api_key = not ov._is_local_openviking_url(
            profile.values.get("endpoint", "")
        )
        ok, message, _role = ov._validate_openviking_setup_values(
            profile.values,
            require_api_key=require_api_key,
        )
        if not ok:
            raise ValueError(message)
    else:
        connection, _role = validate_values(values)
        profile_name = ov._clean_config_value(values.get("profile_name"))
        if not ov._is_valid_ovcli_profile_name(profile_name):
            raise ValueError(
                "Profile names can only contain letters, numbers, '-' and '_'."
            )
        if profile_name == "active":
            raise ValueError(
                "The profile name 'active' is reserved by the OpenViking CLI."
            )
        path = (
            ov._default_ovcli_config_path().parent
            / f"{ov._OVCLI_SAVED_PREFIX}{profile_name}"
        )
        new_data = ov._ovcli_data_from_connection_values(connection)
        if path.exists():
            try:
                existing_data = ov._load_ovcli_config(path)
            except Exception:
                existing_data = {}
            if existing_data != new_data and not overwrite:
                raise MemoryProviderConfigConflictError(
                    "An OpenViking profile with this name already exists with different settings."
                )
        if not path.exists() or overwrite:
            path.parent.mkdir(parents=True, exist_ok=True)
            ov.atomic_json_write(path, new_data, mode=0o600)

    update_process_env = _is_running_hermes_home(hermes_home)
    ov._setup._link_ovcli_profile(
        config=config,
        provider_config=provider_config,
        env_path=Path(hermes_home).expanduser() / ".env",
        ovcli_path=path,
        update_process_env=update_process_env,
    )
    save_config(config)
    if update_process_env:
        reload_env()
    return {"ok": True, "profile_path": str(path)}


def start_local(url: str) -> dict:
    ov = _ov()
    endpoint = ov._normalize_openviking_url(url)
    if not ov._is_local_openviking_url(endpoint):
        return {
            "ok": False,
            "message": "Only a local OpenViking URL can be started from Desktop.",
        }

    reachable, _message = ov._validate_openviking_reachability(endpoint)
    if reachable:
        return {
            "ok": True,
            "message": "OpenViking is already running and ready to use.",
        }

    state, message = ov._start_local_openviking_server(endpoint)
    if state != ov._LOCAL_SERVER_STARTED:
        return {"ok": False, "message": message}
    if not ov._wait_for_openviking_health(endpoint, timeout_seconds=15.0):
        return {
            "ok": False,
            "message": (
                "openviking-server started but did not become ready. It may need server "
                "configuration. Run 'openviking-server init', then try again. "
                f"Logs: {ov._openviking_server_log_path()}"
            ),
        }
    return {"ok": True, "message": "OpenViking started and is ready to use."}
