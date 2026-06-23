"""Interactive setup flow for the OpenViking memory plugin."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .config import (
    _OvcliProfile,
    _clean_config_value,
    _discover_ovcli_profiles,
    _env_writes_from_connection_values,
    _handle_unreachable_endpoint,
    _is_local_openviking_url,
    _is_valid_ovcli_profile_name,
    _load_ovcli_config,
    _normalize_openviking_url,
    _ovcli_config_dir,
    _ovcli_data_from_connection_values,
    _print_validation_progress,
    _reachability_failure_allows_local_autostart,
    _remember_ovcli_path,
    _resolve_ovcli_config_path,
    _retry_or_cancel_manual_setup,
    _validate_openviking_identity_value,
    _validate_openviking_reachability,
    _validate_openviking_setup_values,
    _write_env_vars,
    _write_ovcli_config,
)
from .constants import (
    _AGENT_PROMPT_LABEL,
    _DEFAULT_AGENT,
    _DEFAULT_ENDPOINT,
    _OPENVIKING_ENV_KEYS,
    _OPENVIKING_SERVICE_ENDPOINT,
    _OVCLI_CONFIG_ENV,
    _OVCLI_SAVED_PREFIX,
    _SETUP_CANCELLED,
)


def _facade_attr(name: str, default):
    facade = sys.modules.get(__package__)
    return getattr(facade, name, default) if facade is not None else default


def _prompt_profile_name(prompt, select, cancelled) -> str | object:
    while True:
        name = _clean_config_value(prompt("OpenViking profile name"))
        if _is_valid_ovcli_profile_name(name):
            return name
        retry = _retry_or_cancel_manual_setup(
            select,
            "  Invalid OpenViking profile name",
            "Profile names can only contain letters, numbers, '-' and '_'.",
            cancelled,
        )
        if retry is _SETUP_CANCELLED:
            return _SETUP_CANCELLED


def _confirm_replace_existing_profile(path: Path, values: dict, select, cancelled):
    if not path.exists():
        return True
    try:
        existing_data = _load_ovcli_config(path)
    except Exception:
        existing_data = {}
    if existing_data == _ovcli_data_from_connection_values(values):
        return True
    choice = select(
        "  OpenViking profile already exists",
        [
            ("Choose another name", "leave the existing profile unchanged"),
            ("Replace profile", "overwrite this saved OpenViking profile"),
            ("Cancel setup", "no changes saved"),
        ],
        default=0,
        cancel_returns=cancelled,
    )
    if choice == 1:
        return True
    if choice == 0:
        return False
    return _SETUP_CANCELLED


def _prompt_manual_connection_values(prompt, select, cancelled, *, service: bool = False):
    if service:
        endpoint = _OPENVIKING_SERVICE_ENDPOINT
        print(f"  OpenViking Service endpoint: {endpoint}")
    else:
        while True:
            endpoint = _normalize_openviking_url(prompt("OpenViking server URL", default=_DEFAULT_ENDPOINT))
            _print_validation_progress("Checking OpenViking server...")
            reachable, message = _facade_attr(
                "_validate_openviking_reachability",
                _validate_openviking_reachability,
            )(endpoint)
            if reachable:
                print("  OpenViking server is reachable.")
                break
            retry = _handle_unreachable_endpoint(
                endpoint,
                message,
                select,
                cancelled,
                allow_local_autostart=_reachability_failure_allows_local_autostart(message),
            )
            if retry is True:
                break
            if retry is _SETUP_CANCELLED:
                return _SETUP_CANCELLED

    is_local = _is_local_openviking_url(endpoint)
    api_key_type = "user" if service else ""
    prefilled_api_key = ""
    prefilled_agent = ""
    while True:
        values = {
            "endpoint": endpoint,
            "api_key": "",
            "root_api_key": "",
            "account": "",
            "user": "",
            "agent": "",
        }
        if not api_key_type and is_local:
            credential_choice = select(
                "  OpenViking credential",
                [
                    ("No API key", "local dev mode"),
                    ("User API key", "server derives account/user automatically"),
                    ("Root API key", "requires account and user IDs"),
                ],
                default=0,
                cancel_returns=cancelled,
            )
            if credential_choice == cancelled:
                return _SETUP_CANCELLED
            if credential_choice == 0:
                values["agent"] = _clean_config_value(
                    prompt(_AGENT_PROMPT_LABEL, default=_DEFAULT_AGENT)
                ) or _DEFAULT_AGENT
                _print_validation_progress("Validating OpenViking local dev access...")
                valid, message, _role = _facade_attr(
                    "_validate_openviking_setup_values",
                    _validate_openviking_setup_values,
                )(values)
                if valid:
                    print("  OpenViking local dev access validated.")
                    return values
                retry = _retry_or_cancel_manual_setup(
                    select,
                    "  OpenViking credential failed",
                    message,
                    cancelled,
                )
                if retry is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                continue
            api_key_type = "root" if credential_choice == 2 else "user"
        elif not api_key_type:
            credential_choice = select(
                "  OpenViking API key type",
                [
                    ("User API key", "server derives account/user automatically"),
                    ("Root API key", "requires account and user IDs"),
                ],
                default=0,
                cancel_returns=cancelled,
            )
            if credential_choice == cancelled:
                return _SETUP_CANCELLED
            api_key_type = "root" if credential_choice == 1 else "user"

        values["api_key_type"] = api_key_type
        if service:
            api_key_label = "OpenViking API key"
        else:
            api_key_label = (
                "OpenViking root API key"
                if api_key_type == "root"
                else "OpenViking user API key"
            )
        if prefilled_api_key:
            values["api_key"] = prefilled_api_key
            prefilled_api_key = ""
        else:
            values["api_key"] = _clean_config_value(prompt(api_key_label, secret=True))
        if not values["api_key"]:
            retry = _retry_or_cancel_manual_setup(
                select,
                "  OpenViking API key required",
                f"{api_key_label} is required.",
                cancelled,
            )
            if retry is _SETUP_CANCELLED:
                return _SETUP_CANCELLED
            continue

        if api_key_type == "root":
            _print_validation_progress("Validating OpenViking root API key...")
            valid, message, role = _facade_attr(
                "_validate_openviking_setup_values",
                _validate_openviking_setup_values,
            )(values, require_api_key=True)
            root_ok = valid and role == "root"
            if not root_ok:
                if valid and role == "user":
                    print("  That key is valid, but it is a user API key.")
                    route_choice = select(
                        "  OpenViking key is a user key",
                        [
                            ("Use as User API key", "server derives account/user automatically"),
                            ("Re-enter Root API key", "try another root key"),
                            ("Cancel setup", "no changes saved"),
                        ],
                        default=0,
                        cancel_returns=cancelled,
                    )
                    if route_choice == 0:
                        prefilled_api_key = values["api_key"]
                        api_key_type = "user"
                        continue
                    if route_choice == 1:
                        api_key_type = "root"
                        continue
                    return _SETUP_CANCELLED
                retry = _retry_or_cancel_manual_setup(
                    select,
                    "  OpenViking root API key failed",
                    message,
                    cancelled,
                )
                if retry is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                continue
            print("  OpenViking root API key validated.")
            values["root_api_key"] = values["api_key"]
            account_ok, account_message, account = _validate_openviking_identity_value(
                prompt("OpenViking account"),
                field="account",
            )
            user_ok, user_message, user = _validate_openviking_identity_value(
                prompt("OpenViking user"),
                field="user",
            )
            values["account"] = account
            values["user"] = user
            if not account_ok or not user_ok:
                message = account_message if not account_ok else user_message
                retry = _retry_or_cancel_manual_setup(
                    select,
                    "  OpenViking tenant identity required",
                    message,
                    cancelled,
                )
                if retry is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                prefilled_api_key = values["api_key"]
                continue

        if prefilled_agent:
            values["agent"] = prefilled_agent
            prefilled_agent = ""
        else:
            values["agent"] = _clean_config_value(
                prompt(_AGENT_PROMPT_LABEL, default=_DEFAULT_AGENT)
            ) or _DEFAULT_AGENT
        _print_validation_progress("Validating OpenViking API access...")
        valid, message, role = _facade_attr(
            "_validate_openviking_setup_values",
            _validate_openviking_setup_values,
        )(
            values,
            require_api_key=service or not is_local,
        )
        if valid:
            if api_key_type == "user":
                if role == "root":
                    print("  That key is valid, but it has root access.")
                    route_choice = select(
                        "  OpenViking user API key is root key",
                        [
                            ("Configure as Root API key", "provide account and user IDs"),
                            ("Re-enter User API key", "try another user key"),
                            ("Cancel setup", "no changes saved"),
                        ],
                        default=0,
                        cancel_returns=cancelled,
                    )
                    if route_choice == 0:
                        prefilled_api_key = values["api_key"]
                        prefilled_agent = values["agent"]
                        api_key_type = "root"
                        continue
                    if route_choice == 1:
                        api_key_type = "user"
                        continue
                    return _SETUP_CANCELLED
            if api_key_type == "root" and role != "root":
                retry = _retry_or_cancel_manual_setup(
                    select,
                    "  OpenViking root API key failed",
                    "The supplied key was not accepted as a root API key.",
                    cancelled,
                )
                if retry is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                continue
            print("  OpenViking API access validated.")
            return values
        retry = _retry_or_cancel_manual_setup(
            select,
            "  OpenViking API access failed",
            message,
            cancelled,
        )
        if retry is _SETUP_CANCELLED:
            return _SETUP_CANCELLED


def _set_openviking_provider(config: dict, provider_config: dict) -> None:
    config["memory"]["provider"] = "openviking"
    config["memory"]["openviking"] = provider_config


def _link_ovcli_profile(
    *,
    config: dict,
    provider_config: dict,
    env_path: Path,
    ovcli_path: Path,
) -> None:
    for key in ("endpoint", "api_key", "root_api_key", "account", "user", "agent", "api_key_type"):
        provider_config.pop(key, None)
    provider_config["use_ovcli_config"] = True
    _remember_ovcli_path(provider_config, ovcli_path)
    _set_openviking_provider(config, provider_config)
    _write_env_vars(env_path, {}, remove_keys=_OPENVIKING_ENV_KEYS)
    for key in _OPENVIKING_ENV_KEYS:
        os.environ.pop(key, None)


def _save_hermes_only_config(
    *,
    config: dict,
    provider_config: dict,
    env_path: Path,
    values: dict,
) -> None:
    provider_config["use_ovcli_config"] = False
    provider_config.pop("ovcli_config_path", None)
    _set_openviking_provider(config, provider_config)
    _write_env_vars(
        env_path,
        _env_writes_from_connection_values(values),
        remove_keys=_OPENVIKING_ENV_KEYS,
    )


def _profile_display_name(profile: _OvcliProfile) -> str:
    if profile.source == "env":
        return _OVCLI_CONFIG_ENV
    if profile.source == "active":
        return "ovcli.conf"
    return profile.name


def _profile_description(profile: _OvcliProfile) -> str:
    endpoint = _clean_config_value(profile.values.get("endpoint")) or _DEFAULT_ENDPOINT
    return f"{endpoint} ({profile.path})"


def _validate_profile_for_setup(profile: _OvcliProfile) -> tuple[bool, str, Optional[str]]:
    require_api_key = not _is_local_openviking_url(profile.values.get("endpoint", ""))
    return _facade_attr(
        "_validate_openviking_setup_values",
        _validate_openviking_setup_values,
    )(profile.values, require_api_key=require_api_key)


def _print_openviking_ready(message: str, path: Optional[Path] = None) -> None:
    print("\n  OpenViking memory is ready")
    print(f"  {message}")
    if path is not None:
        print(f"  Config file: {path}")
    print("  Start a new Hermes session to activate.\n")


def _run_existing_profile_setup(
    *,
    profiles: list[_OvcliProfile],
    select,
    cancelled,
    config: dict,
    provider_config: dict,
    env_path: Path,
) -> bool | object:
    while True:
        choice = select(
            "  OpenViking profile",
            [(_profile_display_name(profile), _profile_description(profile)) for profile in profiles],
            default=0,
            cancel_returns=cancelled,
        )
        if choice == cancelled:
            return _SETUP_CANCELLED
        if choice < 0 or choice >= len(profiles):
            return _SETUP_CANCELLED

        profile = profiles[choice]
        _print_validation_progress("Validating OpenViking profile...")
        ok, message, _role = _validate_profile_for_setup(profile)
        if ok:
            _link_ovcli_profile(
                config=config,
                provider_config=provider_config,
                env_path=env_path,
                ovcli_path=profile.path,
            )
            _print_openviking_ready(f"Linked profile: {_profile_display_name(profile)}", profile.path)
            return True

        print(f"  {message}")
        retry = select(
            "  OpenViking profile validation failed",
            [
                ("Choose another profile", "select a different OpenViking profile"),
                ("Retry validation", "try this profile again"),
                ("Cancel setup", "no changes saved"),
            ],
            default=0,
            cancel_returns=cancelled,
        )
        if retry == 0:
            continue
        if retry == 1:
            _print_validation_progress("Validating OpenViking profile...")
            ok, message, _role = _validate_profile_for_setup(profile)
            if ok:
                _link_ovcli_profile(
                    config=config,
                    provider_config=provider_config,
                    env_path=env_path,
                    ovcli_path=profile.path,
                )
                _print_openviking_ready(f"Linked profile: {_profile_display_name(profile)}", profile.path)
                return True
            print(f"  {message}")
            continue
        return _SETUP_CANCELLED


def _mirror_manual_config_to_openviking_store(
    *,
    prompt,
    select,
    cancelled,
    values: dict,
) -> Path | object:
    while True:
        name = _prompt_profile_name(prompt, select, cancelled)
        if name is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        path = _ovcli_config_dir() / f"{_OVCLI_SAVED_PREFIX}{name}"
        replace = _confirm_replace_existing_profile(path, values, select, cancelled)
        if replace is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        if replace is False:
            continue
        _write_ovcli_config(path, values)
        return path


def _run_create_profile_setup(
    *,
    prompt,
    select,
    cancelled,
    config: dict,
    provider_config: dict,
    env_path: Path,
) -> bool | object:
    source_choice = select(
        "  OpenViking connection",
        [
            ("OpenViking Service (VolcEngine Cloud)", "use the managed OpenViking endpoint"),
            ("Custom", "use a local, VPS, or self-hosted OpenViking server"),
        ],
        default=0,
        cancel_returns=cancelled,
    )
    if source_choice == cancelled:
        return _SETUP_CANCELLED

    values = _prompt_manual_connection_values(prompt, select, cancelled, service=(source_choice == 0))
    if values is _SETUP_CANCELLED:
        return _SETUP_CANCELLED
    if values is None:
        return False

    save_choice = select(
        "  Save OpenViking config",
        [
            ("Keep in Hermes only", "write values only to Hermes .env"),
            ("Mirror to OpenViking store", "write ~/.openviking/ovcli.conf.<name> and link it"),
        ],
        default=1,
        cancel_returns=cancelled,
    )
    if save_choice == cancelled:
        return _SETUP_CANCELLED

    if save_choice == 1:
        ovcli_path = _mirror_manual_config_to_openviking_store(
            prompt=prompt,
            select=select,
            cancelled=cancelled,
            values=values,
        )
        if ovcli_path is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        _link_ovcli_profile(
            config=config,
            provider_config=provider_config,
            env_path=env_path,
            ovcli_path=ovcli_path,
        )
        _print_openviking_ready("Created and linked OpenViking profile.", ovcli_path)
        return True

    _save_hermes_only_config(
        config=config,
        provider_config=provider_config,
        env_path=env_path,
        values=values,
    )
    _print_openviking_ready("Connection saved to Hermes .env.")
    return True
