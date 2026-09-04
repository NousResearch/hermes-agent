"""Interactive ``hermes memory setup`` wizard for the OpenViking provider.

Pure UI flow: prompts, menus, and persistence of the chosen connection (Hermes
``.env`` only, or mirrored to an ``ovcli.conf.<name>`` profile that Hermes then
links). Network validation and file writers live in the package ``__init__`` and
are looked up there at call time so tests can monkeypatch them on the plugin module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_SETUP_CANCELLED = object()
_CANCEL_OPTION = ("Cancel setup", "no changes saved")


def _ov():
    """The plugin module — resolved lazily so monkeypatched validators are honored."""
    return sys.modules[__package__]


def _say(message: str) -> None:
    print(f"  {message}", flush=True)


def _retry_or_cancel_manual_setup(select, title: str, message: str, cancelled):
    """-> True (retry) or _SETUP_CANCELLED."""
    _say(message)
    choice = select(title, [("Retry", "try this step again"), _CANCEL_OPTION], default=0, cancel_returns=cancelled)
    return True if choice == 0 else _SETUP_CANCELLED


def _handle_unreachable_endpoint(endpoint: str, message: str, select, cancelled, *, allow_local_autostart: bool = True):
    """-> True (reachable now) / False (re-prompt URL) / _SETUP_CANCELLED."""
    ov = _ov()
    is_local = ov._is_local_openviking_url(endpoint)
    if not (is_local and allow_local_autostart):
        title = "  OpenViking server unhealthy" if is_local else "  OpenViking server unreachable"
        return _retry_or_cancel_manual_setup(select, title, message, cancelled)
    _say(message)
    choice = select("  Local OpenViking server is down",
                    [("Start local OpenViking", "run openviking-server and retry"), ("Retry URL", "enter the server URL again"), _CANCEL_OPTION],
                    default=0, cancel_returns=cancelled)
    if choice == 1:
        return False
    if choice != 0:
        return _SETUP_CANCELLED
    start_state, start_message = ov._start_local_openviking_server(endpoint)
    _say(start_message)
    if start_state != ov._LOCAL_SERVER_STARTED:
        return False
    _say("Waiting for OpenViking server to become reachable...")
    if ov._wait_for_openviking_health(endpoint, timeout_seconds=ov._LOCAL_OPENVIKING_AUTOSTART_TIMEOUT):
        _say("OpenViking server is reachable.")
        return True
    _say("OpenViking server did not become reachable.")
    return False


def _prompt_profile_name(prompt, select, cancelled) -> str | object:
    ov = _ov()
    while True:
        name = ov._clean_config_value(prompt("OpenViking profile name"))
        if ov._is_valid_ovcli_profile_name(name):
            return name
        if _retry_or_cancel_manual_setup(
            select, "  Invalid OpenViking profile name",
            "Profile names can only contain letters, numbers, '-' and '_'.", cancelled,
        ) is _SETUP_CANCELLED:
            return _SETUP_CANCELLED


def _confirm_replace_existing_profile(path: Path, values: dict, select, cancelled):
    """-> True (write) / False (choose another name) / _SETUP_CANCELLED."""
    ov = _ov()
    if not path.exists():
        return True
    try:
        existing_data = ov._load_ovcli_config(path)
    except Exception:
        existing_data = {}
    if existing_data == ov._ovcli_data_from_connection_values(values):
        return True
    choice = select("  OpenViking profile already exists",
                    [("Choose another name", "leave the existing profile unchanged"), ("Replace profile", "overwrite this saved OpenViking profile"), _CANCEL_OPTION],
                    default=0, cancel_returns=cancelled)
    return {1: True, 0: False}.get(choice, _SETUP_CANCELLED)


def _prompt_endpoint(prompt, select, cancelled) -> str | object:
    """Ask for a custom server URL until it normalizes and answers /health."""
    ov = _ov()
    while True:
        try:
            endpoint = ov._normalize_openviking_url(prompt("OpenViking server URL", default=ov._DEFAULT_ENDPOINT))
        except ov._OpenVikingEndpointError as exc:
            if _retry_or_cancel_manual_setup(select, "  Invalid OpenViking endpoint", str(exc), cancelled) is _SETUP_CANCELLED:
                return _SETUP_CANCELLED
            continue
        _say("Checking OpenViking server...")
        reachable, message = ov._validate_openviking_reachability(endpoint)
        if reachable:
            _say("OpenViking server is reachable.")
            return endpoint
        retry = _handle_unreachable_endpoint(endpoint, message, select, cancelled,
                                             allow_local_autostart=not (message or "").startswith(ov._OPENVIKING_RESPONDED_FAILURE_PREFIX))
        if retry is True:
            return endpoint
        if retry is _SETUP_CANCELLED:
            return _SETUP_CANCELLED


# When the entered key turns out to have the other role: (note, menu title, switch option, re-enter label).
_REROUTE = {
    "root": ("That key is valid, but it is a user API key.", "  OpenViking key is a user key",
             ("Use as User API key", "server derives account/user automatically"), "Root API key"),
    "user": ("That key is valid, but it has root access.", "  OpenViking user API key is root key",
             ("Configure as Root API key", "provide account and user IDs"), "User API key"),
}


class _Cancelled(Exception):
    """Internal unwind for the manual-connection loop (converted to _SETUP_CANCELLED at the boundary)."""


def _prompt_manual_connection_values(prompt, select, cancelled, *, service: bool = False):
    """Loop until a validated connection dict is built, or _SETUP_CANCELLED.
    ``continue`` re-enters the loop with ``api_key_type`` / ``prefilled_api_key`` carried over."""
    ov = _ov()
    if service:
        endpoint = ov._OPENVIKING_SERVICE_ENDPOINT
        _say(f"OpenViking Service endpoint: {endpoint}")
    else:
        endpoint = _prompt_endpoint(prompt, select, cancelled)
        if endpoint is _SETUP_CANCELLED:
            return _SETUP_CANCELLED

    is_local = ov._is_local_openviking_url(endpoint)
    api_key_type = "user" if service else ""
    prefilled_api_key = ""

    def retry(title: str, message: str) -> None:
        """Retry/Cancel menu; returns to loop again, raises _Cancelled to abort."""
        if _retry_or_cancel_manual_setup(select, title, message, cancelled) is _SETUP_CANCELLED:
            raise _Cancelled

    def reroute(current: str) -> None:
        """Key has the other role: offer to use it as such (prefilled), re-enter, or cancel."""
        nonlocal api_key_type, prefilled_api_key
        note, title, switch_option, role_label = _REROUTE[current]
        _say(note)
        route_choice = select(title, [switch_option, (f"Re-enter {role_label}", f"try another {current} key"), _CANCEL_OPTION],
                              default=0, cancel_returns=cancelled)
        if route_choice not in (0, 1):
            raise _Cancelled
        api_key_type = ("user" if current == "root" else "root") if route_choice == 0 else current
        prefilled_api_key = values["api_key"] if route_choice == 0 else ""

    try:
        while True:
            values = {"endpoint": endpoint, "api_key": "", "root_api_key": "", "account": "", "user": "", "agent": ""}
            if not api_key_type:
                options = [
                    ("User API key", "recommended; server derives account/user automatically" if is_local else "server derives account/user automatically"),
                    ("Root API key", "requires account and user IDs"),
                ]
                if is_local:
                    options.append(("No API key", "only for explicitly unauthenticated local development"))
                credential_choice = select("  OpenViking credential" if is_local else "  OpenViking API key type", options, default=0, cancel_returns=cancelled)
                if credential_choice == cancelled:
                    raise _Cancelled
                if is_local and credential_choice == 2:
                    _say("Validating OpenViking local dev access...")
                    valid, message, _role = ov._validate_openviking_setup_values(values)
                    if valid:
                        _say("OpenViking local dev access validated.")
                        return values
                    retry("  OpenViking credential failed", message)
                    continue
                api_key_type = "root" if credential_choice == 1 else "user"

            values["api_key_type"] = api_key_type
            api_key_label = "OpenViking API key" if service else f"OpenViking {api_key_type} API key"
            if prefilled_api_key:
                values["api_key"], prefilled_api_key = prefilled_api_key, ""
            else:
                values["api_key"] = ov._clean_config_value(prompt(api_key_label, secret=True))
            if not values["api_key"]:
                retry("  OpenViking API key required", f"{api_key_label} is required.")
                continue

            if api_key_type == "root":
                _say("Validating OpenViking root API key...")
                valid, message, role = ov._validate_openviking_setup_values(values, require_api_key=True)
                if valid and role == "user":
                    reroute("root")
                    continue
                if not (valid and role == "root"):
                    retry("  OpenViking root API key failed", message)
                    continue
                _say("OpenViking root API key validated.")
                values["root_api_key"] = values["api_key"]
                identity_errors = []
                for field, label in (("account", "OpenViking account"), ("user", "OpenViking user")):
                    ok, error, values[field] = ov._validate_openviking_identity_value(prompt(label), field=field)
                    if not ok:
                        identity_errors.append(error)
                if identity_errors:
                    retry("  OpenViking tenant identity required", identity_errors[0])
                    prefilled_api_key = values["api_key"]
                    continue

            _say("Validating OpenViking API access...")
            valid, message, role = ov._validate_openviking_setup_values(values, require_api_key=service or not is_local)
            if not valid:
                retry("  OpenViking API access failed", message)
                continue
            if api_key_type == "user" and role == "root":
                reroute("user")
                continue
            if api_key_type == "root" and role != "root":
                retry("  OpenViking root API key failed", "The supplied key was not accepted as a root API key.")
                continue
            _say("OpenViking API access validated.")
            return values
    except _Cancelled:
        return _SETUP_CANCELLED


def _link_ovcli_profile(*, config: dict, provider_config: dict, env_path: Path, ovcli_path: Path) -> None:
    ov = _ov()
    for key in ("endpoint", "api_key", "root_api_key", "account", "user", "agent", "api_key_type"):
        provider_config.pop(key, None)
    provider_config["use_ovcli_config"] = True
    # Record the path only when it is not the default location (or the env var points elsewhere).
    if os.environ.get(ov._OVCLI_CONFIG_ENV, "").strip() or ovcli_path.expanduser() != ov._default_ovcli_config_path().expanduser():
        provider_config["ovcli_config_path"] = str(ovcli_path)
    else:
        provider_config.pop("ovcli_config_path", None)
    config["memory"]["provider"] = "openviking"
    config["memory"]["openviking"] = provider_config
    ov._write_env_vars(env_path, {}, remove_keys=ov._OPENVIKING_ENV_KEYS)
    for key in ov._OPENVIKING_ENV_KEYS:
        os.environ.pop(key, None)


def _save_hermes_only_config(*, config: dict, provider_config: dict, env_path: Path, values: dict) -> None:
    ov = _ov()
    provider_config["use_ovcli_config"] = False
    provider_config.pop("ovcli_config_path", None)
    # A newly selected connection must not inherit the previous YAML peer; a
    # non-empty peer, if supplied, is saved with the connection below.
    provider_config.pop("agent", None)
    config["memory"]["provider"] = "openviking"
    config["memory"]["openviking"] = provider_config
    # Publish the file writer's cleaned values to the current process as well.
    writes = {env_key: ov._env_line_safe(value) for env_key, key in zip(ov._OPENVIKING_ENV_KEYS, ov._CONNECTION_KEYS)
              if (value := ov._clean_config_value(values.get(key)))}
    ov._write_env_vars(env_path, writes, remove_keys=ov._OPENVIKING_ENV_KEYS)
    os.environ.update(writes)
    for key in set(ov._OPENVIKING_ENV_KEYS) - set(writes):
        os.environ.pop(key, None)


def _profile_display_name(profile) -> str:
    return {"env": _ov()._OVCLI_CONFIG_ENV, "active": "ovcli.conf"}.get(profile.source, profile.name)


def _print_openviking_ready(message: str, path: Optional[Path] = None) -> None:
    print("\n  OpenViking memory is ready")
    _say(message)
    if path is not None:
        _say(f"Config file: {path}")
    print("  Start a new Hermes session to activate.\n")


def _run_existing_profile_setup(*, profiles: list, select, cancelled, config: dict, provider_config: dict, env_path: Path) -> bool | object:
    ov = _ov()
    while True:
        choice = select(
            "  OpenViking profile",
            [(_profile_display_name(p), f"{ov._clean_config_value(p.values.get('endpoint')) or ov._DEFAULT_ENDPOINT} ({p.path})") for p in profiles],
            default=0, cancel_returns=cancelled,
        )
        if choice == cancelled or choice < 0 or choice >= len(profiles):
            return _SETUP_CANCELLED
        profile = profiles[choice]

        for attempt in (0, 1):
            _say("Validating OpenViking profile...")
            require_api_key = not ov._is_local_openviking_url(profile.values.get("endpoint", ""))
            ok, message, _role = ov._validate_openviking_setup_values(profile.values, require_api_key=require_api_key)
            if ok:
                _link_ovcli_profile(config=config, provider_config=provider_config, env_path=env_path, ovcli_path=profile.path)
                _print_openviking_ready(f"Linked profile: {_profile_display_name(profile)}", profile.path)
                return True
            _say(message)
            if attempt == 1:
                break  # second failure returns to the profile picker
            retry = select("  OpenViking profile validation failed",
                           [("Choose another profile", "select a different OpenViking profile"), ("Retry validation", "try this profile again"), _CANCEL_OPTION],
                           default=0, cancel_returns=cancelled)
            if retry == 0:
                break
            if retry != 1:
                return _SETUP_CANCELLED


def _mirror_manual_config_to_openviking_store(*, prompt, select, cancelled, values: dict) -> Path | object:
    ov = _ov()
    while True:
        name = _prompt_profile_name(prompt, select, cancelled)
        if name is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        path = ov._default_ovcli_config_path().parent / f"{ov._OVCLI_SAVED_PREFIX}{name}"
        replace = _confirm_replace_existing_profile(path, values, select, cancelled)
        if replace is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        if replace is False:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # atomic_json_write creates the temp file 0600 and os.replace()s it: no
        # half-written config on crash, no chmod-after-write window for the keys.
        ov.atomic_json_write(path, ov._ovcli_data_from_connection_values(values), mode=0o600)
        return path


def _run_create_profile_setup(*, prompt, select, cancelled, config: dict, provider_config: dict, env_path: Path) -> bool | object:
    source_choice = select("  OpenViking connection",
                           [("OpenViking Service (VolcEngine Cloud)", "use the managed OpenViking endpoint"),
                            ("Custom", "use a local, VPS, or self-hosted OpenViking server")],
                           default=0, cancel_returns=cancelled)
    if source_choice == cancelled:
        return _SETUP_CANCELLED

    values = _prompt_manual_connection_values(prompt, select, cancelled, service=(source_choice == 0))
    if values is _SETUP_CANCELLED:
        return _SETUP_CANCELLED
    if values is None:
        return False

    save_choice = select("  Save OpenViking config",
                         [("Keep in Hermes only", "write values only to Hermes .env"),
                          ("Mirror to OpenViking store", "write ~/.openviking/ovcli.conf.<name> and link it")],
                         default=1, cancel_returns=cancelled)
    if save_choice == cancelled:
        return _SETUP_CANCELLED

    if save_choice == 1:
        ovcli_path = _mirror_manual_config_to_openviking_store(prompt=prompt, select=select, cancelled=cancelled, values=values)
        if ovcli_path is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        _link_ovcli_profile(config=config, provider_config=provider_config, env_path=env_path, ovcli_path=ovcli_path)
        _print_openviking_ready("Created and linked OpenViking profile.", ovcli_path)
        return True

    _save_hermes_only_config(config=config, provider_config=provider_config, env_path=env_path, values=values)
    _print_openviking_ready("Connection saved to Hermes .env.")
    return True


def run_setup(hermes_home: str, config: dict) -> None:
    """Entry point for ``OpenVikingMemoryProvider.post_setup``."""
    from hermes_cli.config import save_config
    from hermes_cli.memory_setup import _CANCELLED, _curses_select, _print_cancelled_setup, _prompt

    env_path = Path(hermes_home) / ".env"
    if not isinstance(config.get("memory"), dict):
        config["memory"] = {}
    provider_config = config["memory"].get("openviking", {})
    provider_config = provider_config if isinstance(provider_config, dict) else {}
    common = dict(select=_curses_select, cancelled=_CANCELLED, config=config, provider_config=provider_config, env_path=env_path)

    print("\n  OpenViking memory setup\n")

    profiles = _ov()._discover_ovcli_profiles()
    if profiles:
        choice = _curses_select("  OpenViking config source",
                                [("Use existing OpenViking profile", "choose from detected ovcli.conf profiles"),
                                 ("Create new OpenViking profile", "enter a new URL/API key")],
                                default=0, cancel_returns=_CANCELLED)
        if choice == _CANCELLED:
            _print_cancelled_setup()
            return
        if choice == 0:
            result = _run_existing_profile_setup(profiles=profiles, **common)
        else:
            result = _run_create_profile_setup(prompt=_prompt, **common)
    else:
        _say("No existing OpenViking CLI profiles found. Creating a new config.")
        result = _run_create_profile_setup(prompt=_prompt, **common)
    if result is _SETUP_CANCELLED:
        _print_cancelled_setup()
    elif result:
        save_config(config)
