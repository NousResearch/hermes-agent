"""Which ``hermes config`` keys live in ``.env`` instead of ``config.yaml``, and their lifecycle.

Platform setting keys such as ``FEISHU_HOME_CHANNEL`` had two writers: the platform setup flows and
``/sethome`` persist them to ``.env`` through ``save_env_value``, while ``hermes config set`` only
routed credential-shaped names there and wrote every other bare name to the top level of
``config.yaml``. The gateway bridges top-level scalars into the environment only when ``.env`` lacks
the name and one-shot CLI readers never bridge, so the two copies diverged silently (#111848).

The routing rule is the key's SHAPE, not a registry: a bare ``UPPER_SNAKE`` name is an environment
setting and goes to ``.env`` — the file every runtime reader (``os.getenv``, the gateway's
``platform_gate_env``) resolves against — whether or not Hermes enumerates it anywhere. Roughly 290
of the ~700 documented variables (``TELEGRAM_GROUP_ALLOWED_USERS``, ``HERMES_TIMEZONE``, ...) are
read straight from the environment without being registered in ``OPTIONAL_ENV_VARS``, so a registry
check alone kept landing them in ``config.yaml``. Provider credentials keep their own rotation
lifecycle in ``hermes_cli.credential_lifecycle``.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Environment-variable shape: what every shell and ``os.getenv`` caller treats as a variable name.
# Case-sensitive on purpose: a lowercase bare name (``my_flag``) stays a config.yaml top-level key.
_ENV_SHAPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

def is_registered_env_name(name: str) -> bool:
    """True when Hermes itself enumerates ``name``: ``OPTIONAL_ENV_VARS`` / ``_EXTRA_ENV_KEYS``, or a
    self-configuring platform suffix so plugin adapters nobody listed (``IRC_HOME_CHANNEL``) count."""
    from hermes_cli.config import _EXTRA_ENV_KEYS, OPTIONAL_ENV_VARS
    from hermes_cli.setup_hidden_env import is_setup_hidden_env

    return name in OPTIONAL_ENV_VARS or name in _EXTRA_ENV_KEYS or is_setup_hidden_env(name)


def is_env_setting_key(key: str) -> bool:
    """True for a bare (undotted) key ``hermes config`` stores in ``.env``: any ``UPPER_SNAKE`` name,
    plus registered names typed in any case (``discord_home_channel``)."""
    if "." in key:
        return False
    return bool(_ENV_SHAPE_RE.match(key)) or is_registered_env_name(key.upper())


def _drop_config_yaml_copies(key: str) -> bool:
    """Remove same-named top-level ``config.yaml`` copies (as typed and upper-cased) so the ``.env``
    value is the only one the gateway bridge and CLI readers can disagree about."""
    from hermes_cli.config import _write_user_config, get_config_path, require_readable_config_before_write

    config_path = get_config_path()
    user_config = require_readable_config_before_write(config_path)
    stale = [name for name in {key, key.upper()} if name in user_config]
    for name in stale:
        del user_config[name]
    if stale:
        _write_user_config(config_path, user_config)
    return bool(stale)


def save_env_setting(key: str, value: str) -> None:
    from hermes_cli.config import require_env_writable, save_env_value

    require_env_writable(key.upper(), "set")
    save_env_value(key.upper(), value)
    _drop_config_yaml_copies(key)


def remove_env_setting(key: str) -> bool:
    """Remove the ``.env`` entry and any stale ``config.yaml`` copy; False when neither existed."""
    from hermes_cli.config import remove_env_value, require_env_writable

    require_env_writable(key.upper(), "remove")
    removed = remove_env_value(key.upper())
    return _drop_config_yaml_copies(key) or removed


def read_env_setting(key: str) -> Optional[str]:
    """Resolve like the gateway does: ``.env`` first, then a not-yet-converged top-level
    ``config.yaml`` copy under the name as typed, which is reported as stale on stderr."""
    from hermes_cli.config import get_env_value, read_raw_config_readonly

    value = get_env_value(key.upper())
    if value is None:
        value = read_raw_config_readonly().get(key)
        if value is not None:
            print(f"  (note: {key} is a stale top-level config.yaml copy; `hermes config set {key} <value>` "
                  f"moves it to .env, `hermes config unset {key}` removes it)", file=sys.stderr)
    return value


def apply_terminal_config_to_env(
    *,
    env: Optional[dict[str, str]] = None,
    config: Optional[dict[str, Any]] = None,
    override: Optional[bool] = None,
) -> dict[str, str]:
    """Bridge ``terminal.*`` config into the env vars terminal tools read.
    ``tools.terminal_tool`` is environment-driven because it also runs in child processes (TUI,
    dashboard PTY, gateway workers); this gives those launch paths the same bridge as the CLI
    without importing ``cli.py``. Explicit keys in the user's raw ``terminal`` section override
    matching env values; merged defaults only backfill missing env vars.

    Lives here (not ``hermes_cli.config``) so the config facade stays within its line cap;
    ``hermes_cli.config`` re-exports it for existing ``from hermes_cli.config import ...``
    callers. Config-side names are imported lazily so this module stays import-safe.
    """
    from hermes_cli.config import (
        TERMINAL_CONFIG_ENV_MAP,
        _is_ssh_remote_tilde_cwd,
        _terminal_config_value_is_bridgeable,
        _terminal_env_value,
        load_config_readonly,
        read_raw_config,
    )

    target = os.environ if env is None else env

    raw_terminal_cfg = read_raw_config().get("terminal")
    file_has_terminal_config = isinstance(raw_terminal_cfg, dict)
    raw_terminal_cfg = raw_terminal_cfg if file_has_terminal_config else {}
    should_override = file_has_terminal_config if override is None else override

    cfg = config if config is not None else load_config_readonly()
    terminal_cfg = cfg.get("terminal", {}) if isinstance(cfg, dict) else {}
    if not isinstance(terminal_cfg, dict):
        return target

    # A caller-supplied config is its own source of explicit keys; otherwise only keys present
    # in raw config.yaml may override existing env values (DEFAULT_CONFIG keys are backfill-only).
    explicit_keys = (
        terminal_cfg.keys() if config is not None else raw_terminal_cfg.keys()
    )
    backend_sources = (terminal_cfg.get("backend"), target.get("TERMINAL_ENV"))
    if not (config is not None or "backend" in raw_terminal_cfg):
        backend_sources = backend_sources[
            ::-1
        ]  # env wins when the file did not set backend
    terminal_backend = str(backend_sources[0] or backend_sources[1] or "")
    # Whether docker_image is the user's choice (config.yaml key, or TERMINAL_DOCKER_IMAGE set before
    # any bridge ran) or the shipped default. DockerEnvironment recreates a persisted container on
    # image mismatch only for a pinned image; a default flip keeps the user's sandbox and asks.
    # Children inherit both vars, so a launcher's verdict is kept unless the file pins it.
    if should_override and "docker_image" in explicit_keys:
        target["TERMINAL_DOCKER_IMAGE_PINNED"] = "1"
    elif "TERMINAL_DOCKER_IMAGE_PINNED" not in target:
        target["TERMINAL_DOCKER_IMAGE_PINNED"] = (
            "1" if "TERMINAL_DOCKER_IMAGE" in target else "0"
        )

    for cfg_key, env_var in TERMINAL_CONFIG_ENV_MAP.items():
        if cfg_key not in terminal_cfg:
            continue
        value = terminal_cfg[cfg_key]
        if not _terminal_config_value_is_bridgeable(cfg_key, value):
            continue
        if cfg_key == "cwd":
            raw_cwd = str(value or "").strip()
            if isinstance(value, str) and not _is_ssh_remote_tilde_cwd(
                terminal_backend, raw_cwd
            ):
                value = os.path.expanduser(value)
        if (should_override and cfg_key in explicit_keys) or env_var not in target:
            target[env_var] = _terminal_env_value(value)
    return target
