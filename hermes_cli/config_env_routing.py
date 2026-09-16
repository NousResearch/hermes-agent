"""Which ``hermes config`` keys live in ``.env`` instead of ``config.yaml``, and their lifecycle.

Platform setting keys such as ``FEISHU_HOME_CHANNEL`` had two writers: the platform setup flows and
``/sethome`` persist them to ``.env`` through ``save_env_value``, while ``hermes config set`` only
routed credential-shaped names there and wrote every other bare name to the top level of
``config.yaml``. The gateway bridges top-level scalars into the environment only when ``.env`` lacks
the name and one-shot CLI readers never bridge, so the two copies diverged silently (#111848).
Every name Hermes itself registers as an environment variable now routes to ``.env`` from
``set``/``get``/``unset``; provider credentials keep their own rotation lifecycle in
``hermes_cli.credential_lifecycle``.
"""

from typing import Optional


def is_env_setting_key(key: str) -> bool:
    """True for a bare (undotted) name Hermes documents as a ``.env`` variable: registered in
    ``OPTIONAL_ENV_VARS`` or ``_EXTRA_ENV_KEYS``, or carrying a self-configuring platform suffix so
    plugin adapters nobody enumerated (``IRC_HOME_CHANNEL``) get the same routing."""
    if "." in key:
        return False
    from hermes_cli.config import _EXTRA_ENV_KEYS, OPTIONAL_ENV_VARS
    from hermes_cli.setup_hidden_env import is_setup_hidden_env

    name = key.upper()
    return name in OPTIONAL_ENV_VARS or name in _EXTRA_ENV_KEYS or is_setup_hidden_env(name)


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
    from hermes_cli.config import save_env_value

    save_env_value(key.upper(), value)
    _drop_config_yaml_copies(key)


def remove_env_setting(key: str) -> bool:
    """Remove the ``.env`` entry and any stale ``config.yaml`` copy; False when neither existed."""
    from hermes_cli.config import remove_env_value

    removed = remove_env_value(key.upper())
    return _drop_config_yaml_copies(key) or removed


def read_env_setting(key: str) -> Optional[str]:
    """Resolve like the gateway does: ``.env`` first, then a not-yet-converged top-level
    ``config.yaml`` copy under the name as typed."""
    from hermes_cli.config import get_env_value, read_raw_config_readonly

    value = get_env_value(key.upper())
    if value is None:
        value = read_raw_config_readonly().get(key)
    return value
