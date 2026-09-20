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

import re
import sys
from pathlib import Path
from typing import Optional

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
    ``config.yaml`` copy under the name as typed, which is reported as stale on stderr."""
    from hermes_cli.config import get_env_value, read_raw_config_readonly

    value = get_env_value(key.upper())
    if value is None:
        value = read_raw_config_readonly().get(key)
        if value is not None:
            print(f"  (note: {key} is a stale top-level config.yaml copy; `hermes config set {key} <value>` "
                  f"moves it to .env, `hermes config unset {key}` removes it)", file=sys.stderr)
    return value
