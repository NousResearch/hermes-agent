"""Config writers keep credentials in ``.env`` and put only a reference into ``config.yaml``.

``config.yaml`` is settings; ``.env`` is secrets (#77471). A writer that receives a credential for a
config field (``hermes config set model.api_key``, the dashboard model / auxiliary pickers, the
``hermes model`` custom-endpoint and auxiliary flows) saves the value to the profile's ``.env`` under
a deterministic name and stores ``${NAME}`` in the field. ``load_config`` expands the reference, so
resolution is unchanged, and ``save_config`` keeps the template (``_preserve_env_ref_templates``).
Inline values a user already has keep working; nothing is migrated.

Names follow the conventions already in the tree:

- ``model.api_key`` -> ``HERMES_MODEL_API_KEY``
- ``auxiliary.<task>.api_key`` -> ``AUXILIARY_<TASK>_API_KEY`` (the legacy aux env-var spelling)
- ``delegation.api_key`` -> ``HERMES_DELEGATION_API_KEY``
- ``custom_providers[N].api_key`` / ``providers.<name>.api_key`` -> ``custom_endpoint_key_env(name)``
- a custom endpoint the dashboard or ``hermes model`` saves -> ``endpoint_api_key_env(base_url)``

Every name is stripped from terminal / subprocess children (``_is_hermes_internal_secret``).
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Optional

_ENV_REF_ONLY_RE = re.compile(r"^\$\{[^}]+\}$")


def _env_slug(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(text or "").upper()).strip("_")


def model_api_key_env() -> str:
    return "HERMES_MODEL_API_KEY"


def aux_api_key_env(task: str) -> str:
    """``AUXILIARY_<TASK>_API_KEY``; ``delegation`` (stored in its own top-level section) gets
    ``HERMES_DELEGATION_API_KEY``."""
    if task == "delegation":
        return "HERMES_DELEGATION_API_KEY"
    return f"AUXILIARY_{_env_slug(task) or 'TASK'}_API_KEY"


def endpoint_identity(base_url: str) -> str:
    """``host[_port]`` — the identity ``hermes model``'s custom flow keys ``.env`` on, so two servers
    on one machine keep separate credentials — plus the path when it is more than ``/v1``, so two
    gateways behind one host:port (``/old/v1``, ``/new/v1``) never overwrite each other's key."""
    parsed = urllib.parse.urlparse(str(base_url or "").strip())
    identity = parsed.hostname or ""
    if parsed.port:
        identity = f"{identity}_{parsed.port}"
    path = parsed.path.strip("/")
    if identity and path and path.lower() != "v1":
        identity = f"{identity}_{path}"
    return identity


def endpoint_api_key_env(base_url: str) -> str:
    """``HERMES_CUSTOM_<HOST[_PORT][_PATH]>_API_KEY`` (the ``hermes model`` custom flow's naming);
    ``HERMES_MODEL_API_KEY`` when the URL has no host."""
    from hermes_cli.config import custom_endpoint_key_env
    identity = endpoint_identity(base_url)
    return custom_endpoint_key_env(identity) if identity else model_api_key_env()


def is_env_reference(value: Any) -> bool:
    return isinstance(value, str) and bool(_ENV_REF_ONLY_RE.match(value.strip()))


def refs_supported() -> bool:
    """Only a file-backed config shares the machine's ``.env``. Under Remote Config the document is
    served to every host while ``.env`` stays local, so a reference would not resolve elsewhere."""
    from hermes_cli.config_backend import supports_file_tooling
    return supports_file_tooling()


def store_secret(env_name: str, value: str) -> str:
    """Save ``value`` to ``.env`` as ``env_name`` and return the ``${env_name}`` reference.
    Raises ``ValueError`` when the ``.env`` write lock forbids the name, so the caller refuses the
    whole write instead of pointing config.yaml at a variable that was never saved."""
    from hermes_cli.config import require_env_writable, save_env_value
    require_env_writable(env_name, "set")
    save_env_value(env_name, value)
    return f"${{{env_name}}}"


def secret_ref_for(env_name: str, value: Any) -> Any:
    """The value a writer should persist for a credential field: ``${env_name}`` after saving a
    literal secret to ``.env``; empty values, existing references and non-strings unchanged (also
    under a backend that cannot share ``.env``)."""
    if not isinstance(value, str) or not value.strip() or is_env_reference(value) or not refs_supported():
        return value
    return store_secret(env_name, value.strip())


def secret_key_env_for(env_name: str, value: Any) -> str:
    """``key_env`` flavour of :func:`secret_ref_for` for blocks that take a variable NAME
    (``custom_providers`` entries): ``env_name`` once the secret is in ``.env``, else ""."""
    return env_name if secret_ref_for(env_name, value) != value else ""


def _custom_provider_entry_name(user_config: dict, index: str) -> str:
    entries = user_config.get("custom_providers")
    if isinstance(entries, list) and index.isdigit() and int(index) < len(entries):
        entry = entries[int(index)]
        if isinstance(entry, dict) and str(entry.get("name") or "").strip():
            return str(entry["name"])
    return f"custom_providers_{index}"


def config_path_api_key_env(parts: list[str], user_config: dict) -> Optional[str]:
    """The ``.env`` name for a credential config path ``hermes config set`` stores by reference, or
    None for paths it writes as given."""
    from hermes_cli.config import custom_endpoint_key_env
    if not parts or parts[-1] != "api_key":
        return None
    head = parts[:-1]
    if head == ["model"]:
        return model_api_key_env()
    if head == ["delegation"]:
        return aux_api_key_env("delegation")
    if len(head) == 2 and head[0] == "auxiliary":
        return aux_api_key_env(head[1])
    if len(head) == 2 and head[0] == "providers":
        return custom_endpoint_key_env(head[1])
    if len(head) == 2 and head[0] == "custom_providers":
        return custom_endpoint_key_env(_custom_provider_entry_name(user_config, head[1]))
    return None


def config_set_secret_value(key: str, value: Any, user_config: dict) -> Any:
    """``hermes config set``'s credential routing: the value to write at ``key``. A literal secret
    for a path in :func:`config_path_api_key_env` is saved to ``.env`` (a note names the variable)
    and its ``${VAR}`` reference returned; a refused ``.env`` write exits without writing."""
    from hermes_cli.config import _exit_invalid, _split_key_path
    env_name = config_path_api_key_env(_split_key_path(key), user_config)
    if env_name is None:
        return value
    try:
        stored = secret_ref_for(env_name, value)
    except ValueError as exc:
        _exit_invalid(f"✗ {exc}")
    if stored != value:
        print(f"  (secret saved to .env as {env_name}; config.yaml stores the reference {stored})")
    return stored


def masked_display_value(key: str, value: Any) -> Any:
    """The value ``hermes config set`` echoes: masked when the (possibly nested) key is
    credential-shaped, e.g. ``model.api_key`` (lowercase, so it misses the .env routing). A
    ``${VAR}`` reference is not a secret and is shown as is."""
    from hermes_cli.config import _is_secret_config_key
    if _is_secret_config_key(key) and isinstance(value, str) and value and not is_env_reference(value):
        from agent.redact import mask_secret
        return mask_secret(value)
    return value
