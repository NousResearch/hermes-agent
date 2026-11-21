"""Guard ``hermes config set`` against writes that leave the dashboard basic-auth
provider holding a credential nobody can authenticate with (#110069).

Split out of :mod:`hermes_cli.config` (code-health size ratchet); ``set_config_value``
late-imports it.
"""

from __future__ import annotations

import base64
from typing import Any

_BASIC_AUTH_SECTION = ("dashboard", "basic_auth")


def is_valid_scrypt_hash(encoded: str) -> bool:
    """True when ``encoded`` is the dashboard basic-auth provider's own encoding.

    ``plugins/dashboard_auth/basic`` stores ``scrypt$N$r$p$<salt_b64>$<derived_key_b64>``
    and its ``_verify_password`` returns ``False`` for any string that does not parse —
    it never raises, so a malformed value is indistinguishable from a wrong password.
    This mirrors that parsing so the CLI accepts exactly what the provider can
    authenticate against: no more (a bare ``"scrypt"`` is not a usable credential) and
    no less (anything ``hash_password`` emits must pass).
    """
    parts = encoded.split("$")
    if len(parts) != 6 or parts[0] != "scrypt":
        return False
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        base64.b64decode(parts[4], validate=True)
        base64.b64decode(parts[5], validate=True)
    except (ValueError, TypeError):
        return False
    return n > 0 and r > 0 and p > 0


def guard_basic_auth_lockout(key: str, value: Any, user_config: dict[str, Any]) -> None:
    """Refuse a ``dashboard.basic_auth`` write that would silently lock the dashboard out.

    Two writes produce the lockout in #110069, and both are rejected here rather than at
    login time:

    * a non-empty ``password_hash`` that does not parse as the provider's scrypt
      encoding. The value is normally pasted into YAML, but
      ``hermes config set dashboard.basic_auth.password_hash "scrypt$..."`` in a
      double-quoted shell argument has every ``$`` segment expanded away, and the
      mangled result was stored silently;
    * clearing ``password`` while no valid ``password_hash`` is stored, which removes
      the last usable credential.

    In either case ``_verify_password`` then returns ``False`` for every password and the
    operator has lost the surface used to administer the agent. An *empty*
    ``password_hash`` stays a legal write: it disables the provider instead of locking it
    out, and removing both keys remains the documented way to turn basic auth off.
    """
    from hermes_cli.config import _exit_invalid, _split_key_path

    segments = _split_key_path(key)
    if len(segments) != 3 or tuple(segments[:2]) != _BASIC_AUTH_SECTION:
        return
    dashboard = user_config.get("dashboard")
    section = dashboard.get("basic_auth") if isinstance(dashboard, dict) else None
    section = section if isinstance(section, dict) else {}
    field = segments[2]

    if field == "password_hash":
        encoded = "" if value is None else str(value).strip()
        if encoded and not is_valid_scrypt_hash(encoded):
            _exit_invalid(
                f"✗ Cannot set '{key}': the value does not parse as the dashboard "
                "basic-auth provider's scrypt encoding "
                "('scrypt$N$r$p$<salt_b64>$<derived_key_b64>'). A double-quoted shell "
                "argument expands the '$' segments away, which stores a hash nothing "
                "can ever authenticate against — re-run with the hash in single "
                "quotes, or set it in config.yaml.")
    elif field == "password" and (value is None or str(value).strip() == ""):
        stored = str(section.get("password_hash") or "").strip()
        if not is_valid_scrypt_hash(stored):
            _exit_invalid(
                f"✗ Cannot set '{key}' to empty: no valid "
                "dashboard.basic_auth.password_hash is stored, so this write would "
                "leave the dashboard with no usable credential. Set a valid "
                "password_hash first, or remove both keys to disable basic auth.")
