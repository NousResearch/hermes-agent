"""Guards for ``hermes config set``/``unset``: a single-segment key overwriting a whole config
section, and a write to a key the managed layer or Remote Config pins.

Split out of :mod:`hermes_cli.config` (code-health size ratchet); ``set_config_value`` late-imports it.
"""

from __future__ import annotations

import sys
from typing import Any, Dict


def _exit_if_key_managed(key: str, action: str) -> None:
    """A key pinned by the managed layer cannot be set/unset (the next load would reinstate it):
    hard-reject and name the source. Distinct from ``is_managed()``; env-shaped keys route to the
    .env writers, which carry their own guard. Callers that rewrite the key before writing
    (``display.platforms`` redirect, bare ``model``) call it again with the key that is written."""
    from hermes_cli.config import _managed_source, get_config_backend, get_config_path, managed_scope
    if managed_scope.is_key_managed(key):
        print(
            f"Cannot {action} '{key}': it is managed by your administrator ({_managed_source('config.yaml')}) "
            f"and cannot be changed. Contact your administrator to modify it.", file=sys.stderr)
        sys.exit(1)
    level = get_config_backend().locked(get_config_path().parent, key)  # remote locks, prefix-aware (D7)
    if level:
        print(f"Cannot {action} '{key}': it is locked by the {level} level of Remote Config and cannot be "
              "changed from this agent.", file=sys.stderr)
        sys.exit(1)


def _guard_section_overwrite(key: str, value: Any, user_config: dict[str, Any], force: bool) -> str:
    """Refuse (or with ``force`` allow) a single-segment key overwriting a mapping with a scalar.
    Bare ``model`` is a documented shorthand — redirected to ``model.default`` so siblings survive;
    a list (or a mapping over an existing section) under it is refused without ``force``.
    Returns the (possibly redirected) key."""
    from hermes_cli.config import _exit_invalid
    existing = user_config.get(key)
    kind = "mapping" if isinstance(value, dict) else "list"
    # Containers under the model-id shorthand have no reader slot (#131435).
    if key == "model" and not force and (
            isinstance(value, list) or (isinstance(value, dict) and isinstance(existing, dict))):
        _exit_invalid(
            f"✗ Cannot set 'model' to a {kind} — "
            "the bare 'model' shorthand takes a model id, and a container value has no "
            "slot there.\n"
            "  Set the section's keys individually instead:\n"
            "    hermes config set model.provider <provider>\n"
            "    hermes config set model.default <model-id>\n"
            "  Or replace the whole section deliberately:\n"
            "    hermes config set --force model '{provider: <provider>, default: <model-id>}'")
    if "." in key or not isinstance(existing, dict):
        return key
    if key == "model":
        if force:
            what = f"the given {kind}" if isinstance(value, (dict, list)) else "a scalar"
            print(
                f"⚠ Replacing entire 'model' section with {what} "
                f"(discarding {len(existing)} existing sub-key(s))")
            return key
        _exit_if_key_managed("model.default", "set")
        print(
            f"✓ Redirecting bare 'model' to 'model.default' "
            f"(preserving {len(existing)} existing model sub-key(s))")
        return "model.default"
    if force:
        return key
    sub = [k for k in existing if isinstance(k, str)]
    err = [
        f"✗ Cannot set '{key}' to a scalar — '{key}' is a "
        f"configuration section with {len(sub)} sub-key(s)."]
    if sub:
        err.append(f"  Sub-keys: {', '.join(sub[:8])}")
        if len(sub) > 8:
            err.append(f"  ... and {len(sub) - 8} more")
    err += [
        "  Use a dotted path to set a specific leaf key:",
        f"    hermes config set {key}.<sub-key> <value>",
        "  Or use --force to replace the entire section:",
        f"    hermes config set --force {key} {value!r}"]
    print("\n".join(err), file=sys.stderr)
    sys.exit(1)
