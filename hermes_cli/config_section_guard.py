"""Guard ``hermes config set`` against a single-segment key overwriting a whole config section.

Split out of :mod:`hermes_cli.config` (code-health size ratchet); ``set_config_value`` late-imports it.
"""

from __future__ import annotations

import sys
from typing import Any, Dict


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
