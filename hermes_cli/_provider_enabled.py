"""Tiny module: ``is_provider_enabled`` — no ``hermes_cli`` imports.

Extracted from ``hermes_cli/config.py`` to ensure ``is_provider_enabled`` is
importable even when ``config.py`` has not finished initializing (e.g. during
a circular import or a stale/partial install where config.py's module body
hasn't executed past line 6976).

All internal callers now import directly from this module rather than via
``config.py``.  ``config.py`` still re-exports the function for backward
compat with any external or third-party code that does
``from hermes_cli.config import is_provider_enabled``.
"""

from typing import Any, Dict, Optional


def is_provider_enabled(provider_cfg: Optional[Dict[str, Any]]) -> bool:
    """Return whether a ``providers.<name>`` config block is enabled.

    A provider is enabled by default.  Only an explicit ``enabled: false`` in
    the block hides it from the model picker, ``/models`` listings, the
    runtime resolver and the doctor / status output.

    Backward-compat: configs without the ``enabled`` key keep working as
    before — the default is ``True``.

    Pass any non-dict (None, list, string) and you get ``True`` too, so
    malformed entries don't disappear silently; they'll still be flagged
    by the existing validation paths.
    """
    if not isinstance(provider_cfg, dict):
        return True
    flag = provider_cfg.get("enabled", True)
    if isinstance(flag, bool):
        return flag
    # YAML can produce strings for "true"/"false" depending on quoting.
    if isinstance(flag, str):
        return flag.strip().lower() not in {"false", "0", "no", "off"}
    return bool(flag)
