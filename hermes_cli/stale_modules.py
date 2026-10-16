"""Heal mixed ``sys.modules`` after an in-place checkout update.

Pre-reexec updaters (Hermes ≤ v2026.9.14) purged only package prefixes
(``hermes_cli``, ``gateway``, ``tools``, ``tui_gateway``, ``agent``) and left
root modules like ``utils`` cached in the updater process. The post-pull
gateway-restart phase then imports new ``hermes_cli.gateway`` /
``hermes_cli.config`` into that process; those need symbols the stale
``utils`` lacks (``file_signature``), and ``hermes update`` exits 1 with
``gateway auto-restart failed: cannot import name 'file_signature' from
'utils'``.

Post-swap hand-off (``hermes_cli.update_handoff``) makes this class dead for
updaters that already include it. This module is the bridge for the one
upgrade from a pre-handoff release onto a tree that needs new root symbols:
freshly imported ``hermes_cli`` code drops the incomplete cache before
importing ``utils``.
"""

from __future__ import annotations

import sys
from typing import Mapping, Sequence

# Root modules the narrow purge left behind, keyed by attributes that must
# exist on the on-disk copy after this release. Extend when a new root-level
# symbol would otherwise break the pre-handoff upgrade path.
_ROOT_MODULE_REQUIRED_ATTRS: dict[str, tuple[str, ...]] = {
    "utils": ("file_signature",),
}


def drop_stale_root_modules(
    required: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Drop cached root modules missing required attrs. Returns dropped names."""
    checks = _ROOT_MODULE_REQUIRED_ATTRS if required is None else required
    dropped: list[str] = []
    for name, attrs in checks.items():
        mod = sys.modules.get(name)
        if mod is None:
            continue
        if any(not hasattr(mod, attr) for attr in attrs):
            sys.modules.pop(name, None)
            dropped.append(name)
    return dropped
