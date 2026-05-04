"""Deprecated shim — use :mod:`hermes_cli.personas` instead.

This module historically held ruflo agent persona discovery + per-role
model configuration.  All logic moved to :mod:`hermes_cli.personas` when
the personas were copied out of the ruflo checkout into
``~/.hermes/personas/`` and ruflo was unwired from the runtime.

This shim re-exports the full public API so existing call sites
(``tools/delegate_tool.py``, ``cli.py``, the older test module) keep
working without churn.  New code should import from
:mod:`hermes_cli.personas` directly.
"""
from __future__ import annotations

from hermes_cli.personas import (
    DEFAULT_PERSONAS_PATH as DEFAULT_RUFLO_PATH,
    Persona,
    Persona as RufloAgent,  # legacy name for the dataclass
    SUGGESTED_ROLE_MODELS,
    _parse_frontmatter,  # re-exported for legacy callers
    _strip_frontmatter,  # re-exported for legacy callers
    apply_suggested_defaults,
    discover_personas,
    discover_ruflo_agents,
    get_personas_path,
    get_personas_path as get_ruflo_path,  # legacy name for the resolver
    get_role_model_map,
    group_by_category,
    lookup_agent,
    lookup_model_for_role,
    set_role_model,
    sync_from_ruflo,
)

__all__ = [
    "DEFAULT_RUFLO_PATH",
    "Persona",
    "RufloAgent",
    "SUGGESTED_ROLE_MODELS",
    "apply_suggested_defaults",
    "discover_personas",
    "discover_ruflo_agents",
    "get_personas_path",
    "get_ruflo_path",
    "get_role_model_map",
    "group_by_category",
    "lookup_agent",
    "lookup_model_for_role",
    "set_role_model",
    "sync_from_ruflo",
]
