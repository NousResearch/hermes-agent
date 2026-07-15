"""Mechanical policy gate for auxiliary Kanban planning.

The main Hermes model owns semantic planning and assignment by default. The
legacy auxiliary decomposer, triage specifier, and profile describer remain
available only after one explicit config opt-in. This module does not inspect
task text, profile names, or user intent; it reads one boolean policy value.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


AUXILIARY_PLANNING_DISABLED_REASON = (
    "Kanban auxiliary planning is disabled by "
    "kanban.auxiliary_planning_enabled=false. The main Hermes model can still "
    "author tasks and assignments directly; set the option to true only to "
    "explicitly enable the legacy auxiliary planning features."
)


def _load_config() -> Any:
    from hermes_cli.config import load_config

    return load_config()


def auxiliary_planning_enabled(config: Any = None) -> bool:
    """Return true only for the exact explicit boolean opt-in.

    Config absence, malformed shapes, non-boolean truthy values, and load
    failures all fail closed. Passing an already-loaded config lets gateway
    loops apply the same policy without a second filesystem read.
    """

    if config is None:
        try:
            config = _load_config()
        except Exception:
            return False
    if not isinstance(config, Mapping):
        return False
    kanban = config.get("kanban")
    if not isinstance(kanban, Mapping):
        return False
    return kanban.get("auxiliary_planning_enabled") is True


__all__ = [
    "AUXILIARY_PLANNING_DISABLED_REASON",
    "auxiliary_planning_enabled",
]
