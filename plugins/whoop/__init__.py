"""WHOOP integration plugin — bundled, auto-loaded.

Registers read-only WHOOP tools for profile, cycles, recovery, sleep, and
workouts. Auth state lives under ``providers.whoop`` in the Hermes auth store
and is configured with ``hermes auth whoop``.
"""

from __future__ import annotations

from plugins.whoop.tools import (
    WHOOP_CYCLES_SCHEMA,
    WHOOP_PROFILE_SCHEMA,
    WHOOP_RECOVERY_SCHEMA,
    WHOOP_SLEEP_SCHEMA,
    WHOOP_WORKOUTS_SCHEMA,
    _check_whoop_available,
    _handle_whoop_cycles,
    _handle_whoop_profile,
    _handle_whoop_recovery,
    _handle_whoop_sleep,
    _handle_whoop_workouts,
)

_TOOLS = (
    ("whoop_profile", WHOOP_PROFILE_SCHEMA, _handle_whoop_profile, "💪"),
    ("whoop_cycles", WHOOP_CYCLES_SCHEMA, _handle_whoop_cycles, "🔄"),
    ("whoop_recovery", WHOOP_RECOVERY_SCHEMA, _handle_whoop_recovery, "🟢"),
    ("whoop_sleep", WHOOP_SLEEP_SCHEMA, _handle_whoop_sleep, "😴"),
    ("whoop_workouts", WHOOP_WORKOUTS_SCHEMA, _handle_whoop_workouts, "🏋️"),
)


def register(ctx) -> None:
    """Register all WHOOP tools. Called once by the plugin loader."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="whoop",
            schema=schema,
            handler=handler,
            check_fn=_check_whoop_available,
            emoji=emoji,
        )
