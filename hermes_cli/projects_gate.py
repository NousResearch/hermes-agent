"""Projects feature gate — the one reader of ``projects.enabled`` (config.yaml).

Three surfaces reach the projects feature without going through each other: the
``hermes project`` CLI verb, the ``projects.*`` JSON-RPCs (Desktop/TUI/dashboard)
and the ``project`` model toolset folded into GUI sessions. All three answer the
same question — "is the projects feature on for this home?" — so the predicate
lives in one module and every surface imports it (#58588). Default on, so an
absent section means today's behavior; ``projects: "off"`` (or any non-mapping
value) is coerced to enabled rather than crashing the surface.
"""

from __future__ import annotations

from hermes_cli.config import DEFAULT_CONFIG

_DISABLED_MESSAGE = (
    "Projects are disabled by config (projects.enabled: false). "
    "Re-enable with: hermes config set projects.enabled true"
)


def projects_enabled(config: dict | None = None) -> bool:
    """Whether first-class Projects are on for the in-scope profile.

    Reads the effective user config (no DEFAULT_CONFIG merge — presence-sensitive)
    when *config* is not supplied, so profile scope decides, not the launch home.
    """
    if config is None:
        from hermes_cli.config_effective import load_user_config_effective
        config = load_user_config_effective()
    section = (config or {}).get("projects")
    if not isinstance(section, dict):
        return True  # absent section or a mis-shaped value: the default is on
    enabled = section.get("enabled", True)
    return bool(enabled) if not isinstance(enabled, str) else enabled.strip().lower() not in ("false", "no", "off", "0")


def projects_disabled_message() -> str:
    return _DISABLED_MESSAGE
