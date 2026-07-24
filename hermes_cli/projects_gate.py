"""Gate predicate for the ``hermes project`` CLI / ``projects.*`` RPC / ``project`` toolset surfaces.

Why this exists
---------------
v0.18 (PR #49037) shipped the first-class project-management subsystem with
**three independent delivery sites** — CLI subcommand registration in
``hermes_cli/main.py``, 11 ``projects.*`` JSON-RPC handlers in
``tui_gateway/server.py``, and an unconditional ``{"project"}`` fold in
``tui_gateway.server._load_enabled_toolsets()`` — none of which consulted
user config.

This module centralises the gate into a single predicate so the three
call sites share one config read and a future surface (web gateway, IDE
plugin, etc.) cannot reintroduce the same defect. See issue #58588 for the
reporter's analysis.

Default behaviour
-----------------
Enabled by default (``projects.enabled: true``). The opt-out is provided
without breaking new-user first-run experience; established users with a
filesystem-anchored project model can flip the flag in ``config.yaml``.

Usage
-----
::

    from hermes_cli.projects_gate import projects_enabled

    if projects_enabled():
        ...  # register CLI subcommand, expose RPC, fold toolset, ...

The predicate is read-only and cheap (single ``dict.get``); it is safe to
call from cold paths (CLI parser build, gateway boot, agent toolset
resolution).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = ["projects_enabled", "projects_disabled_message"]


def projects_enabled(cfg: Optional[Mapping[str, Any]] = None) -> bool:
    """Return ``True`` unless the user has explicitly disabled projects.

    Reads ``projects.enabled`` from the active config; defaults to ``True``
    when the key is missing (preserves upstream v0.18+ behaviour for new
    users and config files that predate the toggle).

    Args:
        cfg: Optional pre-loaded config dict. When ``None`` the predicate
            calls :func:`hermes_cli.config.load_config_readonly` to fetch
            the live config. Callers in hot paths can pass a cached dict
            to skip the load.

    Returns:
        ``True`` if the projects feature is enabled (the common case),
        ``False`` if the user has opted out via ``projects.enabled: false``.
    """
    if cfg is None:
        # Lazy import — config.py transitively imports yaml + the full
        # config schema, which we don't want on every cold CLI start when
        # projects aren't going to be used anyway. The cost is one extra
        # frame for the gated paths; the savings are measurable on TUI
        # gateway boot (where the gate is also consulted).
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()

    try:
        return bool(cfg.get("projects", {}).get("enabled", True))
    except (AttributeError, TypeError):
        # Defensive: a mis-shaped config (e.g. ``projects: "off"``) should
        # not crash the parser or RPC handler. Fall back to the documented
        # default — same answer as a missing key.
        return True


def projects_disabled_message() -> str:
    """User-facing message for the gate's rejection path.

    Surfaced when the CLI is invoked with ``projects.enabled: false`` or
    a ``projects.*`` RPC is called against a disabled feature. Kept as a
    function (not a module constant) so future localisation can hook in
    via the standard ``_("...")`` wrapper without touching call sites.
    """
    return (
        "The projects feature is disabled.\n"
        "Enable it in config.yaml:\n"
        "```\n"
        "projects:\n"
        "  enabled: true\n"
        "```\n"
        "Or remove the override to fall back to the upstream default."
    )