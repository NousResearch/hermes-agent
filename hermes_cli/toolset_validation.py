"""Validation for the ``platform_toolsets`` config section.

Pure, side-effect-free helpers so the logic is unit-testable without importing
the tool registry or launching Hermes (mirrors the decoupled-helper pattern used
elsewhere in the CLI).

Motivated by #38798: a config migration silently rewrote the valid toolset name
``hermes-cli`` to the non-existent ``hermes``. ``resolve_toolset('hermes')``
returns an empty list, so every tool silently disappeared with no error, warning,
or log entry — the agent degraded to text-only replies and the cause took
significant debugging to find. Surfacing invalid toolset names (and the
zero-tools end state) loudly turns that silent failure into an actionable one.

Also cross-references ``known_plugin_toolsets`` (see
``hermes_cli/tools_config.py``): a name that ``is_valid_toolset`` rejects
today but was recorded there for the same platform is almost always a
disabled/uninstalled plugin, not a typo — the two cases warrant different
"did you mean" advice.
"""

from typing import Callable, Dict, List, Optional


def validate_platform_toolsets(
    platform_toolsets: object,
    is_valid_toolset: Callable[[str], bool],
    known_plugin_toolsets: Optional[object] = None,
) -> List[str]:
    """Return human-readable warnings for a ``platform_toolsets`` mapping.

    Two failure modes are reported:

    1. A toolset name that ``is_valid_toolset`` rejects — usually a corrupted or
       renamed entry, or a plugin toolset whose plugin is currently disabled or
       missing. When ``hermes-<platform>`` would have been valid (the exact
       #38798 shape, where ``cli`` held ``hermes`` instead of ``hermes-cli``),
       the warning includes that as a suggestion. When the name instead matches
       an entry previously recorded for this platform in
       ``known_plugin_toolsets`` (populated by ``hermes tools`` — see
       ``_save_platform_tools`` in ``hermes_cli/tools_config.py``), the warning
       points at the plugin being disabled/uninstalled instead, since a stale
       ``hermes-<platform>`` guess would be misleading there.
    2. The mapping is non-empty but resolves to *zero* valid toolsets, so the
       agent would start with no tools at all.

    ``is_valid_toolset`` is injected (normally :func:`toolsets.validate_toolset`)
    so this function performs no imports or I/O and is testable in isolation.

    Args:
        platform_toolsets: The raw ``platform_toolsets`` value from config. Only
            ``dict`` values carry toolset entries; anything else yields no
            warnings (nothing to validate).
        is_valid_toolset: Predicate returning ``True`` for a known toolset name.
        known_plugin_toolsets: The raw ``known_plugin_toolsets`` value from
            config — a ``{platform: [toolset_key, ...]}`` mapping of plugin
            toolset keys seen the last time ``hermes tools`` saved that
            platform. Optional; a missing/malformed value just skips the
            cross-list check (falls back to the generic warning).

    Returns:
        A list of warning strings (empty when everything is valid).
    """
    warnings: List[str] = []
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        return warnings

    if not isinstance(known_plugin_toolsets, dict):
        known_plugin_toolsets = {}

    valid_count = 0
    for platform, raw in platform_toolsets.items():
        names = raw if isinstance(raw, list) else [raw]
        known_for_platform = known_plugin_toolsets.get(platform)
        known_for_platform = (
            set(known_for_platform) if isinstance(known_for_platform, list) else set()
        )
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            if is_valid_toolset(name):
                valid_count += 1
                continue
            if name in known_for_platform:
                warnings.append(
                    f"platform '{platform}' references toolset '{name}', "
                    "which was previously provided by a plugin but is not "
                    "currently available — check that the plugin is still "
                    "enabled (plugins.enabled) and installed"
                )
                continue
            suggestion = f"hermes-{platform}"
            hint = (
                f" — did you mean '{suggestion}'?"
                if is_valid_toolset(suggestion)
                else ""
            )
            warnings.append(
                f"platform '{platform}' references unknown toolset "
                f"'{name}'{hint}"
            )

    if valid_count == 0:
        warnings.append(
            "platform_toolsets resolves to zero valid toolsets — the agent will "
            "have no tools. Run `hermes tools` to reconfigure."
        )
    return warnings
