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

Extended for #59547: a toolset name that is not in the live toolset registry
might still be a *real* plugin toolset that the user previously enabled for
this platform — meaning its current invalidity is almost always a disabled or
uninstalled plugin (``plugins.enabled`` / package removal), not a typo. The
generic ``unknown toolset '<name>'`` warning was misleading in that case
because the fix isn't renaming anything; it's re-enabling or reinstalling the
plugin. ``known_plugin_toolsets`` (written by ``_save_platform_tools`` in
``hermes_cli/tools_config.py``) already records exactly this set, per
platform, so we cross-reference it before falling back to the generic warning.
"""

from typing import Callable, Dict, List, Optional


def validate_platform_toolsets(
    platform_toolsets: object,
    is_valid_toolset: Callable[[str], bool],
    known_plugin_toolsets: Optional[object] = None,
) -> List[str]:
    """Return human-readable warnings for a ``platform_toolsets`` mapping.

    Three failure modes are reported:

    1. A toolset name that ``is_valid_toolset`` rejects *and* that appears in
       ``known_plugin_toolsets[platform]`` — the real cause is almost always
       a plugin that was disabled or uninstalled after ``hermes tools`` last
       saved this platform. The warning names ``plugins.enabled`` and the
       likely uninstall path instead of guessing ``hermes-<platform>``. See
       #59547.
    2. A toolset name that ``is_valid_toolset`` rejects and is *not* in
       ``known_plugin_toolsets[platform]`` — usually a corrupted or renamed
       entry. When ``hermes-<platform>`` would have been valid (the exact
       #38798 shape, where ``cli`` held ``hermes`` instead of
       ``hermes-cli``), the warning includes that as a suggestion.
    3. The mapping is non-empty but resolves to *zero* valid toolsets, so the
       agent would start with no tools at all.

    ``is_valid_toolset`` is injected (normally :func:`toolsets.validate_toolset`)
    so this function performs no imports or I/O and is testable in isolation.

    ``known_plugin_toolsets`` is the raw ``config['known_plugin_toolsets']``
    value (a ``{platform: [toolset_name, ...]}`` mapping written by
    ``_save_platform_tools``); any non-dict value is treated as empty and the
    function degenerates to its pre-#59547 behavior. The third parameter is
    optional and defaults to ``None`` so existing callers/tests aren't
    affected.

    Args:
        platform_toolsets: The raw ``platform_toolsets`` value from config. Only
            ``dict`` values carry toolset entries; anything else yields no
            warnings (nothing to validate).
        is_valid_toolset: Predicate returning ``True`` for a known toolset name.
        known_plugin_toolsets: Optional per-platform snapshot of plugin toolset
            keys seen the last time ``hermes tools`` saved that platform. When
            a toolset name is missing from the live registry but found here,
            we treat it as a disabled/uninstalled plugin rather than a typo.

    Returns:
        A list of warning strings (empty when everything is valid).
    """
    warnings: List[str] = []
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        return warnings

    # Normalize the optional known-plugins snapshot. Anything that's not a dict
    # is treated as "no data" so callers can pass None / [] / a stray string
    # without this function blowing up.
    known_map: Dict[str, set] = {}
    if isinstance(known_plugin_toolsets, dict):
        for platform, raw in known_plugin_toolsets.items():
            if not isinstance(raw, list):
                continue
            known_map[str(platform)] = {
                str(name) for name in raw if isinstance(name, str) and name
            }

    valid_count = 0
    for platform, raw in platform_toolsets.items():
        names = raw if isinstance(raw, list) else [raw]
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            if is_valid_toolset(name):
                valid_count += 1
                continue

            known_for_platform = known_map.get(platform, set())
            if name in known_for_platform:
                # Real, previously-valid plugin toolset that's now gone —
                # the plugin was disabled or uninstalled, not a typo. #59547
                warnings.append(
                    f"platform '{platform}' references toolset '{name}' "
                    f"whose plugin is disabled or uninstalled — re-enable it "
                    f"in plugins.enabled (or reinstall the package) and "
                    f"re-run `hermes tools`."
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