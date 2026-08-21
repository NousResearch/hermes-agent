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
"""

from typing import Callable, List


def validate_platform_toolsets(
    platform_toolsets: object,
    is_valid_toolset: Callable[[str], bool],
) -> List[str]:
    """Return human-readable warnings for a ``platform_toolsets`` mapping.

    Two failure modes are reported:

    1. A toolset name that ``is_valid_toolset`` rejects — usually a corrupted or
       renamed entry. When ``hermes-<platform>`` would have been valid (the exact
       #38798 shape, where ``cli`` held ``hermes`` instead of ``hermes-cli``),
       the warning includes that as a suggestion.
    2. A platform with entries that resolves to zero valid toolsets.  The check
       is **per-platform** (#89050) so that an empty list on the active platform
       is not masked by another platform carrying valid toolsets.

    ``is_valid_toolset`` is injected (normally :func:`toolsets.validate_toolset`)
    so this function performs no imports or I/O and is testable in isolation.

    Args:
        platform_toolsets: The raw ``platform_toolsets`` value from config. Only
            ``dict`` values carry toolset entries; anything else yields no
            warnings (nothing to validate).
        is_valid_toolset: Predicate returning ``True`` for a known toolset name.

    Returns:
        A list of warning strings (empty when everything is valid).
    """
    warnings: List[str] = []
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        return warnings

    all_valid = 0
    for platform, raw in platform_toolsets.items():
        names = raw if isinstance(raw, list) else [raw]
        platform_valid = 0
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            if is_valid_toolset(name):
                platform_valid += 1
                all_valid += 1
                continue
            suggestion = f"hermes-{platform}"
            hint = (
                f" \u2014 did you mean '{suggestion}'?"
                if is_valid_toolset(suggestion)
                else ""
            )
            warnings.append(
                f"platform '{platform}' references unknown toolset "
                f"'{name}'{hint}"
            )

        # Per-platform zero-tools check (#89050): an explicit but empty or
        # entirely-invalid list on a single platform should warn even when
        # other platforms carry valid toolsets.
        if platform_valid == 0 and names:
            warnings.append(
                f"platform '{platform}' has no valid toolsets \u2014 the agent "
                f"will have no tools for this platform. Run `hermes tools` to "
                f"reconfigure."
            )

    if all_valid == 0:
        warnings.append(
            "platform_toolsets resolves to zero valid toolsets \u2014 the agent "
            "will have no tools. Run `hermes tools` to reconfigure."
        )
    return warnings
