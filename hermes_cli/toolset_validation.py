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

from typing import Any, Callable, Collection, Dict, List, Tuple


def _unknown_toolset_warning(
    platform: str,
    name: str,
    is_valid_toolset: Callable[[str], bool],
) -> str:
    """Build a consistent warning for an invalid toolset reference."""
    suggestion = f"hermes-{platform}"
    hint = (
        f" — did you mean '{suggestion}'?"
        if is_valid_toolset(suggestion)
        else ""
    )
    return (
        f"platform '{platform}' references unknown toolset "
        f"'{name}'{hint}"
    )


def _is_known_toolset(
    name: str,
    is_valid_toolset: Callable[[str], bool],
    extra_valid_names: Collection[str] | None = None,
) -> bool:
    """Return whether a toolset name is known in the current environment."""
    return is_valid_toolset(name) or name in (extra_valid_names or ())


def clean_platform_toolsets(
    platform_toolsets: object,
    is_valid_toolset: Callable[[str], bool],
    *,
    extra_valid_names: Collection[str] | None = None,
    removable_names: Collection[str] | None = None,
) -> Tuple[object, List[str], bool]:
    """Drop invalid toolset names while preserving all valid entries.

    If removing invalid entries empties a platform's list override, the platform
    key is removed so Hermes falls back to the platform defaults instead of
    treating an explicit empty list as "disable the defaults".
    """
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        return platform_toolsets, [], False

    cleaned: Dict[str, Any] = {}
    warnings: List[str] = []
    changed = False
    removable = set(removable_names or ())

    for platform, raw in platform_toolsets.items():
        if isinstance(raw, list):
            kept = []
            removed_invalid = False
            for entry in raw:
                if isinstance(entry, str) and entry and not _is_known_toolset(
                    entry,
                    is_valid_toolset,
                    extra_valid_names,
                ):
                    warnings.append(
                        _unknown_toolset_warning(platform, entry, is_valid_toolset)
                    )
                    if not removable or entry in removable:
                        changed = True
                        removed_invalid = True
                        continue
                kept.append(entry)
            if kept:
                cleaned[platform] = kept
            elif removed_invalid:
                changed = True
            else:
                cleaned[platform] = raw
            continue

        if isinstance(raw, str) and raw and not _is_known_toolset(
            raw,
            is_valid_toolset,
            extra_valid_names,
        ):
            warnings.append(_unknown_toolset_warning(platform, raw, is_valid_toolset))
            if not removable or raw in removable:
                changed = True
                continue

        cleaned[platform] = raw

    if not changed:
        return platform_toolsets, warnings, False
    return cleaned, warnings, True


def validate_platform_toolsets(
    platform_toolsets: object,
    is_valid_toolset: Callable[[str], bool],
    *,
    extra_valid_names: Collection[str] | None = None,
) -> List[str]:
    """Return human-readable warnings for a ``platform_toolsets`` mapping.

    Two failure modes are reported:

    1. A toolset name that ``is_valid_toolset`` rejects — usually a corrupted or
       renamed entry. When ``hermes-<platform>`` would have been valid (the exact
       #38798 shape, where ``cli`` held ``hermes`` instead of ``hermes-cli``),
       the warning includes that as a suggestion.
    2. The mapping is non-empty but resolves to *zero* valid toolsets, so the
       agent would start with no tools at all.

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

    valid_count = 0
    for platform, raw in platform_toolsets.items():
        names = raw if isinstance(raw, list) else [raw]
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            if _is_known_toolset(name, is_valid_toolset, extra_valid_names):
                valid_count += 1
                continue
            warnings.append(_unknown_toolset_warning(platform, name, is_valid_toolset))

    if valid_count == 0:
        warnings.append(
            "platform_toolsets resolves to zero valid toolsets — the agent will "
            "have no tools. Run `hermes tools` to reconfigure."
        )
    return warnings
