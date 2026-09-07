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

from hermes_cli.platforms import PLATFORMS
from hermes_cli.toolset_scope import toolset_allowed_for_platform


def _platform_default_toolset(platform: object) -> str:
    info = PLATFORMS.get(platform)
    return info.default_toolset if info is not None else f"hermes-{platform}"


def _platform_default_is_valid(
    platform: object,
    default_toolset: str,
    is_valid_toolset: Callable[[str], bool],
    is_allowed_for_platform: Callable[[str, str], bool],
) -> bool:
    platform_name = str(platform)
    if is_valid_toolset(default_toolset) and is_allowed_for_platform(
        default_toolset, platform_name
    ):
        return True
    # Dynamic plugin platforms are resolved by toolsets.resolve_toolset() even
    # though their synthesized hermes-<platform> name is not in TOOLSETS.
    try:
        from gateway.platform_registry import platform_registry

        return platform_registry.is_registered(platform)
    except Exception:
        return False


def validate_platform_toolsets(
    platform_toolsets: object,
    is_valid_toolset: Callable[[str], bool],
    is_allowed_for_platform: Callable[[str, str], bool] = toolset_allowed_for_platform,
) -> List[str]:
    """Return human-readable warnings for a ``platform_toolsets`` mapping.

    The following failure modes are reported:

    1. A toolset name that ``is_valid_toolset`` rejects — usually a corrupted or
       renamed entry. When ``hermes-<platform>`` would have been valid (the exact
       #38798 shape, where ``cli`` held ``hermes`` instead of ``hermes-cli``),
       the warning includes that as a suggestion.
    2. The mapping is non-empty but resolves to *zero* valid toolsets, so the
       agent would start with no tools at all.
    3. A platform is configured with no valid toolsets.  Checked per-platform
       because the global zero-valid-toolsets net in (2) is suppressed as soon
       as any other platform carries a valid toolset.  This includes empty
       lists and lists containing only invalid entries.  Null values are also
       reported, but match the resolver's platform-default fallback.
    4. A non-list platform value is malformed but matches the resolver's
       platform-default fallback rather than selecting a toolset by itself.

    ``is_valid_toolset`` is injected (normally :func:`toolsets.validate_toolset`)
    so this function performs no tool-registry imports or I/O and is testable in
    isolation. Platform default metadata comes from the shared platform registry.

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
        platform_valid_count = 0
        if not isinstance(raw, list):
            fallback = _platform_default_toolset(platform)
            if _platform_default_is_valid(
                platform, fallback, is_valid_toolset, is_allowed_for_platform
            ):
                valid_count += 1
                platform_valid_count += 1
                fallback_detail = f"falling back to '{fallback}'"
            else:
                fallback_detail = f"falling back to unknown default '{fallback}'"
            if raw is None:
                value_detail = "a null toolset value"
            elif isinstance(raw, str):
                value_detail = f"invalid toolset value '{raw}'"
            else:
                value_detail = f"invalid {type(raw).__name__} toolset value"
            warnings.append(
                f"platform '{platform}' has {value_detail} — "
                f"{fallback_detail}. Run `hermes tools` to configure explicitly."
            )
            if platform_valid_count == 0:
                warnings.append(
                    f"platform '{platform}' has no valid toolsets configured — "
                    f"the agent will have no tools on this platform. "
                    f"Run `hermes tools` to reconfigure."
                )
            continue
        names = raw
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            if is_valid_toolset(name) and is_allowed_for_platform(
                name, str(platform)
            ):
                valid_count += 1
                platform_valid_count += 1
                continue
            if is_valid_toolset(name):
                warnings.append(
                    f"platform '{platform}' references toolset '{name}' "
                    "which is not available on this platform"
                )
                continue
            suggestion = _platform_default_toolset(platform)
            hint = (
                f" — did you mean '{suggestion}'?"
                if _platform_default_is_valid(
                    platform,
                    suggestion,
                    is_valid_toolset,
                    is_allowed_for_platform,
                )
                else ""
            )
            warnings.append(
                f"platform '{platform}' references unknown toolset "
                f"'{name}'{hint}"
            )

        if platform_valid_count == 0:
            if isinstance(raw, list) and not raw:
                reason = "is configured with an empty toolset list"
            else:
                reason = "has no valid toolsets configured"
            warnings.append(
                f"platform '{platform}' {reason} — the agent will have no "
                f"tools on this platform. Run `hermes tools` to reconfigure."
            )

    if valid_count == 0:
        warnings.append(
            "platform_toolsets resolves to zero valid toolsets — the agent will "
            "have no tools. Run `hermes tools` to reconfigure."
        )
    return warnings
