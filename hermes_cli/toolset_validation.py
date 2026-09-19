"""Validation for the ``platform_toolsets`` config section."""

from typing import Callable, List

from hermes_cli.platforms import PLATFORMS
from hermes_cli.toolset_scope import toolset_allowed_for_platform

_NO_TOOLS = "the agent will have no tools on this platform. Run `hermes tools` to reconfigure."


def _platform_default_toolset(platform: object) -> str:
    info = PLATFORMS.get(platform)
    return info.default_toolset if info is not None else f"hermes-{platform}"


def _platform_default_is_valid(
    platform: object, default_toolset: str, is_valid_toolset: Callable[[str], bool],
    is_allowed_for_platform: Callable[[str, str], bool]) -> bool:
    if is_valid_toolset(default_toolset) and is_allowed_for_platform(default_toolset, str(platform)):
        return True
    # Dynamic plugin platforms are resolved by toolsets.resolve_toolset() even though their synthesized
    # hermes-<platform> name is not in TOOLSETS.
    try:
        from gateway.platform_registry import platform_registry

        return platform_registry.is_registered(platform)
    except Exception:
        return False


def _known_plugin_toolsets(known: object, platform: object) -> set:
    """Plugin toolset names *known_plugin_toolsets* declares for *platform* (empty when absent).

    ``hermes tools`` writes this section on every save, so it is the config's own record of which
    plugin-provided toolsets are legitimate. Scoped per platform on purpose: a name declared for
    one platform must not mask a typo under another.
    """
    if not isinstance(known, dict):
        return set()
    names = known.get(platform)
    if isinstance(names, str):  # legacy single-string entry
        return {names}
    if not isinstance(names, list):
        return set()
    return {n for n in names if isinstance(n, str) and n}


def validate_platform_toolsets(
    platform_toolsets: object, is_valid_toolset: Callable[[str], bool],
    is_allowed_for_platform: Callable[[str, str], bool] = toolset_allowed_for_platform,
    known_plugin_toolsets: object = None,
) -> List[str]:
    """Return human-readable warnings for a ``platform_toolsets`` mapping.
    Reports: a toolset name ``is_valid_toolset`` rejects (suggesting ``hermes-<platform>`` when that
    would have been valid); a non-empty mapping resolving to zero valid toolsets (agent would start with
    no tools); a platform with no valid toolsets, checked per-platform because the global net is
    suppressed once any platform is valid; and non-list platform values, which fall back to the platform
    default. ``is_valid_toolset`` is injected so this does no registry imports or I/O.

    ``known_plugin_toolsets`` is the raw config section of the same name (``{platform: [name, ...]}``).
    Plugin toolsets register only after plugins load, which is after config validation runs, so
    ``is_valid_toolset`` alone reports every one of them as unknown; names listed there for the
    platform are accepted rather than warned about."""
    warnings: List[str] = []
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        return warnings

    valid_count = 0
    for platform, raw in platform_toolsets.items():
        known = _known_plugin_toolsets(known_plugin_toolsets, platform)
        default = _platform_default_toolset(platform)
        default_valid = _platform_default_is_valid(platform, default, is_valid_toolset, is_allowed_for_platform)
        platform_valid_count = 0
        if not isinstance(raw, list):
            if default_valid:
                valid_count += 1
                platform_valid_count += 1
            fallback_detail = f"falling back to '{default}'" if default_valid else f"falling back to unknown default '{default}'"
            if raw is None:
                value_detail = "a null toolset value"
            elif isinstance(raw, str):
                value_detail = f"invalid toolset value '{raw}'"
            else:
                value_detail = f"invalid {type(raw).__name__} toolset value"
            warnings.append(
                f"platform '{platform}' has {value_detail} — "
                f"{fallback_detail}. Run `hermes tools` to configure explicitly.")
            if platform_valid_count == 0:
                warnings.append(f"platform '{platform}' has no valid toolsets configured — {_NO_TOOLS}")
            continue

        for name in raw:
            if not isinstance(name, str) or not name:
                continue
            # A declared plugin toolset is valid even though the registry cannot confirm it yet,
            # and is_allowed_for_platform would reject it for the same not-yet-loaded reason.
            known_plugin = name in known
            if not is_valid_toolset(name) and not known_plugin:
                hint = f" — did you mean '{default}'?" if default_valid else ""
                warnings.append(f"platform '{platform}' references unknown toolset '{name}'{hint}")
            elif known_plugin or is_allowed_for_platform(name, str(platform)):
                valid_count += 1
                platform_valid_count += 1
            else:
                warnings.append(
                    f"platform '{platform}' references toolset '{name}' which is not available on this platform"
                )

        if platform_valid_count == 0:
            reason = "is configured with an empty toolset list" if not raw else "has no valid toolsets configured"
            warnings.append(f"platform '{platform}' {reason} — {_NO_TOOLS}")

    if valid_count == 0:
        warnings.append(
            "platform_toolsets resolves to zero valid toolsets — the agent will "
            "have no tools. Run `hermes tools` to reconfigure.")
    return warnings
