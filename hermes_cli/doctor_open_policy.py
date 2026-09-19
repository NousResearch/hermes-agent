"""Read-only doctor check for open messaging platforms with powerful tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hermes_cli.doctor_report import Finding, check_warn, doctor_check


_HIGH_IMPACT_TOOLSETS = frozenset({"terminal", "file", "code_execution"})
_TRUTHY_ALLOW_ALL = frozenset({"true", "1", "yes"})


def _is_allow_all_enabled(env_get: Callable[[str], Any], platform_env: str | None) -> bool:
    """Use the gateway's allow-all tokens for a global or platform opt-in."""
    return any(
        str(env_get(name) or "").strip().lower() in _TRUTHY_ALLOW_ALL
        for name in ("GATEWAY_ALLOW_ALL_USERS", platform_env)
        if name
    )


def _check_open_platform_toolsets_for_config(
    config: dict[str, Any], env_get: Callable[[str], Any],
) -> Finding:
    """Warn only for startup-permitted open own-policy platforms.

    ``OWN_POLICY_OPEN_ENV`` is the gateway's authoritative set of adapters whose
    DM/group policy is resolved from ``extra`` and environment variables.  Keeping
    this check bound to that map prevents doctor from guessing about adapters with
    different authorization boundaries.
    """
    from gateway.config import GatewayConfig
    from gateway.open_policy import OWN_POLICY_OPEN_ENV
    from hermes_cli.tools_config import _get_platform_tools

    finding = Finding()
    gateway_config = GatewayConfig.from_dict(config)
    for platform, platform_config in gateway_config.platforms.items():
        if not platform_config.enabled:
            continue
        policy_env = OWN_POLICY_OPEN_ENV.get(platform)
        if policy_env is None:
            continue
        dm_env, group_env, allow_all_env = policy_env
        extra = platform_config.extra or {}
        dm_policy = str(extra.get("dm_policy") or env_get(dm_env) or "pairing").strip().lower()
        group_policy = str(extra.get("group_policy") or env_get(group_env) or "pairing").strip().lower()
        if dm_policy != "open" and group_policy != "open":
            continue
        if not _is_allow_all_enabled(env_get, allow_all_env):
            # Gateway startup rejects this configuration, so it is not an
            # actionable public exposure for this advisory.
            continue
        enabled = _get_platform_tools(config, platform.value)
        high_impact = sorted(_HIGH_IMPACT_TOOLSETS & {str(toolset) for toolset in enabled})
        if not high_impact:
            continue
        toolsets = ", ".join(high_impact)
        check_warn(
            f"{platform.value}: open access exposes high-impact toolsets",
            f"({toolsets})",
        )
        finding.manual_issues.append(
            f"Review {platform.value} open access with enabled toolsets: {toolsets}; "
            "restrict platform access or disable unneeded high-impact toolsets."
        )
    return finding


@doctor_check(on_error="Open platform toolset review skipped", detail="({e})")
def _check_open_platform_toolsets(should_fix: bool, finding: Finding) -> None:
    """Report public own-policy platforms that can invoke terminal, file, or code tools."""
    from hermes_cli.config import get_env_value, load_config

    result = _check_open_platform_toolsets_for_config(load_config() or {}, get_env_value)
    finding.manual_issues.extend(result.manual_issues)
