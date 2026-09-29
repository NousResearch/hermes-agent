"""Read-only diagnostics for saved toolsets and bundled platform selections."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def config_check_diagnostics(config: dict[str, Any], get_env_value: Callable[[str], str | None]) -> list[str]:
    """Report saved selections that would otherwise be lost in startup output.

    Keep this aligned with the runtime's MCP/plugin exceptions and manifest-based
    platform gate. A configured credential is only a reason to *mention* a
    disabled platform; disabling it may have been intentional.
    """
    from agent.skill_utils import parse_config_string_list
    from hermes_cli.config import _platform_plugin_manifests
    from hermes_cli.tools_config import enabled_mcp_server_names, _get_plugin_toolset_keys
    from hermes_cli.toolset_validation import validate_platform_toolsets
    from toolsets import validate_toolset

    manifests = list(_platform_plugin_manifests(source="bundled"))
    platform_bundles = {f"hermes-{name}" for name, _manifest in manifests}
    platform_bundles.update(
        f"hermes-{name}" for name, _manifest in _platform_plugin_manifests(source="user")
    )
    plugin_toolsets = set(_get_plugin_toolset_keys())
    remembered = config.get("known_plugin_toolsets")
    if isinstance(remembered, dict):
        for names in remembered.values():
            if isinstance(names, list):
                plugin_toolsets.update(str(name) for name in names)

    mcp_config = config.get("mcp_servers") or {}
    mcp_names = set(mcp_config) if isinstance(mcp_config, dict) else set()
    platform_passthrough = (
        enabled_mcp_server_names(config) | plugin_toolsets | platform_bundles | {"no_mcp"}
    )

    def is_valid_platform_name(name: str) -> bool:
        return validate_toolset(name) or name in platform_passthrough

    diagnostics = []
    known_cli_names = mcp_names | plugin_toolsets | {"all", "*"}
    if "toolsets" in config:
        for name in parse_config_string_list(config["toolsets"]):
            if name not in known_cli_names and not validate_toolset(name):
                diagnostics.append(
                    f"toolsets contains unknown name '{name}'. "
                    "Run `hermes config edit` to remove or replace it."
                )
    diagnostics.extend(validate_platform_toolsets(config.get("platform_toolsets"), is_valid_platform_name))

    plugins = config.get("plugins") or {}
    disabled = set(parse_config_string_list(plugins.get("disabled"))) if isinstance(plugins, dict) else set()
    for name, manifest in manifests:
        key = f"platforms/{name}"
        if not {key, str(manifest.get("name") or "")}.intersection(disabled):
            continue
        requirements = manifest.get("requires_env") or []
        required_names = [
            item if isinstance(item, str) else item.get("name")
            for item in requirements
            if isinstance(item, str) or isinstance(item, dict)
        ]
        if required_names and all(
            required_name and get_env_value(required_name) for required_name in required_names
        ):
            diagnostics.append(
                f"platform plugin '{key}' is disabled while its required credentials are configured. "
                f"Run `hermes plugins enable {key}` if you want it active."
            )
    return diagnostics
