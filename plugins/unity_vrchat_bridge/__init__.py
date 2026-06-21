from __future__ import annotations

from . import core
from .cli import register_cli, unity_vrchat_bridge_command

_TOOLS = (
    ("unity_bridge_status", core.STATUS_SCHEMA, core.handle_status, "U"),
    ("unity_bridge_health", core.HEALTH_SCHEMA, core.handle_health, "U"),
    ("unity_bridge_snapshot", core.SNAPSHOT_SCHEMA, core.handle_snapshot, "U"),
    ("unity_bridge_selection_get", core.SELECTION_SCHEMA, core.handle_selection, "U"),
    ("unity_bridge_capabilities", core.CAPABILITIES_SCHEMA, core.handle_capabilities, "U"),
    ("unity_bridge_packages", core.PACKAGES_SCHEMA, core.handle_packages, "U"),
    ("unity_bridge_scene_hierarchy", core.HIERARCHY_SCHEMA, core.handle_hierarchy, "U"),
    ("unity_bridge_console_recent", core.CONSOLE_RECENT_SCHEMA, core.handle_console_recent, "U"),
    ("unity_bridge_asset_search", core.ASSET_SEARCH_SCHEMA, core.handle_asset_search, "U"),
    ("unity_bridge_asset_info", core.ASSET_INFO_SCHEMA, core.handle_asset_info, "U"),
    ("unity_bridge_menu_execute", core.MENU_EXECUTE_SCHEMA, core.handle_menu_execute, "U"),
    ("unity_bridge_operation_plan", core.OPERATION_PLAN_SCHEMA, core.handle_operation_plan, "U"),
    ("unity_bridge_plan_apply", core.PLAN_APPLY_SCHEMA, core.handle_plan_apply, "U"),
    ("unity_project_profile", core.PROJECT_PROFILE_SCHEMA, core.handle_project_profile, "U"),
    ("vrchat_project_health", core.VRCHAT_PROJECT_HEALTH_SCHEMA, core.handle_vrchat_project_health, "V"),
    (
        "commercial_asset_inspect_archive",
        core.COMMERCIAL_ASSET_INSPECT_SCHEMA,
        core.handle_commercial_asset_inspect,
        "A",
    ),
)


def register(ctx) -> None:
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="unity_vrchat_bridge",
            schema=schema,
            handler=handler,
            check_fn=core.check_available,
            description=schema.get("description", ""),
            emoji=emoji,
        )

    ctx.register_command(
        "unity-vrchat-bridge",
        handler=core.handle_slash,
        description="Run Unity/VRChat bridge diagnostics and read-only project audits.",
        args_hint="[status|health|snapshot|selection|capabilities|packages|hierarchy|console|asset-search|asset-info|menu-execute|operation-plan|plan-apply|project-profile|vrchat-health|inspect-archive]",
    )
    ctx.register_cli_command(
        name="unity-vrchat-bridge",
        help="Unity/VRChat Editor bridge diagnostics",
        setup_fn=register_cli,
        handler_fn=unity_vrchat_bridge_command,
        description=(
            "Inspect Unity bridge sessions, VRChat/VCC project health, "
            "and commercial model archives without mutating project files."
        ),
    )
