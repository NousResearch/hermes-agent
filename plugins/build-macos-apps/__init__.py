"""Bundled build-macos-apps plugin."""

from pathlib import Path

from .schemas import (
    MACOS_BUILD_PROJECT_SCHEMA,
    MACOS_COLLECT_CRASH_REPORTS_SCHEMA,
    MACOS_FIND_APP_BUNDLE_SCHEMA,
    MACOS_INSPECT_PROJECT_SCHEMA,
    MACOS_LIST_SCHEMES_SCHEMA,
    MACOS_READ_RECENT_LOGS_SCHEMA,
    MACOS_RUN_APP_SCHEMA,
    MACOS_SHOW_BUILD_SETTINGS_SCHEMA,
    MACOS_STOP_APP_SCHEMA,
    MACOS_TEST_PROJECT_SCHEMA,
)
from .tools import (
    check_macos_dev_requirements,
    handle_macos_build_project,
    handle_macos_collect_crash_reports,
    handle_macos_find_app_bundle,
    handle_macos_inspect_project,
    handle_macos_list_schemes,
    handle_macos_read_recent_logs,
    handle_macos_run_app,
    handle_macos_show_build_settings,
    handle_macos_stop_app,
    handle_macos_test_project,
)

TOOLSET = "macos-dev"


def register(ctx):
    ctx.register_tool(
        name="macos_inspect_project",
        toolset=TOOLSET,
        schema=MACOS_INSPECT_PROJECT_SCHEMA,
        handler=handle_macos_inspect_project,
        check_fn=check_macos_dev_requirements,
        description=MACOS_INSPECT_PROJECT_SCHEMA["description"],
        emoji="🧭",
    )
    ctx.register_tool(
        name="macos_list_schemes",
        toolset=TOOLSET,
        schema=MACOS_LIST_SCHEMES_SCHEMA,
        handler=handle_macos_list_schemes,
        check_fn=check_macos_dev_requirements,
        description=MACOS_LIST_SCHEMES_SCHEMA["description"],
        emoji="📋",
    )
    ctx.register_tool(
        name="macos_build_project",
        toolset=TOOLSET,
        schema=MACOS_BUILD_PROJECT_SCHEMA,
        handler=handle_macos_build_project,
        check_fn=check_macos_dev_requirements,
        description=MACOS_BUILD_PROJECT_SCHEMA["description"],
        emoji="🛠️",
    )
    ctx.register_tool(
        name="macos_test_project",
        toolset=TOOLSET,
        schema=MACOS_TEST_PROJECT_SCHEMA,
        handler=handle_macos_test_project,
        check_fn=check_macos_dev_requirements,
        description=MACOS_TEST_PROJECT_SCHEMA["description"],
        emoji="🧪",
    )
    ctx.register_tool(
        name="macos_find_app_bundle",
        toolset=TOOLSET,
        schema=MACOS_FIND_APP_BUNDLE_SCHEMA,
        handler=handle_macos_find_app_bundle,
        check_fn=check_macos_dev_requirements,
        description=MACOS_FIND_APP_BUNDLE_SCHEMA["description"],
        emoji="📦",
    )
    ctx.register_tool(
        name="macos_run_app",
        toolset=TOOLSET,
        schema=MACOS_RUN_APP_SCHEMA,
        handler=handle_macos_run_app,
        check_fn=check_macos_dev_requirements,
        description=MACOS_RUN_APP_SCHEMA["description"],
        emoji="▶️",
    )
    ctx.register_tool(
        name="macos_stop_app",
        toolset=TOOLSET,
        schema=MACOS_STOP_APP_SCHEMA,
        handler=handle_macos_stop_app,
        check_fn=check_macos_dev_requirements,
        description=MACOS_STOP_APP_SCHEMA["description"],
        emoji="⏹️",
    )
    ctx.register_tool(
        name="macos_read_recent_logs",
        toolset=TOOLSET,
        schema=MACOS_READ_RECENT_LOGS_SCHEMA,
        handler=handle_macos_read_recent_logs,
        check_fn=check_macos_dev_requirements,
        description=MACOS_READ_RECENT_LOGS_SCHEMA["description"],
        emoji="📜",
    )
    ctx.register_tool(
        name="macos_collect_crash_reports",
        toolset=TOOLSET,
        schema=MACOS_COLLECT_CRASH_REPORTS_SCHEMA,
        handler=handle_macos_collect_crash_reports,
        check_fn=check_macos_dev_requirements,
        description=MACOS_COLLECT_CRASH_REPORTS_SCHEMA["description"],
        emoji="💥",
    )
    ctx.register_tool(
        name="macos_show_build_settings",
        toolset=TOOLSET,
        schema=MACOS_SHOW_BUILD_SETTINGS_SCHEMA,
        handler=handle_macos_show_build_settings,
        check_fn=check_macos_dev_requirements,
        description=MACOS_SHOW_BUILD_SETTINGS_SCHEMA["description"],
        emoji="⚙️",
    )

    skills_dir = Path(__file__).parent / "skills"
    if skills_dir.exists():
        for child in sorted(skills_dir.iterdir()):
            skill_md = child / "SKILL.md"
            if child.is_dir() and skill_md.exists():
                ctx.register_skill(child.name, skill_md)
