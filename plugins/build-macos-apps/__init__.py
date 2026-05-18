"""Bundled build-macos-apps plugin."""

from .schemas import (
    MACOS_BUILD_PROJECT_SCHEMA,
    MACOS_INSPECT_PROJECT_SCHEMA,
    MACOS_LIST_SCHEMES_SCHEMA,
)
from .tools import (
    check_macos_dev_requirements,
    handle_macos_build_project,
    handle_macos_inspect_project,
    handle_macos_list_schemes,
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
