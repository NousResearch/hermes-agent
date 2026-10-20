"""cdp-browser plugin — registration (plugin.yaml + register(ctx) native plugin)."""

from . import schemas, tools


def register(ctx):
    """Wire schemas to handlers."""
    ctx.register_tool(
        name="cdp_list",
        toolset="cdp-browser",
        schema=schemas.CDP_LIST,
        handler=tools.cdp_list,
    )
    ctx.register_tool(
        name="cdp_run",
        toolset="cdp-browser",
        schema=schemas.CDP_RUN,
        handler=tools.cdp_run,
    )
    ctx.register_tool(
        name="cdp_spaces",
        toolset="cdp-browser",
        schema=schemas.CDP_SPACES,
        handler=tools.cdp_spaces,
    )
