"""The Discord ``/reload-mcp`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="reload-mcp", description="Reload MCP servers from config")
    async def reload_mcp_command(interaction):
        await adapter._run_simple_slash(interaction, "/reload-mcp")
