"""The Discord ``/usage`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="usage", description="Show token usage for this session")
    async def usage_command(interaction):
        await adapter._run_simple_slash(interaction, "/usage")
