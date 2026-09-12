"""The Discord ``/help`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="help", description="Show available commands")
    async def help_command(interaction):
        await adapter._run_simple_slash(interaction, "/help")
