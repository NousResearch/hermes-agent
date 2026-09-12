"""The Discord ``/sethome`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="sethome", description="Set this chat as the home channel")
    async def sethome_command(interaction):
        await adapter._run_simple_slash(interaction, "/sethome")
