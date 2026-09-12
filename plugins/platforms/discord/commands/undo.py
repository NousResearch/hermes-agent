"""The Discord ``/undo`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="undo", description="Remove the last exchange")
    async def undo_command(interaction):
        await adapter._run_simple_slash(interaction, "/undo")
