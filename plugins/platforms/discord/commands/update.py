"""The Discord ``/update`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="update", description="Update Hermes Agent to the latest version")
    async def update_command(interaction):
        await adapter._run_simple_slash(interaction, "/update", "Update initiated~")
