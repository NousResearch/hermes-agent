"""The Discord ``/compress`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="compress", description="Compress conversation context")
    async def compress_command(interaction):
        await adapter._run_simple_slash(interaction, "/compress")
