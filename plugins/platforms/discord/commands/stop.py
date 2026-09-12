"""The Discord ``/stop`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="stop", description="Stop the running Hermes agent")
    async def stop_command(interaction):
        await adapter._run_simple_slash(interaction, "/stop", "Stop requested~")
