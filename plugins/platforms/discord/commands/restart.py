"""The Discord ``/restart`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="restart", description="Gracefully restart the Hermes gateway")
    async def restart_command(interaction):
        await adapter._run_simple_slash(interaction, "/restart", "Restart requested~")
