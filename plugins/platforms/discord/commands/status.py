"""The Discord ``/status`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="status", description="Show Hermes session status")
    async def status_command(interaction):
        await adapter._run_simple_slash(interaction, "/status", "Status sent~")
