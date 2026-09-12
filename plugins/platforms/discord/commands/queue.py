"""The Discord ``/queue`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="queue", description="Queue a prompt for the next turn (doesn't interrupt)")
    async def queue_command(interaction, prompt: str):
        await adapter._run_simple_slash(interaction, f"/queue {prompt}", "Queued for the next turn.")
