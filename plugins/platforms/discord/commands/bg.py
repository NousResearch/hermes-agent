"""The Discord ``/bg`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="bg", description="Run a prompt in a separate background session")
    async def bg_command(interaction, prompt: str):
        await adapter._run_simple_slash(interaction, f"/bg {prompt}", "Background task started~")
