"""The Discord ``/steer`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="steer", description="Inject a message after the next tool call (no interrupt)")
    async def steer_command(interaction, prompt: str):
        await adapter._run_simple_slash(interaction, f"/steer {prompt}")
