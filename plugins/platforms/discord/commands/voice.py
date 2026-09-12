"""The Discord ``/voice`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="voice", description="Toggle voice reply mode")
    async def voice_command(interaction, mode: str = ""):
        await adapter._run_simple_slash(interaction, f"/voice {mode}".strip())
