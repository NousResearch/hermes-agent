"""The Discord ``/personality`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="personality", description="Set a personality")
    async def personality_command(interaction, name: str = ""):
        await adapter._run_simple_slash(interaction, f"/personality {name}".strip())
