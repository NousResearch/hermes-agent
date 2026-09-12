"""The Discord ``/title`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="title", description="Set or show the session title")
    async def title_command(interaction, name: str = ""):
        await adapter._run_simple_slash(interaction, f"/title {name}".strip())
