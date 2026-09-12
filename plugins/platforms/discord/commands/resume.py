"""The Discord ``/resume`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="resume", description="Resume a previously-named session")
    async def resume_command(interaction, name: str = ""):
        await adapter._run_simple_slash(interaction, f"/resume {name}".strip())
