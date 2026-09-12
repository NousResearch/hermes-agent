"""The Discord ``/reasoning`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="reasoning", description="Show/change reasoning effort, or toggle showing it")
    async def reasoning_command(interaction, effort: str = ""):
        await adapter._run_simple_slash(interaction, f"/reasoning {effort}".strip())
