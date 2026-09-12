"""The Discord ``/reset`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="reset", description="Reset your Hermes session")
    async def reset_command(interaction):
        await adapter._run_simple_slash(
            interaction,
            "/reset",
            "Session reset~",
        )
