"""The Discord ``/new`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="new", description="Start a new conversation")
    async def new_command(interaction):
        await adapter._run_simple_slash(
            interaction,
            "/reset",
            "New conversation started~",
        )
