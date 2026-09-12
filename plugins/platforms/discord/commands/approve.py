"""The Discord ``/approve`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="approve", description="Approve a pending dangerous command")
    async def approve_command(interaction, scope: str = ""):
        await adapter._run_simple_slash(interaction, f"/approve {scope}".strip())
