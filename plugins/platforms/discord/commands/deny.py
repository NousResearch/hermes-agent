"""The Discord ``/deny`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="deny", description="Deny a pending dangerous command")
    async def deny_command(interaction, scope: str = ""):
        await adapter._run_simple_slash(interaction, f"/deny {scope}".strip())
