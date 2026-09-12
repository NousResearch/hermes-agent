"""The Discord ``/insights`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="insights", description="Show usage insights and analytics")
    async def insights_command(interaction, days: int = 7):
        await adapter._run_simple_slash(interaction, f"/insights {days}")
