"""The Discord ``/retry`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="retry", description="Retry your last message")
    async def retry_command(interaction):
        await adapter._run_simple_slash(
            interaction,
            "/retry",
            "Retrying~",
        )
