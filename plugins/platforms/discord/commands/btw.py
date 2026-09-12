"""The Discord ``/btw`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="btw", description="Ask a side question about the current conversation")
    async def btw_command(interaction, question: str):
        await adapter._run_simple_slash(interaction, f"/btw {question}", "Side question dispatched~")
