"""The Discord ``/model`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="model", description="Show or change the model")
    async def model_command(interaction, name: str = ""):
        await adapter._run_simple_slash(interaction, f"/model {name}".strip())
