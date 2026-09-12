"""The Discord ``/plan`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="plan", description="Write a markdown implementation plan (no execution)")
    async def plan_command(interaction, task: str = ""):
        await adapter._run_simple_slash(interaction, f"/plan {task}".strip())
