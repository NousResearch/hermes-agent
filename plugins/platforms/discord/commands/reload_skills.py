"""The Discord ``/reload-skills`` command."""


def register(tree, adapter) -> None:
    @tree.command(name="reload-skills", description="Re-scan ~/.hermes/skills/ for new or removed skills")
    async def reload_skills_command(interaction):
        await adapter._run_simple_slash(interaction, "/reload-skills")
