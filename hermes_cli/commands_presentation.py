"""CLI help and completion projections over canonical command metadata."""

from commands import COMMAND_REGISTRY, CommandDef

def _build_description(cmd: CommandDef) -> str:
    """CLI-facing description including the usage hint."""
    if not cmd.args_hint:
        return cmd.description
    return f"{cmd.description} (usage: /{cmd.name} {cmd.args_hint})"


# Flat "/command" -> description, and the same grouped by category; both exclude gateway_only.
COMMANDS: dict[str, str] = {}
COMMANDS_BY_CATEGORY: dict[str, dict[str, str]] = {}
for _cmd in COMMAND_REGISTRY:
    if _cmd.gateway_only:
        continue
    _entries = {f"/{_cmd.name}": _build_description(_cmd)}
    for _alias in _cmd.aliases:
        _entries[f"/{_alias}"] = f"{_cmd.description} (alias for /{_cmd.name})"
    COMMANDS.update(_entries)
    COMMANDS_BY_CATEGORY.setdefault(_cmd.category, {}).update(_entries)

# /help sub-groups for the large "Session" category (category itself is load-bearing for gateway
# help, so commands are not re-tagged); unlisted Session commands fall under the base header.
HELP_SESSION_SUBGROUPS: dict[str, tuple[str, ...]] = {
    "Context": ("compress", "compact", "context", "ctx", "status"),
    "Background & Automation": (
        "bg", "btw", "agents", "tasks", "queue", "q", "steer", "s", "goal", "subgoal", "heartbeat", "hb",
        "refine", "loop", "proactive", "moa", "journey", "learning", "memory-graph")}
