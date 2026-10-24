"""Public delegation extension contract. Callbacks exchange data, never live agents.

The host owns child construction, identity, ownership, admission and teardown.
One extension per profile may supply the model-facing surface and retention policy.
"""
from __future__ import annotations

from copy import deepcopy


class SubagentExtension:
    def enabled(self) -> bool:
        return True

    def child_options(self, task: dict) -> dict:
        """Optional retain/review_skills/fork_context booleans and a display title.

        fork_context is a default for new children; an explicit task choice wins.
        """
        return {}

    def extend_schema(self, schema: dict) -> dict:
        return schema

    def handle_action(self, action: str, request: dict) -> str | None:
        """Return a tool result, or None to use native dispatch. Spawn stays host-owned."""
        return None

    def format_fork_history(self, messages: list[dict], *, task: dict) -> list[dict]:
        """Format a detached source snapshot for one new child; opt-in, no default policy.

        Includes completed tool history and the current unfinished tool round.
        ``content`` is durable content; ``api_content`` sidecars may include ephemeral
        memory or request-hook injections. Recognized compaction handoffs carry
        ``_compressed_summary=True`` so formatters need no compressor internals.
        Return a complete transcript (no unmatched tool calls/results). The host
        removes persistence bookkeeping and seeds the child's own session, adding
        an assistant boundary if the history ends with a user message.
        """
        raise ValueError("fork_context requires the subagent extension to implement format_fork_history.")

    def format_assignment(self, message, *, session_id: str, timestamp: str):
        return message


def get_subagent_extension():
    from hermes_cli.plugins import get_plugin_manager
    extension = getattr(get_plugin_manager(), "_subagent_extensions", {}).get("default")
    return extension if extension is not None and extension.enabled() else None


def resolve_child_options(extension, task):
    """Resolve policy once before construction; registration alone changes no defaults."""
    options = extension.child_options(deepcopy(task)) if extension is not None else {}
    flags = ("retain", "review_skills", "fork_context")
    if not isinstance(options, dict) or any(type(options[k]) is not bool for k in flags if k in options):
        raise ValueError("Subagent extension options must contain boolean retain/review_skills/fork_context flags.")
    fork = task.get("fork_context", options.get("fork_context", False))
    if type(fork) is not bool:
        raise ValueError("Task fork_context must be a boolean.")
    return {**options, "fork_context": fork}


def extend_delegation_schema(schema):
    extension = get_subagent_extension()
    return extension.extend_schema(deepcopy(schema)) if extension is not None else schema


def dispatch_extension(action, parent, request):
    from agent.subagent_lifecycle import bind_subagent_parent
    extension = get_subagent_extension()
    if extension is None or action in {"", "spawn"}:
        return None
    with bind_subagent_parent(parent):
        return extension.handle_action(action, deepcopy(request))
