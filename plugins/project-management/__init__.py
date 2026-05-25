"""Project management plugin for Hermes.

It provides:
- /newproject
- /archiveproject
- /deleteproject
- /createproject
- /cancelproject

Gateway sessions get a conversational draft flow via `pre_gateway_dispatch`.
CLI slash commands fall back to direct execution using structured input.
"""

from __future__ import annotations

from typing import Any

if __package__:
    from .project_management import handle_direct_command, handle_gateway_command
else:  # pytest may collect this root file as a top-level module.
    from project_management import handle_direct_command, handle_gateway_command


def _register_slash(ctx, name: str, description: str, args_hint: str = "") -> None:
    ctx.register_command(
        name,
        lambda raw_args, _name=name: handle_direct_command(_name, raw_args),
        description=description,
        args_hint=args_hint,
    )


def _on_pre_gateway_dispatch(*, event: Any = None, gateway: Any = None, **_: Any):
    if event is None or gateway is None:
        return None
    result = handle_gateway_command(event, gateway)
    if result == "skip":
        return {"action": "skip", "reason": "project-management handled"}
    return None


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    _register_slash(
        ctx,
        "newproject",
        "Create a new project workspace, board, and triage task",
        args_hint="[project name]",
    )
    _register_slash(
        ctx,
        "createproject",
        "Alias for /newproject",
        args_hint="[project name]",
    )
    _register_slash(
        ctx,
        "archiveproject",
        "Archive a project workspace and its boards",
        args_hint="[project name]",
    )
    _register_slash(
        ctx,
        "deleteproject",
        "Delete a project workspace and remove all boards",
        args_hint="[project name]",
    )
    _register_slash(
        ctx,
        "cancelproject",
        "Cancel the current pending project draft",
    )
