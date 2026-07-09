"""codex-ops — Hermes-native Codex operations guardrails.

This bundled standalone plugin ports the highest-leverage *patterns* from the
awesome-opencode review into Hermes' existing plugin seams:

* ``pre_tool_call`` blocks accidental Codex no-sandbox runs (for example
  ``--danger-full-access``, ``--dangerously-bypass-approvals-and-sandbox``,
  or ``-s danger-full-access``) unless explicitly allowed in config.
* ``transform_terminal_output`` redacts high-confidence secrets and compacts
  long Codex/dev-server/test logs before they hit model context.
* ``post_tool_call`` records a local SQLite run ledger for Codex terminal runs
  without storing message bodies or full command output.
* ``hermes codex-ops ...`` exposes local operator reports.

It deliberately does not run OpenCode plugins directly and does not add a core
model tool. Hermes remains the orchestrator; this plugin is an opt-in edge
capability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from . import core
from .cli import codex_ops_command, register_cli


def _on_pre_tool_call(tool_name: str = "", args: Any = None, **_: Any) -> Optional[dict[str, str]]:
    return core.guard_pre_tool_call(tool_name=tool_name, args=args)


def _on_transform_terminal_output(
    command: str = "",
    output: str = "",
    returncode: int = 0,
    task_id: str = "",
    env_type: str = "",
    **_: Any,
) -> Optional[str]:
    return core.transform_terminal_output(
        command=command,
        output=output,
        returncode=returncode,
        task_id=task_id,
        env_type=env_type,
    )


def _on_post_tool_call(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    duration_ms: int = 0,
    status: str = "",
    error_type: str = "",
    error_message: str = "",
    **_: Any,
) -> None:
    core.record_tool_call(
        tool_name=tool_name,
        args=args,
        result=result,
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        duration_ms=duration_ms,
        status=status,
        error_type=error_type,
        error_message=error_message,
    )


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("transform_terminal_output", _on_transform_terminal_output)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_cli_command(
        name="codex-ops",
        help="Inspect Codex run telemetry and output-hygiene settings",
        setup_fn=register_cli,
        handler_fn=codex_ops_command,
        description=(
            "Local Codex operations ledger plus terminal output compaction and "
            "dangerous Codex invocation guardrails."
        ),
    )
    skill_path = Path(__file__).parent / "skills" / "codex-operations" / "SKILL.md"
    if skill_path.exists():
        ctx.register_skill(
            "codex-operations",
            skill_path,
            description="Run Codex agents safely and observably with codex-ops.",
        )
