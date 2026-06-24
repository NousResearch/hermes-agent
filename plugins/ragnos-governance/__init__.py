"""ragnos-governance plugin: universal pre-tool-call gate + governance telemetry.

OBSERVE by default (records every tool call to the governance ledger). Set
``RAGNOS_GOVERNANCE_ENFORCE=1`` plus ``RAGNOS_GOVERNANCE_FORBIDDEN_TOOLS`` to
BLOCK gated tools so they route through the Hermes Hub (preview + approval).

This is the Sprint 2 governance overlay from the Hermes realignment spec. It is
additive and lives entirely in the RAGnos-owned ``plugins/ragnos-governance/``
surface; it never edits upstream core, so the upstream conformance gate stays
green.
"""
from __future__ import annotations

from typing import Any, Optional

from . import governance


def _on_pre_tool_call(*, tool_name: str = "", args: Optional[dict] = None, **_kwargs: Any) -> Optional[dict]:
    enforce = governance.is_enforcing()
    verdict = governance.evaluate_tool(
        tool_name,
        args,
        enforce=enforce,
        forbidden=governance.forbidden_tools(),
    )
    governance.record_event(
        "pre_tool_call",
        {"tool": tool_name, "enforce": enforce, "blocked": verdict is not None},
    )
    return verdict


def _on_post_tool_call(*, tool_name: str = "", **_kwargs: Any) -> None:
    governance.record_event("post_tool_call", {"tool": tool_name})


def _on_session_end(**_kwargs: Any) -> None:
    governance.record_event("session_end", {})


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)
