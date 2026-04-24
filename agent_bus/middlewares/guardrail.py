"""GuardrailMiddleware — pre-tool-call authorization with pluggable provider.

Inspired by DeerFlow's GuardrailMiddleware (§1.B #6). Complements finalizer
gate (post-action, transition-level) with a **pre-action** check: before a
tool is invoked, run it past a GuardrailProvider and block if denied.

Provider protocol
-----------------
Any object with:
    should_allow(tool_call: dict) -> tuple[bool, str]   # (allowed, reason)

Built-in providers
------------------
- AllowlistProvider: env var `HERMES_TOOL_ALLOWLIST` (comma-separated names).
  Empty allowlist = allow all (zero-friction default).
- DenylistProvider:  env var `HERMES_TOOL_DENYLIST`  (comma-separated names).
  Denies exact-match names.

Order: 60 (between DanglingToolCall and Summarization).

Env vars
--------
HERMES_MW_GUARDRAIL          off | core (default core)
HERMES_GUARDRAIL_MODE        allowlist | denylist | off (default denylist)
HERMES_TOOL_ALLOWLIST        comma-separated tool names (allowlist mode)
HERMES_TOOL_DENYLIST         comma-separated tool names (denylist mode)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

from agent_bus.middleware import BaseMiddleware, MiddlewareContext

logger = logging.getLogger(__name__)


@runtime_checkable
class GuardrailProvider(Protocol):
    """Returns (allowed, reason). `reason` shown to the agent on deny."""

    def should_allow(self, tool_call: dict) -> tuple[bool, str]: ...


# -------- Built-in providers --------
def _tool_name(tool_call: dict) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    return (
        tool_call.get("name")
        or (tool_call.get("function") or {}).get("name")
    )


class AllowlistProvider:
    def __init__(self, names: set[str] | None = None) -> None:
        if names is None:
            raw = os.environ.get("HERMES_TOOL_ALLOWLIST", "").strip()
            names = {n.strip() for n in raw.split(",") if n.strip()}
        self.names = names

    def should_allow(self, tool_call: dict) -> tuple[bool, str]:
        if not self.names:
            return True, ""  # empty allowlist = allow all
        name = _tool_name(tool_call)
        if not name:
            return False, "tool_call missing name"
        if name in self.names:
            return True, ""
        return False, f"tool `{name}` not in allowlist {sorted(self.names)}"


class DenylistProvider:
    def __init__(self, names: set[str] | None = None) -> None:
        if names is None:
            raw = os.environ.get("HERMES_TOOL_DENYLIST", "").strip()
            names = {n.strip() for n in raw.split(",") if n.strip()}
        self.names = names

    def should_allow(self, tool_call: dict) -> tuple[bool, str]:
        if not self.names:
            return True, ""
        name = _tool_name(tool_call)
        if not name:
            return True, ""  # anonymous tool_calls pass (finalizer catches)
        if name in self.names:
            return False, f"tool `{name}` is denylisted"
        return True, ""


class AlwaysAllow:
    """Used when HERMES_GUARDRAIL_MODE=off."""

    def should_allow(self, tool_call: dict) -> tuple[bool, str]:
        return True, ""


def _get_provider() -> GuardrailProvider:
    # Hook for custom providers later — inject via ctx.metadata if needed
    mode = os.environ.get("HERMES_GUARDRAIL_MODE", "denylist").lower()
    if mode == "off":
        return AlwaysAllow()
    if mode == "allowlist":
        return AllowlistProvider()
    if mode == "denylist":
        return DenylistProvider()
    logger.warning("unknown HERMES_GUARDRAIL_MODE=%r — defaulting to denylist", mode)
    return DenylistProvider()


class GuardrailMiddleware(BaseMiddleware):
    """Block disallowed tool calls before they're dispatched."""

    name = "guardrail"

    def before_tool(self, ctx: MiddlewareContext) -> MiddlewareContext:
        tc = ctx.pending_tool_call
        if not tc:
            return ctx
        # Allow custom provider injection via ctx.metadata["guardrail_provider"]
        provider = ctx.metadata.get("guardrail_provider") or _get_provider()
        try:
            allowed, reason = provider.should_allow(tc)
        except Exception as exc:
            logger.warning("guardrail provider raised %s; defaulting to deny", exc)
            allowed, reason = False, f"provider error: {exc}"

        if allowed:
            ctx.record(self.name, "before_tool", "allow", _tool_name(tc) or "")
            return ctx

        # Deny: record decision + set denial flag so caller skips dispatch
        ctx.record(self.name, "before_tool", "deny", reason)
        ctx.metadata.setdefault("guardrail_denials", []).append({
            "tool_call": tc,
            "reason": reason,
        })
        # Inject a synthetic ToolMessage so the conversation stays well-formed
        tid = tc.get("id")
        if tid:
            deny_msg = {
                "role": "tool",
                "tool_call_id": tid,
                "content": f"[guardrail] {reason}",
                "_guardrail_denied": True,
            }
            ctx.messages.append(deny_msg)
        # Clear pending — caller should not dispatch
        ctx.pending_tool_call = None
        return ctx
