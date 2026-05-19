"""Model-aware projection for provider-facing tool shapes.

Hermes keeps file editing semantics in the canonical ``patch`` tool.  Some
providers, notably Codex Responses, perform better when the patch surface is a
freeform ``apply_patch`` custom tool.  This module centralizes that translation
so transports do not grow one-off patch special cases.
"""

from __future__ import annotations

import copy
import json
import os
from enum import Enum
from typing import Any, Iterable, Optional
from urllib.parse import urlparse


INTENT_KEY = "_hermes_patch_tool_intent"

APPLY_PATCH_LARK_GRAMMAR = """start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
"""


class PatchToolSurface(str, Enum):
    HERMES_PATCH = "hermes_patch"
    CODEX_FREEFORM_APPLY_PATCH = "codex_freeform_apply_patch"


def _env_flag_enabled(name: str, *, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _tool_function_name(tool: Any) -> Optional[str]:
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def _tool_names(tools: Optional[Iterable[dict]]) -> set[str]:
    return {name for tool in tools or [] if (name := _tool_function_name(tool))}


def _hostname(value: str | None) -> str:
    if not value:
        return ""
    try:
        return (urlparse(value).hostname or "").lower()
    except Exception:
        return ""


def _is_codex_compatible_responses_runtime(
    *,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    is_codex_backend: bool = False,
) -> bool:
    provider_norm = (provider or "").strip().lower()
    model_norm = (model or "").strip().lower()
    host = _hostname(base_url)
    base_url_norm = (base_url or "").strip().lower()

    return (
        bool(is_codex_backend)
        or provider_norm == "openai-codex"
        or (host == "chatgpt.com" and "/backend-api/codex" in base_url_norm)
        or (provider_norm == "openai" and "codex" in model_norm)
    )


def resolve_patch_tool_surface(
    tools: Optional[list[dict]] = None,
    *,
    api_mode: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    is_codex_backend: bool = False,
    is_github_responses: bool = False,
    is_xai_responses: bool = False,
) -> PatchToolSurface:
    """Return the provider-facing patch surface for the active runtime."""
    if "patch" not in _tool_names(tools):
        return PatchToolSurface.HERMES_PATCH
    if api_mode != "codex_responses":
        return PatchToolSurface.HERMES_PATCH
    if is_github_responses or is_xai_responses:
        return PatchToolSurface.HERMES_PATCH
    if not _env_flag_enabled("HERMES_CODEX_NATIVE_APPLY_PATCH", default=True):
        return PatchToolSurface.HERMES_PATCH
    if not _is_codex_compatible_responses_runtime(
        provider=provider,
        base_url=base_url,
        model=model,
        is_codex_backend=is_codex_backend,
    ):
        return PatchToolSurface.HERMES_PATCH
    return PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH


def freeform_apply_patch_tool() -> dict[str, Any]:
    return {
        "type": "custom",
        "name": "apply_patch",
        "description": (
            "Use the `apply_patch` tool to edit files. This is a FREEFORM "
            "tool, so do not wrap the patch in JSON."
        ),
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": APPLY_PATCH_LARK_GRAMMAR,
        },
    }


def projected_valid_tool_names(
    tools: Optional[list[dict]],
    *,
    patch_surface: PatchToolSurface,
) -> set[str]:
    names = _tool_names(tools)
    if patch_surface == PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH and "patch" in names:
        names.remove("patch")
        names.add("apply_patch")
    else:
        names.discard("apply_patch")
    return names


def _coerce_patch_surface(value: Any) -> Optional[PatchToolSurface]:
    if isinstance(value, PatchToolSurface):
        return value
    if isinstance(value, str):
        try:
            return PatchToolSurface(value)
        except ValueError:
            return None
    return None


def refresh_agent_tool_projection(agent: Any, *, freeze: bool = False) -> None:
    """Refresh runtime patch surface and valid tool names for an agent."""
    patch_surface = None
    if getattr(agent, "_patch_tool_surface_frozen", False):
        patch_surface = _coerce_patch_surface(getattr(agent, "_patch_tool_surface", None))

    if patch_surface is None:
        patch_surface = resolve_patch_tool_surface(
            getattr(agent, "tools", None),
            api_mode=getattr(agent, "api_mode", None),
            provider=getattr(agent, "provider", None),
            base_url=getattr(agent, "base_url", None),
            model=getattr(agent, "model", None),
        )
    agent._patch_tool_surface = patch_surface
    if freeze:
        agent._patch_tool_surface_frozen = True
    agent.valid_tool_names = projected_valid_tool_names(
        getattr(agent, "tools", None),
        patch_surface=patch_surface,
    )


def responses_tools_for_surface(
    tools: Optional[list[dict]],
    *,
    patch_surface: PatchToolSurface,
) -> Optional[list[dict]]:
    """Convert canonical OpenAI tool schemas to Responses tool schemas."""
    if not tools:
        if patch_surface == PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH:
            return [freeform_apply_patch_tool()]
        return None

    converted: list[dict[str, Any]] = []
    for item in tools:
        fn = item.get("function", {}) if isinstance(item, dict) else {}
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        if patch_surface == PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH and name == "patch":
            continue
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description", ""),
            "strict": False,
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })

    if patch_surface == PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH:
        converted.append(freeform_apply_patch_tool())
    return converted or None


def _json_args(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_apply_patch_to_patch_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    intent = tool_call.get(INTENT_KEY)
    if isinstance(intent, dict) and intent.get("canonical_tool") == "patch":
        args = intent.get("canonical_arguments")
        if isinstance(args, dict):
            return copy.deepcopy(args)

    fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    args = _parse_args(fn.get("arguments", "{}") if isinstance(fn, dict) else "{}")
    patch = args.get("patch")
    if isinstance(patch, str):
        return {"mode": "patch", "patch": patch}

    return {"mode": "patch", "patch": str(fn.get("arguments", ""))}


def _patch_args_to_freeform_patch(args: dict[str, Any]) -> Optional[str]:
    mode = str(args.get("mode") or "replace")
    if mode == "patch" and isinstance(args.get("patch"), str):
        return args["patch"]

    if mode != "replace":
        return None
    path = args.get("path")
    old_string = args.get("old_string")
    new_string = args.get("new_string")
    if (
        not isinstance(path, str)
        or not isinstance(old_string, str)
        or not isinstance(new_string, str)
    ):
        return None

    old_lines = old_string.split("\n")
    new_lines = new_string.split("\n")
    return "\n".join([
        "*** Begin Patch",
        f"*** Update File: {path}",
        "@@",
        *(f"-{line}" for line in old_lines),
        *(f"+{line}" for line in new_lines),
        "*** End Patch",
    ])


def _project_tool_call(tool_call: Any, *, patch_surface: PatchToolSurface) -> Any:
    if not isinstance(tool_call, dict):
        return tool_call

    tc = {k: copy.deepcopy(v) for k, v in tool_call.items() if k != INTENT_KEY}
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return tc
    name = fn.get("name")

    if patch_surface == PatchToolSurface.HERMES_PATCH and name == "apply_patch":
        patch_args = _coerce_apply_patch_to_patch_args(tool_call)
        tc["type"] = "function"
        fn["name"] = "patch"
        fn["arguments"] = _json_args(patch_args)
        return tc

    if patch_surface == PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH and name == "patch":
        args = _parse_args(fn.get("arguments", "{}"))
        patch = _patch_args_to_freeform_patch(args)
        if isinstance(patch, str):
            tc["type"] = "apply_patch"
            fn["name"] = "apply_patch"
            fn["arguments"] = _json_args({"patch": patch})
        return tc

    return tc


def project_messages_for_patch_surface(
    messages: list[dict[str, Any]],
    *,
    patch_surface: PatchToolSurface,
) -> list[dict[str, Any]]:
    """Return API-bound message copies projected for the target patch surface."""
    projected: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            projected.append(msg)
            continue
        api_msg = {k: copy.deepcopy(v) for k, v in msg.items() if k != INTENT_KEY}
        tool_calls = api_msg.get("tool_calls")
        if isinstance(tool_calls, list):
            api_msg["tool_calls"] = [
                _project_tool_call(tc, patch_surface=patch_surface)
                for tc in tool_calls
            ]
        if api_msg.get("role") == "tool" and api_msg.get("name") == "apply_patch":
            if patch_surface == PatchToolSurface.HERMES_PATCH:
                api_msg["name"] = "patch"
        elif api_msg.get("role") == "tool" and api_msg.get("name") == "patch":
            if patch_surface == PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH:
                api_msg["name"] = "apply_patch"
        projected.append(api_msg)
    return projected
