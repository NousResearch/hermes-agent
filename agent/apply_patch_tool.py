"""Grammar-constrained ``apply_patch`` file writes for GPT-5+ models.

GPT-5+ models degenerate when they stream a whole file as one JSON-escaped
``write_file`` argument: at the end of an indented block they keep emitting
content (blank lines, cycling lines, invented entries) until the output cap
cuts the call mid-string. OpenAI's own harness never sends these models a JSON
file-write tool. Its only file tool is a freeform ``apply_patch`` custom tool
whose input is raw V4A patch text, constrained by a Lark grammar and closed by
an explicit ``*** End Patch`` line.

This module gives the same surface to those models without adding a tool to
Hermes. On the wire the model sees ``apply_patch`` in place of ``write_file``
and ``patch``. Everywhere else the call is the existing ``patch`` tool in V4A
mode (``{"mode": "patch", "patch": <text>}``), so approvals, checkpoints,
display and file-mutation tracking need no changes:

- :func:`rewrite_request` swaps the tools, keeps other tools from pointing the
  model back at write_file or patch, and translates history on the way out.
- :func:`internal_arguments` (streamed) and :func:`normalize_response`
  (non-streamed) translate a completed call on the way back in.
"""

from __future__ import annotations

import json
from typing import Any, Optional

WIRE_TOOL_NAME = "apply_patch"
INTERNAL_TOOL_NAME = "patch"
END_MARKER = "*** End Patch"

# The tools apply_patch stands in for. Swapping only when one of them is
# offered keeps toolset configuration authoritative: no file tools, no patch.
_REPLACED_TOOL_NAMES = frozenset({"write_file", "patch"})

# Codex's apply_patch grammar (openai/codex codex-rs/core/assets/tools/
# apply_patch.lark) without ``*** Move to:``, which Hermes's V4A parser does
# not implement. A model that wants a rename writes Add File + Delete File.
GRAMMAR = r"""start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
"""

DESCRIPTION = """Create, replace, edit or delete files with a patch. This is a FREEFORM tool: send the patch text itself, not JSON.

This is the only file-writing tool in this session. It replaces write_file and patch: whenever instructions say to write, create, overwrite, edit or patch a file, use apply_patch. Write files whose content you compose yourself with apply_patch, not through scripts or shell commands; a script may still save data it computes.

*** Begin Patch
*** Add File: <path>
+<every line of the new file, each prefixed with +>
*** Update File: <path>
@@ <optional line near the change>
 <unchanged line>
-<line to remove>
+<line to add>
*** Delete File: <path>
*** End Patch

Add File creates a file or replaces an existing one completely. Update File edits an existing file: include about 3 unchanged lines around each change so its location is unambiguous. One patch may touch several files."""

# Auto mode: GPT-5 and later, served on a route that forwards OpenAI custom
# tools to the model. OpenRouter forwards them on chat completions and OpenAI
# enforces the grammar there; other chat-completions gateways may reject or
# rewrite them, so they need an explicit opt-in.
_AUTO_PROVIDERS = frozenset({"openrouter"})


def _gpt_major_version(model: str) -> Optional[int]:
    """Major version of a ``gpt-N`` model id (vendor prefix and ``@preset/…`` suffix allowed),
    else None. A pure preset (``@preset/name``) hides its model, so it never matches."""
    concrete, marker, _slug = (model or "").partition("@preset/")
    name = (concrete if marker and concrete.strip() else model or "").lower().rsplit("/", 1)[-1]
    if not name.startswith("gpt-"):
        return None
    digits = ""
    for ch in name[len("gpt-"):]:
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else None


def _auto_enabled(model: str, provider: str, api_mode: str) -> bool:
    if api_mode != "chat_completions" or (provider or "").lower() not in _AUTO_PROVIDERS:
        return False
    major = _gpt_major_version(model)
    if major is None or major < 5:
        return False
    lowered = (model or "").lower()
    # Open-weight and chat-tuned variants are not served with custom tools.
    return "-oss" not in lowered and "-chat" not in lowered


def is_enabled(setting: Any, model: str, provider: str, api_mode: str) -> bool:
    """Whether a request to ``model`` should offer apply_patch in place of the file tools.

    ``setting`` is ``agent.apply_patch_tool`` from config.yaml: ``"auto"``
    (default), ``True``/``False``, or a list of model-name substrings enabled in
    addition to auto detection, for OpenRouter presets whose names hide the
    serving model. Only the chat-completions wire is implemented, so other API
    modes never enable it.
    """
    if api_mode != "chat_completions":
        return False
    if isinstance(setting, str):
        lowered = setting.strip().lower()
        if lowered in {"true", "on", "yes"}:
            return True
        if lowered in {"false", "off", "no"}:
            return False
        return _auto_enabled(model, provider, api_mode)
    if isinstance(setting, bool):
        return setting
    if isinstance(setting, (list, tuple)):
        lowered = (model or "").lower()
        listed = any(isinstance(s, str) and s.lower() in lowered for s in setting)
        return listed or _auto_enabled(model, provider, api_mode)
    return _auto_enabled(model, provider, api_mode)


def agent_enabled(agent: Any) -> bool:
    """:func:`is_enabled` for the model and route the agent is using right now."""
    return is_enabled(
        getattr(agent, "_apply_patch_tool", "auto"),
        getattr(agent, "model", "") or "",
        getattr(agent, "provider", "") or "",
        getattr(agent, "api_mode", "") or "",
    )


def wire_tool() -> dict:
    """The custom tool in the flat Responses shape, which OpenRouter forwards on chat completions."""
    return {
        "type": "custom",
        "name": WIRE_TOOL_NAME,
        "description": DESCRIPTION,
        "format": {"type": "grammar", "syntax": "lark", "definition": GRAMMAR},
    }


def _tool_name(tool: Any) -> Optional[str]:
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    return fn.get("name") if isinstance(fn, dict) else None


def _patch_text(arguments: Any) -> Optional[str]:
    """The V4A text of an internal ``patch`` call in patch mode, else None."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(arguments, dict) or arguments.get("mode") != "patch":
        return None
    text = arguments.get("patch")
    return text if isinstance(text, str) else None


def _wire_message(message: Any) -> Any:
    """``message`` with its V4A ``patch`` calls written as apply_patch calls."""
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return message
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message
    rewritten = None
    for i, call in enumerate(tool_calls):
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict) or fn.get("name") != INTERNAL_TOOL_NAME:
            continue
        text = _patch_text(fn.get("arguments"))
        if text is None:
            continue
        if rewritten is None:
            rewritten = list(tool_calls)
        rewritten[i] = {**call, "function": {**fn, "name": WIRE_TOOL_NAME, "arguments": text}}
    if rewritten is None:
        return message
    return {**message, "tool_calls": rewritten}


# Other tools' descriptions that point at the replaced tools (terminal: "use
# write_file instead").
_REDIRECTED_PHRASES = (("use write_file instead", "use apply_patch instead"), ("use patch instead", "use apply_patch instead"))


def _redirect_description(tool: Any) -> Any:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(fn, dict):
        return tool
    description = fn.get("description")
    if not isinstance(description, str):
        return tool
    redirected = description
    for old, new in _REDIRECTED_PHRASES:
        redirected = redirected.replace(old, new)
    if redirected == description:
        return tool
    return {**tool, "function": {**fn, "description": redirected}}


def _execute_code_without_file_writes(tool: Any, offered: set) -> Any:
    """execute_code's schema without the write_file and patch script helpers.

    Otherwise the model routes file writes through a script, which streams the
    file as one JSON-escaped ``code`` string again.
    """
    from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, _get_execution_mode, build_execute_code_schema

    sandbox_tools = (SANDBOX_ALLOWED_TOOLS & offered) - _REPLACED_TOOL_NAMES
    return {**tool, "function": build_execute_code_schema(set(sandbox_tools), mode=_get_execution_mode())}


def rewrite_request(tools: Optional[list], messages: list) -> tuple[Optional[list], list]:
    """Offer apply_patch in place of ``write_file``/``patch`` and replay history in its form.

    Returns ``(tools, messages)`` unchanged when neither file tool is offered.
    Neither input is mutated.
    """
    if not tools or not any(_tool_name(t) in _REPLACED_TOOL_NAMES for t in tools):
        return tools, messages
    offered = {name for name in map(_tool_name, tools) if name}
    kept = []
    for tool in tools:
        name = _tool_name(tool)
        if name in _REPLACED_TOOL_NAMES:
            continue
        kept.append(_execute_code_without_file_writes(tool, offered) if name == "execute_code" else _redirect_description(tool))
    return kept + [wire_tool()], [_wire_message(m) for m in messages]


def request_offers_apply_patch(api_kwargs: Any) -> bool:
    """True when ``api_kwargs`` carries the apply_patch custom tool."""
    tools = api_kwargs.get("tools") if isinstance(api_kwargs, dict) else None
    return any(
        isinstance(t, dict) and t.get("type") == "custom" and t.get("name") == WIRE_TOOL_NAME
        for t in tools or ()
    )


def is_complete(text: str) -> bool:
    """True when the patch ends with its end marker, i.e. the call was not cut off."""
    return isinstance(text, str) and text.rstrip().endswith(END_MARKER)


def internal_arguments(text: str) -> str:
    """The ``patch`` tool arguments for a completed apply_patch call."""
    # add_overwrites: Add File replaces an existing file, the semantics the model was trained on.
    return json.dumps({"mode": "patch", "patch": text, "add_overwrites": True}, ensure_ascii=False)


def normalize_response(response: Any) -> None:
    """Translate completed apply_patch calls in a chat-completions response, in place.

    A call cut off before its end marker keeps its raw text, which is not JSON,
    so the existing truncated-tool-call handling refuses and retries it.
    """
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return
    for call in getattr(message, "tool_calls", None) or ():
        fn = getattr(call, "function", None)
        if fn is None or getattr(fn, "name", None) != WIRE_TOOL_NAME:
            continue
        text = getattr(fn, "arguments", "") or ""
        fn.name = INTERNAL_TOOL_NAME
        if is_complete(text):
            fn.arguments = internal_arguments(text)
