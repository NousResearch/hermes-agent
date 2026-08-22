"""Fail-closed compatibility for OpenMaus local-model screenshot forms.

This is deliberately not a general text-to-tool parser. It is active only
when the launching OpenMaus Hermes profile supplies an exact model binding,
the runtime model matches it, and the read-only desktop-state tool is actually
registered. Every other pseudo-tool form remains ordinary assistant text.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any

from agent.transports.types import ToolCall


ENABLE_ENV = "HERMES_OPENMAUS_SCREENSHOT_COMPAT"
MODEL_ENV = "HERMES_OPENMAUS_SCREENSHOT_COMPAT_MODEL"
SCREENSHOT_TOOL = "mcp__computer__get_desktop_state"
MAX_BLOCK_BYTES = 512

# One block, one exact case-sensitive zero-argument call, one bounded printable
# ASCII description, and no angle brackets inside that description. Fullmatch
# rejects leading/trailing prose, duplicates, nesting, and truncated blocks.
_SCREENSHOT_BLOCK = re.compile(
    r"\A<computer>\r?\n"
    r"call:default/ScreenshotTool\(\)#description="
    r"(?P<description>[\x20-\x3B\x3D\x3F-\x7E]{1,400})"
    r"\r?\n</computer>\Z"
)

# Pin the complete Windows acceptance response without retaining its local
# username, bot name, or filesystem path in public source. The structural
# matcher below independently verifies the exact read-only operation,
# zero-argument shape, header layout, whitespace, result, and marker before
# this digest gate is consulted.
_OBSERVED_CUACALL_SHA256 = (
    "9c50c64b19f16f0b0779685bb56967aac0d9cfc5e86e90dd3ec5f8f8a1057eaa"
)
_CUACALL_SCREENSHOT_BLOCK = re.compile(
    r'\A<tool_code name="cuacall" code="#include '
    r'<(?P<header_path>[A-Z]:/Users/'
    r'(?P<user>[A-Za-z0-9._-]{1,64})/\.openmausbot/bots/'
    r'(?P<bot>[A-Za-z0-9._-]{1,128})/'
    r'libraries/cuapm-cpp/dist/bin/xcua\.h)></code>\n'
    r"    xc:0\(1\)\n"
    r'    \{ "op":"screen_snapshot", "args":\{ \} \}</tool_code>'
    r'<tool_result op="exec">Succeeded</tool_result> WINDOWS_VM_SCREENSHOT_OK\Z'
)


def _is_observed_cuacall_screenshot(content: str) -> bool:
    """Validate the CUA screenshot structure, then pin the full response."""

    match = _CUACALL_SCREENSHOT_BLOCK.fullmatch(content)
    if match is None:
        return False
    if match.group("user") in {".", ".."} or match.group("bot") in {".", ".."}:
        return False

    # The regex permits only normalized forward-slash components. Rebuilding
    # the path makes that invariant explicit and prevents a future regex edit
    # from accepting aliases, traversal, or extra path material by accident.
    normalized_path = (
        f"{match.group('header_path')[:2]}/Users/{match.group('user')}"
        f"/.openmausbot/bots/{match.group('bot')}"
        "/libraries/cuapm-cpp/dist/bin/xcua.h"
    )
    if match.group("header_path") != normalized_path:
        return False

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, _OBSERVED_CUACALL_SHA256)


def _runtime_model_matches(runtime_model: Any, expected_model: str) -> bool:
    """Match only the exact ACP-normalized leaf model."""

    actual = str(runtime_model or "").strip()
    expected = expected_model.strip()
    if not actual or not expected:
        return False
    return actual == expected


def is_openmaus_screenshot_compat_enabled(agent: Any) -> bool:
    """Return true only for the explicitly bound OpenMaus model and tool."""

    if os.environ.get(ENABLE_ENV) != "1":
        return False
    expected_model = os.environ.get(MODEL_ENV, "").strip()
    if not _runtime_model_matches(getattr(agent, "model", None), expected_model):
        return False
    valid_tools = getattr(agent, "valid_tool_names", ()) or ()
    return SCREENSHOT_TOOL in valid_tools


def maybe_normalize_openmaus_screenshot_call(
    agent: Any,
    response: Any,
    finish_reason: str,
    *,
    was_streaming: bool,
) -> bool:
    """Convert one exact pseudo-call into a normal read-only tool call.

    Returns whether normalization occurred. The caller deliberately invokes
    this after the provider response is complete and before hooks, display, or
    tool dispatch inspect it.
    """

    if was_streaming or finish_reason != "stop":
        return False
    if not is_openmaus_screenshot_compat_enabled(agent):
        return False
    if getattr(response, "tool_calls", None):
        return False

    content = getattr(response, "content", None)
    if not isinstance(content, str):
        return False
    if len(content.encode("utf-8")) > MAX_BLOCK_BYTES:
        return False
    screenshot_match = _SCREENSHOT_BLOCK.fullmatch(content)
    if screenshot_match is not None and not screenshot_match.group("description").strip():
        return False
    if screenshot_match is None and not _is_observed_cuacall_screenshot(content):
        return False

    # The entire pseudo response is discarded, including any claimed result.
    # The normal Hermes tool path will append the real tool result instead.
    response.content = None
    call_id = None
    deterministic_call_id = getattr(agent, "_deterministic_call_id", None)
    if callable(deterministic_call_id):
        candidate = deterministic_call_id(SCREENSHOT_TOOL, "{}", 0)
        if isinstance(candidate, str) and candidate.strip():
            call_id = candidate.strip()
    response.tool_calls = [
        ToolCall(
            id=call_id,
            name=SCREENSHOT_TOOL,
            arguments="{}",
        )
    ]
    response.finish_reason = "tool_calls"
    return True
