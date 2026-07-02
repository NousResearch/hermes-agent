"""Helpers for adapting OpenAI-style tool schemas to Runware's requirements.

Runware's OpenAI-compatible endpoint (https://api.runware.ai/v1) rejects the
otherwise-universal zero-argument tool shape ``{"type": "object",
"properties": {}}`` with:

    HTTP 400: Invalid value for 'tools[N].schema.properties'. Function
    schema properties must be a non-empty object.

OpenAI, Anthropic, and every other OpenAI-compatible backend Hermes talks to
accept an empty ``properties`` object for no-argument tools — Hermes's own
``tools/schema_sanitizer.py`` injects exactly that shape unconditionally,
since llama.cpp's grammar generator needs the key present at all (see that
module's docstring). Runware is the first backend seen that requires at
least one key *inside* it, so we patch a single harmless, optional
placeholder property into any zero-argument tool right before the request
is sent.
"""

from typing import Any, Dict, List

_PLACEHOLDER_PROPERTY = "_unused"


def sanitize_runware_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of ``tools`` with empty-properties schemas patched.

    Only tools whose ``parameters.properties`` is empty (or missing) are
    touched; tools that already declare real parameters pass through
    unchanged. Input is not mutated.
    """
    if not tools:
        return tools

    patched: List[Dict[str, Any]] = []
    any_change = False
    for tool in tools:
        if not isinstance(tool, dict):
            patched.append(tool)
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            patched.append(tool)
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict) or params.get("properties"):
            patched.append(tool)
            continue

        new_params = {
            **params,
            "type": "object",
            "properties": {
                _PLACEHOLDER_PROPERTY: {
                    "type": "string",
                    "description": "Unused. This tool takes no parameters.",
                }
            },
        }
        patched.append({**tool, "function": {**fn, "parameters": new_params}})
        any_change = True

    return patched if any_change else tools


def is_runware_provider(profile: Any) -> bool:
    """True when the resolved provider profile is Runware."""
    return getattr(profile, "name", None) == "runware"
