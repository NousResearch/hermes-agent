"""Structured-output constraints across backend wire protocols.

The gateway accepts an OpenAI-shaped structured-output request on two surfaces:
Chat Completions ``response_format`` and Responses ``text.format``.  Both are
normalized here into one canonical OpenAI-shaped constraint, and then mapped to
the per-``api_mode`` wire field the active backend actually honours:

    chat_completions    -> top-level ``response_format`` (OpenAI shape, verbatim)
    anthropic_messages  -> ``output_config.format`` (Anthropic Messages structured outputs)

Other api_modes (``codex_responses``, ``codex_app_server``, ``bedrock_converse``)
have no wired mapping yet.  ``unsupported_reason`` returns a message for those so
the gateway can fail fast with a 400 rather than silently dropping the schema and
returning unconstrained text.

Membership in :data:`SUPPORTED_API_MODES` is necessary but not sufficient: some
constraints have no expression on an otherwise-supported wire (a bare
``json_object`` has no ``anthropic_messages`` equivalent), so callers must gate on
:func:`unsupported_reason`, which is constraint-aware.

This module is a leaf — it imports nothing from ``agent`` / ``gateway`` — so the
API server (validation), ``chat_completion_helpers`` (the chat_completions build),
and ``anthropic_adapter`` (the Anthropic build) can all import it without cycles.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# api_modes for which a structured-output constraint can be expressed on the
# wire at all.  Per-constraint feasibility is decided by unsupported_reason.
SUPPORTED_API_MODES = frozenset({"chat_completions", "anthropic_messages"})


def normalize_response_format(rf: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate a Chat Completions ``response_format`` value.

    Returns ``(normalized, error_message)``. ``normalized`` is ``None`` when
    no structured output was requested (plain text); ``error_message`` is set
    only when the field is present but malformed.
    """
    if rf is None:
        return None, None
    if not isinstance(rf, dict):
        return None, "'response_format' must be an object"

    rf_type = rf.get("type")
    if rf_type in (None, "text"):
        return None, None
    if rf_type == "json_object":
        return {"type": "json_object"}, None
    if rf_type == "json_schema":
        schema_block = rf.get("json_schema")
        if not isinstance(schema_block, dict) or not isinstance(schema_block.get("schema"), dict):
            return None, "response_format.json_schema must include a 'schema' object"
        normalized = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_block.get("name") or "response",
                "schema": schema_block["schema"],
            },
        }
        if "strict" in schema_block:
            normalized["json_schema"]["strict"] = bool(schema_block["strict"])
        return normalized, None
    return None, f"Unsupported response_format.type: {rf_type!r}"


def normalize_responses_text_format(text: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Convert a Responses API ``text.format`` block into the equivalent
    Chat Completions ``response_format`` so both endpoints share one
    downstream path. Returns ``(response_format, error_message)``.

    Unlike Chat Completions, the Responses API nests ``name``/``schema``/
    ``strict`` directly under ``format`` (not under a ``json_schema`` key).
    """
    if not isinstance(text, dict):
        return None, None
    fmt = text.get("format")
    if not isinstance(fmt, dict):
        return None, None

    fmt_type = fmt.get("type")
    if fmt_type in (None, "text"):
        return None, None
    if fmt_type == "json_object":
        return {"type": "json_object"}, None
    if fmt_type == "json_schema":
        schema = fmt.get("schema")
        if not isinstance(schema, dict):
            return None, "text.format.schema must be an object for json_schema"
        rf: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.get("name") or "response",
                "schema": schema,
            },
        }
        if "strict" in fmt:
            rf["json_schema"]["strict"] = bool(fmt["strict"])
        return rf, None
    return None, f"Unsupported text.format.type: {fmt_type!r}"


def unsupported_reason(constraint: Optional[Dict[str, Any]], api_mode: Optional[str]) -> Optional[str]:
    """Why the given (constraint, api_mode) pair can't be honoured, or ``None``.

    Resolves up-front so the gateway returns a 400 instead of a plain-text reply
    that silently fails schema validation.  ``None`` constraint (plain text) and
    an unresolved ``api_mode`` both pass — the latter defers to the underlying
    provider error rather than guessing.
    """
    if not constraint:
        return None
    if not api_mode:
        return None
    if api_mode not in SUPPORTED_API_MODES:
        return (
            "Structured output (response_format / json_schema) is not supported "
            f"for the configured backend (api_mode={api_mode!r}). Supported: "
            f"{', '.join(sorted(SUPPORTED_API_MODES))}."
        )
    if api_mode == "anthropic_messages" and constraint.get("type") == "json_object":
        return (
            "Structured output 'json_object' is not supported on api_mode="
            "'anthropic_messages'; provide a json_schema instead."
        )
    return None


def apply(kwargs: Dict[str, Any], constraint: Optional[Dict[str, Any]], api_mode: str) -> Dict[str, Any]:
    """Attach ``constraint`` to outgoing API ``kwargs`` in the wire shape for
    ``api_mode``, mutating and returning ``kwargs``.

    No-op when ``constraint`` is falsy.  Assumes the pair is supported — callers
    gate on :func:`unsupported_reason` first.  Existing values are preserved:
    ``response_format`` is set only if absent, and ``output_config.format`` is
    merged alongside any ``output_config.effort`` the adaptive-thinking path set.
    """
    if not constraint:
        return kwargs
    if api_mode == "chat_completions":
        kwargs.setdefault("response_format", constraint)
        return kwargs
    if api_mode == "anthropic_messages":
        fmt = _anthropic_output_format(constraint)
        if fmt is not None:
            output_config = dict(kwargs.get("output_config") or {})
            output_config.setdefault("format", fmt)
            kwargs["output_config"] = output_config
        return kwargs
    return kwargs


def _anthropic_output_format(constraint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The Anthropic ``output_config.format`` payload for a canonical constraint,
    or ``None`` when the constraint has no Anthropic expression (e.g. json_object).
    Anthropic structured outputs take a bare ``{type, schema}`` — the OpenAI
    ``name``/``strict`` fields have no equivalent and are dropped."""
    if constraint.get("type") != "json_schema":
        return None
    schema = constraint.get("json_schema", {}).get("schema")
    if not isinstance(schema, dict):
        return None
    return {"type": "json_schema", "schema": schema}
