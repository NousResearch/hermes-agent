"""Structured payloads for tool calls that did not complete — one source of truth.

Tool results that are not real outputs (pre-execution rejections, interrupt
skips, timeouts, orphan recovery) are stated in the tool-result content the model
sees, and the context compressor reads them to summarize the call as "not run" —
otherwise compaction renders a never-completed call as a completed action and the
post-compaction agent reports its effect as done.

Producers build messages from these constants and wrap them with the
:func:`rejected_result` / :func:`outcome_result` builders into a
``{"status": …, "error": …}`` payload, which the compressor reads structurally
(the same way it reads ``status: blocked`` / ``pending_approval``). The
compressor's prose matchers exist only for transcripts recorded before the
structured statuses existed; both sides build from these constants, so a
rewording cannot silently decouple them (each side's tests alone would stay
green). This module must stay import-light: it is imported by the compressor,
which the loop imports.
"""

import json

# Structured marker: rejections carry this ``status`` so the compressor never has
# to parse prose to know a call did not run.
REJECTED_STATUS = "rejected"

# Execution-layer outcomes for calls that never returned a result. ``cancelled``
# matches the payload ``_ToolCallRef.emit_cancelled`` already emits; ``timeout``
# matches the sequential runner's hook outcome; ``unknown`` marks calls whose
# effect cannot be determined (orphan recovery, a worker that vanished).
CANCELLED_STATUS = "cancelled"
TIMEOUT_STATUS = "timeout"
UNKNOWN_STATUS = "unknown"


def rejected_result(message: str) -> str:
    """Wrap rejection prose into the structured payload the compressor reads first.

    Same shape as ``tools.registry.tool_error`` (a JSON body with an ``error``
    string), plus the ``status`` field the refused-result summarizer keys on.
    """
    return json.dumps({"status": REJECTED_STATUS, "error": message}, ensure_ascii=False)


def outcome_result(status: str, message: str) -> str:
    """Wrap an execution-layer non-completion (skip / timeout / unknown effect)
    into the structured payload the compressor reads first — same shape as
    :func:`rejected_result`, for outcomes that are not validation rejections."""
    return json.dumps({"status": status, "error": message}, ensure_ascii=False)


def error_result(message: str) -> str:
    """Wrap a raised error into the plain ``{"error": …}`` body the compressor's
    failure suffix reads — the import-light twin of ``tools.registry.tool_error``
    (registry is import-heavy from the executor's helpers)."""
    return json.dumps({"error": message}, ensure_ascii=False)


# Legacy prose shapes (pre-status transcripts). The compressor builds its legacy
# matchers from these constants; live producers no longer emit them bare.
TOOL_EXECUTION_CANCELLED_PREFIX = "[Tool execution cancelled — "
TOOL_EXECUTION_SKIPPED_PREFIX = "[Tool execution skipped — "
ERROR_EXECUTING_TOOL_PREFIX = "Error executing tool"
ORPHAN_RECOVERY_PREFIX = "[Orphan recovery: "


# Unknown tool name — str.format template; the compressor builds its matcher from it.
UNKNOWN_TOOL_NAME_TEMPLATE = "Tool '{name}' does not exist."

# Empty tool name — a model echoing tool-call syntax found in data (#47967).
EMPTY_TOOL_NAME_PREFIX = "Tool call rejected:"

# Valid sibling skipped because another call in the same response had invalid JSON.
SKIPPED_SIBLING_INVALID_JSON_PREFIX = "Skipped: other tool call in this response had invalid JSON."

# A call's own arguments were invalid JSON (recovery message once the retry budget is spent).
INVALID_JSON_ARGS_PREFIX = "Error: Invalid JSON arguments."

# Legacy shape: sibling skipped because another call used an invalid name. Emitted by
# pre-mixed-batch builds; kept only so the compressor can summarize old transcripts.
LEGACY_SKIPPED_SIBLING_INVALID_NAME_PREFIX = "Skipped: another tool call in this turn"
