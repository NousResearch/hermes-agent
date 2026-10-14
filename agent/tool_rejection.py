"""Pre-execution rejection messages for tool calls — one source of truth.

The dispatch loop's rejection shapes (invalid tool name, invalid JSON arguments)
are stated in the tool-result content the model sees, and the context compressor
reads them to summarize the call as "not run" — otherwise compaction renders a
rejected call as a completed action and the post-compaction agent reports its
effect as done.

Producers (``agent.conversation_loop``, ``agent.turn_tool_validation``) build
messages from these constants and wrap them with :func:`rejected_result` into a
``{"status": "rejected", "error": …}`` payload, which the compressor reads
structurally (the same way it reads ``status: blocked`` / ``pending_approval``).
The compressor's prose matcher exists only for transcripts recorded before the
structured status existed; both sides build from these constants, so a rewording
cannot silently decouple them (each side's tests alone would stay green). This
module must stay import-light: it is imported by the compressor, which the loop
imports.
"""

import json

# Structured marker: rejections carry this ``status`` so the compressor never has
# to parse prose to know a call did not run.
REJECTED_STATUS = "rejected"


def rejected_result(message: str) -> str:
    """Wrap rejection prose into the structured payload the compressor reads first.

    Same shape as ``tools.registry.tool_error`` (a JSON body with an ``error``
    string), plus the ``status`` field the refused-result summarizer keys on.
    """
    return json.dumps({"status": REJECTED_STATUS, "error": message}, ensure_ascii=False)


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
