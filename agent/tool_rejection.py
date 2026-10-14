"""Pre-execution rejection messages for tool calls — one source of truth.

The dispatch loop's rejection shapes (invalid tool name, invalid JSON arguments)
are stated in the tool-result content the model sees, and the context compressor
matches those same messages to summarize them as "not run" — otherwise compaction
renders a rejected call as a completed action and the post-compaction agent
reports its effect as done.

Producers (``agent.conversation_loop``, ``agent.turn_tool_validation``) build
messages from these constants and the compressor matches on them, so a rewording
cannot silently decouple the two sides (each side's tests alone would stay
green). This module must stay import-light: it is imported by the compressor,
which the loop imports.
"""

# Unknown tool name — f-string template; the compressor builds its matcher from it.
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
