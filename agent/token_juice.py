"""TokenJuice — tool output compression safety gates.

Three safety valves adapted from OpenHuman's tokenjuice Rust crate:
  1. PASSTHROUGH_BYTES — outputs shorter than this are kept verbatim.
  2. PASSTHROUGH_COMMANDS — file-inspection tools (read_file/cat/tail/head/bat) never compressed.
  3. MIN_COMPRESSION_RATIO — LLM-summarised text that gained < 5% savings is
     discarded in favour of the original (summarisation cost without benefit).

These gates run BEFORE the LLM summariser so cheap decisions don't incur API
calls, and the ratio gate runs AFTER to detect useless summaries.
"""

from typing import Any, Dict, FrozenSet, List

# ── Constants ──────────────────────────────────────────────────────────────

# Outputs shorter than this (chars) are always kept verbatim.
TOKENJUICE_PASSTHROUGH_BYTES: int = 240

# Tools/commands whose output is never compressed — file-content inspection.
TOKENJUICE_PASSTHROUGH_COMMANDS: FrozenSet[str] = frozenset({
    "read_file", "cat", "tail", "head", "bat", "batcat",
})

# If the summarised text is >= this fraction of the original length,
# the summary is discarded and the original is kept.
TOKENJUICE_MIN_COMPRESSION_RATIO: float = 0.95


# ── Gate 1 & 2 ─────────────────────────────────────────────────────────────

def _content_char_count(message: Dict[str, Any]) -> int:
    """Count the effective character length of a message's content."""
    content = message.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
    return len(str(content))


def _extract_tool_command_args(arguments_json: str) -> str:
    """Extract the actual command argv[0] from terminal tool arguments JSON.

    The terminal tool stores the shell command in an ``arguments`` field,
    like ``{"command": "cat ./file.txt"}``.  We tokenise and return argv[0].
    Returns empty string on parse failure or non-terminal tool.
    """
    if not arguments_json:
        return ""
    import json as _json
    try:
        args = _json.loads(arguments_json)
    except Exception:
        return ""
    command = args.get("command", "")
    if not command:
        return ""
    # Simple shell-like tokenisation — grab the first word
    import shlex
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ""
    if not tokens:
        return ""
    return tokens[0]


def should_passthrough(msg: Dict[str, Any], tool_name: str = "", tool_args: str = "") -> bool:
    """Return True if this tool_result message should NOT be compressed.

    Applies two gates:
      Gate 1: content length < TOKENJUICE_PASSTHROUGH_BYTES
      Gate 2: the tool or executed command is in TOKENJUICE_PASSTHROUGH_COMMANDS
              (read_file/cat/tail/head/bat/batcat — file-inspection tools)

    Args:
        msg: The tool_result message dict.
        tool_name: The function name of the tool call (e.g. \"terminal\").
        tool_args: JSON string of the tool call arguments.
    """
    if msg.get("role") != "tool":
        return False

    # Gate 1 — tiny output
    if _content_char_count(msg) < TOKENJUICE_PASSTHROUGH_BYTES:
        return True

    # Gate 2 — never-compress file inspection tools/commands
    if tool_name in TOKENJUICE_PASSTHROUGH_COMMANDS:
        return True

    if tool_name == "terminal":
        cmd = _extract_tool_command_args(tool_args)
        if cmd and cmd in TOKENJUICE_PASSTHROUGH_COMMANDS:
            return True

    return False


def classify_messages(
    messages: List[Dict[str, Any]],
    call_id_to_tool: Dict[str, tuple] | None = None,
) -> tuple:
    """Split messages into (protected, compressible) based on safety gates.

    Returns (protected_messages: list, compressible_messages: list).
    Protected messages are excluded from the compression pipeline entirely.
    """
    protected: List[Dict[str, Any]] = []
    compressible: List[Dict[str, Any]] = []

    for msg in messages:
        tool_name = ""
        tool_args = ""
        if call_id_to_tool is not None:
            call_id = msg.get("tool_call_id", "")
            info = call_id_to_tool.get(call_id)
            if info:
                tool_name, tool_args = info
        if should_passthrough(msg, tool_name=tool_name, tool_args=tool_args):
            protected.append(msg)
        else:
            compressible.append(msg)

    return protected, compressible


# ── Gate 3 ─────────────────────────────────────────────────────────────────

# Gate 3 absolute minimum — reject summaries shorter than this many characters
# regardless of ratio.  Prevents near-empty / one-word summaries from
# replacing the original tool output.
_TOKENJUICE_MIN_COMPRESSED_CHARS: int = 50

# Gate 3 relative minimum — compressed text must be at least this fraction
# of the original length.  Prevents summaries that discard > 97% of content,
# which typically means the LLM returned garbage rather than a real summary.
# Set low enough (3%) that legitimate dense summaries still pass.
_TOKENJUICE_MIN_COMPRESSED_RATIO: float = 0.03

# Only apply the "too short to be safe" checks to sizeable compression windows.
# Small conversations often have valid one-sentence summaries; rejecting them
# causes the compressor to keep the full context even though the summary is
# useful and intentionally brief.
_TOKENJUICE_SHORT_SUMMARY_GUARD_ORIGINAL_MIN: int = 500


def compression_safe_to_apply(original_len: int, compressed_len: int) -> bool:
    """Return True if the compressed text is meaningfully shorter.

    Gate 3: three-stage safety check —
      1. Absolute floor: compressed_len >= 50 chars.
      2. Relative floor: compressed_len >= 3% of original_len.
      3. Ratio gate: compressed_len < 95% of original_len
         (i.e. the compression actually saved ≥5%).

    Rejects summaries that are pathologically short (empty, one-word, etc.)
    which would otherwise pass the ratio check by being tiny.
    """
    if original_len <= 0:
        return False
    if original_len < _TOKENJUICE_SHORT_SUMMARY_GUARD_ORIGINAL_MIN:
        return True
    if original_len >= _TOKENJUICE_SHORT_SUMMARY_GUARD_ORIGINAL_MIN:
        if compressed_len < _TOKENJUICE_MIN_COMPRESSED_CHARS:
            return False
        if compressed_len < original_len * _TOKENJUICE_MIN_COMPRESSED_RATIO:
            return False
    return (compressed_len / original_len) < TOKENJUICE_MIN_COMPRESSION_RATIO
