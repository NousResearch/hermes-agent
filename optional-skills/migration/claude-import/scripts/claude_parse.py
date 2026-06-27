"""
claude_parse — Extract memory units from a Claude.ai data export.

Usage:
    python claude_parse.py <export_dir> [--output <path>]

Reads conversations.json (streaming) and memories.json from the export
directory, produces structured memory units suitable for import via
Hermes' built-in ``memory`` tool.

Output: JSON array of memory units to stdout (or --output file). Each unit:

.. code-block:: json

    {
      "content": "Conversation summary and key exchanges...",
      "source": "claude",
      "source_type": "conversation",
      "conversation_uuid": "550e8400-e29b-41d4-a716-446655440000",
      "conversation_name": "Project Planning",
      "timestamp": "2025-08-15T14:30:00Z",
      "updated_at": "2025-08-15T15:30:00Z"
    }

No external dependencies — uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Streaming JSON array reader (stdlib only)
# ---------------------------------------------------------------------------

def stream_json_array(filepath: str) -> Iterator[Dict[str, Any]]:
    """Yield top-level objects from a JSON array one at a time.

    Uses ``json.JSONDecoder.raw_decode`` with an accumulating buffer so
    that a 175 MB ``conversations.json`` never needs to be fully resident.
    Falls back to ``json.loads`` for files under 50 MB (faster path).
    """
    size = os.path.getsize(filepath)
    if size < 50 * 1024 * 1024:
        # Small enough to load in one shot — faster
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            yield from data
        return

    # Streaming path for large files
    decoder = json.JSONDecoder()
    buffer = ""
    with open(filepath, "r", encoding="utf-8") as f:
        # Consume up to the opening '['
        while True:
            ch = f.read(1)
            if not ch:
                return
            if ch == "[":
                break

        while True:
            ch = f.read(1)
            if not ch:
                break
            buffer += ch

            stripped = buffer.strip()
            if not stripped or stripped == ",":
                buffer = ""
                continue
            if stripped == "]":
                return

            try:
                obj, idx = decoder.raw_decode(buffer)
                yield obj
                # Keep any trailing content (e.g. "," or "]}") for next iter
                buffer = buffer[idx:].lstrip().lstrip(",").lstrip()
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Claude data model helpers
# ---------------------------------------------------------------------------

def _flatten_content(content: Any) -> str:
    """Extract readable text from Claude's rich content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return ""


def _extract_key_exchanges(messages: List[Dict]) -> str:
    """Summarise the first substantive exchange in a conversation.

    Keeps the first human message and the first assistant response to
    give context without bloating the memory unit.
    """
    exchanges: List[str] = []
    seen_uuid: set = set()

    for msg in messages:
        mid = msg.get("uuid", "")
        if mid in seen_uuid:
            continue
        seen_uuid.add(mid)

        sender = msg.get("sender", "")
        text = msg.get("text", "") or _flatten_content(msg.get("content", ""))
        if not text.strip():
            continue

        if sender == "human":
            # Truncate long human messages
            if len(text) > 500:
                text = text[:500] + "…"
            exchanges.append(f"Q: {text.strip()}")
        elif sender == "assistant" and exchanges:
            # First assistant response after a human message
            if len(text) > 800:
                text = text[:800] + "…"
            exchanges.append(f"A: {text.strip()}")
            break  # One Q&A pair is enough for context

    return "\n\n".join(exchanges)


def _make_memory_unit(
    conv: Dict[str, Any],
    *,
    source_type: str = "conversation",
    content_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a memory unit dict from a conversation object."""
    conv_name = conv.get("name", "").strip() or "(untitled)"
    conv_summary = conv.get("summary", "").strip()
    created = conv.get("created_at", "")
    updated = conv.get("updated_at", "")

    if content_override:
        content = content_override
    else:
        parts: List[str] = []
        if conv_summary:
            parts.append(f"Conversation: {conv_name}\nSummary: {conv_summary}")
        else:
            parts.append(f"Conversation: {conv_name}")

        messages = conv.get("chat_messages", [])
        key = _extract_key_exchanges(messages)
        if key:
            parts.append(key)

        content = "\n\n".join(parts)

    return {
        "content": content,
        "source": "claude",
        "source_type": source_type,
        "conversation_uuid": conv.get("uuid", ""),
        "conversation_name": conv_name,
        "timestamp": created,
        "updated_at": updated,
    }


# ---------------------------------------------------------------------------
# Memories.json parser
# ---------------------------------------------------------------------------

def parse_memories(memories_path: str) -> List[Dict[str, Any]]:
    """Parse Claude's memories.json into memory units.

    memories.json contains per-conversation and per-project memory texts
    that Claude maintains. These are higher-signal than the raw conversation
    data because they represent the user's curated knowledge.
    """
    if not os.path.isfile(memories_path):
        return []

    with open(memories_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    units: List[Dict[str, Any]] = []

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            # Per-conversation memories
            conv_memory = entry.get("conversations_memory")
            if conv_memory and isinstance(conv_memory, str) and conv_memory.strip():
                # These are freeform memory texts without conversation UUIDs
                # in the memory section itself — group them under a generic source
                units.append({
                    "content": conv_memory.strip(),
                    "source": "claude",
                    "source_type": "conversation_memory",
                    "conversation_uuid": entry.get("account_uuid", ""),
                    "conversation_name": "Claude Memory",
                    "timestamp": "",
                    "updated_at": "",
                })

            # Per-project memories
            project_memories = entry.get("project_memories")
            if project_memories and isinstance(project_memories, dict):
                for project_uuid, memory_text in project_memories.items():
                    if isinstance(memory_text, str) and memory_text.strip():
                        # Extract project name from first line if possible
                        first_line = memory_text.strip().split("\n")[0]
                        proj_name = first_line.replace("**", "").replace("###", "").strip()
                        if len(proj_name) > 80:
                            proj_name = f"project_{project_uuid[:8]}"
                        units.append({
                            "content": memory_text.strip(),
                            "source": "claude",
                            "source_type": "project_memory",
                            "conversation_uuid": project_uuid,
                            "conversation_name": proj_name,
                            "timestamp": "",
                            "updated_at": "",
                        })

    return units


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_export(
    export_dir: str,
    *,
    include_memories: bool = True,
) -> List[Dict[str, Any]]:
    """Parse a Claude data export and return memory units.

    Args:
        export_dir: Path to the root of the Claude export directory.
        include_memories: Also parse ``memories.json`` if present.

    Returns:
        List of memory unit dicts.
    """
    if not os.path.isdir(export_dir):
        print(f"Error: directory not found: {export_dir}", file=sys.stderr)
        sys.exit(1)

    convs_path = os.path.join(export_dir, "conversations.json")
    if not os.path.isfile(convs_path):
        # Maybe it's inside a batch subdirectory
        batch_dirs = [
            d for d in os.listdir(export_dir)
            if d.startswith("data-") and os.path.isdir(os.path.join(export_dir, d))
        ]
        if batch_dirs:
            convs_path = os.path.join(export_dir, batch_dirs[0], "conversations.json")

    if not os.path.isfile(convs_path):
        print(f"Error: conversations.json not found in {export_dir}", file=sys.stderr)
        sys.exit(1)

    units: List[Dict[str, Any]] = []
    conv_count = 0

    for conv in stream_json_array(convs_path):
        conv_count += 1
        unit = _make_memory_unit(conv)
        units.append(unit)

    if include_memories:
        # Try the root directory first, then the batch subdirectory
        mem_path = os.path.join(export_dir, "memories.json")
        if not os.path.isfile(mem_path):
            batch_dirs = [
                d for d in os.listdir(export_dir)
                if d.startswith("data-") and os.path.isdir(os.path.join(export_dir, d))
            ]
            if batch_dirs:
                mem_path = os.path.join(export_dir, batch_dirs[0], "memories.json")

        if os.path.isfile(mem_path):
            mem_units = parse_memories(mem_path)
            units.extend(mem_units)

    return units


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract memory units from a Claude.ai data export."
    )
    parser.add_argument(
        "export_dir",
        help="Path to the Claude data export directory",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--no-memories",
        action="store_true",
        help="Skip importing project/conversation memories from memories.json",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.export_dir):
        print(f"Error: directory not found: {args.export_dir}", file=sys.stderr)
        sys.exit(1)

    units = parse_export(args.export_dir, include_memories=not args.no_memories)

    output = json.dumps(units, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Wrote {len(units)} memory units to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
