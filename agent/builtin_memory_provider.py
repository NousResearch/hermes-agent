"""Built-in file-based memory provider for Hermes.

Implements MemoryProvider to persist agent memory as plain markdown files
(MEMORY.md, USER.md) inside the active HERMES_HOME/.hermes/ directory.

This provider is ALWAYS active and cannot be removed.  It serves as the
primary long-term memory store and the fallback when no external provider
(Honcho, Mem0, etc.) is configured.

Lifecycle (called by MemoryManager):
  initialize()         — resolve memory file paths, run warm-up
  system_prompt_block() — static system-prompt fragment
  prefetch(query)      — background recall before each turn
  sync_turn(user, asst) — persist a completed turn
  get_tool_schemas()   — tool schemas (memory_read, memory_write)
  handle_tool_call()   — dispatch memory tool calls
  shutdown()           — no-op (files are already on disk)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MEMORY_FNAME = "MEMORY.md"
USER_FNAME = "USER.md"
_MEMORY_VERSION = "1.0"


def _memory_dir(hermes_home: str) -> Path:
    return Path(hermes_home) / ".hermes"


def _memory_path(hermes_home: str, filename: str = MEMORY_FNAME) -> Path:
    d = _memory_dir(hermes_home)
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^#{1,3}\s+(.+?)\s*$",
    re.MULTILINE,
)

_ENTRY_RE = re.compile(
    r"^(#{1,3}\s+[^\n]+\n)(.+?)(?=\n#{1,3}\s|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _parse_sections(content: str) -> Dict[str, str]:
    """Split a memory file into sections keyed by heading text."""
    sections: Dict[str, str] = {}
    parts = _ENTRY_RE.split(content)
    # parts[0] may be leading text before first heading
    for heading, body in zip(parts[1::2], parts[2::2]):
        heading = heading.strip()
        body = body.strip()
        if heading:
            sections[heading] = body
    return sections


def _section_name_to_id(name: str) -> str:
    """Convert a section name to a URL-safe id for linking."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_MEMORY_READ_SCHEMA = {
    "name": "memory_read",
    "description": (
        "Read the current agent memory (MEMORY.md) or user profile (USER.md). "
        "Use 'target=memory' (default) for the agent's own memory, or 'target=user' "
        "for the user's profile.  Returns the full file content grouped by sections. "
        "Omit 'query' to return all sections."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "default": "memory",
                "description": "Which memory file to read.",
            },
            "query": {
                "type": "string",
                "default": "",
                "description": (
                    "Optional search term. If provided, only sections whose heading "
                    "or content contains this substring (case-insensitive) are returned."
                ),
            },
        },
        "required": [],
    },
}

_MEMORY_WRITE_SCHEMA = {
    "name": "memory_write",
    "description": (
        "Add, replace, or remove a named section in the agent's memory (MEMORY.md) "
        "or user profile (USER.md).  Sections are created automatically on first add. "
        "Use action='add' to append or create, action='replace' to overwrite an "
        "existing section (by exact heading match), or action='remove' to delete "
        "a section by heading.  Returns a brief confirmation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The write action to perform.",
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "default": "memory",
                "description": "Which memory file to modify.",
            },
            "heading": {
                "type": "string",
                "description": (
                    "Section heading. Use '## Section Name' format. "
                    "For 'remove', provide the exact heading to delete."
                ),
            },
            "content": {
                "type": "string",
                "default": "",
                "description": "Section body content. Required for 'add' and 'replace'.",
            },
        },
        "required": ["action", "heading"],
    },
}


# ---------------------------------------------------------------------------
# Memory provider
# ---------------------------------------------------------------------------


class BuiltinMemoryProvider(MemoryProvider):
    """Built-in markdown-file memory provider.

    Manages MEMORY.md and USER.md in ``~/.hermes/.hermes/``.
    Always available; cannot be removed via config.
    """

    name: str = "builtin"

    def __init__(self) -> None:
        self._hermes_home: Optional[str] = None
        self._session_id: Optional[str] = None
        self._initialized: bool = False
        self._turn_count: int = 0

    # -- MemoryProvider required methods -------------------------------------

    def is_available(self) -> bool:
        """Builtin provider is always available."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._hermes_home = str(kwargs.get("hermes_home", ""))
        if not self._hermes_home:
            try:
                from hermes_constants import get_hermes_home

                self._hermes_home = str(get_hermes_home())
            except Exception:
                pass
        if not self._hermes_home:
            raise ValueError(
                "BuiltinMemoryProvider requires hermes_home kwarg or HERMES_HOME env."
            )
        self._session_id = session_id
        self._turn_count = 0
        self._ensure_memory_files_exist()
        self._initialized = True
        logger.info(
            "BuiltinMemoryProvider initialized for session '%s' (home=%s)",
            session_id,
            self._hermes_home,
        )

    def system_prompt_block(self) -> str:
        if not self._initialized:
            return ""
        home = self._hermes_home or ""
        sections = self._read_sections(MEMORY_FNAME)
        if not sections:
            return (
                "[Built-in memory is empty. Use memory_write to add entries.]\n"
                f"[Memory location: {home}/.hermes/{MEMORY_FNAME}]"
            )
        lines = [
            "[Built-in memory — use memory_read(target=\"memory\") to view. Sections:]",
        ]
        for heading in sections:
            safe_id = _section_name_to_id(heading)
            lines.append(f"  ## {heading} (id={safe_id})")
        lines.append(f"[Memory file: {home}/.hermes/{MEMORY_FNAME}]")
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return matching memory sections for the query."""
        if not self._initialized:
            return ""
        sections = self._read_sections(MEMORY_FNAME)
        if not query.strip():
            # No query → return all sections (truncated)
            combined: List[str] = []
            for heading, body in sections.items():
                excerpt = body[:300] + ("..." if len(body) > 300 else "")
                combined.append(f"## {heading}\n{excerpt}")
            return "\n\n".join(combined) if combined else ""

        query_lower = query.lower()
        matched: List[str] = []
        for heading, body in sections.items():
            if query_lower in heading.lower() or query_lower in body.lower():
                matched.append(f"## {heading}\n{body}")
        return "\n\n".join(matched)

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Append a turn summary to MEMORY.md under an auto-named section."""
        if not self._initialized:
            return
        if not user_content.strip() and not assistant_content.strip():
            return
        self._turn_count += 1
        heading = f"Turn {self._turn_count}"
        user_snippet = user_content[:500].strip() if user_content else "(no user)"
        asst_snippet = assistant_content[:500].strip() if assistant_content else ""
        content = (
            f"**User:** {user_snippet}\n\n"
            f"**Assistant:** {asst_snippet}"
        )
        self._append_entry(MEMORY_FNAME, heading, content)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [_MEMORY_READ_SCHEMA, _MEMORY_WRITE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        import json

        if tool_name == "memory_read":
            return self._tool_memory_read(args)
        elif tool_name == "memory_write":
            return self._tool_memory_write(args)
        raise NotImplementedError(f"BuiltinMemoryProvider does not handle '{tool_name}'")

    def shutdown(self) -> None:
        """No-op — files are always on disk."""
        self._initialized = False

    # -- Optional MemoryProvider hooks --------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = turn_number

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract insights from messages before they are compressed."""
        if not self._initialized:
            return ""
        # Summarize the conversation trajectory into a single memory entry
        lines: List[str] = []
        for msg in messages[-10:]:  # last 10 messages only
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            snippet = str(content)[:200].strip()
            if snippet:
                lines.append(f"[{role}]: {snippet}")
        if not lines:
            return ""
        body = "\n".join(lines)
        return (
            "[Pre-compression summary — consider adding to memory:]\n"
            f"{body}\n"
            "[Use memory_write to persist important facts]"
        )

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror external provider writes into built-in memory."""
        logger.debug(
            "on_memory_write called (action=%s target=%s) — builtin ignores external writes",
            action,
            target,
        )
        # Builtin is the source of writes; no need to mirror from external providers.

    # -- Internal helpers ----------------------------------------------------

    def _ensure_memory_files_exist(self) -> None:
        if not self._hermes_home:
            return
        for fname in (MEMORY_FNAME, USER_FNAME):
            p = _memory_path(self._hermes_home, fname)
            if not p.exists():
                p.write_text(
                    f"# {fname.replace('.md', '')}\n\n"
                    f"[Auto-created by Hermes v{_MEMORY_VERSION} on {datetime.now().isoformat()}]",
                    encoding="utf-8",
                )

    def _read_sections(self, filename: str) -> Dict[str, str]:
        if not self._hermes_home:
            return {}
        path = _memory_path(self._hermes_home, filename)
        if not path.exists():
            return {}
        try:
            content = path.read_text(encoding="utf-8")
            return _parse_sections(content)
        except Exception as e:
            logger.warning("Failed to read memory file '%s': %s", path, e)
            return {}

    def _write_sections(
        self, filename: str, sections: Dict[str, str]
    ) -> None:
        if not self._hermes_home:
            return
        path = _memory_path(self._hermes_home, filename)
        # Preserve any leading text (before first heading) as-is
        try:
            existing = ""
            if path.exists():
                existing = path.read_text(encoding="utf-8")
        except Exception:
            existing = ""

        # Check for leading non-heading text
        lines = existing.splitlines()
        lead_lines: List[str] = []
        body_lines: List[str] = []
        in_body = False
        for line in lines:
            if not in_body and not line.strip().startswith("#"):
                lead_lines.append(line)
            else:
                in_body = True
                body_lines.append(line)

        lead = "\n".join(lead_lines).strip()
        parts: List[str] = []
        if lead:
            parts.append(lead)
        for heading, body in sections.items():
            parts.append(f"## {heading}")
            parts.append(body)
        new_content = "\n\n".join(parts) + "\n"
        path.write_text(new_content, encoding="utf-8")

    def _append_entry(
        self, filename: str, heading: str, content: str
    ) -> None:
        sections = self._read_sections(filename)
        sections[heading] = content
        self._write_sections(filename, sections)

    def _remove_section(self, filename: str, heading: str) -> bool:
        sections = self._read_sections(filename)
        # Try exact match first
        if heading in sections:
            del sections[heading]
            self._write_sections(filename, sections)
            return True
        # Try case-insensitive match
        heading_lower = heading.lower()
        for key in list(sections.keys()):
            if key.lower() == heading_lower:
                del sections[key]
                self._write_sections(filename, sections)
                return True
        return False

    # -- Tool handlers -------------------------------------------------------

    def _tool_memory_read(self, args: Dict[str, Any]) -> str:
        import json

        target = args.get("target", "memory")
        query = args.get("query", "")
        filename = USER_FNAME if target == "user" else MEMORY_FNAME
        sections = self._read_sections(filename)

        if not query:
            if not sections:
                return json.dumps({"found": False, "content": "", "sections": []})
            return json.dumps(
                {
                    "found": True,
                    "content": "\n\n".join(f"## {h}\n{b}" for h, b in sections.items()),
                    "sections": list(sections.keys()),
                }
            )

        query_lower = query.lower()
        matched: Dict[str, str] = {}
        for heading, body in sections.items():
            if query_lower in heading.lower() or query_lower in body.lower():
                matched[heading] = body
        if not matched:
            return json.dumps({"found": False, "query": query, "sections": []})
        return json.dumps(
            {
                "found": True,
                "query": query,
                "matched_sections": len(matched),
                "content": "\n\n".join(f"## {h}\n{b}" for h, b in matched.items()),
            }
        )

    def _tool_memory_write(self, args: Dict[str, Any]) -> str:
        import json

        action = args.get("action", "add")
        target = args.get("target", "memory")
        heading = args.get("heading", "")
        content = args.get("content", "")
        filename = USER_FNAME if target == "user" else MEMORY_FNAME

        if not heading:
            return json.dumps({"success": False, "error": "heading is required"})
        if action in ("add", "replace") and not content:
            return json.dumps({"success": False, "error": "content is required for add/replace"})

        if action == "remove":
            ok = self._remove_section(filename, heading)
            return json.dumps({"success": ok, "action": "remove", "heading": heading})

        sections = self._read_sections(filename)
        sections[heading] = content
        self._write_sections(filename, sections)

        # Notify external providers (MemoryManager handles the broadcast)
        # We fire-and-forget here; MemoryManager.on_memory_write will call
        # any registered external providers.
        return json.dumps(
            {
                "success": True,
                "action": action,
                "heading": heading,
                "target": target,
            }
        )
