"""Obsidian memory provider for Hermes Agent.

Gives the agent persistent recall from your Obsidian vault using hybrid
BM25 + semantic search with graph-augmented retrieval (Graph RAG).

Features
--------
- Hybrid search: BM25 full-text + cosine semantic similarity, fused with RRF
- Graph RAG: backlink traversal for richer context (follows [[wikilinks]])
- Live index: watchdog-based incremental updates as you edit notes
- Session notes: each conversation auto-saved to AI/Hermes/Sessions/
- Daily note sync: activity logged to your daily note
- Claude mirror: mirrors built-in memory writes to AI/Hermes/Memory/
- Pre-compress extraction: key facts saved before context is compressed

Tools exposed to the agent
--------------------------
  obsidian_search        — semantic + full-text search across the vault
  obsidian_read          — read a specific note by title or path
  obsidian_write         — create or update a note
  obsidian_list          — list notes in a folder
  obsidian_graph_context — get notes linked to/from a given note

Configuration
-------------
  OBSIDIAN_VAULT_PATH   env var or config key (default: ~/Documents/Obsidian Vault)
  OBSIDIAN_DAILY_FOLDER daily notes folder (default: Daily Notes)
  OBSIDIAN_AI_FOLDER    AI output folder (default: AI)
  OBSIDIAN_RECALL_MODE  "hybrid" | "bm25" | "tools" (default: hybrid)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_VAULT = "~/Documents/Obsidian Vault"
_DEFAULT_DAILY = "Daily Notes"
_DEFAULT_AI_FOLDER = "AI"
_CRON_CONTEXTS = {"cron", "flush", "subagent"}


def _vault_path() -> Optional[Path]:
    raw = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip() or _DEFAULT_VAULT
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


# ---------------------------------------------------------------------------
# ObsidianMemoryProvider
# ---------------------------------------------------------------------------

class ObsidianMemoryProvider(MemoryProvider):
    """Full Obsidian vault memory provider with hybrid search and Graph RAG."""

    @property
    def name(self) -> str:
        return "obsidian"

    def is_available(self) -> bool:
        return _vault_path() is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", str(Path.home() / ".hermes"))
        self._platform = kwargs.get("platform", "cli")
        self._agent_context = kwargs.get("agent_context", "primary")
        self._is_primary = self._agent_context not in _CRON_CONTEXTS
        self._turn_count = 0
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._session_turns: List[Dict[str, str]] = []

        vault = _vault_path()
        if vault is None:
            logger.warning("obsidian-memory: vault not found, provider inactive")
            return

        self._vault = vault
        self._daily_folder = os.environ.get("OBSIDIAN_DAILY_FOLDER", _DEFAULT_DAILY)
        self._ai_folder = os.environ.get("OBSIDIAN_AI_FOLDER", _DEFAULT_AI_FOLDER)
        self._recall_mode = os.environ.get("OBSIDIAN_RECALL_MODE", "hybrid")

        from plugins.memory.obsidian.vault import VaultReader, VaultWriter
        from plugins.memory.obsidian.index import VaultIndex
        from plugins.memory.obsidian.watcher import make_watcher

        self._reader = VaultReader(vault)
        self._writer = VaultWriter(vault)
        self._index = VaultIndex(vault)
        self._watcher = None

        # Build index in background so agent startup isn't blocked
        def _build():
            try:
                self._index.build()
                self._watcher = make_watcher(vault, self._index)
                self._watcher.start()
            except Exception as exc:
                logger.warning("obsidian-memory: index build failed: %s", exc)

        t = threading.Thread(target=_build, daemon=True, name="obsidian-index-build")
        t.start()
        # Give it 0.3s to finish for small vaults; continue either way
        t.join(timeout=0.3)

        logger.info("obsidian-memory: initialised (vault=%s, mode=%s)", vault, self._recall_mode)

    def shutdown(self) -> None:
        if self._watcher:
            try:
                self._watcher.stop()
                self._watcher.join(timeout=2)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        vault = getattr(self, "_vault", None)
        if vault is None:
            return ""
        hub_notes = ""
        try:
            hubs = self._index._graph.hub_notes(top_n=5)
            if hubs:
                hub_notes = "Most-linked notes: " + ", ".join(f"[[{s}]]" for s, _ in hubs)
        except Exception:
            pass
        return (
            f"You have access to the user's Obsidian vault at `{self._vault}`.\n"
            f"Use the obsidian_search, obsidian_read, obsidian_write, and obsidian_graph_context "
            f"tools to read existing notes, create new ones, and retrieve context.\n"
            + (f"{hub_notes}\n" if hub_notes else "")
        )

    # ------------------------------------------------------------------
    # Prefetch (background recall before each turn)
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not getattr(self, "_index", None) or not self._index._built:
            return

        def _run():
            try:
                snippets = self._index.search(query, top_k=5)
                if snippets:
                    formatted = "\n\n".join(snippets)
                    with self._prefetch_lock:
                        self._prefetch_result = formatted
            except Exception as exc:
                logger.debug("obsidian-memory: prefetch failed: %s", exc)

        t = threading.Thread(target=_run, daemon=True, name="obsidian-prefetch")
        t.start()

    # ------------------------------------------------------------------
    # Sync (persist turn to vault)
    # ------------------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not self._is_primary or not getattr(self, "_writer", None):
            return
        self._turn_count += 1
        self._session_turns.append({"user": user_content, "assistant": assistant_content})

        # Append a brief log to today's daily note every 5 turns or if first turn
        if self._turn_count == 1 or self._turn_count % 5 == 0:
            try:
                snippet = user_content[:200].replace("\n", " ")
                self._writer.append_to_daily(
                    f"> {snippet}…" if len(user_content) > 200 else f"> {user_content}",
                    section=f"Hermes ({self._platform})",
                    folder=self._daily_folder,
                )
            except Exception as exc:
                logger.debug("obsidian-memory: daily note sync failed: %s", exc)

    # ------------------------------------------------------------------
    # on_session_end: synthesise session into a note
    # ------------------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._is_primary or not getattr(self, "_writer", None):
            return
        if not self._session_turns:
            return
        try:
            title = _derive_session_title(self._session_turns)
            body = _format_session_note(self._session_turns, self._session_id, self._platform)
            self._writer.write_session_note(
                self._session_id,
                title=title,
                content=body,
                agent="hermes",
                folder=f"{self._ai_folder}/Hermes/Sessions",
            )
            # After writing, update the index so future searches find it
            if self._index._built:
                session_path = self._vault / f"{self._ai_folder}/Hermes/Sessions"
                for p in sorted(session_path.glob(f"*{self._session_id[:8]}*")):
                    self._index.update_note(p)
        except Exception as exc:
            logger.warning("obsidian-memory: session note write failed: %s", exc)

    # ------------------------------------------------------------------
    # on_pre_compress: extract insights before context is discarded
    # ------------------------------------------------------------------

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._is_primary or not getattr(self, "_writer", None):
            return ""
        try:
            key_turns = [
                m for m in messages
                if m.get("role") in ("user", "assistant")
                and len(str(m.get("content", ""))) > 50
            ][-6:]  # last 6 substantive turns
            if not key_turns:
                return ""
            summary_lines = []
            for m in key_turns:
                role = m["role"].capitalize()
                text = str(m.get("content", ""))[:300]
                summary_lines.append(f"**{role}**: {text}")
            note_content = "\n\n".join(summary_lines)
            today = datetime.now().strftime("%Y-%m-%d")
            ts = datetime.now().strftime("%H%M")
            rel = f"{self._ai_folder}/Hermes/Sessions/compress-{today}-{ts}.md"
            self._writer.write(rel, note_content, frontmatter={
                "type": "compression-extract",
                "session_id": self._session_id,
                "date": today,
                "tags": ["ai/hermes", "compression"],
            })
            return f"(Key context saved to Obsidian: {rel})"
        except Exception as exc:
            logger.debug("obsidian-memory: pre-compress write failed: %s", exc)
        return ""

    # ------------------------------------------------------------------
    # on_memory_write: mirror built-in memory tool writes to vault
    # ------------------------------------------------------------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not getattr(self, "_writer", None):
            return
        try:
            self._writer.mirror_memory_entry(action, target, content, agent="hermes")
        except Exception as exc:
            logger.debug("obsidian-memory: memory mirror failed: %s", exc)

    # ------------------------------------------------------------------
    # on_session_switch
    # ------------------------------------------------------------------

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        if reset:
            self._session_turns = []
            self._turn_count = 0
        self._session_id = new_session_id

    # ------------------------------------------------------------------
    # on_delegation
    # ------------------------------------------------------------------

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        if not getattr(self, "_writer", None):
            return
        try:
            ts = datetime.now().strftime("%H:%M")
            entry = f"**Delegated task** ({ts})\n> {task[:300]}\n\n**Result:** {result[:500]}"
            self._writer.append_to_daily(entry, section="Hermes Delegations", folder=self._daily_folder)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # backup_paths
    # ------------------------------------------------------------------

    def backup_paths(self) -> List[str]:
        vault = _vault_path()
        return [str(vault)] if vault else []

    # ------------------------------------------------------------------
    # Tool schemas
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "obsidian_search",
                "description": (
                    "Search the user's Obsidian vault using hybrid semantic + keyword search. "
                    "Returns relevant note excerpts ranked by relevance. "
                    "Use for finding existing notes, past conversations, research, or any stored knowledge."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query — natural language or keywords",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results to return (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "obsidian_read",
                "description": "Read the full content of an Obsidian note by title or relative path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "Note title or relative path (e.g. 'My Note' or 'AI/Hermes/Sessions/2025-01-01-abc.md')",
                        },
                    },
                    "required": ["note"],
                },
            },
            {
                "name": "obsidian_write",
                "description": (
                    "Create or update an Obsidian note. If the note exists, overwrites it unless append=true. "
                    "Path is relative to the vault root."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path in vault, e.g. 'Projects/MyProject.md'",
                        },
                        "content": {
                            "type": "string",
                            "description": "Markdown content to write",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to add to frontmatter",
                        },
                        "append": {
                            "type": "boolean",
                            "description": "If true, append to existing note rather than overwriting",
                            "default": False,
                        },
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "obsidian_list",
                "description": "List notes in a vault folder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "folder": {
                            "type": "string",
                            "description": "Folder path relative to vault root (e.g. 'Projects'). Leave empty for root.",
                            "default": "",
                        },
                    },
                },
            },
            {
                "name": "obsidian_graph_context",
                "description": (
                    "Get notes linked to/from a given note via wikilinks. "
                    "Useful for exploring related concepts, finding context around a topic."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Note title to explore links from/to",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Hop depth (1=direct links, 2=links of links). Default 1.",
                            "default": 1,
                        },
                    },
                    "required": ["title"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool call handler
    # ------------------------------------------------------------------

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "obsidian_search":
                return self._tool_search(args)
            if tool_name == "obsidian_read":
                return self._tool_read(args)
            if tool_name == "obsidian_write":
                return self._tool_write(args)
            if tool_name == "obsidian_list":
                return self._tool_list(args)
            if tool_name == "obsidian_graph_context":
                return self._tool_graph(args)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as exc:
            logger.warning("obsidian-memory: tool %s failed: %s", tool_name, exc)
            return json.dumps({"error": str(exc)})

    def _tool_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        top_k = min(int(args.get("top_k", 5)), 10)
        if not self._index._built:
            return json.dumps({"results": [], "note": "Index still building, try again shortly"})
        results = self._index.search(query, top_k=top_k)
        return json.dumps({"results": results, "count": len(results)})

    def _tool_read(self, args: Dict[str, Any]) -> str:
        note_ref = args.get("note", "")
        note = self._reader.read(note_ref)
        if note is None:
            return json.dumps({"error": f"Note not found: {note_ref}"})
        return json.dumps({
            "title": note.title,
            "path": str(note.path.relative_to(self._vault)),
            "tags": note.tags,
            "content": note.raw[:8000],
            "links": note.links[:20],
        })

    def _tool_write(self, args: Dict[str, Any]) -> str:
        path = args.get("path", "")
        content = args.get("content", "")
        tags = args.get("tags", [])
        append = bool(args.get("append", False))
        fm = {"tags": tags} if tags else None
        if append:
            written = self._writer.append(path, content)
        else:
            written = self._writer.write(path, content, frontmatter=fm)
        if self._index._built:
            self._index.update_note(written)
        rel = str(written.relative_to(self._vault))
        return json.dumps({"written": rel, "success": True})

    def _tool_list(self, args: Dict[str, Any]) -> str:
        folder = args.get("folder", "")
        if folder:
            paths = self._reader.list_folder(folder)
        else:
            paths = self._reader.list_all()
        rel_paths = [str(p.relative_to(self._vault)) for p in paths[:100]]
        return json.dumps({"notes": rel_paths, "count": len(rel_paths)})

    def _tool_graph(self, args: Dict[str, Any]) -> str:
        title = args.get("title", "")
        depth = min(int(args.get("depth", 1)), 3)
        if not self._index._built:
            return json.dumps({"context": [], "note": "Index still building"})
        results = self._index.graph_context(title, depth=depth)
        return json.dumps({"context": results, "count": len(results)})

    # ------------------------------------------------------------------
    # Config schema
    # ------------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "vault_path",
                "description": "Path to your Obsidian vault directory",
                "required": False,
                "default": _DEFAULT_VAULT,
                "env_var": "OBSIDIAN_VAULT_PATH",
            },
            {
                "key": "recall_mode",
                "description": "How to surface vault context: hybrid (auto-inject + tools), bm25 (keyword only), tools (on-demand only)",
                "required": False,
                "default": "hybrid",
                "choices": ["hybrid", "bm25", "tools"],
                "env_var": "OBSIDIAN_RECALL_MODE",
            },
            {
                "key": "daily_folder",
                "description": "Folder for daily notes",
                "required": False,
                "default": _DEFAULT_DAILY,
                "env_var": "OBSIDIAN_DAILY_FOLDER",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        cfg_path = Path(hermes_home) / "obsidian.json"
        existing: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text())
            except Exception:
                pass
        existing.update(values)
        cfg_path.write_text(json.dumps(existing, indent=2))


# ---------------------------------------------------------------------------
# Session note helpers
# ---------------------------------------------------------------------------

def _derive_session_title(turns: List[Dict[str, str]]) -> str:
    first_user = turns[0].get("user", "") if turns else ""
    snippet = first_user[:60].strip().replace("\n", " ")
    return snippet or "Untitled session"


def _format_session_note(
    turns: List[Dict[str, str]],
    session_id: str,
    platform: str,
) -> str:
    lines = [f"Session `{session_id}` on {platform}\n"]
    for i, turn in enumerate(turns, 1):
        u = turn.get("user", "")[:500]
        a = turn.get("assistant", "")[:500]
        lines.append(f"### Turn {i}\n**User:** {u}\n\n**Hermes:** {a}\n")
    return "\n".join(lines)
