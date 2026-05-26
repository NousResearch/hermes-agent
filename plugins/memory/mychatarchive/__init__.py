"""MyChatArchive memory plugin -- MemoryProvider for cross-model chat archive memory.

Provides persistent recall across sessions by querying a local MyChatArchive
database containing imported conversations from ChatGPT, Claude, Cursor, Grok,
and other platforms.  Read-heavy by design: the archive is the source of truth
and Hermes queries it for context via semantic search, keyword search, and
thread summary retrieval.

The 4 tools (mca_search, mca_recall, mca_remember, mca_provenance) are
exposed through the MemoryProvider interface.

Config via $HERMES_HOME/mychatarchive.json (profile-scoped):
  db_path:         path to the MCA SQLite database (default: ~/.mychatarchive/archive.db)
  recall_mode:     hybrid | context | tools (default: hybrid)
  prefetch_limit:  max chunks injected per turn (default: 5)
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

MCA_SEARCH_SCHEMA = {
    "name": "mca_search",
    "description": (
        "Search the MyChatArchive database for past conversations across all "
        "platforms (ChatGPT, Claude, Cursor, Grok, etc.).\n\n"
        "Supports semantic search (vector similarity), keyword search (FTS5), "
        "or hybrid (both merged). Filter by platform, time range, or thread group.\n\n"
        "Use this when the user asks about something they discussed before, "
        "or when you need to find prior context on a topic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in the archive.",
            },
            "mode": {
                "type": "string",
                "enum": ["semantic", "keyword", "hybrid"],
                "description": "Search mode. semantic (default) uses vector similarity, keyword uses full-text search, hybrid merges both.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default: 10).",
            },
            "platform": {
                "type": "string",
                "description": "Filter to a specific platform (chatgpt, anthropic, grok, claude_code, cursor).",
            },
            "group": {
                "type": "string",
                "description": "Filter to a named thread group.",
            },
            "hours_back": {
                "type": "integer",
                "description": "Only search messages from the last N hours.",
            },
        },
        "required": ["query"],
    },
}

MCA_RECALL_SCHEMA = {
    "name": "mca_recall",
    "description": (
        "Rich contextual retrieval from MyChatArchive. Combines message chunks, "
        "thread summaries, and captured thoughts for a given topic.\n\n"
        "Returns a structured context bundle with messages, summaries, and thoughts. "
        "More comprehensive than mca_search -- use this when you need deep context "
        "on a topic the user has discussed across multiple conversations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Topic to recall context about.",
            },
            "limit": {
                "type": "integer",
                "description": "Max items per category (default: 5).",
            },
            "platform": {
                "type": "string",
                "description": "Filter to a specific platform.",
            },
            "group": {
                "type": "string",
                "description": "Filter to a named thread group.",
            },
        },
        "required": ["topic"],
    },
}

MCA_REMEMBER_SCHEMA = {
    "name": "mca_remember",
    "description": (
        "Capture a thought, insight, or fact into the MyChatArchive database. "
        "Stored as a 'thought' with a vector embedding for future retrieval.\n\n"
        "Use this to save important information the user wants to remember "
        "across sessions. Tagged with the current Hermes session for provenance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The thought or fact to remember.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags for categorization.",
            },
        },
        "required": ["content"],
    },
}

MCA_PROVENANCE_SCHEMA = {
    "name": "mca_provenance",
    "description": (
        "Look up the full source context for a chunk or thought returned by "
        "mca_search or mca_recall. Returns the thread title, platform, "
        "timestamps, and thread summary.\n\n"
        "Use this when the user wants to know where a piece of recalled "
        "information originally came from."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chunk_id": {
                "type": "string",
                "description": "A chunk ID from mca_search or mca_recall results.",
            },
            "thought_id": {
                "type": "string",
                "description": "A thought ID from mca_search or mca_recall results.",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_plugin_config(hermes_home: str) -> dict:
    """Load config from $HERMES_HOME/mychatarchive.json."""
    config_path = Path(hermes_home) / "mychatarchive.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _parse_meta(meta_json: Any) -> dict:
    """Parse chunk/thought metadata, handling double-encoded JSON."""
    if not meta_json:
        return {}
    try:
        parsed = json.loads(meta_json) if isinstance(meta_json, str) else meta_json
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _load_plugin_config_from_hermes() -> dict:
    """Load config from the active Hermes home directory.

    Resolution: HERMES_HOME env var, then hermes_constants.get_hermes_home().
    """
    import os
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        cfg = _load_plugin_config(env_home)
        if cfg:
            return cfg
    try:
        from hermes_constants import get_hermes_home
        cfg = _load_plugin_config(str(get_hermes_home()))
        if cfg:
            return cfg
    except Exception:
        pass
    return {}


def _resolve_db_path(config: dict) -> Path:
    """Resolve the MCA database path from plugin config or MCA defaults."""
    custom = config.get("db_path", "")
    if custom:
        return Path(custom).expanduser()
    try:
        from mychatarchive.config import get_db_path
        return get_db_path()
    except Exception:
        return Path.home() / ".mychatarchive" / "archive.db"


def _resolve_group_thread_ids(con: Any, group_name: str) -> set:
    """Resolve a group name to a set of thread IDs.

    Returns empty set if the group does not exist, so callers that pass the
    result to search_chunks/fts_search get zero results instead of silently
    falling back to an unfiltered global search.
    """
    try:
        from mychatarchive import db
        row = db.get_group_by_name(con, group_name)
        if row:
            return db.get_group_thread_ids(con, row[0])
    except Exception as exc:
        logger.debug("Group resolution failed for %r: %s", group_name, exc)
    return set()


def _get_stored_embedding_dim(con: Any) -> Optional[int]:
    """Read the embedding dimension from the vec_chunks table definition."""
    import re
    try:
        row = con.execute(
            "SELECT sql FROM sqlite_master WHERE name='vec_chunks'"
        ).fetchone()
        if not row or not row[0]:
            return None
        match = re.search(r"float\[(\d+)\]", row[0])
        return int(match.group(1)) if match else None
    except Exception:
        return None


def _validate_embedding_dimension(con: Any, current_dim: int) -> None:
    """Raise if the archive's stored vectors have a different dimension."""
    stored_dim = _get_stored_embedding_dim(con)
    if stored_dim is None:
        return
    if stored_dim != current_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: archive has {stored_dim}-dim "
            f"vectors but current model produces {current_dim}-dim. "
            f"Re-run `mychatarchive embed --force` to rebuild vectors with "
            f"the current model, or restore the original embedding model "
            f"in ~/.mychatarchive/config.json."
        )


def _cutoff_iso(hours_back: int) -> str:
    """Return an ISO timestamp N hours in the past."""
    from datetime import timedelta
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class MyChatArchiveProvider(MemoryProvider):
    """MyChatArchive memory with semantic search over cross-platform chat history."""

    def __init__(self) -> None:
        self._config: dict = {}
        self._con: Any = None
        self._db: Any = None
        self._embeddings: Any = None
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._recall_mode: str = "hybrid"
        self._prefetch_limit: int = 5
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "mychatarchive"

    def is_available(self) -> bool:
        """Check that the mychatarchive package is importable and the DB exists.

        No network calls. Reads saved config from $HERMES_HOME/mychatarchive.json
        to resolve the DB path (which may differ from the default).
        """
        try:
            importlib.import_module("mychatarchive")
        except ImportError:
            return False
        config = self._config
        if not config:
            config = _load_plugin_config_from_hermes()
        db_path = _resolve_db_path(config)
        return db_path.exists()

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config fields for the hermes memory setup wizard."""
        return [
            {
                "key": "db_path",
                "description": "Path to MyChatArchive SQLite database",
                "default": "~/.mychatarchive/archive.db",
            },
            {
                "key": "recall_mode",
                "description": "Memory integration mode",
                "default": "hybrid",
                "choices": ["hybrid", "context", "tools"],
            },
            {
                "key": "prefetch_limit",
                "description": "Max chunks auto-injected per turn",
                "default": "5",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to $HERMES_HOME/mychatarchive.json."""
        config_path = Path(hermes_home) / "mychatarchive.json"
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(
            json.dumps(existing, indent=2) + "\n",
            encoding="utf-8",
        )

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Interactive setup wizard with dependency installation."""
        import shutil
        import subprocess
        import sys

        from hermes_cli.config import save_config

        print("\n  Configuring MyChatArchive memory:\n")

        # Step 1: install mychatarchive if missing
        try:
            importlib.import_module("mychatarchive")
            print("  mychatarchive package: installed")
        except ImportError:
            print("  mychatarchive package: not installed")
            print("  Installing from GitHub...\n")
            uv_path = shutil.which("uv")
            pip_cmd = (
                [uv_path, "pip", "install", "--python", sys.executable, "--quiet"]
                if uv_path
                else [sys.executable, "-m", "pip", "install", "--quiet"]
            )
            try:
                subprocess.run(
                    pip_cmd + ["mychatarchive"],
                    check=True, timeout=120, capture_output=True,
                )
                print("  mychatarchive installed successfully")
            except Exception as exc:
                print(f"  Install failed: {exc}")
                print("  Run manually: pip install git+https://github.com/1ch1n/mychatarchive\n")
                return

        # Step 2: resolve DB path
        try:
            from mychatarchive.config import get_db_path
            default_db = str(get_db_path())
        except Exception:
            default_db = "~/.mychatarchive/archive.db"

        sys.stdout.write(f"  Database path [{default_db}]: ")
        sys.stdout.flush()
        db_input = sys.stdin.readline().strip()
        db_path = db_input or default_db

        # Step 3: recall mode
        print("\n  Recall mode:")
        print("    hybrid  -- auto-injected context + tools (default)")
        print("    context -- auto-injected context only, tools hidden")
        print("    tools   -- tools only, no auto-injection")
        sys.stdout.write("  Recall mode [hybrid]: ")
        sys.stdout.flush()
        mode_input = sys.stdin.readline().strip()
        recall_mode = mode_input if mode_input in ("hybrid", "context", "tools") else "hybrid"

        # Step 4: prefetch limit
        sys.stdout.write("  Max chunks per turn [5]: ")
        sys.stdout.flush()
        limit_input = sys.stdin.readline().strip()
        try:
            prefetch_limit = int(limit_input) if limit_input else 5
        except ValueError:
            prefetch_limit = 5

        # Step 5: save config
        provider_config = {
            "db_path": db_path,
            "recall_mode": recall_mode,
            "prefetch_limit": str(prefetch_limit),
        }
        self.save_config(provider_config, hermes_home)

        config.setdefault("memory", {})["provider"] = "mychatarchive"
        save_config(config)

        # Step 6: verify DB exists
        resolved = Path(db_path).expanduser()
        if resolved.exists():
            try:
                from mychatarchive import db
                con = db.get_connection(resolved)
                msgs = db.message_count(con)
                chunks = db.chunk_count(con)
                con.close()
                print(f"\n  Database: {resolved}")
                print(f"  Messages: {msgs:,}, Chunks: {chunks:,}")
            except Exception as exc:
                print(f"\n  Database found but could not read: {exc}")
        else:
            print(f"\n  Database not found at {resolved}")
            print("  Run: mychatarchive sync && mychatarchive embed")

        print(f"\n  Memory provider set to 'mychatarchive'")
        print("  Start a new session to activate.\n")

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Open DB connection and load the embedding model.

        Called once at agent startup. Receives hermes_home for profile-scoped
        config storage. The MCA database path itself is user-wide (not
        profile-scoped) because the archive predates Hermes and is shared
        across tools.
        """
        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home", ""))

        self._config = _load_plugin_config(self._hermes_home)
        if not self._config:
            self._config = _load_plugin_config_from_hermes()
        self._recall_mode = self._config.get("recall_mode", "hybrid")
        if self._recall_mode not in ("hybrid", "context", "tools"):
            self._recall_mode = "hybrid"
        try:
            self._prefetch_limit = int(self._config.get("prefetch_limit", 5))
        except (ValueError, TypeError):
            self._prefetch_limit = 5

        db_path = _resolve_db_path(self._config)
        if not db_path.exists():
            logger.warning(
                "MyChatArchive DB not found at %s -- plugin will be read-only with no data",
                db_path,
            )
            return

        try:
            from mychatarchive import db as mca_db
            from mychatarchive import embeddings as mca_embeddings
            from mychatarchive.config import get_embedding_model, get_embedding_dim

            self._db = mca_db
            self._embeddings = mca_embeddings
            self._con = mca_db.get_connection(db_path)
            mca_db.ensure_schema(self._con)

            _validate_embedding_dimension(self._con, get_embedding_dim())

            logger.info(
                "MyChatArchive initialized: db=%s, model=%s, dim=%d, "
                "messages=%d, chunks=%d, thoughts=%d",
                db_path,
                get_embedding_model(),
                get_embedding_dim(),
                mca_db.message_count(self._con),
                mca_db.chunk_count(self._con),
                mca_db.thought_count(self._con),
            )
        except Exception as exc:
            logger.warning("MyChatArchive initialization failed: %s", exc)
            self._con = None

    def system_prompt_block(self) -> str:
        """Return archive stats for the system prompt.

        Tells the model what data is available so it knows when to use
        the mca_* tools.
        """
        if not self._con or not self._db:
            return ""
        try:
            messages = self._db.message_count(self._con)
            chunks = self._db.chunk_count(self._con)
            thoughts = self._db.thought_count(self._con)
            threads = self._db.thread_count(self._con)
            summaries = self._db.summarized_thread_count(self._con)
            platforms = self._db.platform_counts(self._con)
            platform_str = ", ".join(f"{p} ({c})" for p, c in platforms) if platforms else "none"

            if self._recall_mode == "context":
                mode_note = (
                    "Relevant archive context is automatically injected. "
                    "No archive tools are available."
                )
            elif self._recall_mode == "tools":
                mode_note = (
                    "Use mca_search to find past conversations, mca_recall for deep context, "
                    "mca_remember to save insights, mca_provenance to trace sources. "
                    "No automatic context injection."
                )
            else:
                mode_note = (
                    "Relevant archive context is automatically injected. "
                    "Use mca_search, mca_recall, mca_remember, mca_provenance for deeper queries."
                )

            return (
                f"# MyChatArchive\n"
                f"Active. {messages} messages across {threads} threads "
                f"({summaries} summarized), {chunks} chunks, {thoughts} thoughts.\n"
                f"Platforms: {platform_str}.\n"
                f"{mode_note}"
            )
        except Exception as exc:
            logger.debug("MyChatArchive system_prompt_block failed: %s", exc)
            return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Retrieve relevant archive context for the upcoming turn.

        Runs semantic search over chunks and returns formatted results
        for injection into the conversation context.

        Returns empty when recall_mode is 'tools' (no auto-injection).
        """
        if self._recall_mode == "tools":
            return ""
        if not self._con or not self._db or not self._embeddings or not query:
            return ""
        if not query.strip():
            return ""

        try:
            embedding = self._embeddings.embed_single(query)
            results = self._db.search_chunks(
                self._con, embedding, limit=self._prefetch_limit,
            )
            if not results:
                return ""

            lines: List[str] = []
            for chunk_id, distance in results:
                row = self._db.get_chunk_by_id(self._con, chunk_id)
                if not row:
                    continue
                text, thread_id, ts_start, ts_end, meta_json = row
                score = max(0.0, 1.0 - distance)
                preview = text[:300].replace("\n", " ")
                lines.append(f"- [{score:.2f}] {preview}")

            if not lines:
                return ""
            return "## MyChatArchive\n" + "\n".join(lines)
        except Exception as exc:
            logger.debug("MyChatArchive prefetch failed: %s", exc)
            return ""

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = "",
    ) -> None:
        """Persist a completed turn to the archive (non-blocking).

        Currently a lightweight no-op for content storage because MCA's
        message schema is import-oriented.  Fires a daemon thread for
        future expansion without blocking the agent loop.
        """
        if not self._con or not self._db:
            return

        def _sync() -> None:
            pass

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="mca-sync",
        )
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas, respecting recall_mode.

        context-only mode hides all tools.
        """
        if self._recall_mode == "context":
            return []
        return [
            MCA_SEARCH_SCHEMA,
            MCA_RECALL_SCHEMA,
            MCA_REMEMBER_SCHEMA,
            MCA_PROVENANCE_SCHEMA,
        ]

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs: Any,
    ) -> str:
        """Dispatch a tool call to the appropriate handler."""
        if not self._con or not self._db:
            return tool_error("MyChatArchive is not initialized.")

        try:
            if tool_name == "mca_search":
                return self._handle_search(args)
            elif tool_name == "mca_recall":
                return self._handle_recall(args)
            elif tool_name == "mca_remember":
                return self._handle_remember(args)
            elif tool_name == "mca_provenance":
                return self._handle_provenance(args)
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as exc:
            logger.error("MyChatArchive tool %s failed: %s", tool_name, exc)
            return tool_error(f"{tool_name} failed: {exc}")

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs: Any,
    ) -> None:
        """Update cached session_id when the agent switches sessions."""
        self._session_id = new_session_id

    def shutdown(self) -> None:
        """Close DB connection and wait for pending sync."""
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        if self._con:
            try:
                self._con.close()
            except Exception:
                pass
            self._con = None
        self._db = None
        self._embeddings = None

    # -- Tool handlers -------------------------------------------------------

    def _handle_search(self, args: dict) -> str:
        """Handle mca_search: semantic, keyword, or hybrid search."""
        query = args.get("query", "").strip()
        if not query:
            return tool_error("Missing required parameter: query")

        mode = args.get("mode", "semantic")
        if mode not in ("semantic", "keyword", "hybrid"):
            return tool_error(f"Invalid search mode: {mode}. Use semantic, keyword, or hybrid.")
        limit = int(args.get("limit", 10))
        platform = args.get("platform")
        group_name = args.get("group")
        hours_back = args.get("hours_back")

        cutoff = _cutoff_iso(int(hours_back)) if hours_back else None
        group_ids = _resolve_group_thread_ids(self._con, group_name) if group_name else None

        results: List[dict] = []
        seen_ids: set = set()

        if mode in ("semantic", "hybrid"):
            embedding = self._embeddings.embed_single(query)
            hits = self._db.search_chunks(
                self._con, embedding, limit=limit,
                platform=platform, cutoff_iso=cutoff,
                group_thread_ids=group_ids,
            )
            for chunk_id, distance in hits:
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                row = self._db.get_chunk_by_id(self._con, chunk_id)
                if not row:
                    continue
                text, thread_id, ts_start, ts_end, meta_json = row
                meta = _parse_meta(meta_json)
                results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "thread_id": thread_id,
                    "platform": meta.get("platform", ""),
                    "title": meta.get("title", ""),
                    "ts_start": ts_start,
                    "score": round(max(0.0, 1.0 - distance), 4),
                    "match": "semantic",
                })

        if mode in ("keyword", "hybrid"):
            fts_hits = self._db.fts_search(
                self._con, query, limit=limit,
                platform=platform, cutoff_iso=cutoff,
                group_thread_ids=group_ids,
            )
            for row in fts_hits:
                msg_id, text, thread_id, ts, role, title = (
                    row[0], row[1], row[2], row[3], row[4], row[5],
                )
                key = f"fts-{msg_id}"
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                results.append({
                    "message_id": msg_id,
                    "text": text[:500],
                    "thread_id": thread_id,
                    "timestamp": ts,
                    "role": role,
                    "title": title or "",
                    "match": "keyword",
                })

        return json.dumps({"results": results[:limit], "count": len(results)})

    def _handle_recall(self, args: dict) -> str:
        """Handle mca_recall: rich multi-source context retrieval."""
        topic = args.get("topic", "").strip()
        if not topic:
            return tool_error("Missing required parameter: topic")

        limit = int(args.get("limit", 5))
        platform = args.get("platform")
        group_name = args.get("group")
        group_ids = _resolve_group_thread_ids(self._con, group_name) if group_name else None

        embedding = self._embeddings.embed_single(topic)

        # Layer 1: message chunks
        messages: List[dict] = []
        chunk_hits = self._db.search_chunks(
            self._con, embedding, limit=limit,
            platform=platform, group_thread_ids=group_ids,
        )
        for chunk_id, distance in chunk_hits:
            row = self._db.get_chunk_by_id(self._con, chunk_id)
            if not row:
                continue
            text, thread_id, ts_start, ts_end, meta_json = row
            meta = _parse_meta(meta_json)
            messages.append({
                "chunk_id": chunk_id,
                "text": text,
                "thread_id": thread_id,
                "platform": meta.get("platform", ""),
                "title": meta.get("title", ""),
                "ts_start": ts_start,
                "score": round(max(0.0, 1.0 - distance), 4),
            })

        # Layer 2: thread summaries (post-filtered by platform/group since
        # search_thread_summaries does not accept those params)
        summaries: List[dict] = []
        summary_hits = self._db.search_thread_summaries(
            self._con, embedding, limit=limit * 3,
        )
        for summary_id, distance in summary_hits:
            row = self._db.get_summary_by_id(self._con, summary_id)
            if not row:
                continue
            # 10-col: summary_id, thread_id, segment_index, title,
            #         platform, message_count, ts_start, ts_end, summary, key_topics
            row_platform = row[4] or ""
            row_thread_id = row[1]
            if platform and row_platform != platform:
                continue
            if group_ids is not None and row_thread_id not in group_ids:
                continue
            try:
                key_topics = json.loads(row[9]) if row[9] else []
            except (json.JSONDecodeError, TypeError):
                key_topics = []
            summaries.append({
                "summary_id": row[0],
                "thread_id": row_thread_id,
                "title": row[3] or "",
                "platform": row_platform,
                "message_count": row[5],
                "ts_start": row[6],
                "ts_end": row[7],
                "summary": row[8],
                "key_topics": key_topics,
                "score": round(max(0.0, 1.0 - distance), 4),
            })
            if len(summaries) >= limit:
                break

        # Layer 3: captured thoughts
        thoughts: List[dict] = []
        thought_hits = self._db.search_thoughts(self._con, embedding, limit=limit)
        for thought_id, distance in thought_hits:
            row = self._db.get_thought_by_id(self._con, thought_id)
            if not row:
                continue
            text, created_at, meta_json = row
            thoughts.append({
                "thought_id": thought_id,
                "text": text,
                "created_at": created_at,
                "score": round(max(0.0, 1.0 - distance), 4),
            })

        return json.dumps({
            "messages": messages,
            "summaries": summaries,
            "thoughts": thoughts,
        })

    def _handle_remember(self, args: dict) -> str:
        """Handle mca_remember: capture a thought into the archive."""
        content = args.get("content", "").strip()
        if not content:
            return tool_error("Missing required parameter: content")

        tags_str = args.get("tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

        thought_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        embedding = self._embeddings.embed_single(content)

        meta = {
            "source": "hermes",
            "session_id": self._session_id,
        }
        if tags:
            meta["tags"] = tags

        self._db.insert_thought(
            self._con, thought_id, content, now, embedding, meta,
        )
        self._con.commit()

        return json.dumps({
            "thought_id": thought_id,
            "created_at": now,
            "status": "saved",
        })

    def _handle_provenance(self, args: dict) -> str:
        """Handle mca_provenance: source context lookup for a chunk or thought."""
        chunk_id = (args.get("chunk_id") or "").strip()
        thought_id = (args.get("thought_id") or "").strip()

        if not chunk_id and not thought_id:
            return tool_error("Provide either chunk_id or thought_id.")
        if chunk_id and thought_id:
            return tool_error("Provide only one of chunk_id or thought_id, not both.")

        if chunk_id:
            row = self._db.get_chunk_by_id(self._con, chunk_id)
            if not row:
                return tool_error(f"Chunk not found: {chunk_id}")
            text, thread_id, ts_start, ts_end, meta_json = row
            meta = _parse_meta(meta_json)

            # Fetch thread summary for additional context
            thread_title = meta.get("title", "")
            thread_summary = ""
            thread_platform = meta.get("platform", "")
            summary_row = self._db.get_thread_summary(self._con, thread_id)
            if summary_row:
                thread_title = thread_title or summary_row[3] or ""
                thread_platform = thread_platform or summary_row[4] or ""
                thread_summary = summary_row[8] or ""

            return json.dumps({
                "type": "chunk",
                "chunk_id": chunk_id,
                "text": text,
                "thread_id": thread_id,
                "thread_title": thread_title,
                "platform": thread_platform,
                "ts_start": ts_start,
                "ts_end": ts_end,
                "thread_summary": thread_summary,
                "meta": meta,
            })

        # thought_id path
        row = self._db.get_thought_by_id(self._con, thought_id)
        if not row:
            return tool_error(f"Thought not found: {thought_id}")
        text, created_at, meta_json = row
        meta = _parse_meta(meta_json)

        return json.dumps({
            "type": "thought",
            "thought_id": thought_id,
            "text": text,
            "created_at": created_at,
            "meta": meta,
        })


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    """Register MyChatArchive as a memory provider plugin."""
    ctx.register_memory_provider(MyChatArchiveProvider())
