"""Custom memory plugin — MemoryProvider interface.

A generic, bring-your-own-backend memory provider. It lets Hermes recall from
and write back to a knowledge store you already maintain — an "LLM wiki",
"second brain", or Obsidian-style markdown vault — instead of a hosted service.

This first version ships the ``files`` backend: point Hermes at a directory
(local path, or an NFS/SMB-mounted share) of markdown/text notes. Hermes
recalls relevant notes before each turn and, because it is self-improving,
writes new memories back into the same vault so future sessions can recall
them. A pluggable backend seam is in place so an ``http`` mode (recall/write
against a configurable API endpoint) can be added without touching this
provider's lifecycle.

Config (``~/.hermes/config.yaml``)::

    memory:
      provider: custom
      custom:
        mode: files                 # "files" (this version); "http" planned
        dir: "/path/to/obsidian-vault"   # your LLM wiki / second brain root
        write_subdir: "hermes-memory"    # where Hermes writes new notes
        write_format: markdown      # "markdown" (vault-native, recalled) | "jsonl" (export log)
        max_results: 5              # notes injected per recall

``dir`` is read for recall and (under ``write_subdir``) written to. With
``write_format: markdown`` the notes Hermes writes are themselves recalled on
later turns (a full read+write loop — ideal for an Obsidian vault). With
``jsonl`` each write is a timestamped JSON record appended to the file — not
recalled, but suitable as an append-only audit trail of what the agent stored
and when (review, compliance, or replay into another system).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Recall/store tuning
_MIN_QUERY_LEN = 8          # ignore trivially short recall queries
_MIN_TERM_LEN = 3           # ignore very short search terms ("a", "is")
_SNIPPET_CHARS = 600        # max chars returned per matched note
_DEFAULT_MAX_RESULTS = 5
_DEFAULT_READ_GLOBS = ("*.md", "*.txt")
_DEFAULT_WRITE_SUBDIR = "hermes-memory"


def _load_custom_config() -> Dict[str, Any]:
    """Read the ``memory.custom`` section from config.yaml (no network)."""
    try:
        from hermes_cli.config import load_config, cfg_get
        return cfg_get(load_config(), "memory", "custom", default={}) or {}
    except Exception as e:  # config not loadable -> provider simply inactive
        logger.debug("custom memory: could not load config: %s", e)
        return {}


def _safe_name(value: str) -> str:
    """Make *value* safe to use as a filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip("-")
    return cleaned or "session"


# ---------------------------------------------------------------------------
# Files backend — recall from / write to a local (or mounted) note vault
# ---------------------------------------------------------------------------

class _FilesBackend:
    """Recall from and write to a directory of markdown/text notes.

    Designed for an Obsidian vault / LLM wiki / second brain that lives on disk
    or an NFS/SMB share. Recall is a dependency-free keyword scan; writes append
    notes under ``write_subdir`` so they become part of the same vault.
    """

    def __init__(
        self,
        root: Path,
        write_subdir: str,
        write_format: str,
        read_globs: tuple,
        max_results: int,
    ):
        self.root = root
        self.write_dir = root / write_subdir
        self.write_format = write_format if write_format in {"markdown", "jsonl"} else "markdown"
        self.read_globs = read_globs
        self.max_results = max_results
        self._write_lock = threading.Lock()

    def available(self) -> bool:
        """The vault directory must exist. No network, cheap to call."""
        try:
            return self.root.is_dir()
        except Exception:
            return False

    # -- recall --------------------------------------------------------------

    def _iter_files(self):
        for pattern in self.read_globs:
            try:
                yield from self.root.rglob(pattern)
            except Exception:
                continue

    @staticmethod
    def _snippet(text: str, terms: List[str]) -> str:
        """Return the most relevant lines of *text* (those mentioning a term)."""
        hits = [ln.strip() for ln in text.splitlines()
                if ln.strip() and any(t in ln.lower() for t in terms)]
        snippet = "\n".join(hits) if hits else text.strip()
        return snippet[:_SNIPPET_CHARS]

    def recall(self, query: str, max_results: int) -> List[str]:
        """Return up to *max_results* note snippets relevant to *query*."""
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= _MIN_TERM_LEN]
        if not terms:
            return []
        scored = []
        for path in self._iter_files():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            low = text.lower()
            score = sum(low.count(t) for t in terms)
            if score > 0:
                scored.append((score, path, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for _score, path, text in scored[:max_results]:
            # Use the file name only — never leak the absolute vault path.
            results.append(f"[{path.name}]\n{self._snippet(text, terms)}")
        return results

    # -- writes --------------------------------------------------------------

    def _append(self, path: Path, content: str) -> None:
        with self._write_lock:
            self.write_dir.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists()
            with open(path, "a", encoding="utf-8") as f:
                if new_file and self.write_format == "markdown":
                    f.write(
                        f"---\ncreated: {self._now()}\nsource: hermes-agent\n---\n"
                        f"# Hermes memory\n\n"
                    )
                f.write(content)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def write_turn(self, user_content: str, assistant_content: str, session_id: str) -> None:
        stem = f"session-{_safe_name(session_id)}"
        if self.write_format == "jsonl":
            line = json.dumps({
                "ts": self._now(),
                "session": session_id,
                "user": user_content,
                "assistant": assistant_content,
            }, ensure_ascii=False)
            self._append(self.write_dir / f"{stem}.jsonl", line + "\n")
        else:
            block = (
                f"## {self._now()}\n\n"
                f"**User:** {user_content.strip()}\n\n"
                f"**Assistant:** {assistant_content.strip()}\n\n"
            )
            self._append(self.write_dir / f"{stem}.md", block)

    def write_fact(self, content: str) -> None:
        if self.write_format == "jsonl":
            line = json.dumps({"ts": self._now(), "fact": content}, ensure_ascii=False)
            self._append(self.write_dir / "facts.jsonl", line + "\n")
        else:
            self._append(self.write_dir / "facts.md", f"- {self._now()}: {content.strip()}\n")


def _make_backend(cfg: Dict[str, Any]) -> Optional[_FilesBackend]:
    """Build the configured backend, or None if unavailable/unsupported.

    The ``mode`` switch is the seam for future backends (e.g. ``http``). This
    version implements ``files`` only; any other mode is treated as inactive.
    """
    mode = str(cfg.get("mode", "files")).strip().lower() or "files"
    if mode != "files":
        logger.debug("custom memory: mode '%s' not supported in this version", mode)
        return None
    dir_value = str(cfg.get("dir", "")).strip()
    if not dir_value:
        return None
    read_globs = tuple(cfg.get("read_globs") or _DEFAULT_READ_GLOBS)
    try:
        max_results = int(cfg.get("max_results", _DEFAULT_MAX_RESULTS) or _DEFAULT_MAX_RESULTS)
    except (TypeError, ValueError):
        max_results = _DEFAULT_MAX_RESULTS
    return _FilesBackend(
        root=Path(dir_value).expanduser(),
        write_subdir=str(cfg.get("write_subdir", _DEFAULT_WRITE_SUBDIR)).strip() or _DEFAULT_WRITE_SUBDIR,
        write_format=str(cfg.get("write_format", "markdown")).strip().lower(),
        read_globs=read_globs,
        max_results=max_results,
    )


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": (
        "Search your connected knowledge store (LLM wiki / second brain / "
        "Obsidian vault) for relevant notes from earlier sessions. Use when "
        "past context would help answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "memory_add",
    "description": (
        "Save an important note to your connected knowledge store so future "
        "sessions can recall it. Use for decisions, preferences, and durable facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The note to remember."},
        },
        "required": ["content"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class CustomMemoryProvider(MemoryProvider):
    """Recall from and write to a user-supplied knowledge store (LLM wiki)."""

    def __init__(self):
        self._backend: Optional[_FilesBackend] = None
        self._session_id = ""
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "custom"

    def is_available(self) -> bool:
        """True when a backend is configured and reachable. No network."""
        backend = _make_backend(_load_custom_config())
        return bool(backend and backend.available())

    def get_config_schema(self):
        return [
            {
                "key": "mode",
                "description": "Backend type ('files' = local/mounted note vault)",
                "default": "files",
                "choices": ["files"],
            },
            {
                "key": "dir",
                "description": "Path to your LLM wiki / second brain / Obsidian vault",
                "required": True,
            },
            {
                "key": "write_subdir",
                "description": "Subdirectory the agent writes new memories into",
                "default": _DEFAULT_WRITE_SUBDIR,
            },
            {
                "key": "write_format",
                "description": "How writes are stored ('markdown' = vault-native + recalled; 'jsonl' = export log)",
                "default": "markdown",
                "choices": ["markdown", "jsonl"],
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._backend = _make_backend(_load_custom_config())

    def system_prompt_block(self) -> str:
        if not self._backend:
            return ""
        return (
            "# Custom Memory\n"
            "Connected to your knowledge store (LLM wiki / second brain). "
            "Use memory_search to recall past notes and memory_add to save "
            "durable facts for future sessions."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant notes before the model is called (synchronous)."""
        if not self._backend or len(query.strip()) < _MIN_QUERY_LEN:
            return ""
        try:
            hits = self._backend.recall(query.strip(), self._backend.max_results)
        except Exception as e:
            logger.debug("custom memory recall failed: %s", e)
            return ""
        if not hits:
            return ""
        return "## Memory (from your knowledge store)\n\n" + "\n\n".join(hits)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch() runs synchronously at turn start."""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist the completed turn to the store in the background."""
        if not self._backend or not user_content.strip():
            return

        def _write():
            try:
                self._backend.write_turn(user_content, assistant_content, session_id or self._session_id)
            except Exception as e:
                logger.debug("custom memory sync_turn failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_write, daemon=True, name="custom-mem-sync")
        self._sync_thread.start()

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in MEMORY.md / USER.md facts into the store."""
        if not self._backend or action not in {"add", "replace"} or not content:
            return

        def _write():
            try:
                label = "User profile" if target == "user" else "Agent memory"
                self._backend.write_fact(f"[{label}] {content}")
            except Exception as e:
                logger.debug("custom memory mirror failed: %s", e)

        threading.Thread(target=_write, daemon=True, name="custom-mem-fact").start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Wait for any pending background write to land."""
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not self._backend:
            return []
        return [SEARCH_SCHEMA, ADD_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._backend:
            return tool_error("Custom memory is not configured.")
        if tool_name == "memory_search":
            query = args.get("query", "")
            if not query:
                return tool_error("query is required")
            hits = self._backend.recall(query.strip(), self._backend.max_results)
            if not hits:
                return json.dumps({"result": "No relevant notes found."})
            return json.dumps({"result": "\n\n".join(hits)})
        if tool_name == "memory_add":
            content = args.get("content", "")
            if not content:
                return tool_error("content is required")
            try:
                self._backend.write_fact(content)
            except Exception as e:
                return tool_error(f"Failed to save note: {e}")
            return json.dumps({"result": "Note saved to your knowledge store."})
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the custom knowledge-store memory provider."""
    ctx.register_memory_provider(CustomMemoryProvider())
