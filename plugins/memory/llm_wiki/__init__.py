"""LLM Wiki memory provider.

First-class MemoryProvider wrapper for Hermes LLM Wiki. The provider is
intentionally read-first: rich durable memory should be retrieved explicitly
or via bounded prefetch, not dumped into every prompt.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from agent.memory_provider import MemoryProvider

WikiEngine = None


class LLMWikiMemoryProvider(MemoryProvider):
    """MemoryProvider facade for the Hermes LLM Wiki."""

    def __init__(self) -> None:
        self._session_id = ""
        self._hermes_home = ""
        self._agent_context = "primary"
        self.wiki_path = Path.home() / ".hermes" / "wiki" / "personal"
        self.wiki_name = "personal"
        self._engine_instance = None
        self.prefetch_limit = 3
        self.prefetch_max_chars = 1200
        self.search_max_limit = 20
        self.read_max_chars = 64_000

    @property
    def name(self) -> str:
        return "llm_wiki"

    def is_available(self) -> bool:
        return importlib.util.find_spec("hermes_wiki.engine") is not None

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home") or "")
        self._agent_context = str(kwargs.get("agent_context") or "primary").strip().lower() or "primary"
        self._load_wiki_config()

    def system_prompt_block(self) -> str:
        return (
            "LLM Wiki memory is available as source-backed durable knowledge. "
            "Use wiki_search/wiki_read/wiki_query when rich memory or policy is relevant; "
            "do not assume the whole wiki is already in context. Queries default to read-only."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        query = (query or "").strip()
        if not query:
            return ""
        try:
            searcher = getattr(self._engine(), "search", None)
            if searcher is None:
                return ""
            results = searcher.search(query, limit=self.prefetch_limit, exclude_sources=True)
        except Exception:
            return ""
        return self._format_prefetch_results(results)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        return None

    def _format_prefetch_results(self, results: Any) -> str:
        if not results:
            return ""
        header = "LLM Wiki relevant context (source-backed; use wiki_read/wiki_query for more):"
        lines = [header]
        remaining = self.prefetch_max_chars - len(header) - 1
        for result in results:
            data = self._search_result_to_dict(result)
            page_path = data["page_path"]
            title = data["title"] or page_path or "Untitled"
            text = " ".join(data["text"].split())
            score = data["score"]
            score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
            prefix = f"- [{page_path}] {title}{score_text}: "
            if len(prefix) >= remaining:
                break
            snippet = text[: max(0, min(260, remaining - len(prefix) - 1))]
            line = prefix + snippet
            if len(line) > remaining:
                break
            lines.append(line)
            remaining -= len(line) + 1
            if remaining <= 40:
                break
        return "\n".join(lines)[: self.prefetch_max_chars]

    def _search_result_to_dict(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return {
                "page_path": str(result.get("page_path", "")),
                "title": str(result.get("title", "")),
                "page_type": str(result.get("page_type", "")),
                "chunk_index": result.get("chunk_index", 0),
                "text": str(result.get("text", "")),
                "score": result.get("score"),
                "tags": result.get("tags", []),
            }
        return {
            "page_path": str(getattr(result, "page_path", "")),
            "title": str(getattr(result, "title", "")),
            "page_type": str(getattr(result, "page_type", "")),
            "chunk_index": getattr(result, "chunk_index", 0),
            "text": str(getattr(result, "text", "")),
            "score": getattr(result, "score", None),
            "tags": getattr(result, "tags", []),
        }

    def _load_wiki_config(self) -> None:
        config = self._wiki_config()
        self.wiki_name = config.wiki_name
        self.wiki_path = config.wiki_path

    def _config_path(self) -> Path:
        hermes_home = Path(self._hermes_home).expanduser() if self._hermes_home else Path.home() / ".hermes"
        return hermes_home / "config.yaml"

    def _load_config_data(self) -> Dict[str, Any]:
        config_path = self._config_path()
        if not config_path.exists():
            return {}
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}

    def _wiki_config(self):
        from hermes_wiki.config import WikiConfig

        data = self._load_config_data()
        if "wiki" not in data:
            hermes_home = Path(self._hermes_home).expanduser() if self._hermes_home else Path.home() / ".hermes"
            data = {"wiki": {"path": str(hermes_home / "wiki" / "personal"), "name": "personal"}}
        return WikiConfig.from_dict(data)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "path", "description": "Filesystem path for the LLM Wiki", "default": "~/.hermes/wiki/personal"},
            {"key": "name", "description": "Wiki name used for vector collection suffix", "default": "personal"},
            {"key": "embedding_url", "description": "OpenAI-compatible embedding endpoint", "default": "http://localhost:22222"},
            {"key": "embedding_model", "description": "Embedding model name", "default": "Qwen3-Embedding-8B"},
            {"key": "embedding_dim", "description": "Embedding vector dimension", "default": 4096},
            {"key": "qdrant_url", "description": "Qdrant HTTP URL", "default": "http://localhost:6333"},
            {"key": "collection_prefix", "description": "Qdrant collection prefix", "default": "hermes_wiki"},
            {"key": "llm_url", "description": "OpenAI-compatible LLM endpoint for wiki_query", "default": "http://localhost:8011/v1"},
            {"key": "llm_model", "description": "LLM model used by wiki_query", "default": "gpt-5.5"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home).expanduser() / "config.yaml"
        data: Dict[str, Any] = {}
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            data = loaded if isinstance(loaded, dict) else {}

        wiki = data.setdefault("wiki", {})
        if not isinstance(wiki, dict):
            wiki = {}
            data["wiki"] = wiki

        def set_if_present(key: str, section: Dict[str, Any], dest: str | None = None, *, cast=None) -> None:
            if key not in values or values[key] in (None, ""):
                return
            value = values[key]
            if cast is not None:
                value = cast(value)
            section[dest or key] = value

        set_if_present("path", wiki)
        set_if_present("name", wiki)
        embedding = wiki.setdefault("embedding", {})
        vector = wiki.setdefault("vector_store", {})
        llm = wiki.setdefault("llm", {})
        set_if_present("embedding_url", embedding, "url")
        set_if_present("embedding_model", embedding, "model")
        set_if_present("embedding_dim", embedding, "dim", cast=int)
        set_if_present("qdrant_url", vector, "url")
        set_if_present("collection_prefix", vector)
        set_if_present("llm_url", llm, "url")
        set_if_present("llm_model", llm, "model")

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def _ensure_optional_deps(self) -> None:
        try:
            from tools.lazy_deps import ensure as lazy_ensure

            lazy_ensure("memory.llm_wiki", prompt=False)
        except ImportError:
            return

    def _engine(self):
        if self._engine_instance is None:
            if self._writes_allowed():
                self._ensure_optional_deps()
            engine_cls = WikiEngine
            if engine_cls is None:
                try:
                    from hermes_wiki.engine import WikiEngine as imported_engine
                except Exception as exc:  # pragma: no cover - exercised by integration tests later
                    raise RuntimeError("LLM Wiki engine is not importable") from exc
                engine_cls = imported_engine
            self._engine_instance = engine_cls(self._wiki_config(), read_only=not self._writes_allowed())
        return self._engine_instance

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "wiki_status",
                "description": "Show LLM Wiki status for the active Hermes profile.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "wiki_orient",
                "description": "Orient to the LLM Wiki index and recent activity without writing logs.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "wiki_search",
                "description": "Search the LLM Wiki for source-backed durable memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results.",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "wiki_read",
                "description": "Read a wiki page by slug or relative path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "string", "description": "Page slug or relative markdown path."},
                    },
                    "required": ["page"],
                },
            },
            {
                "name": "wiki_query",
                "description": "Ask a question against the LLM Wiki. Defaults to read-only: no filed result and no query log.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to answer from wiki context."},
                        "file_result": {
                            "type": "boolean",
                            "description": "Whether to file a durable query page if the result is worth keeping.",
                            "default": False,
                        },
                        "log_query": {
                            "type": "boolean",
                            "description": "Whether to append this query to log.md.",
                            "default": False,
                        },
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "wiki_lint",
                "description": "Lint the LLM Wiki. Defaults to non-mutating: write_log=false.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "write_log": {"type": "boolean", "description": "Append result to log.md.", "default": False},
                    },
                    "required": [],
                },
            },
            {
                "name": "wiki_ingest",
                "description": "Ingest a curated source file into the LLM Wiki. Defaults to dry_run=true.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to source file to ingest."},
                        "dry_run": {"type": "boolean", "description": "Plan ingest without writing wiki files.", "default": True},
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "wiki_reindex",
                "description": "Rebuild the LLM Wiki vector index. Blocked outside primary agent contexts.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        """Parse bool-like tool arguments without treating "false" as truthy."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", ""}:
                return False
        return bool(value)

    def _clamp_limit(self, value: Any, default: int = 5) -> int:
        try:
            limit = int(value)
        except (TypeError, ValueError):
            limit = default
        return max(1, min(self.search_max_limit, limit))

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        args = args or {}
        if tool_name == "wiki_status":
            return json.dumps(self._engine().status(), indent=2, default=str)
        if tool_name == "wiki_orient":
            return str(self._engine().orient())
        if tool_name == "wiki_query":
            question = str(args.get("question") or "").strip()
            if not question:
                return "Error: question is required"
            file_result = self._as_bool(args.get("file_result"), False)
            log_query = self._as_bool(args.get("log_query"), False)
            if (file_result or log_query) and not self._writes_allowed():
                return self._write_blocked("wiki_query")
            return str(
                self._engine().query(
                    question,
                    file_result=file_result,
                    log_query=log_query,
                )
            )
        if tool_name == "wiki_read":
            page = str(args.get("page") or "").strip()
            if not page:
                return "Error: page is required"
            return self._read_page(page)
        if tool_name == "wiki_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return "Error: query is required"
            limit = self._clamp_limit(args.get("limit", 5), 5)
            engine = self._engine()
            searcher = getattr(engine, "search", None)
            if searcher is None:
                return "Error: wiki search is unavailable"
            results = searcher.search(query, limit=limit) if hasattr(searcher, "search") else searcher(query, limit=limit)
            return json.dumps([self._search_result_to_dict(r) for r in results], indent=2, default=str)
        if tool_name == "wiki_lint":
            write_log = self._as_bool(args.get("write_log"), False)
            if write_log and not self._writes_allowed():
                return self._write_blocked("wiki_lint")
            return json.dumps(self._engine().lint(write_log=write_log), indent=2, default=str)
        if tool_name == "wiki_ingest":
            file_path = str(args.get("file_path") or "").strip()
            if not file_path:
                return "Error: file_path is required"
            if not self._writes_allowed():
                return self._write_blocked("wiki_ingest")
            dry_run = self._as_bool(args.get("dry_run"), True)
            return json.dumps(self._engine().ingest_file(file_path, dry_run=dry_run), indent=2, default=str)
        if tool_name == "wiki_reindex":
            if not self._writes_allowed():
                return self._write_blocked("wiki_reindex")
            return json.dumps(self._engine().reindex(), indent=2, default=str)
        return super().handle_tool_call(tool_name, args, **kwargs)

    def _writes_allowed(self) -> bool:
        return self._agent_context == "primary"

    def _write_blocked(self, operation: str) -> str:
        return (
            f"Blocked {operation}: LLM Wiki file ingest and durable writes are disabled in "
            f"agent_context={self._agent_context!r}; use a primary agent context."
        )

    def _read_page(self, page: str) -> str:
        path = self._resolve_page_path(page)
        if path is None:
            return f"Error: wiki page not found: {page}"
        text = path.read_text(encoding="utf-8")
        if len(text) <= self.read_max_chars:
            return text
        return text[: self.read_max_chars] + f"\n\n[truncated: {len(text) - self.read_max_chars} additional characters omitted]"

    def _resolve_page_path(self, page: str) -> Path | None:
        raw = str(page or "").strip()
        if not raw:
            return None

        requested = Path(raw)
        candidates: list[Path]
        if requested.is_absolute():
            candidates = [requested]
        else:
            candidates = [self.wiki_path / requested]
            if requested.suffix == "":
                candidates.append(self.wiki_path / f"{raw}.md")
            if len(requested.parts) == 1:
                slug = requested.stem if requested.suffix == ".md" else raw
                for subdir in ("entities", "concepts", "comparisons", "queries"):
                    candidates.append(self.wiki_path / subdir / f"{slug}.md")

        wiki_root = self.wiki_path.resolve()
        for candidate in candidates:
            resolved = candidate.resolve()
            if wiki_root not in resolved.parents and resolved != wiki_root:
                return None
            if resolved.exists() and resolved.is_file():
                return resolved
        return None


def register(ctx: Any) -> None:
    """Register the LLM Wiki memory provider with Hermes plugin context."""

    ctx.register_memory_provider(LLMWikiMemoryProvider())
