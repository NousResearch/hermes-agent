"""Ferrosa Memory — MemoryProvider plugin backed by the ferrosa-memory MCP server.

Connects to ferrosa-memory via its HTTP JSON-RPC endpoint.  No client SDK needed;
talks raw JSON-RPC over urllib.

Config (env var takes precedence over saved config):
  FERROSA_MEMORY_URL   — full MCP endpoint, e.g.
                         http://ferrosa_user:ferrosa_user@127.0.0.1:18765/mcp
  FERROSA_MEMORY_TENANT_ID — optional tenant override (default: read from server)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent.memory_provider import MemoryProvider
from agent.skill_providers import SkillMetadata, SkillPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple JSON-RPC MCP client (raw HTTP, no SDK)
# ---------------------------------------------------------------------------

class _McpClient:
    def __init__(self, url: str):
        self.url, self._headers = self._prepare_url_and_headers(url)
        self._ctx = ssl.create_default_context()
        self._ctx.check_hostname = False
        self._ctx.verify_mode = ssl.CERT_NONE

    @staticmethod
    def _prepare_url_and_headers(url: str) -> tuple[str, Dict[str, str]]:
        """Strip URL userinfo into an Authorization header.

        urllib does not consistently turn http://user:pass@host URLs into Basic
        auth headers. Keeping the sanitized URL separately also prevents
        accidental credential logging.
        """
        parts = urllib.parse.urlsplit(url)
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if parts.username is None:
            return url, headers
        username = urllib.parse.unquote(parts.username)
        password = urllib.parse.unquote(parts.password or "")
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        sanitized = urllib.parse.urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
        return sanitized, headers

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool and return the parsed result."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(4):
            req = urllib.request.Request(
                self.url,
                data=data,
                headers=self._headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30, context=self._ctx) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    if "error" in body:
                        raise RuntimeError(body["error"])
                    result = body.get("result", {})
                    # MCP text content
                    if "content" in result:
                        for item in result["content"]:
                            if item.get("type") == "text":
                                try:
                                    return json.loads(item["text"])
                                except json.JSONDecodeError:
                                    return item["text"]
                    return result
            except Exception as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                if code != 429 or attempt == 3:
                    break
                time.sleep(1.5 * (attempt + 1))
        if last_error:
            raise last_error
        return {}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_saved_config(hermes_home: str) -> Dict[str, Any]:
    cfg_path = Path(hermes_home) / "plugins" / "ferrosa" / "config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            pass
    return {}


def _save_config(values: Dict[str, Any], hermes_home: str) -> None:
    cfg_dir = Path(hermes_home) / "plugins" / "ferrosa"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.json"
    try:
        cfg_path.write_text(json.dumps(values, indent=2))
    except Exception as e:
        logger.debug("Failed to write ferrosa config: %s", e)


def _resolve_url(saved: Dict[str, Any]) -> Optional[str]:
    # 1. env var
    url = os.environ.get("FERROSA_MEMORY_URL", "").strip()
    if url:
        return url
    # 2. saved config
    url = (saved.get("url") or "").strip()
    if url:
        return url
    # 3. fallback
    return "http://ferrosa_user:ferrosa_user@127.0.0.1:18765/mcp"


def _csv_env(name: str, default: Iterable[str]) -> List[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return [str(item).strip() for item in default if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


_DEFAULT_SKILL_LIST_CONTEXTS = [
    "skill",
    "blueprint project architecture plan",
    "task-level software development testing security audit",
    "tech language cloud database infrastructure",
    "management product marketing",
]


def _format_skill_payload_content(data: Dict[str, Any]) -> str:
    name = str(data.get("skill_name") or data.get("name") or "").strip()
    description = str(data.get("description") or "").strip()
    category = str(data.get("category") or "fmem").strip()
    version = str(data.get("version") or "").strip()
    steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    output_artifacts = data.get("output_artifacts") if isinstance(data.get("output_artifacts"), list) else []
    completion = str(data.get("completion_criteria") or "").strip()
    first_step = str(data.get("first_step_prompt") or "").strip()

    frontmatter = ["---", f"name: {json.dumps(name)}"]
    if description:
        frontmatter.append(f"description: {json.dumps(description)}")
    if category:
        frontmatter.append(f"category: {json.dumps(category)}")
    if version:
        frontmatter.append(f"version: {json.dumps(version)}")
    frontmatter.append("---")

    body = ["", f"# {name}", ""]
    if description:
        body.extend([description, ""])
    if first_step:
        body.extend(["## First step", "", first_step, ""])
    if steps:
        body.extend(["## Steps", ""])
        for index, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                continue
            phase = str(step.get("phase") or f"Step {index}").strip()
            instruction = str(step.get("instruction") or "").strip()
            body.extend([f"### {phase}", "", instruction, ""])
    if output_artifacts:
        body.extend(["## Output artifacts", ""])
        for artifact in output_artifacts:
            body.append(f"- {artifact}")
        body.append("")
    if completion:
        body.extend(["## Completion criteria", "", completion, ""])
    return "\n".join(frontmatter + body).rstrip() + "\n"


class FerrosaSkillProvider:
    """Virtual Hermes skills backed by fmem's global skill catalog."""

    namespace = "fmem"

    def __init__(self, client: Optional[_McpClient] = None, url: Optional[str] = None):
        self._client = client
        self._url = url
        self._metadata_cache: Optional[List[SkillMetadata]] = None

    @property
    def client(self) -> _McpClient:
        if self._client is None:
            url = self._url or _resolve_url({})
            if not url:
                raise RuntimeError("ferrosa-memory URL is not configured")
            self._client = _McpClient(url)
        return self._client

    def list_skills(self) -> List[SkillMetadata]:
        if self._metadata_cache is not None:
            return list(self._metadata_cache)
        by_name: Dict[str, SkillMetadata] = {}
        contexts = _csv_env("FERROSA_MEMORY_SKILL_LIST_CONTEXTS", _DEFAULT_SKILL_LIST_CONTEXTS)
        limit = int(os.environ.get("FERROSA_MEMORY_SKILL_LIST_LIMIT", "200") or "200")
        for context in contexts:
            try:
                result = self.client.call(
                    "retrieve_skills_for_context",
                    {"context": context, "limit": limit, "min_score": 0},
                )
            except Exception as exc:
                logger.debug("ferrosa skill list query failed for %r: %s", context, exc)
                continue
            for hit in result.get("results", []) if isinstance(result, dict) else []:
                name = str(hit.get("skill_name") or hit.get("name") or "").strip()
                if not name or name in by_name:
                    continue
                by_name[name] = SkillMetadata(
                    name=name,
                    description=str(hit.get("description") or ""),
                    category=str(hit.get("category") or "fmem"),
                    tags=[str(hit.get("category") or "fmem")],
                )
        self._metadata_cache = sorted(by_name.values(), key=lambda item: item.name)
        return list(self._metadata_cache)

    def resolve_skill(self, name: str) -> Optional[SkillPayload]:
        name = str(name or "").strip()
        if not name:
            return None
        result = self.client.call("invoke_skill", {"skill_name": name})
        if not isinstance(result, dict) or result.get("error"):
            return None
        skill_name = str(result.get("skill_name") or name)
        description = str(result.get("description") or "")
        category = str(result.get("category") or "fmem")
        return SkillPayload(
            name=skill_name,
            description=description,
            content=_format_skill_payload_content(result),
            linked_files=None,
            tags=[category] if category else [],
            metadata={"fmem": {"entity_id": result.get("entity_id"), "version": result.get("version")}},
        )

    def read_supporting_file(self, name: str, file_path: str) -> None:
        return None


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class FerrosaMemoryProvider(MemoryProvider):
    def __init__(self):
        self._url: Optional[str] = None
        self._client: Optional[_McpClient] = None
        self._session_id: str = ""
        self._tenant_id: str = ""
        self._hermes_home: str = ""
        self._saved_config: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "ferrosa"

    def is_available(self) -> bool:
        """Return True if we can reach the ferrosa-memory MCP endpoint."""
        url = _resolve_url(self._saved_config)
        if not url:
            return False
        try:
            _McpClient(url).call("get_stats", {})
            return True
        except Exception:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "url",
                "description": "Ferrosa Memory MCP HTTP endpoint (include credentials if auth required). Example: http://ferrosa_user:ferrosa_user@127.0.0.1:18765/mcp",
                "default": "http://ferrosa_user:ferrosa_user@127.0.0.1:18765/mcp",
                "required": True,
            },
            {
                "key": "tenant_id",
                "description": "Tenant ID for multi-tenant deployments (optional)",
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        _save_config(values, hermes_home)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = str(kwargs.get("hermes_home", "~/.hermes"))
        self._saved_config = _load_saved_config(self._hermes_home)
        self._url = _resolve_url(self._saved_config)
        if not self._url:
            logger.warning("ferrosa-memory: no URL configured; skipping activation")
            return
        self._client = _McpClient(self._url)
        tenant = os.environ.get("FERROSA_MEMORY_TENANT_ID", "")
        if tenant:
            self._tenant_id = tenant
        else:
            self._tenant_id = self._saved_config.get("tenant_id", "")
        logger.info("ferrosa-memory: connected to %s", self._url.rsplit("@", 1)[-1] if "@" in self._url else self._url)

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        try:
            stats = self._client.call("get_stats", {})
            e = stats.get("entity_count", 0)
            f = stats.get("fold_count", 0)
            if e == 0 and f == 0:
                return "# Ferrosa Memory\nActive. No memories yet — use `smart_ingest` via the MCP tools to save persistent knowledge.\n"
            return f"# Ferrosa Memory\nActive. {e} entities, {f} folds indexed across sessions.\n"
        except Exception:
            return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._client or not query:
            return ""
        effective_session = self._effective_session_id(session_id)
        segment_session = self._context_segment_session_id(effective_session)
        try:
            segment_args: Dict[str, Any] = {
                "query": query,
                "limit": 5,
                "expand": {"prev": 1, "next": 2, "max_tokens": 4000},
            }
            if segment_session:
                segment_args["session_id"] = segment_session
                if effective_session and effective_session != segment_session:
                    segment_args["source_session_id"] = effective_session
            segment_result = self._client.call("search_context_segments", segment_args)
            segment_hits = segment_result.get("results", []) if isinstance(segment_result, dict) else []
            if not segment_hits and segment_session:
                broad_args = dict(segment_args)
                broad_args.pop("session_id", None)
                broad_args.pop("source_session_id", None)
                segment_result = self._client.call("search_context_segments", broad_args)
                segment_hits = segment_result.get("results", []) if isinstance(segment_result, dict) else []
            if segment_hits:
                lines = ["## Ferrosa Memory Context Segments"]
                for hit in segment_hits[:5]:
                    expanded = hit.get("expanded_context") if isinstance(hit, dict) else None
                    if not expanded and isinstance(hit, dict):
                        segment = hit.get("segment", {}) if isinstance(hit.get("segment"), dict) else {}
                        segment_id = str(segment.get("segment_id") or "").strip()
                        if segment_id and segment_session:
                            window_args: Dict[str, Any] = {
                                "session_id": segment_session,
                                "segment_id": segment_id,
                                "prev": 1,
                                "next": 2,
                                "max_tokens": 4000,
                            }
                            if effective_session and effective_session != segment_session:
                                window_args["source_session_id"] = effective_session
                            try:
                                window_result = self._client.call("get_context_window", window_args)
                                if isinstance(window_result, dict):
                                    expanded = window_result.get("segments", [])
                            except Exception as e:
                                logger.debug("ferrosa context window expansion failed: %s", e)
                    if expanded:
                        for item in expanded[:6]:
                            segment = item.get("segment", {}) if isinstance(item, dict) else {}
                            text = str(segment.get("segment_text") or "").strip()
                            direction = str(item.get("direction") or "context").strip()
                            if text:
                                lines.append(f"- [{direction}] {text[:500]}")
                    else:
                        segment = hit.get("segment", {}) if isinstance(hit, dict) else {}
                        text = str(segment.get("segment_text") or "").strip()
                        if text:
                            lines.append(f"- {text[:500]}")
                if len(lines) > 1:
                    return "\n".join(lines)
        except Exception as e:
            logger.debug("ferrosa context segment prefetch failed: %s", e)
        try:
            # Use hybrid_search for broad recall fallback.
            result = self._client.call("hybrid_search", {
                "query": query,
                "limit": 5,
            })
            results = result.get("results", []) if isinstance(result, dict) else []
            if not results:
                return ""
            lines = ["## Ferrosa Memory"]
            for r in results:
                name = r.get("entity_name", "?")
                etype = r.get("entity_type", "?")
                content = r.get("content", "")[:250]
                lines.append(f"- [{etype}] {name}: {content}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug("ferrosa prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._client:
            return
        try:
            self._client.call("smart_ingest", {
                "content": f"User turn in session {session_id or self._session_id}: {user_content[:800]}",
                "entity_type": "conversation_turn",
                "entity_name": f"user-{session_id or self._session_id}",
                "session_id": session_id or self._session_id,
            })
        except Exception as e:
            logger.debug("ferrosa sync_turn user failed: %s", e)
        try:
            self._client.call("smart_ingest", {
                "content": f"Assistant turn in session {session_id or self._session_id}: {assistant_content[:800]}",
                "entity_type": "conversation_turn",
                "entity_name": f"assistant-{session_id or self._session_id}",
                "session_id": session_id or self._session_id,
            })
        except Exception as e:
            logger.debug("ferrosa sync_turn assistant failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # We do NOT replicate fmem tools here — they are already exposed
        # as MCP tools via the ferrosa-memory MCP server. This provider is
        # context-only: prefetch + sync_turn.
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        return json.dumps({"error": "ferrosa memory provider has no provider-local tools; use the fmem MCP tools directly."})

    def _effective_session_id(self, session_id: str = "") -> str:
        return str(session_id or self._session_id or "").strip()

    @staticmethod
    def _context_segment_session_id(session_id: str) -> str:
        """Return the UUID session key required by fmem context segment APIs.

        Hermes session identifiers are often timestamp/slugs (Discord/TUI), while
        fmem's first-class context segment storage is UUID-keyed. Keep real UUIDs
        unchanged; deterministically map non-UUID Hermes ids so ingest/search/window
        all address the same partition.
        """
        raw = str(session_id or "").strip()
        if not raw:
            return ""
        try:
            return str(uuid.UUID(raw))
        except (TypeError, ValueError):
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hermes-context-segments:{raw}"))

    @staticmethod
    def _entity_id_from_result(result: Any) -> str:
        if isinstance(result, dict):
            return str(result.get("entity_id") or result.get("id") or "").strip()
        return ""

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return " ".join(p for p in parts if p).strip()
        return str(content or "").strip()

    @staticmethod
    def _message_text(message: Dict[str, Any]) -> str:
        role = str(message.get("role") or "unknown")
        content_text = FerrosaMemoryProvider._content_text(message.get("content", ""))
        if not content_text:
            return ""
        return f"{role}: {content_text}"

    def _context_segment_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for index, message in enumerate(messages):
            content_text = self._content_text(message.get("content", ""))
            if not content_text:
                continue
            row: Dict[str, Any] = {
                "role": str(message.get("role") or "unknown"),
                "content": content_text,
                "turn_index": int(message.get("turn_index", index) or index),
            }
            created_at = message.get("created_at") or message.get("timestamp")
            if created_at:
                row["created_at"] = str(created_at)
            metadata = message.get("metadata")
            if isinstance(metadata, dict):
                row["metadata"] = metadata
            rows.append(row)
        return rows

    def _conversation_transcript(self, messages: List[Dict[str, Any]], *, limit: int = 6000) -> str:
        lines = [text for msg in messages for text in [self._message_text(msg)] if text]
        return "\n".join(lines)[-limit:]

    def _semantic_context_chunks(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_messages: int = 2,
        max_chars: int = 4000,
    ) -> List[str]:
        """Chunk raw context on message boundaries for later page-style traversal."""
        chunks: List[str] = []
        current: List[str] = []
        current_chars = 0
        for msg in messages:
            text = self._message_text(msg)
            if not text:
                continue
            would_exceed_messages = len(current) >= max_messages
            would_exceed_chars = current and current_chars + len(text) + 1 > max_chars
            if would_exceed_messages or would_exceed_chars:
                chunks.append("\n".join(current))
                current = []
                current_chars = 0
            current.append(text)
            current_chars += len(text) + 1
        if current:
            chunks.append("\n".join(current))
        return chunks

    def _ensure_session_entity(self, session_id: str, messages: List[Dict[str, Any]]) -> str:
        if not self._client:
            return ""
        transcript = self._conversation_transcript(messages, limit=1000)
        result = self._client.call("smart_ingest", {
            "content": f"Hermes session {session_id} transcript summary/context.\n{transcript}",
            "entity_type": "event",
            "entity_name": f"Hermes session {session_id}",
        })
        return self._entity_id_from_result(result)

    def _record_session_temporal_fact(self, session_id: str, messages: List[Dict[str, Any]], fact_text: str) -> None:
        if not self._client or not session_id or not messages:
            return
        try:
            entity_id = self._ensure_session_entity(session_id, messages)
            if not entity_id:
                return
            self._client.call("write_temporal_fact", {
                "entity_id": entity_id,
                "fact_text": fact_text,
                "session_id": session_id,
            })
            # Verify the write/read path while still in the same session scope.
            self._client.call("get_temporal_chain", {
                "entity_id": entity_id,
                "session_id": session_id,
            })
        except Exception as e:
            logger.debug("ferrosa temporal fact recording failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._client or not messages:
            return ""
        session_id = self._effective_session_id()
        segment_session = self._context_segment_session_id(session_id)
        if not session_id or not segment_session:
            return ""
        segment_messages = self._context_segment_messages(messages)
        if not segment_messages:
            return ""
        segment_payload: Dict[str, Any] = {
            "session_id": segment_session,
            "source_session_id": session_id,
            "conversation_id": f"hermes-{session_id}",
            "messages": segment_messages,
            "embed_missing": True,
            "segmentation": {
                "strategy": "deterministic_v1",
                "target_tokens": 1000,
                "max_tokens": 1800,
                "time_gap_seconds": 900,
                "semantic_drift_threshold": 0.72,
            },
        }
        try:
            try:
                result = self._client.call("ingest_context_segments", segment_payload)
            except Exception as first_error:
                # If the embedding service is down, still persist lexical/BM25
                # context pages and temporal links. The fmem search path can use
                # context_segment_terms immediately and ANN can be backfilled later.
                retry_payload = dict(segment_payload)
                retry_payload["embed_missing"] = False
                logger.debug(
                    "ferrosa context segment ingest with embeddings failed; retrying lexical-only: %s",
                    first_error,
                )
                result = self._client.call("ingest_context_segments", retry_payload)
            segments_created = int(result.get("segments_created", 0)) if isinstance(result, dict) else 0
            segments_skipped = int(result.get("segments_skipped", 0)) if isinstance(result, dict) else 0
            total_segments = segments_created + segments_skipped
            if total_segments:
                self._record_session_temporal_fact(
                    session_id,
                    messages,
                    f"Context compacted: persisted {total_segments} context segments for temporal page traversal",
                )
                return f"Ferrosa Memory: Persisted {total_segments} pre-compression context segments."
        except Exception as e:
            logger.debug("ferrosa context segment ingest failed; falling back to entity chunks: %s", e)

        chunks = self._semantic_context_chunks(messages)
        if not chunks:
            return ""
        chunk_ids: List[str] = []
        try:
            self._ensure_session_entity(session_id, messages)
            for index, chunk in enumerate(chunks, 1):
                result = self._client.call("smart_ingest", {
                    "content": chunk,
                    "entity_type": "section",
                    "entity_name": f"Hermes session {session_id} context chunk {index:04d}",
                })
                entity_id = self._entity_id_from_result(result)
                if entity_id:
                    chunk_ids.append(entity_id)
            for index, (src, dst) in enumerate(zip(chunk_ids, chunk_ids[1:]), 1):
                self._client.call("create_edge", {
                    "src_entity_id": src,
                    "dst_entity_id": dst,
                    "edge_type": "related_to",
                    "metadata": f"next_context_chunk page={index} session_id={session_id}",
                    "session_id": session_id,
                })
                self._client.call("create_edge", {
                    "src_entity_id": dst,
                    "dst_entity_id": src,
                    "edge_type": "related_to",
                    "metadata": f"previous_context_chunk page={index + 1} session_id={session_id}",
                    "session_id": session_id,
                })
            if chunk_ids:
                self._record_session_temporal_fact(
                    session_id,
                    messages,
                    f"Context compacted: persisted {len(chunk_ids)} semantic chunks for page traversal",
                )
        except Exception as e:
            logger.debug("ferrosa pre-compress chunk persistence failed: %s", e)
            return ""
        return f"Ferrosa Memory: Persisted {len(chunk_ids)} pre-compression context chunks linked in order."

    def _run_consolidation_async(self, args: Dict[str, Any]) -> None:
        if not self._client:
            return

        def worker() -> None:
            try:
                self._client.call("run_consolidation", args)
                logger.info("ferrosa-memory: ran session-end consolidation")
            except Exception as e:
                logger.debug("ferrosa consolidation failed: %s", e)

        threading.Thread(
            target=worker,
            name="ferrosa-memory-session-end-consolidation",
            daemon=True,
        ).start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._client:
            return
        session_id = self._effective_session_id()
        if session_id:
            self._record_session_temporal_fact(
                session_id,
                messages,
                f"Session completed: {len(messages or [])} messages captured for durable temporal recall",
            )
        args = {"session_id": session_id} if session_id else {}
        self._run_consolidation_async(args)

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self._client or action != "add" or not content:
            return
        try:
            self._client.call("smart_ingest", {
                "content": content,
                "entity_type": target if target in ("memory", "user") else "memory",
                "session_id": self._session_id,
            })
        except Exception as e:
            logger.debug("ferrosa on_memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        self._client = None


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register ferrosa memory and virtual fmem skill providers."""
    ctx.register_memory_provider(FerrosaMemoryProvider())
    if hasattr(ctx, "register_skill_provider"):
        ctx.register_skill_provider(FerrosaSkillProvider(), namespace="fmem")

