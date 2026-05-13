"""Local operational memory context engine.

This bundled context-engine plugin keeps durable operational context in a
local SQLite side store and exposes retrieval tools for the agent. It is
inactive until selected with ``context.engine: operational_lcm``.

The engine is intentionally local-only: it reads Hermes-owned local state
surfaces such as Kanban task metadata, sanitized session summaries, handoffs,
daily notes, progress notes, wiki pages, live message snippets, artifact paths,
and hashes. It does not make network calls and it avoids indexing secret-shaped
content or credential/runtime files.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent.context_engine import ContextEngine

_MAX_TEXT_CHARS = 6000
_MAX_FILE_BYTES = 160_000
_DEFAULT_CONTEXT_LENGTH = 200_000
_DEFAULT_THRESHOLD_PERCENT = 0.78

_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "password",
    "passwd",
    "secret",
    "bearer ",
    "private key",
    "auth.json",
    ".env",
    "credential",
    "totp",
    "recovery code",
)
_DENIED_PATH_PARTS = {
    "auth.json",
    ".env",
    "credentials",
    "sessions",
    "logs",
    "browser_profiles",
}
_SECRET_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
_HASH_RE = re.compile(r"\b(?:sha256:|sha256=|sha:|sha=)?[a-fA-F0-9]{6,64}\b")
_PATH_RE = re.compile(r"(?:/Users/[^\s`'\"]+|~?/[^\s`'\"]+|[A-Za-z0-9_.-]+/[A-Za-z0-9_./@+=:-]+)")
_TASK_RE = re.compile(r"\bt_[a-fA-F0-9]{6,12}\b")
_DECISION_RE = re.compile(r"\b(decision|decided|recommendation|verdict|approved|blocked|supersedes)\b", re.I)
_ARTIFACT_RE = re.compile(r"\b(artifact|sha256|hash|receipt|report|handoff|\.md|\.json|\.py|\.yaml|\.yml|\.toml)\b", re.I)

TOOL_NAMES = (
    "operational_context_search",
    "operational_context_get_task",
    "operational_context_recent_decisions",
    "operational_context_artifacts",
    "operational_context_status",
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif "content" in item:
                    parts.append(str(item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _safe_text(text: Any) -> bool:
    s = "" if text is None else str(text)
    low = s.lower()
    if any(marker in low for marker in _SECRET_MARKERS):
        return False
    if re.search(r"\b[A-Za-z0-9_]*(?:token|secret|password|key)\s*[:=]", low):
        return False
    return not any(pattern.search(s) for pattern in _SECRET_PATTERNS)


def _safe_path(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return not bool(parts & _DENIED_PATH_PARTS)


def _clip(text: Any, limit: int = _MAX_TEXT_CHARS) -> str:
    s = "" if text is None else str(text)
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s[:limit]


def _safe_display(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    s = str(value)
    if not s:
        return fallback
    return _clip(s, 1200) if _safe_text(s) else fallback


def _extract_hashes(text: str) -> str:
    vals: List[str] = []
    for match in _HASH_RE.findall(text or ""):
        val = str(match)
        if val and val not in vals:
            vals.append(val)
    return ",".join(vals[:12])


def _extract_paths(text: str) -> str:
    vals: List[str] = []
    for val in _PATH_RE.findall(text or ""):
        cleaned = val.rstrip(".,);]")
        if not cleaned:
            continue
        if not _safe_text(cleaned):
            continue
        if cleaned not in vals:
            vals.append(cleaned)
    return ",".join(vals[:20])


def _extract_task_id(text: str) -> str:
    match = _TASK_RE.search(text or "")
    return match.group(0) if match else ""


def _safe_json_text(raw: Any, max_chars: int = 4000) -> str:
    """Render structured metadata while dropping secret-shaped keys/values."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw
        try:
            raw = json.loads(raw)
        except Exception:
            return _clip(text, max_chars) if _safe_text(text) else ""

    def clean(value: Any, depth: int = 0) -> Any:
        if depth > 5:
            return "..."
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for key, val in value.items():
                key_text = str(key)[:80]
                if not key_text or not _safe_text(key_text):
                    continue
                if any(marker.strip().lower() in key_text.lower() for marker in _SECRET_MARKERS):
                    continue
                cleaned = clean(val, depth + 1)
                if cleaned not in ({}, [], ""):
                    out[key_text] = cleaned
            return out
        if isinstance(value, list):
            return [clean(item, depth + 1) for item in value[:40] if clean(item, depth + 1) not in ({}, [], "")]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        s = str(value)
        return _clip(s, 500) if _safe_text(s) else "[redacted]"

    try:
        rendered = json.dumps(clean(raw), ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered = str(raw)
    return _clip(rendered, max_chars) if _safe_text(rendered) else ""


@dataclass
class ContextRecord:
    source_type: str
    source_id: str
    title: str
    text: str
    tags: str = ""
    task_id: str = ""
    artifact_paths: str = ""
    hashes: str = ""


class OperationalLCMContextEngine(ContextEngine):
    """Local-only structured operational context engine."""

    threshold_percent = _DEFAULT_THRESHOLD_PERCENT
    protect_first_n = 3
    protect_last_n = 35

    def __init__(self, hermes_home: Optional[str] = None, context_length: int = _DEFAULT_CONTEXT_LENGTH) -> None:
        self.hermes_home = Path(hermes_home).expanduser() if hermes_home else _hermes_home()
        self.context_length = int(context_length or _DEFAULT_CONTEXT_LENGTH)
        self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self.session_id = ""
        self.platform = ""
        self.model = ""
        self._store_path = self.hermes_home / "data" / "operational-lcm" / "context.db"
        self._last_ingest_count = 0

    @property
    def name(self) -> str:
        return "operational_lcm"

    def is_available(self) -> bool:
        return True

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        usage = usage or {}
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or self.last_prompt_tokens or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or self.last_completion_tokens or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or (self.last_prompt_tokens + self.last_completion_tokens))

    def update_model(self, model: str, context_length: int, base_url: str = "", provider: str = "", **kwargs: Any) -> None:
        self.model = model or self.model
        self.context_length = int(context_length or self.context_length or _DEFAULT_CONTEXT_LENGTH)
        self.threshold_tokens = int(self.context_length * self.threshold_percent)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = int(prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens or 0)
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        char_count = sum(len(_stringify_content(message.get("content"))) for message in messages or [])
        return char_count > int(self.threshold_tokens * 2.5)

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return bool(messages and len(messages) > (self.protect_first_n + self.protect_last_n))

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id or ""
        self.hermes_home = Path(kwargs.get("hermes_home") or self.hermes_home).expanduser()
        self.platform = kwargs.get("platform") or self.platform
        self.model = kwargs.get("model") or self.model
        if kwargs.get("context_length"):
            self.context_length = int(kwargs["context_length"])
            self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self._store_path = self.hermes_home / "data" / "operational-lcm" / "context.db"
        self._ensure_store()
        self._last_ingest_count = self._ingest_operational_surfaces()

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._ingest_messages(messages or [], source_id=session_id or self.session_id or "session-end")

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None, focus_topic: str = None) -> List[Dict[str, Any]]:
        messages = list(messages or [])
        self._ensure_store()
        self._ingest_messages(messages, source_id=self.session_id or "active-session")
        self.compression_count += 1
        query = focus_topic or self._infer_focus(messages)
        records = self._search(query=query, limit=8) if query else self._recent(limit=8)
        summary = self._summary_message(records, query)
        if not messages:
            return [{"role": "assistant", "content": summary}]
        head = messages[: self.protect_first_n]
        tail = messages[-self.protect_last_n :] if len(messages) > self.protect_last_n else messages[self.protect_first_n :]
        summary_role = "assistant"
        if head and head[-1].get("role") == "assistant":
            summary_role = "user"
        return self._dedupe_adjacent_summary(head + [{"role": summary_role, "content": summary}] + tail)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {"name": "operational_context_search", "description": "Search local structured operational context records.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 8}}, "required": ["query"]}},
            {"name": "operational_context_get_task", "description": "Get one local Kanban task context record by task id.", "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}},
            {"name": "operational_context_recent_decisions", "description": "Return recent decision, verdict, and recommendation records.", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 8}}}},
            {"name": "operational_context_artifacts", "description": "Return artifact, file, and hash evidence records.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "default": ""}, "limit": {"type": "integer", "default": 8}}}},
            {"name": "operational_context_status", "description": "Return local operational context engine status.", "parameters": {"type": "object", "properties": {}}},
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        args = args or {}
        self._ensure_store()
        live_messages = kwargs.get("messages")
        if isinstance(live_messages, list):
            self._ingest_messages(live_messages, source_id=(self.session_id or "live-tool-context"))
        if name == "operational_context_search":
            return json.dumps({"ok": True, "results": self._search(str(args.get("query") or ""), int(args.get("limit") or 8))})
        if name == "operational_context_get_task":
            task = self._get_task(str(args.get("task_id") or ""))
            return json.dumps({"ok": bool(task), "task": task})
        if name == "operational_context_recent_decisions":
            return json.dumps({"ok": True, "decisions": self._decisions(int(args.get("limit") or 8))})
        if name == "operational_context_artifacts":
            return json.dumps({"ok": True, "artifacts": self._artifacts(str(args.get("query") or ""), int(args.get("limit") or 8))})
        if name == "operational_context_status":
            return json.dumps({"ok": True, "status": self.get_status()})
        return json.dumps({"ok": False, "error": f"unknown tool {name}"})

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "engine": self.name,
            "store_path": str(self._store_path),
            "last_ingest_count": self._last_ingest_count,
            "session_id": self.session_id,
            "local_only": True,
            "tools": list(TOOL_NAMES),
        })
        try:
            with sqlite3.connect(str(self._store_path)) as conn:
                status["record_count"] = conn.execute("select count(*) from records").fetchone()[0]
        except Exception:
            status["record_count"] = 0
        return status

    def _ensure_store(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._store_path)) as conn:
            conn.execute("""
                create table if not exists records (
                    record_key text primary key,
                    source_type text not null,
                    source_id text not null,
                    title text,
                    text text not null,
                    tags text,
                    task_id text,
                    artifact_paths text,
                    hashes text,
                    updated_at text not null
                )
            """)
            conn.execute("create index if not exists idx_records_source on records(source_type, source_id)")
            conn.execute("create index if not exists idx_records_task on records(task_id)")
            conn.execute("create index if not exists idx_records_updated on records(updated_at)")

    def _upsert(self, rec: ContextRecord) -> bool:
        if not rec.text or not _safe_text(rec.text):
            return False
        text = _clip(rec.text)
        source_type = _safe_display(rec.source_type, "unknown")
        source_id = _safe_display(rec.source_id, "redacted-source")
        title = _safe_display(rec.title, "redacted-title")
        tags = _safe_display(rec.tags, "")
        task_id = _safe_display(rec.task_id, "")
        artifact_paths = _safe_display(rec.artifact_paths or _extract_paths(text), "")
        hashes = _safe_display(rec.hashes or _extract_hashes(text), "")
        key = f"{source_type}:{source_id}:{title}"[:512]
        with sqlite3.connect(str(self._store_path)) as conn:
            conn.execute(
                """
                insert into records(record_key, source_type, source_id, title, text, tags, task_id, artifact_paths, hashes, updated_at)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(record_key) do update set
                    text=excluded.text,
                    tags=excluded.tags,
                    task_id=excluded.task_id,
                    artifact_paths=excluded.artifact_paths,
                    hashes=excluded.hashes,
                    updated_at=excluded.updated_at
                """,
                (key, source_type, source_id, title, text, tags, task_id, artifact_paths, hashes, _now()),
            )
        return True

    def _ingest_operational_surfaces(self) -> int:
        count = 0
        count += self._ingest_kanban()
        count += self._ingest_state_db_sessions()
        for rel, source_type in (("memories/handoffs", "handoff"), ("memories/daily", "daily"), ("memories/progress", "progress"), ("wiki", "wiki")):
            count += self._ingest_markdown_dir(self.hermes_home / rel, source_type)
        return count

    def _ingest_kanban(self) -> int:
        db = self.hermes_home / "kanban.db"
        if not db.exists():
            return 0
        count = 0
        try:
            with sqlite3.connect(str(db)) as conn:
                conn.row_factory = sqlite3.Row
                tables = {row[0] for row in conn.execute("select name from sqlite_master where type='table'").fetchall()}
                if "tasks" in tables:
                    cols = {row[1] for row in conn.execute("pragma table_info(tasks)").fetchall()}
                    order_candidates = [c for c in ("completed_at", "started_at", "created_at") if c in cols]
                    order_expr = "coalesce(" + ", ".join(order_candidates) + ") desc" if order_candidates else "rowid desc"
                    for row in conn.execute(f"select * from tasks order by {order_expr} limit 250").fetchall():
                        data = dict(row)
                        task_id = str(data.get("id") or "")
                        text = "\n".join(str(data.get(k) or "") for k in ("title", "body", "result", "status", "assignee") if k in data)
                        if self._upsert(ContextRecord("kanban", task_id, str(data.get("title") or task_id), text, tags=str(data.get("status") or ""), task_id=task_id)):
                            count += 1
                if "task_runs" in tables:
                    for row in conn.execute("select * from task_runs order by coalesce(ended_at, started_at, id) desc limit 300").fetchall():
                        data = dict(row)
                        task_id = str(data.get("task_id") or "")
                        metadata = _safe_json_text(data.get("metadata"))
                        text = "\n".join(x for x in [
                            f"task_id={task_id}",
                            f"profile={data.get('profile') or ''}",
                            f"status={data.get('status') or ''}",
                            f"outcome={data.get('outcome') or ''}",
                            str(data.get("summary") or ""),
                            metadata,
                        ] if x)
                        tags = "run,decision" if _DECISION_RE.search(text) else "run"
                        if _ARTIFACT_RE.search(text):
                            tags += ",artifact"
                        if self._upsert(ContextRecord("kanban_run", f"{task_id}:{data.get('id')}", f"run {data.get('id')} {task_id}", text, tags=tags, task_id=task_id)):
                            count += 1
                if "task_comments" in tables:
                    for row in conn.execute("select * from task_comments order by created_at desc limit 300").fetchall():
                        data = dict(row)
                        task_id = str(data.get("task_id") or "")
                        text = "\n".join(str(data.get(k) or "") for k in ("author", "body") if k in data)
                        tags = "comment,decision" if _DECISION_RE.search(text) else "comment"
                        if _ARTIFACT_RE.search(text):
                            tags += ",artifact"
                        if self._upsert(ContextRecord("kanban_comment", f"{task_id}:{data.get('id')}", f"comment {data.get('id')} {task_id}", text, tags=tags, task_id=task_id)):
                            count += 1
                if "task_events" in tables:
                    for row in conn.execute("select * from task_events order by created_at desc limit 300").fetchall():
                        data = dict(row)
                        task_id = str(data.get("task_id") or "")
                        payload = _safe_json_text(data.get("payload"))
                        text = "\n".join(x for x in [f"kind={data.get('kind') or ''}", payload] if x)
                        tags = "event,decision" if _DECISION_RE.search(text) else "event"
                        if _ARTIFACT_RE.search(text):
                            tags += ",artifact"
                        if self._upsert(ContextRecord("kanban_event", f"{task_id}:{data.get('id')}", f"event {data.get('kind') or ''} {task_id}", text, tags=tags, task_id=task_id)):
                            count += 1
        except Exception:
            return count
        return count

    def _ingest_state_db_sessions(self) -> int:
        db = self.hermes_home / "state.db"
        if not db.exists():
            return 0
        count = 0
        try:
            with sqlite3.connect(str(db)) as conn:
                conn.row_factory = sqlite3.Row
                tables = {row[0] for row in conn.execute("select name from sqlite_master where type='table'").fetchall()}
                if "sessions" not in tables:
                    return 0
                cols = {row[1] for row in conn.execute("pragma table_info(sessions)").fetchall()}
                safe_cols = [c for c in ("id", "source", "model", "parent_session_id", "started_at", "ended_at", "end_reason", "message_count", "tool_call_count", "input_tokens", "output_tokens", "title", "api_call_count") if c in cols]
                for row in conn.execute("select " + ",".join(safe_cols) + " from sessions order by started_at desc limit 200").fetchall():
                    data = dict(row)
                    session_id = str(data.get("id") or "")
                    title = str(data.get("title") or session_id)
                    fields = []
                    for key in safe_cols:
                        if key == "id":
                            continue
                        val = data.get(key)
                        if val is not None and _safe_text(str(val)):
                            fields.append(f"{key}={val}")
                    text = "\n".join(fields)
                    tags = "session_summary"
                    if _DECISION_RE.search(text):
                        tags += ",decision"
                    if self._upsert(ContextRecord("session_summary", session_id, title, text, tags=tags)):
                        count += 1
        except Exception:
            return count
        return count

    def _ingest_markdown_dir(self, root: Path, source_type: str) -> int:
        if not root.exists():
            return 0
        count = 0
        for path in sorted(root.rglob("*.md"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:200]:
            try:
                if not _safe_path(path) or path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = path.read_text(errors="ignore")
                if not _safe_text(text):
                    continue
                tags: List[str] = []
                if _DECISION_RE.search(text):
                    tags.append("decision")
                if _ARTIFACT_RE.search(text):
                    tags.append("artifact")
                task_id = _extract_task_id(text)
                if self._upsert(ContextRecord(source_type, str(path.relative_to(self.hermes_home)), path.name, text, tags=",".join(tags), task_id=task_id)):
                    count += 1
            except Exception:
                continue
        return count

    def _ingest_messages(self, messages: List[Dict[str, Any]], source_id: str) -> int:
        count = 0
        for idx, message in enumerate(messages[-80:]):
            text = _stringify_content(message.get("content"))
            if not text or not _safe_text(text):
                continue
            tags = []
            if _DECISION_RE.search(text):
                tags.append("decision")
            if _ARTIFACT_RE.search(text):
                tags.append("artifact")
            if not tags and not _extract_task_id(text):
                continue
            if self._upsert(ContextRecord("session", f"{source_id}:{idx}", str(message.get("role") or "message"), text, tags=",".join(tags), task_id=_extract_task_id(text))):
                count += 1
        return count

    def _infer_focus(self, messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages[-10:]):
            text = _stringify_content(message.get("content"))
            if text and _safe_text(text):
                return text[:500]
        return ""

    def _rows(self, sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with sqlite3.connect(str(self._store_path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]

    def _format_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{
            "source_type": _safe_display(row.get("source_type"), "unknown"),
            "source_id": _safe_display(row.get("source_id"), "redacted-source"),
            "title": _safe_display(row.get("title"), "redacted-title"),
            "task_id": _safe_display(row.get("task_id") or "", ""),
            "tags": _safe_display(row.get("tags") or "", ""),
            "artifact_paths": _safe_display(row.get("artifact_paths") or "", ""),
            "hashes": _safe_display(row.get("hashes") or "", ""),
            "text": _clip(row.get("text") or "", 1200) if _safe_text(row.get("text") or "") else "",
            "updated_at": row.get("updated_at"),
        } for row in rows]

    def _search(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query or not _safe_text(query):
            return []
        terms = [term.lower() for term in re.findall(r"[A-Za-z0-9_./:-]{3,}", query)[:8]]
        rows = self._rows("select * from records order by updated_at desc limit 500")
        scored = []
        for row in rows:
            haystack = " ".join(str(row.get(key) or "") for key in ("title", "text", "tags", "task_id", "artifact_paths", "hashes")).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, row))
        scored.sort(key=lambda item: (item[0], item[1].get("updated_at") or ""), reverse=True)
        return self._format_rows([row for _, row in scored[: max(1, min(int(limit or 8), 25))]])

    def _recent(self, limit: int = 8) -> List[Dict[str, Any]]:
        return self._format_rows(self._rows("select * from records order by updated_at desc limit ?", (max(1, min(limit, 25)),)))

    def _get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        rows = self._rows("select * from records where source_type='kanban' and (task_id=? or source_id=?) order by updated_at desc limit 1", (task_id, task_id))
        if not rows:
            rows = self._rows("select * from records where task_id=? or source_id=? order by case source_type when 'kanban' then 0 else 1 end, updated_at desc limit 1", (task_id, task_id))
        return self._format_rows(rows)[0] if rows else None

    def _decisions(self, limit: int = 8) -> List[Dict[str, Any]]:
        rows = self._rows("select * from records where tags like '%decision%' or lower(text) like '%decision%' or lower(text) like '%verdict%' order by updated_at desc limit ?", (max(1, min(limit, 25)),))
        return self._format_rows(rows)

    def _artifacts(self, query: str = "", limit: int = 8) -> List[Dict[str, Any]]:
        if query:
            return [row for row in self._search(query, limit=limit) if row.get("artifact_paths") or row.get("hashes") or "artifact" in row.get("tags", "")]
        rows = self._rows("select * from records where tags like '%artifact%' or artifact_paths!='' or hashes!='' order by updated_at desc limit ?", (max(1, min(limit, 25)),))
        return self._format_rows(rows)

    def _summary_message(self, records: List[Dict[str, Any]], query: str) -> str:
        lines = [
            "[Operational structured context] Earlier operational context was compacted into a local structured store and can be retrieved with operational_context_* tools.",
            f"Session: {self.session_id or 'unknown'} | Store: {self._store_path}",
        ]
        if query:
            lines.append(f"Focus: {_clip(query, 300)}")
        if records:
            lines.append("Relevant records:")
            for record in records[:8]:
                label = f"- {record.get('source_type')}:{record.get('source_id')}"
                task = f" task={record.get('task_id')}" if record.get("task_id") else ""
                hashes = f" hashes={record.get('hashes')}" if record.get("hashes") else ""
                lines.append(f"{label}{task}{hashes} — {record.get('title') or ''}: {_clip(record.get('text') or '', 260)}")
        else:
            lines.append("No matching structured records yet; use operational_context_search when needed.")
        return "\n".join(lines)

    def _dedupe_adjacent_summary(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for message in messages:
            if cleaned and cleaned[-1].get("role") == message.get("role") and "Operational structured context" in str(message.get("content", "")):
                message = dict(message)
                message["role"] = "user" if message.get("role") == "assistant" else "assistant"
            cleaned.append(message)
        return cleaned


def register(ctx) -> None:
    ctx.register_context_engine(OperationalLCMContextEngine())
