"""Utilities for building redacted Hermes operator fine-tuning corpora.

The helpers in this module are intentionally file/SQLite based and do not
import the live Hermes runtime. They are safe to use against a copied or live
``state.db`` because callers open SQLite in read-only mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from agent.redact import redact_sensitive_text

SCHEMA_VERSION = "hermes.training.corpus.v1"
SFT_SCHEMA_VERSION = "hermes.operator.sft.v1"
DPO_SCHEMA_VERSION = "hermes.operator.dpo.v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are a Hermes operator assistant. Keep personal facts in memory/RAG, "
    "learn reusable operational procedure, verify tool effects, and require "
    "confirmation before write, publish, destructive, or external actions."
)

_WINDOWS_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z]:[\\/](?:[^\\/\s\"'<>|]+[\\/])*[^\\/\s\"'<>|]*)"
)
_POSIX_HOME_RE = re.compile(r"(?<![A-Za-z0-9_])(/home/[^/\s\"']+|/Users/[^/\s\"']+)")
_NGROK_URL_RE = re.compile(
    r"https?://[A-Za-z0-9-]+\.(?:ngrok-free\.app|ngrok\.io)(?:/[^\s\"']*)?",
    re.IGNORECASE,
)
_SESSIONISH_RE = re.compile(
    r"\b(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|"
    r"019[0-9a-f]{29}|[0-9a-f]{24,64})\b",
    re.IGNORECASE,
)


def _json_or_text(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sqlite_readonly_uri(path: Path) -> str:
    resolved = path.expanduser().resolve()
    return f"file:{resolved.as_posix()}?mode=ro"


def iter_state_db_records(
    db_path: Path,
    *,
    limit_sessions: int = 200,
    since_days: int | None = None,
    source: str | None = None,
    include_system_prompt: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield session records from Hermes ``state.db`` without mutating it."""

    cutoff = None
    if since_days is not None:
        cutoff = time.time() - since_days * 86400

    where = ["1=1"]
    params: list[Any] = []
    if cutoff is not None:
        where.append("started_at >= ?")
        params.append(cutoff)
    if source:
        where.append("source = ?")
        params.append(source)
    params.append(max(1, limit_sessions))

    conn = sqlite3.connect(_sqlite_readonly_uri(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        sessions = conn.execute(
            f"""
            SELECT id, source, user_id, model, model_config, system_prompt,
                   parent_session_id, started_at, ended_at, end_reason,
                   message_count, tool_call_count, cwd, title, api_call_count
            FROM sessions
            WHERE {' AND '.join(where)}
            ORDER BY started_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        for session in sessions:
            messages = conn.execute(
                """
                SELECT role, content, tool_call_id, tool_calls, tool_name,
                       timestamp, token_count, finish_reason, active
                FROM messages
                WHERE session_id = ? AND active = 1
                ORDER BY timestamp ASC, id ASC
                """,
                (session["id"],),
            ).fetchall()
            if not messages:
                continue

            session_payload = {
                "id": session["id"],
                "source": session["source"],
                "user_id": session["user_id"],
                "model": session["model"],
                "model_config": _json_or_text(session["model_config"]),
                "parent_session_id": session["parent_session_id"],
                "started_at": _safe_float(session["started_at"]),
                "ended_at": _safe_float(session["ended_at"]),
                "end_reason": session["end_reason"],
                "message_count": session["message_count"],
                "tool_call_count": session["tool_call_count"],
                "cwd": session["cwd"],
                "title": session["title"],
                "api_call_count": session["api_call_count"],
            }
            if include_system_prompt:
                session_payload["system_prompt"] = session["system_prompt"]

            yield {
                "schema": SCHEMA_VERSION,
                "redacted": False,
                "source": "state_db",
                "session": session_payload,
                "messages": [
                    {
                        "role": row["role"],
                        "content": row["content"],
                        "tool_call_id": row["tool_call_id"],
                        "tool_calls": _json_or_text(row["tool_calls"]),
                        "tool_name": row["tool_name"],
                        "timestamp": _safe_float(row["timestamp"]),
                        "token_count": row["token_count"],
                        "finish_reason": row["finish_reason"],
                    }
                    for row in messages
                ],
            }
    finally:
        conn.close()


def iter_log_records(log_paths: Iterable[Path], *, max_lines: int = 2000) -> Iterator[dict[str, Any]]:
    """Yield bounded log-line records for optional harness/gateway context."""

    for path in log_paths:
        expanded = path.expanduser()
        if not expanded.exists() or not expanded.is_file():
            continue
        emitted = 0
        with expanded.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                if emitted >= max_lines:
                    break
                text = line.rstrip("\r\n")
                if not text:
                    continue
                emitted += 1
                yield {
                    "schema": SCHEMA_VERSION,
                    "redacted": False,
                    "source": "log",
                    "log": {
                        "path": str(expanded),
                        "line": line_no,
                        "content": text,
                    },
                }


def iter_harness_result_records(result_paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield structured Harness result files as trainable operator records."""

    for path in result_paths:
        expanded = path.expanduser()
        if not expanded.exists() or not expanded.is_file():
            continue
        for index, result in enumerate(_iter_json_objects(expanded), start=1):
            if not isinstance(result, dict):
                continue
            status = _harness_status(result)
            title = result.get("name") or result.get("task") or result.get("id") or expanded.stem
            yield {
                "schema": SCHEMA_VERSION,
                "redacted": False,
                "source": "harness_result",
                "session": {
                    "id": f"{expanded.stem}:{index}",
                    "source": "harness_result",
                    "title": str(title),
                    "end_reason": "success" if status in {"ok", "pass", "passed", "success", "healthy"} else status,
                    "cwd": str(expanded.parent),
                },
                "messages": [
                    {
                        "role": "user",
                        "content": "Review this Hermes Harness execution result and decide the operator follow-up.",
                    },
                    {
                        "role": "assistant",
                        "content": _harness_operator_summary(result, status),
                    },
                ],
            }


def _iter_json_objects(path: Path) -> Iterator[Any]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return
    try:
        yield json.loads(text)
        return
    except json.JSONDecodeError:
        pass
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid Harness JSON/JSONL: {exc}") from exc


def _harness_status(result: Mapping[str, Any]) -> str:
    for key in ("status", "state", "result", "outcome", "health"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    ok = result.get("ok") or result.get("success") or result.get("healthy")
    if isinstance(ok, bool):
        return "success" if ok else "failed"
    return "unknown"


def _harness_operator_summary(result: Mapping[str, Any], status: str) -> str:
    compact = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return (
        f"Harness result status: {status}. Preserve the structured result for evidence, "
        f"verify any dependent service before declaring completion, and avoid exposing secrets. "
        f"Result: {compact}"
    )


def iter_codex_rollout_records(
    rollout_paths: Iterable[Path],
    *,
    max_events_per_rollout: int = 10000,
) -> Iterator[dict[str, Any]]:
    """Yield Codex rollout JSONL sessions as Hermes operator corpus records."""

    for path in rollout_paths:
        expanded = path.expanduser()
        if not expanded.exists() or not expanded.is_file():
            continue

        session_meta: dict[str, Any] = {
            "id": expanded.stem,
            "source_path": str(expanded),
            "started_at": None,
        }
        messages: list[dict[str, Any]] = []
        pending_assistant_calls: list[dict[str, Any]] = []
        emitted = 0

        with expanded.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if emitted >= max_events_per_rollout:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                emitted += 1
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                payload = event.get("payload")
                timestamp = _safe_float(event.get("timestamp"))
                if event_type == "session_meta" and isinstance(payload, dict):
                    session_meta.update(_extract_codex_session_meta(payload))
                    if timestamp is not None and session_meta.get("started_at") is None:
                        session_meta["started_at"] = timestamp
                    continue
                if event_type != "response_item" or not isinstance(payload, dict):
                    continue

                item_type = payload.get("type")
                if item_type == "message":
                    _flush_pending_tool_calls(messages, pending_assistant_calls, timestamp)
                    role = str(payload.get("role") or "").strip()
                    content = _codex_content_to_text(payload.get("content"))
                    if role in {"user", "assistant", "system"} and content:
                        messages.append({
                            "role": role,
                            "content": content,
                            "timestamp": timestamp,
                            "tool_calls": None,
                            "tool_call_id": None,
                            "tool_name": None,
                        })
                elif item_type in {"function_call", "custom_tool_call", "tool_search_call"}:
                    pending_assistant_calls.append(_codex_tool_call(payload))
                elif item_type in {"function_call_output", "custom_tool_call_output", "tool_search_output"}:
                    _flush_pending_tool_calls(messages, pending_assistant_calls, timestamp)
                    call_id = payload.get("call_id")
                    messages.append({
                        "role": "tool",
                        "content": _codex_output_to_text(payload),
                        "tool_call_id": call_id,
                        "tool_calls": None,
                        "tool_name": _tool_name_for_call_id(messages, str(call_id) if call_id else ""),
                        "timestamp": timestamp,
                    })
                elif item_type == "web_search_call":
                    pending_assistant_calls.append(_codex_tool_call(payload))

        _flush_pending_tool_calls(messages, pending_assistant_calls, None)
        if messages:
            yield {
                "schema": SCHEMA_VERSION,
                "redacted": False,
                "source": "codex_rollout",
                "session": session_meta,
                "messages": messages,
            }


def _extract_codex_session_meta(payload: Mapping[str, Any]) -> dict[str, Any]:
    nested = payload.get("payload")
    if isinstance(nested, dict):
        payload = nested
    return {
        "id": payload.get("id") or payload.get("thread_id") or payload.get("threadId"),
        "cwd": payload.get("cwd") or payload.get("workdir"),
        "title": payload.get("title") or payload.get("objective"),
        "source": "codex_rollout",
    }


def _codex_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return ""


def _codex_output_to_text(payload: Mapping[str, Any]) -> str:
    output = payload.get("output")
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def _codex_tool_call(payload: Mapping[str, Any]) -> dict[str, Any]:
    item_type = str(payload.get("type") or "function_call")
    stable_payload_id = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    call_id = str(payload.get("call_id") or f"call_{stable_payload_id}")
    name = payload.get("name") or payload.get("namespace") or item_type
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("input") or payload.get("action") or {}
    if isinstance(arguments, str):
        arg_text = arguments
    else:
        arg_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": arg_text,
        },
    }


def _flush_pending_tool_calls(
    messages: list[dict[str, Any]],
    pending_assistant_calls: list[dict[str, Any]],
    timestamp: float | None,
) -> None:
    if not pending_assistant_calls:
        return
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": list(pending_assistant_calls),
        "tool_call_id": None,
        "tool_name": None,
        "timestamp": timestamp,
    })
    pending_assistant_calls.clear()


def _tool_name_for_call_id(messages: list[dict[str, Any]], call_id: str) -> str | None:
    for message in reversed(messages):
        for tool_call in _iter_tool_calls(message.get("tool_calls")):
            if str(tool_call.get("id") or "") == call_id:
                function = tool_call.get("function")
                if isinstance(function, dict) and function.get("name"):
                    return str(function["name"])
    return None


class CorpusRedactor:
    """Stable anonymizer for local Hermes training records."""

    def __init__(
        self,
        *,
        user_home: Path | None = None,
        hermes_home: Path | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.user_home = (user_home or Path.home()).expanduser()
        self.hermes_home = (hermes_home or self.user_home / ".hermes").expanduser()
        self.repo_root = repo_root.expanduser() if repo_root else None
        self._path_map: dict[str, str] = {}
        self._session_map: dict[str, str] = {}

    def redact_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        redacted = self._redact_value(dict(record))
        if isinstance(redacted, dict):
            redacted["schema"] = record.get("schema", SCHEMA_VERSION)
            redacted["redacted"] = True
            redacted["redaction_policy"] = {
                "version": 1,
                "secret_values": "agent.redact.redact_sensitive_text(force=True)",
                "paths": "stable placeholders",
                "session_ids": "stable placeholders",
                "public_tunnels": "<PUBLIC_TUNNEL_URL>",
            }
        return redacted

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._redact_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._redact_value(item) for key, item in value.items()}
        return value

    def redact_text(self, text: str) -> str:
        text = redact_sensitive_text(text, force=True)
        text = _NGROK_URL_RE.sub("<PUBLIC_TUNNEL_URL>", text)
        text = self._replace_known_path(text, self.repo_root, "<HERMES_REPO>")
        text = self._replace_known_path(text, self.hermes_home, "<HERMES_HOME>")
        text = self._replace_known_path(text, self.user_home, "<USER_HOME>")
        text = _POSIX_HOME_RE.sub("<USER_HOME>", text)
        text = _WINDOWS_PATH_RE.sub(lambda match: self._path_placeholder(match.group(1)), text)
        text = _SESSIONISH_RE.sub(lambda match: self._session_placeholder(match.group(0)), text)
        return text

    @staticmethod
    def _replace_known_path(text: str, path: Path | None, placeholder: str) -> str:
        if path is None:
            return text
        raw = str(path)
        variants: set[str] = {raw, raw.replace("\\", "/")}
        for variant in sorted(variants, key=len, reverse=True):
            variant_text = str(variant)
            if variant_text:
                text = re.sub(re.escape(variant_text), placeholder, text)
        return text

    def _path_placeholder(self, path: str) -> str:
        key = path.replace("\\", "/").lower()
        if key not in self._path_map:
            self._path_map[key] = f"<LOCAL_PATH:{len(self._path_map) + 1}>"
        return self._path_map[key]

    def _session_placeholder(self, session_id: str) -> str:
        key = session_id.lower()
        if key not in self._session_map:
            self._session_map[key] = f"<SESSION_ID:{len(self._session_map) + 1}>"
        return self._session_map[key]


def write_jsonl(records: Iterable[Mapping[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl(input_path: Path) -> Iterator[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{input_path}:{line_no}: expected object record")
            yield value


def build_sft_record(
    record: Mapping[str, Any],
    *,
    allow_unredacted: bool = False,
    min_messages: int = 2,
    max_message_chars: int = 8000,
    max_tool_chars: int = 2000,
    max_messages: int = 120,
) -> dict[str, Any] | None:
    if not record.get("redacted") and not allow_unredacted:
        raise ValueError("refusing to build SFT from unredacted corpus record")
    if record.get("source") not in {"state_db", "codex_rollout", "harness_result"}:
        return None

    raw_messages = record.get("messages") or []
    if not isinstance(raw_messages, list) or len(raw_messages) < min_messages:
        return None
    raw_messages = _select_sft_messages(raw_messages, max_messages)

    raw_session = record.get("session")
    session: Mapping[str, Any] = raw_session if isinstance(raw_session, dict) else {}
    messages: list[dict[str, Any]] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    tools_seen: set[str] = set()
    role_counts: dict[str, int] = {}

    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip()
        if role not in {"user", "assistant", "tool", "system"}:
            continue
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        tool_name = msg.get("tool_name")
        max_chars = max_tool_chars if role == "tool" else max_message_chars
        out: dict[str, Any] = {
            "role": role,
            "content": _truncate_content(content if content is not None else "", max_chars),
        }
        if role == "assistant" and tool_calls:
            out["tool_calls"] = tool_calls
            for tool_call in _iter_tool_calls(tool_calls):
                name = tool_call.get("function", {}).get("name") or tool_call.get("name")
                if name:
                    tools_seen.add(str(name))
        if role == "tool":
            if msg.get("tool_call_id"):
                out["tool_call_id"] = msg["tool_call_id"]
            if tool_name:
                out["name"] = tool_name
                tools_seen.add(str(tool_name))
        messages.append(out)
        role_counts[role] = role_counts.get(role, 0) + 1

    if role_counts.get("user", 0) < 1 or role_counts.get("assistant", 0) < 1:
        return None

    return {
        "schema": SFT_SCHEMA_VERSION,
        "messages": messages,
        "tools": sorted(tools_seen),
        "metadata": {
            "source": record.get("source"),
            "session_id": session.get("id"),
            "session_source": session.get("source"),
            "title": session.get("title"),
            "outcome": _infer_outcome(session),
            "tags": _infer_tags(record, tools_seen),
        },
    }


def _truncate_content(content: Any, max_chars: int) -> str:
    text = str(content)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n<TRUNCATED chars={omitted}>"


def _select_sft_messages(messages: list[Any], max_messages: int) -> list[Any]:
    if max_messages <= 0 or len(messages) <= max_messages:
        return messages
    head_count = min(2, max_messages)
    tail_count = max_messages - head_count
    return [*messages[:head_count], *messages[-tail_count:]]


def build_dpo_record(
    record: Mapping[str, Any],
    *,
    allow_unredacted: bool = False,
) -> dict[str, Any] | None:
    """Build an Axolotl-style DPO row from a redacted preference record."""

    if not record.get("redacted") and not allow_unredacted:
        raise ValueError("refusing to build DPO from unredacted corpus record")
    preference = record.get("preference")
    if not isinstance(preference, dict):
        return None
    prompt_messages = preference.get("prompt_messages")
    chosen = preference.get("chosen")
    rejected = preference.get("rejected")
    if not isinstance(prompt_messages, list) or not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    return {
        "schema": DPO_SCHEMA_VERSION,
        "prompt": prompt_messages,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": preference.get("metadata") if isinstance(preference.get("metadata"), dict) else {},
    }


def _iter_tool_calls(tool_calls: Any) -> Iterator[dict[str, Any]]:
    if isinstance(tool_calls, dict):
        yield tool_calls
    elif isinstance(tool_calls, list):
        for item in tool_calls:
            if isinstance(item, dict):
                yield item


def _infer_outcome(session: Mapping[str, Any]) -> str:
    end_reason = session.get("end_reason")
    if end_reason in {"error", "interrupted"}:
        return "needs_review"
    return "unknown"


def _infer_tags(record: Mapping[str, Any], tools_seen: set[str]) -> list[str]:
    text = json.dumps(record, ensure_ascii=False).lower()
    tags = {"hermes", "operator"}
    for word, tag in (
        ("gateway", "gateway"),
        ("scheduled", "scheduled-task"),
        ("task scheduler", "scheduled-task"),
        ("dashboard", "dashboard"),
        ("harness", "harness"),
        ("gguf", "gguf"),
        ("ci", "ci"),
        ("github actions", "ci"),
        ("restart", "restart"),
        ("powershell", "windows"),
        ("scheduled task", "windows"),
        ("uac", "windows"),
        ("gguf", "training"),
        ("fine-tuning", "training"),
        ("post-training", "training"),
        ("qlora", "training"),
        ("redact", "redaction"),
        ("redaction", "redaction"),
        ("secret", "redaction"),
    ):
        if word in text:
            tags.add(tag)
    if tools_seen:
        tags.add("tool-calling")
    return sorted(tags)


def add_common_redaction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--user-home", type=Path, default=Path.home())
    parser.add_argument("--hermes-home", type=Path, default=Path.home() / ".hermes")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())


def cli_export(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Hermes sessions for operator training.")
    parser.add_argument("--state-db", type=Path, default=Path.home() / ".hermes" / "state.db")
    parser.add_argument("--output", type=Path, default=Path("training/corpora/hermes_operator_corpus.redacted.jsonl"))
    parser.add_argument("--limit-sessions", type=int, default=200)
    parser.add_argument("--since-days", type=int)
    parser.add_argument("--source")
    parser.add_argument("--include-system-prompt", action="store_true")
    parser.add_argument("--include-logs", action="store_true")
    parser.add_argument("--logs-dir", type=Path, default=Path.home() / ".hermes" / "logs")
    parser.add_argument("--harness-log", action="append", type=Path, default=[])
    parser.add_argument("--harness-result", action="append", type=Path, default=[])
    parser.add_argument("--codex-rollout", action="append", type=Path, default=[])
    parser.add_argument("--codex-sessions-dir", type=Path)
    parser.add_argument("--limit-codex-rollouts", type=int, default=20)
    parser.add_argument("--raw", action="store_true", help="Write unredacted records. Unsafe; never commit this output.")
    add_common_redaction_args(parser)
    args = parser.parse_args(argv)

    records: list[dict[str, Any]] = list(
        iter_state_db_records(
            args.state_db,
            limit_sessions=args.limit_sessions,
            since_days=args.since_days,
            source=args.source,
            include_system_prompt=args.include_system_prompt,
        )
    )
    if args.include_logs:
        log_paths = sorted(args.logs_dir.glob("*.log")) if args.logs_dir.exists() else []
        log_paths.extend(args.harness_log)
        records.extend(iter_log_records(log_paths))
    if args.harness_result:
        records.extend(iter_harness_result_records(args.harness_result))
    codex_paths = list(args.codex_rollout)
    if args.codex_sessions_dir and args.codex_sessions_dir.exists():
        codex_paths.extend(
            sorted(
                args.codex_sessions_dir.rglob("*.jsonl"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )[: max(1, args.limit_codex_rollouts)]
        )
    if codex_paths:
        records.extend(iter_codex_rollout_records(codex_paths))

    if not args.raw:
        redactor = CorpusRedactor(
            user_home=args.user_home,
            hermes_home=args.hermes_home,
            repo_root=args.repo_root,
        )
        records = [redactor.redact_record(record) for record in records]

    count = write_jsonl(records, args.output)
    print(f"wrote {count} record(s) to {args.output}")
    return 0


def cli_redact(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Redact a Hermes training corpus JSONL file.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    add_common_redaction_args(parser)
    args = parser.parse_args(argv)

    redactor = CorpusRedactor(
        user_home=args.user_home,
        hermes_home=args.hermes_home,
        repo_root=args.repo_root,
    )
    count = write_jsonl((redactor.redact_record(record) for record in read_jsonl(args.input)), args.output)
    print(f"wrote {count} redacted record(s) to {args.output}")
    return 0


def cli_build_sft(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build OpenAI-message SFT JSONL from redacted Hermes corpus.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-unredacted", action="store_true")
    parser.add_argument("--min-messages", type=int, default=2)
    parser.add_argument("--max-message-chars", type=int, default=8000)
    parser.add_argument("--max-tool-chars", type=int, default=2000)
    parser.add_argument("--max-messages", type=int, default=120)
    args = parser.parse_args(argv)

    def _records() -> Iterator[dict[str, Any]]:
        for record in read_jsonl(args.input):
            sft = build_sft_record(
                record,
                allow_unredacted=args.allow_unredacted,
                min_messages=args.min_messages,
                max_message_chars=args.max_message_chars,
                max_tool_chars=args.max_tool_chars,
                max_messages=args.max_messages,
            )
            if sft is not None:
                yield sft

    count = write_jsonl(_records(), args.output)
    print(f"wrote {count} SFT record(s) to {args.output}")
    return 0


def cli_build_dpo(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build DPO JSONL from redacted Hermes preference records.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-unredacted", action="store_true")
    args = parser.parse_args(argv)

    def _records() -> Iterator[dict[str, Any]]:
        for record in read_jsonl(args.input):
            dpo = build_dpo_record(record, allow_unredacted=args.allow_unredacted)
            if dpo is not None:
                yield dpo

    count = write_jsonl(_records(), args.output)
    print(f"wrote {count} DPO record(s) to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("expected subcommand: export | redact | build-sft | build-dpo", file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "export":
        return cli_export(rest)
    if command == "redact":
        return cli_redact(rest)
    if command == "build-sft":
        return cli_build_sft(rest)
    if command == "build-dpo":
        return cli_build_dpo(rest)
    print(f"unknown subcommand: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
