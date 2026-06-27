"""Hermes context engine backed by RecursiveIntell context-governor.

This engine shells out to the local Rust context-governor binary.  It keeps
Hermes' context-engine contract small and explicit: the Rust crate owns
allocation/receipts/fallback storage; Hermes owns message transport and config.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agent.context_engine import ContextEngine
from agent.model_metadata import estimate_messages_tokens_rough
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def _resolve_governor_dir() -> Path:
    env_dir = os.getenv("HERMES_CONTEXT_GOVERNOR_DIR")
    if env_dir:
        resolved = Path(env_dir).expanduser()
        if resolved.exists():
            return resolved

    home = Path.home()
    fallback = home / "Coding" / "Libraries" / "context-governor"
    if fallback.exists():
        return fallback
    return Path(__file__).resolve().parents[4]


def _resolve_binary(crate_dir: Path) -> Path:
    env_bin = os.getenv("HERMES_CONTEXT_GOVERNOR_BIN")
    if env_bin:
        candidate = Path(env_bin).expanduser()
        if candidate.exists():
            return candidate

    which_bin = shutil.which("context-governor")
    if which_bin:
        return Path(which_bin)

    release = crate_dir / "target" / "release" / "context-governor"
    if release.exists():
        return release
    debug = crate_dir / "target" / "debug" / "context-governor"
    return debug


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_str_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if not raw:
        return default
    return raw


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "on", "yes", "y"}


class ContextGovernorEngine(ContextEngine):
    threshold_percent = _read_int_env("HERMES_CONTEXT_GOVERNOR_THRESHOLD_PERCENT", 50) / 100.0
    protect_first_n = _read_int_env("HERMES_CONTEXT_GOVERNOR_PROTECT_FIRST", 3)
    protect_last_n = _read_int_env("HERMES_CONTEXT_GOVERNOR_PROTECT_LAST", 1)

    def __init__(self) -> None:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.context_length = 0
        self.threshold_tokens = 0
        self.compression_count = 0
        self.session_id: Optional[str] = None
        self.store_dir = get_hermes_home() / "context-governor" / "receipts"
        self.crate_dir = _resolve_governor_dir()
        self.binary = _resolve_binary(self.crate_dir)

    @property
    def name(self) -> str:
        return "context_governor"

    def is_available(self) -> bool:
        return self.binary.exists() or (self.crate_dir / "target" / "release" / "context-governor").exists()

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.context_length = int(context_length or 0)
        self.threshold_tokens = int(self.context_length * self.threshold_percent) if self.context_length else 0

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or (self.last_prompt_tokens + self.last_completion_tokens))

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        tokens = int(prompt_tokens or self.last_prompt_tokens or 0)
        return bool(self.threshold_tokens and tokens >= self.threshold_tokens)

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        rough = estimate_messages_tokens_rough(messages)
        return self.should_compress(rough)

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return len(messages) > (self.protect_first_n + self.protect_last_n + 1)

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.session_id = None

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not messages:
            return messages
        self._ensure_binary()
        request = {
            "session_id": self.session_id or "hermes-context-governor",
            "messages": [self._to_governor_message(i, m) for i, m in enumerate(messages)],
            "policy": {
                "target_tokens": self._target_tokens(current_tokens),
                "protect_first_n": self.protect_first_n,
                "protect_last_n": self.protect_last_n,
                "summary_max_chars": _read_int_env("HERMES_CONTEXT_GOVERNOR_SUMMARY_MAX_CHARS", 2400),
                "allocator": _read_str_env("HERMES_CONTEXT_GOVERNOR_ALLOCATOR", "aggressive_v1"),
                "semantic_memory_enabled": _read_bool_env("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_ENABLED", False),
                "archive_memory_enabled": _read_bool_env("HERMES_CONTEXT_GOVERNOR_ARCHIVE_MEMORY_ENABLED", False),
                "budget_mode": _read_str_env("HERMES_CONTEXT_GOVERNOR_BUDGET_MODE", "hard_cascade"),
                "token_counter": _read_str_env("HERMES_CONTEXT_GOVERNOR_TOKEN_COUNTER", "approx_chars"),
            },
            "focus": focus_topic,
        }
        response = self._run_compact_with_fallback(request, messages)
        compacted = [self._from_governor_message(m) for m in response.get("compacted_messages", [])]
        compacted = self._ensure_latest_user_last(compacted, messages)
        if not compacted:
            logger.warning("context-governor returned no compacted messages; keeping original")
            return messages
        response["compacted_messages"] = [self._to_governor_message(i, m) for i, m in enumerate(compacted)]
        self._archive_response_to_semantic_memory(response)
        try:
            self._persist_response(response)
        except Exception as exc:
            logger.exception("context-governor receipt persistence failed; continuing with compacted messages")
            response.setdefault("receipt", {}).setdefault("warnings", []).append(
                f"receipt persistence failed after successful compaction: {exc}"
            )
        self.compression_count += 1
        self.last_prompt_tokens = int(response.get("receipt", {}).get("compacted_approx_tokens") or 0)
        return compacted

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "context_governor_search",
                "description": "Search receipt-backed compacted context and exact fallback records.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Literal text to search for"},
                        "top_k": {"type": "integer", "description": "Maximum hits", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "context_governor_expand",
                "description": "Expand exact fallback content from a context-governor receipt item id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "receipt": {"type": "string", "description": "Receipt id"},
                        "item": {"type": "string", "description": "Context item id"},
                        "max_chars": {"type": "integer", "description": "Maximum characters", "default": 4000},
                    },
                    "required": ["receipt", "item"],
                },
            },
            {
                "name": "context_governor_diff",
                "description": "Compute a compaction diff for a stored context-governor receipt.",
                "parameters": {
                    "type": "object",
                    "properties": {"receipt": {"type": "string", "description": "Receipt id"}},
                    "required": ["receipt"],
                },
            },
            {
                "name": "context_governor_receipts",
                "description": "List stored context-governor receipt ids.",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "description": "Maximum receipts", "default": 20}},
                },
            },
            {
                "name": "context_governor_status",
                "description": "Return context-governor engine status, policy, binary, and receipt store information.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            self._ensure_binary()
            if name == "context_governor_search":
                query = str(args.get("query") or "")
                top_k = int(args.get("top_k") or 5)
                proc = self._run_cli(["search", "--dir", str(self.store_dir), "--query", query, "--top-k", str(top_k)])
                return proc.stdout
            if name == "context_governor_expand":
                receipt = str(args.get("receipt") or "")
                item = str(args.get("item") or "")
                max_chars = int(args.get("max_chars") or 4000)
                proc = self._run_cli([
                    "expand",
                    "--dir",
                    str(self.store_dir),
                    "--receipt",
                    receipt,
                    "--item",
                    item,
                    "--max-chars",
                    str(max_chars),
                ])
                return proc.stdout
            if name == "context_governor_diff":
                receipt = str(args.get("receipt") or "")
                proc = self._run_cli(["diff"], input_text=self._receipt_path(receipt).read_text())
                return proc.stdout
            if name == "context_governor_receipts":
                limit = int(args.get("limit") or 20)
                receipts = [p.stem for p in sorted(self.store_dir.glob("*.json"), reverse=True)[:limit]]
                return json.dumps({"receipts": receipts, "store_dir": str(self.store_dir)})
            if name == "context_governor_status":
                return json.dumps(self.get_status(), default=str)
        except Exception as exc:
            logger.exception("context-governor tool failed")
            return json.dumps({"error": str(exc)})
        return json.dumps({"error": f"Unknown context-governor tool: {name}"})

    def _run_compact_with_fallback(
        self,
        request: Dict[str, Any],
        original_messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            return self._run_context_governor(["compact"], request)
        except subprocess.CalledProcessError as exc:
            budget_mode = str((request.get("policy") or {}).get("budget_mode") or "")
            fallback_mode = _read_str_env("HERMES_CONTEXT_GOVERNOR_FALLBACK_BUDGET_MODE", "soft_warn")
            if budget_mode != fallback_mode and fallback_mode.lower() not in {"", "off", "none"}:
                logger.warning(
                    "context-governor compact failed in %s; retrying with %s: %s",
                    budget_mode,
                    fallback_mode,
                    exc.stderr or exc,
                )
                retry_request = json.loads(json.dumps(request))
                retry_request.setdefault("policy", {})["budget_mode"] = fallback_mode
                try:
                    response = self._run_context_governor(["compact"], retry_request)
                    response.setdefault("receipt", {}).setdefault("warnings", []).append(
                        f"primary budget_mode {budget_mode} failed; retried with {fallback_mode}"
                    )
                    return response
                except subprocess.CalledProcessError as retry_exc:
                    logger.exception("context-governor fallback compact failed; fail-open returning original messages")
                    warning = (
                        f"primary budget_mode {budget_mode} failed: {exc.stderr or exc}; "
                        f"fallback budget_mode {fallback_mode} failed: {retry_exc.stderr or retry_exc}"
                    )
                    return self._original_messages_response(request, original_messages, warning)
            logger.exception("context-governor compact failed; fail-open returning original messages")
            return self._original_messages_response(request, original_messages, f"context-governor compact failed: {exc}")

    def _original_messages_response(
        self,
        request: Dict[str, Any],
        messages: List[Dict[str, Any]],
        warning: str,
    ) -> Dict[str, Any]:
        compacted = [self._to_governor_message(i, m) for i, m in enumerate(messages)]
        return {
            "receipt": {
                "receipt_id": f"ctxr_fail_open_{uuid.uuid4().hex}",
                "compacted_approx_tokens": estimate_messages_tokens_rough(messages),
                "semantic_memory_fact_ids": [],
                "warnings": [warning],
            },
            "allocation_plan": {"items": [], "archived_item_ids": []},
            "compacted_messages": compacted,
            "exact_store": [],
        }

    def _receipt_path(self, receipt_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in receipt_id)
        path = self.store_dir / f"{safe}.json"
        if not safe or not path.exists():
            raise FileNotFoundError(f"context-governor receipt not found: {receipt_id}")
        return path

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update(
            {
                "engine": self.name,
                "store_dir": str(self.store_dir),
                "binary": str(self.binary),
                "crate_dir": str(self.crate_dir),
                "policy": {
                    "target": "env:HERMES_CONTEXT_GOVERNOR_TARGET_TOKENS",
                    "allocator": _read_str_env("HERMES_CONTEXT_GOVERNOR_ALLOCATOR", "aggressive_v1"),
                    "token_counter": _read_str_env("HERMES_CONTEXT_GOVERNOR_TOKEN_COUNTER", "approx_chars"),
                    "budget_mode": _read_str_env("HERMES_CONTEXT_GOVERNOR_BUDGET_MODE", "hard_cascade"),
                    "semantic_memory": _read_bool_env("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_ENABLED", False),
                    "archive_memory": _read_bool_env("HERMES_CONTEXT_GOVERNOR_ARCHIVE_MEMORY_ENABLED", False),
                },
            }
        )
        return status

    def _target_tokens(self, current_tokens: Optional[int] = None) -> int:
        if current_tokens:
            return min(12_000, max(4_000, int(current_tokens * 0.25)))
        if self.last_prompt_tokens:
            return min(12_000, max(4_000, int(self.last_prompt_tokens * 0.25)))
        if self.context_length:
            return min(12_000, max(4_000, int(self.context_length * 0.03)))
        return 8_000

    def _to_governor_message(self, index: int, message: Dict[str, Any]) -> Dict[str, Any]:
        role = str(message.get("role") or "assistant")
        if role not in {"system", "user", "assistant", "tool"}:
            role = "assistant"
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, sort_keys=True)
        out: Dict[str, Any] = {"id": str(message.get("id") or f"m{index}"), "role": role, "content": content}
        if message.get("name"):
            out["name"] = str(message.get("name"))
        return out

    def _from_governor_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        role = message.get("role") or "assistant"
        content = message.get("content") or ""
        if role == "tool":
            role = "assistant"
            content = f"[historical tool output preserved by context-governor]\n{content}"
        out = {"role": role, "content": content}
        if message.get("name") and role != "assistant":
            out["name"] = message.get("name")
        if message.get("id"):
            out["_context_governor_id"] = message.get("id")
        return out

    def _ensure_latest_user_last(
        self,
        compacted: List[Dict[str, Any]],
        original: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        latest_user = next((m for m in reversed(original) if m.get("role") == "user"), None)
        if latest_user is None:
            return compacted
        latest_content = latest_user.get("content")
        if not isinstance(latest_content, str):
            latest_content = json.dumps(latest_content, ensure_ascii=False, sort_keys=True)
        filtered = [
            m for m in compacted
            if not (m.get("role") == "user" and m.get("content") == latest_content)
        ]
        latest = {"role": "user", "content": latest_content}
        if latest_user.get("name"):
            latest["name"] = str(latest_user.get("name"))
        filtered.append(latest)
        return filtered

    def _ensure_binary(self) -> None:
        if self.binary.exists():
            return
        self.crate_dir = _resolve_governor_dir()
        self.binary = _resolve_binary(self.crate_dir)
        if self.binary.exists():
            return
        if shutil.which("cargo") is None:
            raise FileNotFoundError("context-governor binary missing and cargo is not installed")
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(self.crate_dir),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.binary = _resolve_binary(self.crate_dir)
        if not self.binary.exists():
            raise FileNotFoundError(f"context-governor binary not found: {self.binary}")

    def _run_context_governor(self, args: List[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        proc = self._run_cli(args, input_text=json.dumps(payload, ensure_ascii=False))
        return json.loads(proc.stdout)

    def _run_cli(self, args: List[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.binary), *args],
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    def _persist_response(self, response: Dict[str, Any]) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._run_context_governor(["store", "--dir", str(self.store_dir)], response)

    def _archive_response_to_semantic_memory(self, response: Dict[str, Any]) -> None:
        if not _read_bool_env("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_ENABLED", False):
            return

        receipt = response.setdefault("receipt", {})
        fact_ids = receipt.setdefault("semantic_memory_fact_ids", [])
        warnings = receipt.setdefault("warnings", [])
        payloads = self._semantic_memory_archive_payloads(response)
        if not payloads:
            warnings.append("semantic-memory archive enabled but no archiveable exact records were found")
            return

        for payload in payloads:
            try:
                existing = self._semantic_memory_existing_fact_id(payload)
                if existing:
                    if existing not in fact_ids:
                        fact_ids.append(existing)
                    continue
                result = self._semantic_memory_post_json(
                    "/add",
                    payload,
                    timeout=_read_int_env("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_TIMEOUT", 10),
                )
                if result and result.get("ok") and result.get("fact_id"):
                    fact_id = str(result["fact_id"])
                    if fact_id not in fact_ids:
                        fact_ids.append(fact_id)
                    self._clear_no_memory_sink_warnings(warnings)
                else:
                    warnings.append(f"semantic-memory archive add failed for {payload.get('source')}: {result}")
            except Exception as exc:
                logger.exception("context-governor semantic-memory archive failed")
                warnings.append(f"semantic-memory archive failed: {exc}")

    def _semantic_memory_archive_payloads(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        receipt = response.get("receipt") or {}
        receipt_id = str(receipt.get("receipt_id") or "unknown")
        plan = response.get("allocation_plan") or {}
        archived_ids: Set[str] = {str(item_id) for item_id in (plan.get("archived_item_ids") or [])}
        items = plan.get("items") or []
        exact_by_id = {
            str(stored.get("item_id")): stored
            for stored in (response.get("exact_store") or [])
            if stored.get("item_id")
        }
        namespace = _read_str_env("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_NAMESPACE", "projects")
        max_chars = _read_int_env("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_MAX_CHARS", 4000)
        payloads: List[Dict[str, Any]] = []
        for item in items:
            item_id = str(item.get("item_id") or "")
            if not item_id:
                continue
            authority = str(item.get("authority_class") or "")
            should_archive = item_id in archived_ids or authority in {"DurableMemoryCandidate", "EvidenceCritical"}
            if not should_archive:
                continue
            stored = exact_by_id.get(item_id)
            if not stored:
                continue
            content = str(stored.get("content") or "")
            content_blake3 = str(stored.get("content_blake3") or "")
            item_type = str(item.get("item_type") or "Unknown")
            content_kind = str(item.get("content_kind") or "Unknown")
            clipped = content[:max_chars]
            if len(content) > max_chars:
                clipped += f"\n[truncated by context-governor semantic archive: {len(content) - max_chars} chars omitted]"
            fact_content = (
                "Context-governor archived exact record.\n"
                f"receipt_id: {receipt_id}\n"
                f"item_id: {item_id}\n"
                f"content_blake3: {content_blake3}\n"
                f"item_type: {item_type}\n"
                f"content_kind: {content_kind}\n"
                f"authority_class: {authority}\n"
                "archive_reason: semantic-memory-enabled context compaction archival\n\n"
                "Archived content:\n"
                f"{clipped}"
            )
            payloads.append(
                {
                    "content": fact_content,
                    "namespace": namespace,
                    "source": f"context-governor receipt {receipt_id} item {item_id}",
                    "memory_kind": "project_state",
                    "sensitivity": "internal",
                    "evidence_refs": [f"context-governor:{receipt_id}:{item_id}"],
                }
            )
        return payloads

    def _clear_no_memory_sink_warnings(self, warnings: List[str]) -> None:
        warnings[:] = [
            warning for warning in warnings
            if "no memory sink" not in warning and "semantic-memory IDs are intentionally empty" not in warning
        ]

    def _semantic_memory_existing_fact_id(self, payload: Dict[str, Any]) -> Optional[str]:
        marker = "content_blake3: "
        content = str(payload.get("content") or "")
        if marker not in content:
            return None
        content_blake3 = content.split(marker, 1)[1].splitlines()[0].strip()
        if not content_blake3:
            return None
        result = self._semantic_memory_post_json(
            "/search",
            {"query": content_blake3, "top_k": 3, "namespaces": [payload.get("namespace", "projects")]},
            timeout=_read_int_env("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_TIMEOUT", 10),
        )
        for hit in (result or {}).get("results", []):
            if content_blake3 in str(hit.get("content") or ""):
                return str(hit.get("result_id") or hit.get("fact_id") or "") or None
        return None

    def _semantic_memory_post_json(self, path: str, payload: Dict[str, Any], timeout: int = 10) -> Optional[Dict[str, Any]]:
        port = _read_int_env("SEMANTIC_MEMORY_HTTP_PORT", 1738)
        base_url = _read_str_env("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_URL", f"http://127.0.0.1:{port}").rstrip("/")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}{path}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response_obj:
            return json.loads(response_obj.read().decode("utf-8"))


def register(ctx) -> None:
    ctx.register_context_engine(ContextGovernorEngine())
