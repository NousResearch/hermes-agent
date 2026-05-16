"""Opt-in DAG-backed context engine runtime assembly.

PR4 keeps this engine read-only during context builds: it loads raw transcript
messages, the active projection, and summary nodes from ``ContextDAGStore`` and
uses the pure assembler to construct model context.  It deliberately returns a
projection-only compression result so callers do not rotate sessions or rewrite
raw transcripts with the projection.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

from agent.context_dag_assembler import ContextAssemblyError, assemble_context, estimate_message_tokens
from agent.context_dag_models import AssemblyBudget
from agent.context_dag_store import ContextDAGStore
from agent.context_engine import ContextCompressionResult, ContextEngine

logger = logging.getLogger(__name__)


class DAGContextEngine(ContextEngine):
    """Read-only runtime assembler for persisted DAG context state."""

    ENGINE_VERSION = "dag-v1"
    projection_only_compression = True
    preserves_session_on_compress = True
    gateway_compression_enabled = False

    def __init__(
        self,
        *,
        session_db=None,
        enabled: bool = True,
        max_context_tokens: Optional[int] = None,
        threshold_percent: float = 0.75,
        gateway_enabled: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.session_db = session_db
        self.store = ContextDAGStore(session_db) if session_db is not None else None
        self.session_id: Optional[str] = None
        self.platform = "cli"
        self.context_length = int(max_context_tokens or 0)
        self.threshold_percent = float(threshold_percent)
        self.threshold_tokens = int(self.context_length * self.threshold_percent) if self.context_length else 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self.gateway_enabled = bool(gateway_enabled)
        self.gateway_compression_enabled = False
        self._last_fallback_reason: Optional[str] = None
        self._last_projection_token_estimate: Optional[int] = None
        self._last_fresh_tail_start_message_id: Optional[int] = None
        self._last_checkpoint: Optional[Dict[str, Any]] = None
        self._last_summary_count = 0

    @property
    def name(self) -> str:
        return "dag"

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        self.context_length = int(context_length or 0)
        self.threshold_tokens = int(self.context_length * self.threshold_percent) if self.context_length else 0

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_id = session_id
        self.platform = str(kwargs.get("platform") or self.platform or "cli")
        if self.store is None and self.session_db is None:
            session_db = kwargs.get("session_db")
            if session_db is not None:
                self.session_db = session_db
                self.store = ContextDAGStore(session_db)

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if not isinstance(usage, dict):
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or (self.last_prompt_tokens + self.last_completion_tokens))

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = int(prompt_tokens or self.last_prompt_tokens or 0)
        return bool(self.enabled and self.threshold_tokens and tokens >= self.threshold_tokens)

    _API_MESSAGE_KEYS = {"role", "content", "name", "tool_call_id", "tool_calls", "function_call"}

    def _sanitize_message_for_api(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Copy a DAG/internal message into the chat-completions-safe shape."""
        sanitized: Dict[str, Any] = {}
        for key in self._API_MESSAGE_KEYS:
            if key not in message:
                continue
            value = message.get(key)
            if value in ("", [], {}):
                continue
            sanitized[key] = copy.deepcopy(value)
        if "role" not in sanitized and message.get("role"):
            sanitized["role"] = message.get("role")
        if "content" not in sanitized and "content" in message:
            sanitized["content"] = copy.deepcopy(message.get("content"))
        if "name" not in sanitized and message.get("tool_name"):
            sanitized["name"] = copy.deepcopy(message.get("tool_name"))
        return sanitized

    def _sanitize_messages_for_api(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._sanitize_message_for_api(message) for message in messages]

    def _raw_message_to_openai(self, message: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": message.get("id"),
            "role": message.get("role"),
            "content": message.get("content"),
        }
        for key in (
            "tool_call_id",
            "tool_name",
            "name",
            "tool_calls",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
        ):
            value = message.get(key)
            if value not in (None, "", [], {}):
                out[key] = copy.deepcopy(value)
        if out.get("tool_name") and not out.get("name"):
            out["name"] = out["tool_name"]
        return out

    def _read_raw_messages(self, fallback_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.store or not self.session_id:
            self._last_fallback_reason = "missing_store_or_session"
            return copy.deepcopy(fallback_messages)
        raw_rows = self.store.db.get_messages(self.session_id)
        if not raw_rows:
            self._last_fallback_reason = "missing_raw_messages"
            return copy.deepcopy(fallback_messages)
        return [self._raw_message_to_openai(dict(row)) for row in raw_rows]

    def _assembly_budget(self) -> AssemblyBudget:
        max_tokens = int(self.context_length or self.threshold_tokens or 200_000)
        return AssemblyBudget(
            max_tokens=max_tokens,
            fresh_tail_min_tokens=max(1, min(max_tokens, max_tokens // 4)),
            summary_max_tokens=max(1, max_tokens // 2),
        )

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> ContextCompressionResult:
        if not self.enabled:
            sanitized = self._sanitize_messages_for_api(copy.deepcopy(messages))
            return ContextCompressionResult(
                messages=sanitized,
                projection_only=True,
                preserves_session=True,
                changed=sanitized != messages,
                warning="DAG context engine disabled; using legacy/raw messages.",
            )

        self._last_fallback_reason = None
        try:
            raw_messages = self._read_raw_messages(messages)
            if not self.store or not self.session_id:
                assembled = raw_messages
            else:
                projection = self.store.read_active_projection(self.session_id, self.ENGINE_VERSION)
                if projection is None:
                    self._last_fallback_reason = "missing_projection"
                summaries = self.store.list_summary_nodes(self.session_id)
                checkpoint = self.store.read_checkpoint(self.session_id)
                self._last_summary_count = len(summaries)
                self._last_projection_token_estimate = projection.token_estimate if projection else None
                self._last_fresh_tail_start_message_id = projection.fresh_tail_start_message_id if projection else None
                self._last_checkpoint = checkpoint.__dict__ if checkpoint else None
                assembled = assemble_context(
                    raw_messages=raw_messages,
                    summaries=summaries,
                    projection=projection,
                    budget=self._assembly_budget(),
                )
        except (ContextAssemblyError, ValueError, TypeError) as exc:
            self._last_fallback_reason = str(exc)
            logger.warning("DAG context assembly failed; using caller messages: %s", exc)
            assembled = copy.deepcopy(messages)
        except Exception as exc:
            self._last_fallback_reason = str(exc)
            logger.warning("DAG context assembly unavailable; using caller messages: %s", exc)
            assembled = copy.deepcopy(messages)

        sanitized = self._sanitize_messages_for_api(assembled)
        self.compression_count += 1
        self.last_prompt_tokens = sum(estimate_message_tokens(m) for m in sanitized)
        return ContextCompressionResult(
            messages=sanitized,
            projection_only=True,
            preserves_session=True,
            changed=sanitized != messages,
            raw_checkpoint=self._last_checkpoint,
            warning=self._last_fallback_reason,
        )

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update(
            {
                "engine": self.name,
                "enabled": self.enabled,
                "projection_only": True,
                "preserves_session": True,
                "summary_count": self._last_summary_count,
                "projection_token_estimate": self._last_projection_token_estimate,
                "fresh_tail_start_message_id": self._last_fresh_tail_start_message_id,
                "last_checkpoint": self._last_checkpoint,
                "gateway_compression_enabled": self.gateway_compression_enabled,
                "gateway_compression_status": "skipped_until_gateway_safe_compression_lands",
                "fallback_reason": self._last_fallback_reason,
            }
        )
        return status
