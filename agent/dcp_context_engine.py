"""DCP-style model-guided context engine for Hermes Agent."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict, deque
from typing import Any

from agent.context_engine import ContextEngine
from agent.dcp_config import (
    DCP_DEFAULT_PROTECTED_TOOLS,
    DCPConfig,
    parse_dcp_config,
    resolve_model_limit,
)
from agent.dcp_state import CompressionBlock, DCPSessionState

_ERROR_RE = re.compile(r"\b(error|exception|traceback|failed|failure|timed out|timeout)\b", re.I)

_DCP_SYSTEM_EXTENSION = (
    "DCP context management is active. Message refs look like m0001; "
    "compressed blocks look like b1. Use the compress tool when older work "
    "is complete or stale. Preserve concrete file paths, commands, errors, "
    "test results, decisions, constraints, and open questions. Do not compress "
    "the active task or very recent user turns."
)

# Maximum deactivated blocks to retain; older ones are evicted to bound memory.
_MAX_INACTIVE_BLOCKS = 50


class DCPContextEngine(ContextEngine):
    """Model-guided context engine inspired by Dynamic Context Pruning.

    The engine keeps canonical history intact. ``compress`` creates DCP state;
    ``transform_api_messages`` applies that state to the provider-bound copy.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any] | DCPConfig | None = None,
        context_length: int = 0,
        model: str = "",
        provider: str = "",
        quiet_mode: bool = False,
    ) -> None:
        self.config = config if isinstance(config, DCPConfig) else parse_dcp_config(config)
        self.context_length = context_length or 0
        self.model = model or ""
        self.provider = provider or ""
        self.quiet_mode = quiet_mode
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self.threshold_tokens = self._min_limit()
        self.state = DCPSessionState()
        # Cache for message signatures keyed by id(msg) — invalidated when
        # the canonical message list changes.  Avoids re-hashing the same
        # dicts on every API call.
        self._sig_cache: dict[int, str] = {}

    @property
    def name(self) -> str:
        return "dcp"

    def update_from_response(self, usage: dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or (self.last_prompt_tokens + self.last_completion_tokens))
        self.state.last_prompt_tokens = self.last_prompt_tokens

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return False

    def should_compress_preflight(self, messages: list[dict[str, Any]]) -> bool:
        return False

    def has_content_to_compress(self, messages: list[dict[str, Any]]) -> bool:
        self._ensure_refs(messages)
        return len(self.state.index_by_ref) > 4

    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> list[dict[str, Any]]:
        if focus_topic:
            self.state.manual_mode = "compress-pending"
            self.state.pending_manual_focus = focus_topic
        return messages

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        self.state.session_id = session_id
        model = kwargs.get("model")
        context_length = kwargs.get("context_length")
        if isinstance(model, str) and model:
            self.model = model
        if isinstance(context_length, int) and context_length > 0:
            self.context_length = context_length
            self.threshold_tokens = self._min_limit()

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.state = DCPSessionState(session_id=self.state.session_id)
        self._sig_cache.clear()

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.model = model or self.model
        self.provider = provider or self.provider
        self.context_length = context_length
        self.threshold_tokens = self._min_limit()

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        if not self.config.enabled or self.config.compress.permission == "deny":
            return []
        return [self._compress_tool_schema()]

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if name != "compress":
            return json.dumps({"ok": False, "error": f"Unknown context engine tool: {name}"})
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return json.dumps({"ok": False, "error": "compress requires current messages"})
        try:
            result = self._handle_compress(args, messages)
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        return json.dumps(result)

    def transform_api_messages(
        self,
        api_messages: list[dict[str, Any]],
        *,
        canonical_messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict[str, Any]] | None,
        api_call_count: int,
        model: str,
        provider: str | None,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return api_messages

        self.model = model or self.model
        self.provider = provider or self.provider
        self.state.session_id = session_id or self.state.session_id

        # Build / refresh refs from canonical messages.  This also updates
        # turn counters and message-since-last-user tracking.
        self._ensure_refs(canonical_messages)

        # Match API messages to refs by content signature.  We use a
        # role+content-based signature that excludes tool_calls (whose JSON
        # may have been re-serialised by _canonicalize_api_tool_calls between
        # the canonical list and the API copy) so that assistant tool-calling
        # messages still match.
        ref_by_api_index = self._match_api_messages_to_refs(api_messages, canonical_messages)

        # Shallow-copy the list and structurally clone messages we will
        # mutate.  This avoids the O(n) cost of copy.deepcopy on every call
        # while still protecting the caller's message dicts.
        transformed = list(api_messages)
        mutated: set[int] = set()

        self._annotate_refs(transformed, ref_by_api_index, mutated)
        self._apply_blocks(transformed, ref_by_api_index, mutated)

        if self._automatic_strategies_enabled():
            if self.config.deduplication.enabled:
                self._apply_deduplication(transformed, mutated)
            if self.config.purge_errors.enabled:
                self._apply_purge_errors(transformed, mutated)

        self._inject_system_extension(transformed, mutated)
        self._inject_nudge(transformed, api_call_count=api_call_count, mutated=mutated)
        return transformed

    def get_status(self) -> dict[str, Any]:
        active_blocks = self.state.active_blocks()
        return {
            **super().get_status(),
            "engine": "dcp",
            "active_blocks": len(active_blocks),
            "message_refs": len(self.state.ref_by_message_key),
            "min_context_limit": self._min_limit(),
            "max_context_limit": self._max_limit(),
            "compress_mode": self.config.compress.mode,
            "compress_permission": self.config.compress.permission,
        }

    # -- Tool schema ------------------------------------------------------

    def _compress_tool_schema(self) -> dict[str, Any]:
        is_message_mode = self.config.compress.mode == "message"
        if is_message_mode:
            item_properties = {
                "messageId": {"type": "string", "description": "Message ref, e.g. m0042."},
                "topic": {"type": "string", "description": "Short label for this message."},
                "summary": {"type": "string", "description": "Complete technical summary replacing this message."},
            }
            item_required = ["messageId", "topic", "summary"]
            description = (
                "Compress individual high-volume messages by ref. Preserve concrete "
                "technical facts, file paths, commands, decisions, errors, and open questions."
            )
        else:
            item_properties = {
                "startId": {"type": "string", "description": "Starting message or block ref, e.g. m0004 or b2."},
                "endId": {"type": "string", "description": "Ending message or block ref, e.g. m0018 or b3."},
                "summary": {"type": "string", "description": "Complete technical summary replacing the range."},
            }
            item_required = ["startId", "endId", "summary"]
            description = (
                "Compress completed, stale context ranges by message/block ref. "
                "Use this when prior work is closed and a concise technical summary "
                "will preserve the useful state. Do not compress the active task."
            )
        return {
            "name": "compress",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Short 3-5 word label for this compression batch."},
                    "content": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": item_properties,
                            "required": item_required,
                        },
                    },
                },
                "required": ["topic", "content"],
            },
        }

    # -- Tool handling ----------------------------------------------------

    def _handle_compress(self, args: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_refs(messages)
        topic = self._require_str(args, "topic")
        content = args.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError("compress.content must be a non-empty array")

        is_message_mode = self.config.compress.mode == "message"
        created: list[int] = []
        deactivated: list[int] = []
        run_id = self.state.new_run_id()

        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Each compression entry must be an object")
            summary = self._require_str(item, "summary")

            if is_message_mode:
                ref = self._require_str(item, "messageId")
                if ref not in self.state.index_by_ref:
                    raise ValueError(f"Unknown message ref: {ref}")
                item_topic = item.get("topic") if isinstance(item.get("topic"), str) else topic
                message_refs = [ref]
                included_blocks: list[int] = []
                block = CompressionBlock(
                    block_id=self.state.new_block_id(),
                    run_id=run_id,
                    mode="message",
                    topic=item_topic,
                    summary=self._augment_summary(summary, message_refs),
                    message_refs=message_refs,
                    included_block_ids=[],
                    consumed_block_ids=[],
                    created_at=time.time(),
                )
            else:
                start_ref = self._require_str(item, "startId")
                end_ref = self._require_str(item, "endId")
                message_refs, included_blocks = self._resolve_range(start_ref, end_ref)
                if not message_refs:
                    raise ValueError(f"Range {start_ref}-{end_ref} does not cover any messages")
                item_topic = topic
                consumed_blocks: list[int] = []
                block_id = self.state.new_block_id()
                for included in included_blocks:
                    old = self.state.blocks_by_id.get(included)
                    if old and old.active:
                        old.active = False
                        old.deactivated_at = time.time()
                        old.deactivated_by_block_id = block_id
                        self.state.active_block_ids.discard(included)
                        consumed_blocks.append(included)
                        deactivated.append(included)
                block = CompressionBlock(
                    block_id=block_id,
                    run_id=run_id,
                    mode="range",
                    topic=item_topic,
                    summary=self._augment_summary(summary, message_refs),
                    start_ref=start_ref,
                    end_ref=end_ref,
                    message_refs=message_refs,
                    included_block_ids=included_blocks,
                    consumed_block_ids=consumed_blocks,
                    created_at=time.time(),
                )

            self.state.blocks_by_id[block.block_id] = block
            self.state.active_block_ids.add(block.block_id)
            created.append(block.block_id)

        self.compression_count += len(created)
        self.state.turns_since_last_compress = 0
        self._evict_inactive_blocks()
        mode = "message" if is_message_mode else "range"
        return {
            "ok": True,
            "mode": mode,
            "created_blocks": created,
            "deactivated_blocks": deactivated,
            "active_blocks": sorted(self.state.active_block_ids),
            "message": f"Compressed {len(created)} {mode}(s) into {', '.join(f'b{i}' for i in created)}.",
        }

    # -- Transforms -------------------------------------------------------

    def _ensure_refs(self, messages: list[dict[str, Any]]) -> None:
        self._sig_cache.clear()
        self.state.index_by_ref.clear()
        for idx, msg in enumerate(messages):
            key = self._message_key(msg, idx)
            ref = self.state.ref_by_message_key.get(key)
            if ref is None:
                ref = self.state.new_message_ref()
                self.state.ref_by_message_key[key] = ref
                self.state.message_key_by_ref[ref] = key
            self.state.index_by_ref[ref] = idx
        user_indices = [idx for idx, msg in enumerate(messages) if msg.get("role") == "user"]
        if user_indices:
            last_user = user_indices[-1]
            if last_user != self.state.last_user_turn_index:
                self.state.turns_since_last_compress += 1
            self.state.last_user_turn_index = last_user
            self.state.messages_since_last_user = len(messages) - last_user - 1

    def _match_api_messages_to_refs(
        self,
        api_messages: list[dict[str, Any]],
        canonical_messages: list[dict[str, Any]],
    ) -> dict[int, str]:
        # Build a mapping from content signature (role + content only,
        # NOT tool_calls) to a deque of refs.  We exclude tool_calls from
        # the signature because _canonicalize_api_tool_calls may have
        # re-serialised tool-call argument JSON with sort_keys=True on the
        # API copy, producing a different hash than the canonical message.
        refs_by_sig: dict[str, deque[str]] = defaultdict(deque)
        for idx, msg in enumerate(canonical_messages):
            key = self._message_key(msg, idx)
            ref = self.state.ref_by_message_key.get(key)
            if ref:
                refs_by_sig[self._content_signature(msg)].append(ref)

        out: dict[int, str] = {}
        for api_idx, msg in enumerate(api_messages):
            if msg.get("role") == "system":
                continue
            sig = self._content_signature(msg)
            queue = refs_by_sig.get(sig)
            if queue:
                out[api_idx] = queue.popleft()
        return out

    def _clone_if_needed(self, messages: list[dict[str, Any]], idx: int, mutated: set[int]) -> dict[str, Any]:
        """Clone a message dict before mutating it (copy-on-write)."""
        if idx not in mutated:
            messages[idx] = dict(messages[idx])
            mutated.add(idx)
        return messages[idx]

    def _annotate_refs(self, messages: list[dict[str, Any]], ref_by_api_index: dict[int, str], mutated: set[int]) -> None:
        for idx, ref in ref_by_api_index.items():
            msg = self._clone_if_needed(messages, idx, mutated)
            content = msg.get("content")
            marker = f'<dcp-ref id="{ref}" />'
            if isinstance(content, str):
                if marker not in content:
                    msg["content"] = f"{content}\n\n{marker}" if content else marker
            elif isinstance(content, list):
                msg["content"] = content + [{"type": "text", "text": marker}]

    def _apply_blocks(self, messages: list[dict[str, Any]], ref_by_api_index: dict[int, str], mutated: set[int]) -> None:
        ref_to_api_index = {ref: idx for idx, ref in ref_by_api_index.items()}
        for block in self.state.active_blocks():
            covered = [ref for ref in block.message_refs if ref in ref_to_api_index]
            if not covered:
                continue
            anchor_ref = covered[0]
            anchor_idx = ref_to_api_index[anchor_ref]
            anchor = self._clone_if_needed(messages, anchor_idx, mutated)
            anchor["content"] = self._block_summary_text(block)
            for ref in covered[1:]:
                idx = ref_to_api_index[ref]
                msg = self._clone_if_needed(messages, idx, mutated)
                msg["content"] = f"[DCP: content moved into compressed block {block.ref}.]"

    def _apply_deduplication(self, messages: list[dict[str, Any]], mutated: set[int]) -> None:
        protected = DCP_DEFAULT_PROTECTED_TOOLS | self.config.deduplication.protected_tools
        latest_by_sig: dict[str, int] = {}
        result_by_call_id: dict[str, int] = {}
        calls: list[tuple[int, str, str]] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    name = self._tool_name(tc)
                    if not name or name in protected:
                        continue
                    call_id = tc.get("id") if isinstance(tc, dict) else None
                    if not isinstance(call_id, str):
                        continue
                    sig = self._tool_signature(tc)
                    calls.append((idx, call_id, sig))
                    latest_by_sig[sig] = idx
            elif msg.get("role") == "tool":
                call_id = msg.get("tool_call_id")
                if isinstance(call_id, str):
                    result_by_call_id[call_id] = idx
        protected_indices = self._turn_protected_indices(messages)
        for call_idx, call_id, sig in calls:
            if latest_by_sig.get(sig) == call_idx:
                continue
            result_idx = result_by_call_id.get(call_id)
            if result_idx is not None and result_idx not in protected_indices:
                msg = self._clone_if_needed(messages, result_idx, mutated)
                msg["content"] = "[DCP: duplicate tool output removed. Same tool and arguments were called again later.]"

    def _apply_purge_errors(self, messages: list[dict[str, Any]], mutated: set[int]) -> None:
        protected = DCP_DEFAULT_PROTECTED_TOOLS | self.config.purge_errors.protected_tools
        keep_tail = max(0, self.config.purge_errors.turns * 2)
        cutoff = max(0, len(messages) - keep_tail)
        call_name_by_id: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                call_id = tc.get("id") if isinstance(tc, dict) else None
                name = self._tool_name(tc)
                if isinstance(call_id, str) and name:
                    call_name_by_id[call_id] = name
        protected_indices = self._turn_protected_indices(messages)
        for idx, msg in enumerate(messages[:cutoff]):
            if idx in protected_indices or msg.get("role") != "tool":
                continue
            call_id = msg.get("tool_call_id")
            name = call_name_by_id.get(call_id) if isinstance(call_id, str) else None
            if name in protected:
                continue
            content = msg.get("content")
            if isinstance(content, str) and len(content) > 240 and _ERROR_RE.search(content):
                first_line = content.strip().splitlines()[0][:240]
                cloned = self._clone_if_needed(messages, idx, mutated)
                cloned["content"] = f"[DCP: old failed tool output pruned after {self.config.purge_errors.turns} turns. Error preserved: {first_line}]"

    def _inject_system_extension(self, messages: list[dict[str, Any]], mutated: set[int]) -> None:
        if self.config.compress.permission == "deny":
            return
        if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
            if _DCP_SYSTEM_EXTENSION not in messages[0]["content"]:
                msg = self._clone_if_needed(messages, 0, mutated)
                msg["content"] = f"{msg['content']}\n\n{_DCP_SYSTEM_EXTENSION}"

    def _inject_nudge(self, messages: list[dict[str, Any]], *, api_call_count: int, mutated: set[int]) -> None:
        # Use the provider-reported token count from the last response if
        # available — avoids re-estimating tokens on every API call.
        prompt_tokens = self.last_prompt_tokens
        max_limit = self._max_limit()
        min_limit = self._min_limit()
        nudge: str | None = None
        if max_limit and prompt_tokens >= max_limit:
            force = (
                "Before continuing, call compress on any completed range if safe."
                if self.config.compress.nudge_force == "strong"
                else "Consider calling compress on completed older ranges before continuing."
            )
            nudge = f"DCP context pressure is high (~{prompt_tokens:,} tokens). {force}"
        elif min_limit and prompt_tokens >= min_limit and self.state.turns_since_last_compress >= self.config.compress.nudge_frequency:
            nudge = "DCP: context is growing. If an older topic is complete, use compress with the visible refs."
        elif self.state.messages_since_last_user >= self.config.compress.iteration_nudge_threshold:
            nudge = "DCP: many assistant/tool messages have accumulated since the last user turn. Compress closed context if safe."
        elif self.state.manual_mode == "compress-pending":
            focus = f" Focus: {self.state.pending_manual_focus}." if self.state.pending_manual_focus else ""
            nudge = f"DCP manual compression requested.{focus} Call compress before continuing if there is safe completed context."
            self.state.manual_mode = False
            self.state.pending_manual_focus = None

        if not nudge:
            return
        # Only inject into user messages — never into tool results or
        # assistant messages, which could violate provider message semantics.
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                cloned = self._clone_if_needed(messages, idx, mutated)
                cloned["content"] = f"{cloned['content']}\n\n<dcp-nudge>{nudge}</dcp-nudge>"
                return

    # -- Helpers ----------------------------------------------------------

    def _message_key(self, msg: dict[str, Any], idx: int) -> str:
        return f"{idx}:{self._content_signature(msg)}"

    def _content_signature(self, msg: dict[str, Any]) -> str:
        """Signature based on role + content only.

        Excludes tool_calls and tool_call_id because the API copy may have
        been re-serialised (sorted JSON keys) by _canonicalize_api_tool_calls,
        which would produce a different hash than the canonical message.
        """
        cache_key = id(msg)
        cached = self._sig_cache.get(cache_key)
        if cached is not None:
            return cached
        clean = {
            "role": msg.get("role"),
            "content": msg.get("content"),
        }
        raw = json.dumps(clean, sort_keys=True, default=str, separators=(",", ":"))
        sig = hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()
        self._sig_cache[cache_key] = sig
        return sig

    def _require_str(self, args: dict[str, Any], key: str) -> str:
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"compress.{key} must be a non-empty string")
        return value.strip()

    def _resolve_range(self, start_ref: str, end_ref: str) -> tuple[list[str], list[int]]:
        start_idx = self._resolve_ref_to_index(start_ref)
        end_idx = self._resolve_ref_to_index(end_ref)
        if start_idx is None:
            raise ValueError(f"Unknown startId: {start_ref}")
        if end_idx is None:
            raise ValueError(f"Unknown endId: {end_ref}")
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        refs = [ref for ref, idx in self.state.index_by_ref.items() if start_idx <= idx <= end_idx]
        refs.sort(key=lambda ref: self.state.index_by_ref[ref])
        included_blocks = [
            block.block_id
            for block in self.state.active_blocks()
            if any(ref in refs for ref in block.message_refs)
        ]
        return refs, included_blocks

    def _resolve_ref_to_index(self, ref: str) -> int | None:
        if ref.startswith("m"):
            return self.state.index_by_ref.get(ref)
        if ref.startswith("b"):
            try:
                block_id = int(ref[1:])
            except ValueError:
                return None
            block = self.state.blocks_by_id.get(block_id)
            if not block or not block.message_refs:
                return None
            return self.state.index_by_ref.get(block.message_refs[0])
        return None

    def _augment_summary(self, summary: str, message_refs: list[str]) -> str:
        parts = [summary.strip()]
        if self.config.compress.protect_user_messages:
            parts.append(f"Covered refs: {', '.join(message_refs)}")
        return "\n\n".join(part for part in parts if part)

    def _block_summary_text(self, block: CompressionBlock) -> str:
        covers = f"{block.start_ref}-{block.end_ref}" if block.start_ref and block.end_ref else ", ".join(block.message_refs)
        return (
            f'<dcp-compressed-block id="{block.ref}" topic="{block.topic}">\n'
            f"Summary: {block.summary}\n"
            f"Covers: {covers}\n"
            "</dcp-compressed-block>"
        )

    def _tool_name(self, tool_call: Any) -> str | None:
        if not isinstance(tool_call, dict):
            return None
        function = tool_call.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
        return None

    def _tool_signature(self, tool_call: dict[str, Any]) -> str:
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = function.get("name", "")
        args = function.get("arguments", "")
        try:
            args_obj = json.loads(args) if isinstance(args, str) else args
            args_norm = json.dumps(args_obj, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            args_norm = str(args)
        return f"{name}::{args_norm}"

    def _automatic_strategies_enabled(self) -> bool:
        if self.config.manual_mode.enabled and not self.config.manual_mode.automatic_strategies:
            return False
        return True

    def _turn_protected_indices(self, messages: list[dict[str, Any]]) -> set[int]:
        if not self.config.turn_protection.enabled or self.config.turn_protection.turns <= 0:
            return set()
        user_indices = [idx for idx, msg in enumerate(messages) if msg.get("role") == "user"]
        if not user_indices:
            return set()
        start = user_indices[-self.config.turn_protection.turns] if len(user_indices) >= self.config.turn_protection.turns else user_indices[0]
        return set(range(start, len(messages)))

    def _evict_inactive_blocks(self) -> None:
        """Bound memory by evicting old deactivated blocks."""
        inactive = sorted(
            (bid for bid, b in self.state.blocks_by_id.items() if not b.active),
            key=lambda bid: self.state.blocks_by_id[bid].deactivated_at or 0,
        )
        for bid in inactive[_MAX_INACTIVE_BLOCKS:]:
            del self.state.blocks_by_id[bid]

    def _min_limit(self) -> int:
        return resolve_model_limit(
            self.config.compress.model_min_limits,
            provider=self.provider,
            model=self.model,
            context_length=self.context_length,
            fallback=self.config.compress.min_context_limit,
        )

    def _max_limit(self) -> int:
        return resolve_model_limit(
            self.config.compress.model_max_limits,
            provider=self.provider,
            model=self.model,
            context_length=self.context_length,
            fallback=self.config.compress.max_context_limit,
        )
