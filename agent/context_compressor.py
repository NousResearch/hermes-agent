"""Automatic context window compression for long conversations.

Self-contained class with its own OpenAI client for summarization.
Uses Gemini Flash (cheap/fast) to summarize middle turns while
protecting head and tail context.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import get_text_auxiliary_client
from agent.model_metadata import (
    get_model_context_length,
    estimate_messages_tokens_rough,
)

logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work.
"""
SUMMARY_PREFIX = """Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:"""
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"
SUMMARY_PREFIX_MARKERS = (
    SUMMARY_PREFIX,
    LEGACY_SUMMARY_PREFIX,
    "[CONTEXT COMPACTION]",
)
APPROX_BYTES_PER_TOKEN = 4
DEFAULT_USER_MESSAGE_TOKEN_BUDGET = 20_000
MULTI_COMPACTION_WARNING = (
    "Heads up: Long threads and multiple compactions can cause the model to be less accurate. "
    "Start a new thread when possible to keep threads small and targeted."
)


class ContextCompressor:
    """Compresses conversation context when approaching the model's context limit.

    Algorithm: protect first N + last N turns, summarize everything in between.
    Token tracking uses actual counts from API responses for accuracy.
    """

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.85,
        protect_first_n: int = 3,
        protect_last_n: int = 4,
        summary_target_tokens: int = 2500,
        quiet_mode: bool = False,
        summary_model_override: str = None,
        compaction_prompt_override: str = None,
        user_message_token_budget: int = DEFAULT_USER_MESSAGE_TOKEN_BUDGET,
        base_url: str = "",
    ):
        self.model = model
        self.base_url = base_url
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_tokens = summary_target_tokens
        self.quiet_mode = quiet_mode

        self.context_length = get_model_context_length(model, base_url=base_url)
        self.threshold_tokens = int(self.context_length * threshold_percent)
        self.compression_count = 0
        self._context_probed = False  # True after a step-down from context error

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

        self.client, default_model = get_text_auxiliary_client()
        self.summary_model = summary_model_override or default_model
        self.compaction_prompt = (
            compaction_prompt_override.strip()
            if compaction_prompt_override and compaction_prompt_override.strip()
            else DEFAULT_COMPACTION_PROMPT
        )
        self.user_message_token_budget = max(0, int(user_message_token_budget))

    def update_from_response(self, usage: Dict[str, Any]):
        """Update tracked token usage from API response."""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Check if context exceeds the compression threshold."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick pre-flight check using rough estimate (before API call)."""
        rough_estimate = estimate_messages_tokens_rough(messages)
        return rough_estimate >= self.threshold_tokens

    def get_status(self) -> Dict[str, Any]:
        """Get current compression status for display/logging."""
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (self.last_prompt_tokens / self.context_length * 100)
            if self.context_length
            else 0,
            "compression_count": self.compression_count,
        }

    def _generate_summary(
        self, turns_to_summarize: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate a concise summary of conversation turns.

        Tries the auxiliary model first, then falls back to the user's main
        model.  Returns None if all attempts fail — the caller should drop
        the middle turns without a summary rather than inject a useless
        placeholder.
        """
        parts = []
        for msg in turns_to_summarize:
            role = msg.get("role", "unknown")
            content = self._content_to_text(msg.get("content"))
            if len(content) > 2000:
                content = content[:1000] + "\n...[truncated]...\n" + content[-500:]
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tool_names = [
                    tc.get("function", {}).get("name", "?")
                    for tc in tool_calls
                    if isinstance(tc, dict)
                ]
                content += f"\n[Tool calls: {', '.join(tool_names)}]"
            parts.append(f"[{role.upper()}]: {content}")

        content_to_summarize = "\n\n".join(parts)
        prompt = (
            f"{self.compaction_prompt.strip()}\n\n"
            f"Target roughly {self.summary_target_tokens} tokens.\n\n"
            "---\n"
            f"TURNS TO SUMMARIZE:\n{content_to_summarize}\n"
            "---\n\n"
            "Write only the handoff summary."
        )

        # 1. Try the auxiliary model (cheap/fast)
        if self.client:
            try:
                return self._call_summary_model(self.client, self.summary_model, prompt)
            except Exception as e:
                logging.warning(
                    f"Failed to generate context summary with auxiliary model: {e}"
                )

        # 2. Fallback: try the user's main model endpoint
        fallback_client, fallback_model = self._get_fallback_client()
        if fallback_client is not None:
            try:
                logger.info(
                    "Retrying context summary with main model (%s)", fallback_model
                )
                summary = self._call_summary_model(
                    fallback_client, fallback_model, prompt
                )
                self.client = fallback_client
                self.summary_model = fallback_model
                return summary
            except Exception as fallback_err:
                logging.warning(f"Main model summary also failed: {fallback_err}")

        # 3. All models failed — return None so the caller drops turns without a summary
        logging.warning(
            "Context compression: no model available for summary. Middle turns will be dropped without summary."
        )
        return None

    def _call_summary_model(self, client, model: str, prompt: str) -> str:
        """Make the actual LLM call to generate a summary. Raises on failure."""
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "timeout": 30.0,
        }
        # Most providers (OpenRouter, local models) use max_tokens.
        # Direct OpenAI with newer models (gpt-4o, o-series, gpt-5+)
        # requires max_completion_tokens instead.
        try:
            kwargs["max_tokens"] = self.summary_target_tokens * 2
            response = client.chat.completions.create(**kwargs)
        except Exception as first_err:
            if "max_tokens" in str(first_err) or "unsupported_parameter" in str(
                first_err
            ):
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = self.summary_target_tokens * 2
                response = client.chat.completions.create(**kwargs)
            else:
                raise

        summary = response.choices[0].message.content.strip()
        return self._with_summary_prefix(summary)

    def _with_summary_prefix(self, summary: str) -> str:
        normalized = (summary or "").strip()
        for prefix in SUMMARY_PREFIX_MARKERS:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].lstrip()
                break
        return f"{SUMMARY_PREFIX}\n{normalized}"

    def _get_fallback_client(self):
        """Try to build a fallback client from the main model's endpoint config.

        When the primary auxiliary client fails (e.g. stale OpenRouter key), this
        creates a client using the user's active custom endpoint (OPENAI_BASE_URL)
        so compression can still produce a real summary instead of a static string.

        Returns (client, model) or (None, None).
        """
        custom_base = os.getenv("OPENAI_BASE_URL")
        custom_key = os.getenv("OPENAI_API_KEY")
        if not custom_base or not custom_key:
            return None, None

        # Don't fallback to the same provider that just failed
        from hermes_constants import OPENROUTER_BASE_URL

        if custom_base.rstrip("/") == OPENROUTER_BASE_URL.rstrip("/"):
            return None, None

        model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or self.model
        try:
            from openai import OpenAI as _OpenAI

            client = _OpenAI(api_key=custom_key, base_url=custom_base)
            logger.debug(
                "Built fallback auxiliary client: %s via %s", model, custom_base
            )
            return client, model
        except Exception as exc:
            logger.debug("Could not build fallback auxiliary client: %s", exc)
            return None, None

    # ------------------------------------------------------------------
    # Tool-call / tool-result pair integrity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_call_id(tc) -> str:
        """Extract the call ID from a tool_call entry (dict or SimpleNamespace)."""
        if isinstance(tc, dict):
            return tc.get("id", "")
        return getattr(tc, "id", "") or ""

    def _sanitize_tool_pairs(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs after compression.

        Two failure modes:
        1. A tool *result* references a call_id whose assistant tool_call was
           removed (summarized/truncated).  The API rejects this with
           "No tool call found for function call output with call_id ...".
        2. An assistant message has tool_calls whose results were dropped.
           The API rejects this because every tool_call must be followed by
           a tool result with the matching call_id.

        This method removes orphaned results and inserts stub results for
        orphaned calls so the message list is always well-formed.
        """
        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = self._get_tool_call_id(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Remove tool results whose call_id has no matching assistant tool_call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m
                for m in messages
                if not (
                    m.get("role") == "tool"
                    and m.get("tool_call_id") in orphaned_results
                )
            ]
            if not self.quiet_mode:
                logger.info(
                    "Compression sanitizer: removed %d orphaned tool result(s)",
                    len(orphaned_results),
                )

        # 2. Add stub results for assistant tool_calls whose results were dropped
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = self._get_tool_call_id(tc)
                        if cid in missing_results:
                            patched.append(
                                {
                                    "role": "tool",
                                    "content": "[Result from earlier conversation — see context summary above]",
                                    "tool_call_id": cid,
                                }
                            )
            messages = patched
            if not self.quiet_mode:
                logger.info(
                    "Compression sanitizer: added %d stub tool result(s)",
                    len(missing_results),
                )

        return messages

    def _align_boundary_forward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Push a compress-start boundary forward past any orphan tool results.

        If ``messages[idx]`` is a tool result, slide forward until we hit a
        non-tool message so we don't start the summarised region mid-group.
        """
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _align_boundary_backward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Pull a compress-end boundary backward to avoid splitting a
        tool_call / result group.

        If the message just before ``idx`` is an assistant message with
        tool_calls, those tool results will start at ``idx`` and would be
        separated from their parent.  Move backwards to include the whole
        group in the summarised region.
        """
        if idx <= 0 or idx >= len(messages):
            return idx
        prev = messages[idx - 1]
        if prev.get("role") == "assistant" and prev.get("tool_calls"):
            # The results for this assistant turn sit at idx..idx+k.
            # Include the assistant message in the summarised region too.
            idx -= 1
        return idx

    def _is_compaction_summary_message(self, message: Dict[str, Any]) -> bool:
        if message.get("role") != "user":
            return False
        content = message.get("content")
        if not isinstance(content, str):
            return False
        content = content.strip()
        return any(content.startswith(prefix) for prefix in SUMMARY_PREFIX_MARKERS)

    def _approx_compaction_tokens(self, text: str) -> int:
        if not text:
            return 0
        byte_len = len(text.encode("utf-8"))
        return max((byte_len + APPROX_BYTES_PER_TOKEN - 1) // APPROX_BYTES_PER_TOKEN, 1)

    def _approx_message_content_tokens(self, content: Any) -> int:
        if isinstance(content, str):
            return self._approx_compaction_tokens(content)
        if content is None:
            return 0
        try:
            serialized = json.dumps(content, ensure_ascii=False)
        except TypeError:
            serialized = str(content)
        return self._approx_compaction_tokens(serialized)

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append(item.get("text", ""))
                    elif item_type == "image_url":
                        parts.append("[image]")
                    elif item_type:
                        parts.append(f"[{item_type}]")
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(content)

    @staticmethod
    def _prefix_on_utf8_boundary(data: bytes, budget: int) -> bytes:
        end = min(max(budget, 0), len(data))
        while end > 0 and end < len(data) and (data[end] & 0xC0) == 0x80:
            end -= 1
        return data[:end]

    @staticmethod
    def _suffix_on_utf8_boundary(data: bytes, budget: int) -> bytes:
        start = max(len(data) - max(budget, 0), 0)
        while start < len(data) and (data[start] & 0xC0) == 0x80:
            start += 1
        return data[start:]

    def _truncate_preserved_user_content(self, content: Any, token_budget: int) -> str:
        if token_budget <= 0:
            return ""

        content = self._content_to_text(content)
        data = content.encode("utf-8")
        max_bytes = token_budget * APPROX_BYTES_PER_TOKEN
        if len(data) <= max_bytes:
            return content

        left_budget = max_bytes // 2
        right_budget = max_bytes - left_budget
        prefix = self._prefix_on_utf8_boundary(data, left_budget)
        suffix = self._suffix_on_utf8_boundary(data, right_budget)
        if len(data) - len(suffix) < len(prefix):
            suffix = data[len(prefix) :]

        removed_bytes = max(len(data) - len(prefix) - len(suffix), 0)
        removed_tokens = (
            removed_bytes + APPROX_BYTES_PER_TOKEN - 1
        ) // APPROX_BYTES_PER_TOKEN
        marker = f"…{removed_tokens} tokens truncated…"
        return (
            prefix.decode("utf-8", errors="ignore")
            + marker
            + suffix.decode("utf-8", errors="ignore")
        )

    def _collect_preserved_user_messages(
        self, messages: List[Dict[str, Any]], *, keep_latest_user_full: bool = False
    ) -> List[tuple[int, Dict[str, Any]]]:
        preserved_messages: List[tuple[int, Dict[str, Any]]] = []
        remaining_tokens = self.user_message_token_budget
        start_idx = len(messages) - 1

        if keep_latest_user_full:
            for idx in range(len(messages) - 1, -1, -1):
                message = messages[idx]
                if message.get("role") == "user" and not self._is_compaction_summary_message(
                    message
                ):
                    preserved_messages.append((idx, message.copy()))
                    start_idx = idx - 1
                    break

        for idx in range(start_idx, -1, -1):
            if remaining_tokens == 0:
                break
            message = messages[idx]
            if message.get("role") != "user" or self._is_compaction_summary_message(
                message
            ):
                continue

            content = message.get("content")
            message_tokens = max(self._approx_message_content_tokens(content), 1)
            if message_tokens <= remaining_tokens:
                preserved_messages.append((idx, message.copy()))
                remaining_tokens = max(remaining_tokens - message_tokens, 0)
                continue

            truncated_message = message.copy()
            truncated_message["content"] = self._truncate_preserved_user_content(
                content, remaining_tokens
            )
            if truncated_message["content"]:
                preserved_messages.append((idx, truncated_message))
            break

        preserved_messages.reverse()
        return preserved_messages

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        *,
        keep_latest_user_full: bool = False,
    ) -> List[Dict[str, Any]]:
        n_messages = len(messages)
        if n_messages <= self.protect_first_n + self.protect_last_n + 1:
            if not self.quiet_mode:
                print(
                    f"⚠️  Cannot compress: only {n_messages} messages (need > {self.protect_first_n + self.protect_last_n + 1})"
                )
            return messages

        display_tokens = (
            current_tokens
            if current_tokens
            else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)
        )

        preserved_user_messages = self._collect_preserved_user_messages(
            messages, keep_latest_user_full=keep_latest_user_full
        )
        preserved_user_indexes = {idx for idx, _ in preserved_user_messages}
        stale_summary_indexes = {
            idx
            for idx, message in enumerate(messages)
            if self._is_compaction_summary_message(message)
        }
        turns_to_summarize = [
            message
            for idx, message in enumerate(messages)
            if idx not in preserved_user_indexes
            and idx not in stale_summary_indexes
            and message.get("role") != "system"
        ]

        if not turns_to_summarize and not stale_summary_indexes:
            return messages

        if not self.quiet_mode:
            print(
                f"\n📦 Context compression triggered ({display_tokens:,} tokens ≥ {self.threshold_tokens:,} threshold)"
            )
            print(
                f"   📊 Model context limit: {self.context_length:,} tokens ({self.threshold_percent * 100:.0f}% = {self.threshold_tokens:,})"
            )

        if not self.quiet_mode:
            print(
                f"   🗜️  Summarizing {len(turns_to_summarize)} message(s) after preserving recent user requests"
            )

        summary = (
            self._generate_summary(turns_to_summarize) if turns_to_summarize else None
        )

        compressed = [
            message.copy() for message in messages if message.get("role") == "system"
        ]

        current_user_message = (
            preserved_user_messages[-1][1]
            if keep_latest_user_full and preserved_user_messages
            else None
        )
        older_preserved_user_messages = (
            preserved_user_messages[:-1]
            if current_user_message is not None
            else preserved_user_messages
        )

        for _, message in older_preserved_user_messages:
            compressed.append(message)

        if summary:
            compressed.append({"role": "user", "content": summary})
        else:
            if not self.quiet_mode:
                print(
                    "   ⚠️  No summary model available — middle turns dropped without summary"
                )

        if current_user_message is not None:
            compressed.append(current_user_message)

        self.compression_count += 1

        compressed = self._sanitize_tool_pairs(compressed)

        logger.warning(MULTI_COMPACTION_WARNING)
        if not self.quiet_mode:
            print(f"   ⚠️  {MULTI_COMPACTION_WARNING}")

        if not self.quiet_mode:
            new_estimate = estimate_messages_tokens_rough(compressed)
            saved_estimate = display_tokens - new_estimate
            print(
                f"   ✅ Compressed: {n_messages} → {len(compressed)} messages (~{saved_estimate:,} tokens saved)"
            )
            print(f"   💡 Compression #{self.compression_count} complete")

        return compressed
