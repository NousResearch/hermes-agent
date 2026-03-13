"""Automatic context window compression for long conversations.

Self-contained class with its own OpenAI client for summarization.
Uses Gemini Flash (cheap/fast) to summarize middle turns while
protecting head and tail context.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm
from agent.model_metadata import (
    get_model_context_length,
    estimate_messages_tokens_rough,
)

logger = logging.getLogger(__name__)


class ContextCompressor:
    """Compresses conversation context when approaching the model's context limit.

    Algorithm: protect first N + last N turns, summarize everything in between.
    Token tracking uses actual counts from API responses for accuracy.
    """

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 4,
        summary_target_tokens: int = 2500,
        quiet_mode: bool = False,
        summary_model_override: str = None,
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

        self.summary_model = summary_model_override or ""

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
            "usage_percent": (self.last_prompt_tokens / self.context_length * 100) if self.context_length else 0,
            "compression_count": self.compression_count,
        }

    def _generate_summary(self, turns_to_summarize: List[Dict[str, Any]]) -> Optional[str]:
        """Generate a concise summary of conversation turns.

        Tries the auxiliary model first, then falls back to the user's main
        model.  Returns None if all attempts fail — the caller should drop
        the middle turns without a summary rather than inject a useless
        placeholder.
        """
        parts = []
        for msg in turns_to_summarize:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            if len(content) > 2000:
                content = content[:1000] + "\n...[truncated]...\n" + content[-500:]
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls if isinstance(tc, dict)]
                content += f"\n[Tool calls: {', '.join(tool_names)}]"
            parts.append(f"[{role.upper()}]: {content}")

        content_to_summarize = "\n\n".join(parts)
        prompt = f"""Summarize these conversation turns concisely. This summary will replace these turns in the conversation history.

Write from a neutral perspective describing:
1. What actions were taken (tool calls, searches, file operations)
2. Key information or results obtained
3. Important decisions or findings
4. Relevant data, file names, or outputs

Keep factual and informative. Target ~{self.summary_target_tokens} tokens.

---
TURNS TO SUMMARIZE:
{content_to_summarize}
---

Write only the summary, starting with "[CONTEXT SUMMARY]:" prefix."""

        # Use the centralized LLM router — handles provider resolution,
        # auth, and fallback internally.
        try:
            call_kwargs = {
                "task": "compression",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": self.summary_target_tokens * 2,
                "timeout": 30.0,
            }
            if self.summary_model:
                call_kwargs["model"] = self.summary_model
            response = call_llm(**call_kwargs)
            summary = response.choices[0].message.content.strip()
            if not summary.startswith("[CONTEXT SUMMARY]:"):
                summary = "[CONTEXT SUMMARY]: " + summary
            return summary
        except RuntimeError:
            logging.warning("Context compression: no provider available for "
                            "summary. Middle turns will be dropped without summary.")
            return None
        except Exception as e:
            logging.warning("Failed to generate context summary: %s", e)
            return None

    # ------------------------------------------------------------------
    # Tool-call / tool-result pair integrity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_call_id(tc) -> str:
        """Extract the call ID from a tool_call entry (dict or SimpleNamespace)."""
        if isinstance(tc, dict):
            return tc.get("id", "")
        return getattr(tc, "id", "") or ""

    def _sanitize_tool_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            if not self.quiet_mode:
                logger.info("Compression sanitizer: removed %d orphaned tool result(s)", len(orphaned_results))

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
                            patched.append({
                                "role": "tool",
                                "content": "[Result from earlier conversation — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            if not self.quiet_mode:
                logger.info("Compression sanitizer: added %d stub tool result(s)", len(missing_results))

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

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None) -> List[Dict[str, Any]]:
        """Compress conversation messages by summarizing middle turns.

        Keeps first N + last N turns, summarizes everything in between.
        After compression, orphaned tool_call / tool_result pairs are cleaned
        up so the API never receives mismatched IDs.
        """
        n_messages = len(messages)
        if n_messages <= self.protect_first_n + self.protect_last_n + 1:
            if not self.quiet_mode:
                print(f"⚠️  Cannot compress: only {n_messages} messages (need > {self.protect_first_n + self.protect_last_n + 1})")
            return messages

        compress_start = self.protect_first_n
        compress_end = n_messages - self.protect_last_n
        if compress_start >= compress_end:
            return messages

        # Adjust boundaries to avoid splitting tool_call/result groups.
        compress_start = self._align_boundary_forward(messages, compress_start)
        compress_end = self._align_boundary_backward(messages, compress_end)
        if compress_start >= compress_end:
            return messages

        turns_to_summarize = messages[compress_start:compress_end]
        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        if not self.quiet_mode:
            print(f"\n📦 Context compression triggered ({display_tokens:,} tokens ≥ {self.threshold_tokens:,} threshold)")
            print(f"   📊 Model context limit: {self.context_length:,} tokens ({self.threshold_percent*100:.0f}% = {self.threshold_tokens:,})")

        if not self.quiet_mode:
            print(f"   🗜️  Summarizing turns {compress_start+1}-{compress_end} ({len(turns_to_summarize)} turns)")

        summary = self._generate_summary(turns_to_summarize)

        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system" and self.compression_count == 0:
                msg["content"] = (msg.get("content") or "") + "\n\n[Note: Some earlier conversation turns may be summarized to preserve context space.]"
            compressed.append(msg)

        if summary:
            last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
            summary_role = "user" if last_head_role in ("assistant", "tool") else "assistant"
            compressed.append({"role": summary_role, "content": summary})
        else:
            if not self.quiet_mode:
                print("   ⚠️  No summary model available — middle turns dropped without summary")

        for i in range(compress_end, n_messages):
            compressed.append(messages[i].copy())

        self.compression_count += 1

        compressed = self._sanitize_tool_pairs(compressed)

        if not self.quiet_mode:
            new_estimate = estimate_messages_tokens_rough(compressed)
            saved_estimate = display_tokens - new_estimate
            print(f"   ✅ Compressed: {n_messages} → {len(compressed)} messages (~{saved_estimate:,} tokens saved)")
            print(f"   💡 Compression #{self.compression_count} complete")

        return compressed
