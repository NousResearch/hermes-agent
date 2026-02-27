"""Automatic context window compression for long conversations.

Self-contained class with its own OpenAI client for summarization.
Uses Gemini Flash (cheap/fast) to summarize middle turns while
protecting head and tail context.
"""

import logging
import os
from typing import Any, Dict, List

from agent.auxiliary_client import get_text_auxiliary_client
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
        threshold_percent: float = 0.85,
        protect_first_n: int = 3,
        protect_last_n: int = 4,
        summary_target_tokens: int = 500,
        quiet_mode: bool = False,
    ):
        self.model = model
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_tokens = summary_target_tokens
        self.quiet_mode = quiet_mode

        self.context_length = get_model_context_length(model)
        self.threshold_tokens = int(self.context_length * threshold_percent)
        self.compression_count = 0

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

        self.client, self.summary_model = get_text_auxiliary_client()

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

    def _generate_summary(self, turns_to_summarize: List[Dict[str, Any]]) -> str:
        """Generate a concise summary of conversation turns using a fast model."""
        if not self.client:
            return "[CONTEXT SUMMARY]: Previous conversation turns have been compressed to save space. The assistant performed various actions and received responses."

        parts = []
        for msg in turns_to_summarize:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
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

        try:
            kwargs = {
                "model": self.summary_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "timeout": 30.0,
            }
            # Most providers (OpenRouter, local models) use max_tokens.
            # Direct OpenAI with newer models (gpt-4o, o-series, gpt-5+)
            # requires max_completion_tokens instead.
            try:
                kwargs["max_tokens"] = self.summary_target_tokens * 2
                response = self.client.chat.completions.create(**kwargs)
            except Exception as first_err:
                if "max_tokens" in str(first_err) or "unsupported_parameter" in str(first_err):
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = self.summary_target_tokens * 2
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    raise

            summary = response.choices[0].message.content.strip()
            if not summary.startswith("[CONTEXT SUMMARY]:"):
                summary = "[CONTEXT SUMMARY]: " + summary
            return summary
        except Exception as e:
            logging.warning(f"Failed to generate context summary: {e}")
            return "[CONTEXT SUMMARY]: Previous conversation turns have been compressed. The assistant performed tool calls and received responses."

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None) -> List[Dict[str, Any]]:
        """Compress conversation messages by summarizing middle turns.

        Keeps first N + last N turns, summarizes everything in between.
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

        turns_to_summarize = messages[compress_start:compress_end]
        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        if not self.quiet_mode:
            print(f"\n📦 Context compression triggered ({display_tokens:,} tokens ≥ {self.threshold_tokens:,} threshold)")
            print(f"   📊 Model context limit: {self.context_length:,} tokens ({self.threshold_percent*100:.0f}% = {self.threshold_tokens:,})")

        # Truncation fallback when no auxiliary model is available
        if self.client is None:
            print("⚠️  Context compression: no auxiliary model available. Falling back to message truncation.")
            # Keep system message(s) at the front and the protected tail;
            # simply drop the oldest non-system messages until under threshold.
            kept = []
            for msg in messages:
                if msg.get("role") == "system":
                    kept.append(msg.copy())
                else:
                    break
            tail = messages[-self.protect_last_n:]
            kept.extend(m.copy() for m in tail)
            self.compression_count += 1
            if not self.quiet_mode:
                print(f"   ✂️  Truncated: {len(messages)} → {len(kept)} messages (dropped middle turns)")
            return kept

        if not self.quiet_mode:
            print(f"   🗜️  Summarizing turns {compress_start+1}-{compress_end} ({len(turns_to_summarize)} turns)")

        summary = self._generate_summary(turns_to_summarize)

        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system" and self.compression_count == 0:
                msg["content"] = msg.get("content", "") + "\n\n[Note: Some earlier conversation turns may be summarized to preserve context space.]"
            compressed.append(msg)

        compressed.append({"role": "user", "content": summary})

        for i in range(compress_end, n_messages):
            compressed.append(messages[i].copy())

        self.compression_count += 1

        if not self.quiet_mode:
            new_estimate = estimate_messages_tokens_rough(compressed)
            saved_estimate = display_tokens - new_estimate
            print(f"   ✅ Compressed: {n_messages} → {len(compressed)} messages (~{saved_estimate:,} tokens saved)")
            print(f"   💡 Compression #{self.compression_count} complete")

        return compressed
