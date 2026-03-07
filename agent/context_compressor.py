"""Automatic context window compression for long conversations.

Self-contained class with its own OpenAI client for summarization.
Uses Gemini Flash (cheap/fast) to summarize middle turns while
protecting head and tail context.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.auxiliary_client import get_text_auxiliary_client
from agent.model_metadata import (
    get_model_context_length,
    estimate_messages_tokens_rough,
)

logger = logging.getLogger(__name__)

NEVER_PRUNE_TOOLS = {"clarify", "memory", "skill_view", "todo", "read_file"}

COMPACTION_TEMPLATE = """Summarize the compressed conversation turns into a structured resumption context.
Use EXACTLY this format:

## Goal
[What is the user trying to accomplish?]

## Standing Instructions
[Any persistent instructions or constraints from the user]

## Key Discoveries
[Important findings, relevant code, data, error messages discovered]

## Accomplished So Far
[What has been completed -- files changed, commands run, tests passed]

## Relevant Files & Paths
[All file paths, URLs, and resources that matter for the ongoing task]

## Next Steps
[What was the agent about to do when compression triggered?]

Keep each section concise but complete. Omit sections that have no content.
"""


def _adaptive_prune_protect(context_length: int) -> int:
    """Scale PRUNE_PROTECT tokens based on model context window."""
    if context_length >= 500_000:
        return 100_000
    if context_length >= 128_000:
        return 40_000
    if context_length >= 64_000:
        return 20_000
    return 10_000


def _adaptive_prune_minimum(context_length: int) -> int:
    """Minimum tokens to reclaim before pruning is worthwhile."""
    return max(5_000, context_length // 20)


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
        self._prune_protect_tokens = _adaptive_prune_protect(self.context_length)
        self._prune_minimum_tokens = _adaptive_prune_minimum(self.context_length)

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

    def _is_protected_tool(self, message: Dict[str, Any]) -> bool:
        """Return True if this tool message should never be pruned."""
        name = message.get("name", "") or ""
        return name in NEVER_PRUNE_TOOLS

    def _prune_tool_outputs(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Phase 1: Replace old tool outputs with short placeholders. No LLM call needed.

        Walks backwards through messages, keeping recent tool outputs within
        _prune_protect_tokens. Older tool outputs beyond that window are replaced
        with a short placeholder. Protected tools (memory, clarify, etc.) are
        never pruned.

        Returns:
            (pruned_messages, chars_saved)
        """
        pruned = list(messages)
        chars_saved = 0
        recent_tool_tokens = 0
        prune_protect = self._prune_protect_tokens

        for i in range(len(pruned) - 1, self.protect_first_n - 1, -1):
            msg = pruned[i]
            if msg.get("role") != "tool":
                continue
            if self._is_protected_tool(msg):
                continue
            content = msg.get("content") or ""
            token_estimate = len(content) // 4
            if recent_tool_tokens < prune_protect:
                # Still within the protected recent window — keep as-is
                recent_tool_tokens += token_estimate
            else:
                # Beyond protected window — prune this tool output
                original_len = len(content)
                placeholder = f"[Tool output pruned — was {original_len:,} chars]"
                pruned[i] = {**msg, "content": placeholder}
                chars_saved += original_len - len(placeholder)

        return pruned, chars_saved

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
        prompt = (
            COMPACTION_TEMPLATE
            + f"\nTarget ~{self.summary_target_tokens} tokens total.\n"
            + f"\n---\nTURNS TO SUMMARIZE:\n{content_to_summarize}"
            + "\n---\n\nWrite only the structured summary starting with [CONTEXT SUMMARY]: prefix."
        )

        # 1. Try the auxiliary model (cheap/fast)
        if self.client:
            try:
                return self._call_summary_model(self.client, self.summary_model, prompt)
            except Exception as e:
                logging.warning(f"Failed to generate context summary with auxiliary model: {e}")

        # 2. Fallback: try the user's main model endpoint
        fallback_client, fallback_model = self._get_fallback_client()
        if fallback_client is not None:
            try:
                logger.info("Retrying context summary with main model (%s)", fallback_model)
                summary = self._call_summary_model(fallback_client, fallback_model, prompt)
                self.client = fallback_client
                self.summary_model = fallback_model
                return summary
            except Exception as fallback_err:
                logging.warning(f"Main model summary also failed: {fallback_err}")

        # 3. All models failed — return None so the caller drops turns without a summary
        logging.warning("Context compression: no model available for summary. Middle turns will be dropped without summary.")
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
            if "max_tokens" in str(first_err) or "unsupported_parameter" in str(first_err):
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = self.summary_target_tokens * 2
                response = client.chat.completions.create(**kwargs)
            else:
                raise

        summary = response.choices[0].message.content.strip()
        if not summary.startswith("[CONTEXT SUMMARY]:"):
            summary = "[CONTEXT SUMMARY]: " + summary
        return summary

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
            logger.debug("Built fallback auxiliary client: %s via %s", model, custom_base)
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

        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        if not self.quiet_mode:
            print(f"\n📦 Context compression triggered ({display_tokens:,} tokens ≥ {self.threshold_tokens:,} threshold)")
            print(f"   📊 Model context limit: {self.context_length:,} tokens ({self.threshold_percent*100:.0f}% = {self.threshold_tokens:,})")

        # Phase 1: prune old tool outputs — free, no LLM call needed
        pruned_messages, chars_saved = self._prune_tool_outputs(messages)
        pruned_tokens = estimate_messages_tokens_rough(pruned_messages)
        tokens_saved_phase1 = max(0, display_tokens - pruned_tokens)

        if chars_saved > 0 and not self.quiet_mode:
            print(f"   ✂️  Phase 1 (prune): removed {chars_saved:,} chars of old tool outputs (~{tokens_saved_phase1:,} tokens saved)")

        if chars_saved > 0 and pruned_tokens < self.threshold_tokens:
            self.compression_count += 1
            if not self.quiet_mode:
                print(f"   ✅ Phase 1 sufficient: {n_messages} → {len(pruned_messages)} messages, now {pruned_tokens:,} tokens")
                print(f"   💡 Compression #{self.compression_count} complete (prune only — no LLM call needed)")
            return self._sanitize_tool_pairs(pruned_messages)

        # Phase 2: LLM summarization of middle turns
        messages = pruned_messages
        n_messages = len(messages)
        compress_start = self.protect_first_n
        compress_end = n_messages - self.protect_last_n
        if compress_start >= compress_end:
            return messages
        compress_start = self._align_boundary_forward(messages, compress_start)
        compress_end = self._align_boundary_backward(messages, compress_end)
        if compress_start >= compress_end:
            return messages

        turns_to_summarize = messages[compress_start:compress_end]

        if not self.quiet_mode:
            print(f"   🗜️  Phase 2 (compact): summarizing turns {compress_start+1}-{compress_end} ({len(turns_to_summarize)} turns)")

        summary = self._generate_summary(turns_to_summarize)

        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system" and self.compression_count == 0:
                msg["content"] = (msg.get("content") or "") + "\n\n[Note: Some earlier conversation turns may be summarized to preserve context space.]"
            compressed.append(msg)

        if summary:
            compressed.append({"role": "user", "content": summary})
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
