"""
Agentic stall-retry (dflash Q4 premature-EOS workaround).

dflash (Qwen3.6-27B Q4_K_M, lucebox spec-decode) sometimes emits EOS right
after a short action preamble ("Let me check X:") on agentic decision turns,
ending the turn with NO tool_call -> the agent loop treats it as a final
answer and stops mid-task. Higher-precision weights (the stock Q6 lane on the
same host) continue to a real tool call on the identical prompt.

This module detects that stall signature on a no-tool-call turn and retries
the SAME turn once against a higher-quality model lane. If the retry produces
tool_calls, the loop adopts that response and continues; otherwise the
original response stands (no behavior change).

Entirely opt-in: does nothing unless ``HERMES_STALL_RETRY_MODEL`` is set
(e.g. ``qwen3.6-27b-256k``). Default-off => zero change to existing behavior.

Env:
  HERMES_STALL_RETRY_MODEL  retry lane/model name (required to enable)
  HERMES_STALL_RETRY_MAX_CHARS  max content length to still count as a stall
                                (default 400; real final answers are longer)
"""
from __future__ import annotations

import os
import re

# Action-preamble signature: the turn announced an action but produced no tool
# call. These end mid-thought, typically with a colon, or open with intent.
_ACTION_RE = re.compile(
    r"(let me\b|let's\b|i'?ll\b|i will\b|i'?m going to\b|i am going to\b|"
    r"now i\b|first,?\s+i\b|next,?\s+i\b|i need to\b|i should\b|"
    r"going to (check|look|run|start|examine|search|read|list|create|write|edit|use))",
    re.IGNORECASE,
)
# Genuine completion signature: the model declared it is done / nothing to do.
# These must NOT be retried (they are correct no-tool-call turns).
_COMPLETION_RE = re.compile(
    r"(\bdone\b|\bcomplete(d)?\b|nothing to (do|save|change|report|fix)|"
    r"no changes?\b|no action\b|already (complete|done|finished)|\bfinished\b|"
    r"all set\b|no further\b|nothing left\b|here('?s| is| are)\b|"
    r"in summary\b|to summarize\b|the answer is\b)",
    re.IGNORECASE,
)


def looks_like_stall(content: str, finish_reason: str, has_tool_calls: bool,
                     max_chars: int) -> bool:
    """True when a no-tool-call turn looks like a premature agentic stall
    (announced an action, didn't call a tool) rather than a real final answer."""
    if has_tool_calls:
        return False
    if finish_reason not in ("stop", "length"):
        return False
    c = (content or "").strip()
    # Strip a leading <think>...</think> block if present; judge the visible tail.
    c = re.sub(r"^<think>.*?</think>\s*", "", c, flags=re.IGNORECASE | re.DOTALL).strip()
    if not c:
        return True  # empty visible turn mid-task => stall
    if len(c) > max_chars:
        return False  # long => almost certainly a real answer
    if _COMPLETION_RE.search(c):
        return False  # model said it's done => respect it
    if _ACTION_RE.search(c):
        return True   # announced an action, no tool call => stall
    # Short prose that doesn't declare completion and isn't an obvious answer:
    # a trailing colon strongly implies "about to do something".
    if c.endswith(":"):
        return True
    return False


def retry_on_stall(agent, api_messages, finish_reason):
    """If the just-finished no-tool-call turn looks like a stall and a retry
    lane is configured, re-issue the SAME turn against that lane (same provider
    / client / endpoint — only the model name changes) ONCE.

    Returns the normalized assistant_message from the retry IF it produced tool
    calls (caller should adopt it + its finish_reason='tool_calls'), else None.
    Never raises into the caller — any failure returns None (original stands).
    """
    retry_model = os.environ.get("HERMES_STALL_RETRY_MODEL", "").strip()
    if not retry_model:
        return None
    try:
        max_chars = int(os.environ.get("HERMES_STALL_RETRY_MAX_CHARS", "400"))
    except ValueError:
        max_chars = 400

    try:
        # Build kwargs exactly as the normal turn would, then override only the
        # model name. Safe when the retry lane is served by the SAME provider/
        # endpoint as agent.model (e.g. taro serves both dflash and the Q6 lane),
        # so no client rebuild is needed.
        api_kwargs = agent._build_api_kwargs(api_messages)
        orig_model = api_kwargs.get("model")
        if retry_model == orig_model:
            return None  # nothing to gain retrying the same model
        api_kwargs = dict(api_kwargs)
        api_kwargs["model"] = retry_model
        # Force non-streaming for the retry (simpler, we only inspect the result).
        api_kwargs.pop("stream", None)
        api_kwargs["stream"] = False

        try:
            agent._vprint(
                f"{getattr(agent, 'log_prefix', '')}↻ stall detected "
                f"(no tool call) — retrying turn on '{retry_model}'",
                force=True,
            )
        except Exception:
            pass

        response = agent._interruptible_api_call(api_kwargs)
        if response is None:
            return None
        transport = agent._get_transport()
        normalize_kwargs = {}
        if getattr(agent, "api_mode", None) == "anthropic_messages":
            normalize_kwargs["strip_tool_prefix"] = getattr(agent, "_is_anthropic_oauth", False)
        normalized = transport.normalize_response(response, **normalize_kwargs)
        if getattr(normalized, "tool_calls", None):
            return normalized
        return None
    except Exception:
        # Any error => silently fall back to the original response.
        return None
