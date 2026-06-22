"""Strip model chain-of-thought from final assistant turns.

Reasoning models (``minimax-m3:cloud``, ``sonar-reasoning-pro``, Claude with
extended thinking, etc.) emit their internal reasoning as part of the final
assistant turn — typically wrapped in ``<tool_call>...</think>`` blocks, but
also in HTML comments, ``Reasoning:`` prefixes, and ``Chain of Thought:``
prefixes depending on the model and provider.

The Hermes gateway's job is to relay the final reply to a messaging
platform. For 1:1 chat surfaces (Discord, Slack), the chain-of-thought is
noise — readers want the answer, not the deliberation. The CLI / TUI /
Desktop clients have a separate ``display.show_reasoning`` toggle for users
who want to see the CoT locally.

This module is the pure-function kernel. The integration point in
``gateway.run._sanitize_gateway_final_response`` calls ``strip_for_platform``
with the platform name; only ``discord`` and ``slack`` get stripped in this
PR (Telegram keeps its existing secret-redaction-only path; see the
companion plan doc for the rationale).
"""
from __future__ import annotations

import re
from typing import Optional

# Patterns we recognize as "this is the model thinking out loud, not the
# answer". Order matters: `` first because it's the most common in modern
# reasoning models. Each pattern is compiled once at import.
_REASONING_PATTERNS: tuple[re.Pattern[str], ...] = (
    # `` — DeepSeek / Qwen / o1-style / MiniMax-M3 style. Non-greedy so we
    # stop at the first closing tag, not the last in the string.
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    # Anthropic's hidden reasoning is sometimes shipped as an HTML comment
    # when not stripped by the SDK. Defensive.
    re.compile(r"<!--\s*reasoning:.*?-->", re.DOTALL | re.IGNORECASE),
    # "Reasoning:" followed by a blank line, then the actual answer. The
    # positive look-ahead requires a newline + a non-lowercase-leading
    # sentence (real prose tends to start with a capital or a quote).
    re.compile(
        r"(?im)^Reasoning:\s*\n.*?(?=\n\s*\n[A-Z\"']|\Z)",
        re.DOTALL,
    ),
    # "Chain of Thought:" prefix form, same shape.
    re.compile(
        r"(?im)^Chain of Thought:\s*\n.*?(?=\n\s*\n[A-Z\"']|\Z)",
        re.DOTALL,
    ),
    # The gateway's own prepended "💭 **Reasoning:**\n```\n...\n```\n\n"
    # block (built in gateway/run.py at the `_show_reasoning_effective`
    # site). Belt-and-suspenders: if a future refactor reorders the
    # pipeline so the sanitizer runs after the prepender, the stripper
    # still cleans it up.
    re.compile(
        r"💭\s*\*\*Reasoning:\*\*\s*\n```[\s\S]*?```\s*\n\n?",
        re.MULTILINE,
    ),
)

# Collapse 3+ consecutive newlines down to 2 (i.e. one blank line). After
# stripping a CoT block we often leave behind a few stray newlines; this
# keeps the output clean without aggressive whitespace mutation.
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Platforms that opt in to CoT stripping today. Telegram is intentionally
# excluded — its sanitizer does different work (provider error replacement
# and secret redaction), and changing that behavior is a separate concern.
_PLATFORMS_THAT_STRIP_REASONING: frozenset[str] = frozenset({"discord", "slack"})

# User-facing fallback when the entire model turn was CoT (no visible
# answer survives the strip). The gateway sends this rather than empty.
_NO_ANSWER_FALLBACK = "(the model produced no visible response)"


def strip_reasoning_blocks(text: str) -> str:
    """Remove model chain-of-thought blocks from ``text``.

    Returns the cleaned string. If the entire input was reasoning (no
    visible answer remains), returns an empty string — the caller is
    responsible for the user-facing fallback.
    """
    if not text:
        return text

    stripped = text
    for pattern in _REASONING_PATTERNS:
        stripped = pattern.sub("", stripped)

    # Collapse the whitespace that stripping usually leaves behind, then
    # strip leading/trailing whitespace so the result is clean.
    stripped = _EXCESS_NEWLINES.sub("\n\n", stripped).strip()
    return stripped


def strip_for_platform(platform: str, text: Optional[str]) -> Optional[str]:
    """Platform-aware wrapper around ``strip_reasoning_blocks``.

    Only Discord and Slack have the stripper enabled in this PR. Telegram
    and unknown platforms return the text unchanged so the rest of the
    gateway's existing pipeline runs as before.

    If the stripper eats the entire message, returns a short user-facing
    fallback rather than ``""`` so the user sees *something* in chat.
    """
    if text is None or text == "":
        return text

    if platform not in _PLATFORMS_THAT_STRIP_REASONING:
        return text

    stripped = strip_reasoning_blocks(text)
    if not stripped:
        return _NO_ANSWER_FALLBACK
    return stripped
