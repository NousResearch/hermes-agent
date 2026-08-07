"""Boundary repair for providers that stream reasoning as discrete summary parts.

Reasoning-summary models (OpenAI's gpt-5.x family, and anything relaying the
Responses API onto the OpenAI chat wire) do not stream a chain of thought token
by token. They emit one ``reasoning_content`` delta per *completed* summary
part, each opening with a bold markdown heading::

    {"delta": {"reasoning_content": "**Investigating likely culprit PRs**"}}
    {"delta": {"reasoning_content": "**Inspecting message schema**"}}

On the Responses API those parts are delimited by ``summary_index``
(``response.reasoning_summary_part.added`` / ``.done``). The OpenAI chat wire
carries no such field — verified live against Nous Portal's
``openai/gpt-5.6-sol``, whose reasoning chunks contain nothing but
``delta.reasoning_content`` — so the boundary cannot be recovered from
metadata, and consumers that concatenate deltas glue the parts together:

    **Investigating likely culprit PRs****Inspecting message schema**

That ``****`` run is neither a bold close nor a bold open to a markdown parser,
so the whole trace renders as one unbroken, unspaced, half-bold paragraph.

The AI SDK hit exactly this (vercel/ai#6742) and fixed it upstream by starting
a new reasoning part per ``summary_index``. That route needs the index, which
this wire does not give us, so we re-derive the boundary from the one signal it
does carry: a delta opening a bold heading. Hermes' own Responses adapter
already joins its summary parts with a blank line
(``agent/codex_responses_adapter.py``), so this brings the chat-completions
stream in line with the path that keeps the structure.
"""

from __future__ import annotations

__all__ = ["separate_glued_reasoning_blocks"]


def separate_glued_reasoning_blocks(previous: str, delta: str) -> str:
    """Return *delta*, prefixed with a paragraph break when it glues onto *previous*.

    *previous* is the reasoning text accumulated so far (only its tail matters).
    A break is inserted when *delta* opens a bold heading and *previous* is
    mid-line, which is the summary-part boundary the chat wire drops. Both
    shapes the upstream issue reports are covered: a heading-only part butting
    against the next heading (``**One****Two**``), and a part whose prose body
    butts against the next heading (``...interaction!**Next**``).

    Token-streamed reasoning is left alone: its deltas carry their own leading
    whitespace, so *previous* ends mid-line only when the model really did run
    two parts together.
    """
    if not previous or not delta:
        return delta

    if not delta.startswith("**"):
        return delta

    # Already separated — the provider (or an earlier part) ended the line.
    if previous[-1].isspace():
        return delta

    # Require a *closed* heading. A token-streamed fragment that merely opens
    # emphasis ("**" then "bold" then "**" across three deltas) is not a part
    # boundary; a summary part always carries its whole heading in one delta.
    if "**" not in delta[2:]:
        return delta

    return f"\n\n{delta}"
