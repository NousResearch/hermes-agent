"""Whether a reasoning-only clean stop may be promoted to the visible answer.

A field name is not provenance: OpenRouter, DeepSeek, Bedrock and several adapters all put
private chain-of-thought in ``reasoning``. Promotion is therefore limited to trusted routes,
decided from the agent's LIVE route on every call so fallback activation, primary restore and
model switches never carry a decision to another route.

Trust is necessary, not sufficient: the promoted text must also look WHOLE. A reasoning field
the provider cut mid-token is not an answer, and promoting it reports a fragment as the
completed result of an unattended run.
"""

from __future__ import annotations

import logging
from typing import Any

from utils import base_url_host_matches

logger = logging.getLogger(__name__)

_ANSWER_IN_REASONING_CAPABILITY = "answer_in_reasoning"
_NEMOTRON_PARSER_MODEL_MARKER = "nemotron-3.5-lightning"


def answer_in_reasoning_capability(agent: Any) -> bool:
    """True when the live route may return a clean-stop reasoning payload as the answer.

    Trusted: a custom_providers per-model ``answer_in_reasoning`` opt-in or provider-level
    ``capabilities: {answer_in_reasoning: ...}`` block on the live route, or the local
    Nemotron-3.5-Lightning parser route from #109205. OpenRouter and non-chat-completions
    transports are never trusted.
    """
    base_url = str(getattr(agent, "base_url", "") or "")
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    if provider == "openrouter" or base_url_host_matches(base_url, "openrouter.ai"):
        return False

    api_mode = str(getattr(agent, "api_mode", "") or "").strip().lower()
    if api_mode and api_mode != "chat_completions":
        return False

    # Returns None on unreadable config, so no guard is needed here.
    from hermes_cli.config import _entries_for_route, get_custom_provider_model_capability

    model = str(getattr(agent, "model", "") or "")
    custom_providers = getattr(agent, "_custom_providers", None)
    configured = get_custom_provider_model_capability(
        model=model,
        base_url=base_url,
        capability=_ANSWER_IN_REASONING_CAPABILITY,
        custom_providers=custom_providers,
    )
    if configured is not None:
        return configured
    # The provider-level ``capabilities:`` block, re-read on the live route like the per-model key
    # so it holds on CLI/TUI (no constructor ``capabilities=``) and survives /model switches.
    # Several entries may share one endpoint (LiteLLM/vLLM router): only the live ``custom:<slug>``
    # entry counts; without one, every flagged sibling on the endpoint must opt in. Startup, gateway
    # and TUI keep ``provider="custom"`` and the named identity in ``requested_provider``.
    from hermes_cli.providers import custom_provider_slug

    requested = str(getattr(agent, "requested_provider", "") or "").strip().lower()
    ids = {custom_provider_slug(p) for p in (provider, requested) if p.startswith("custom:")}
    entries = list(_entries_for_route(base_url, custom_providers, None))
    live = [e for e in entries
            if custom_provider_slug(e.get("name"), e.get("provider_key")) in ids]
    flags = {v for e in live or entries
             if isinstance(v := (e.get("capabilities") or {}).get(_ANSWER_IN_REASONING_CAPABILITY), bool)}
    if flags:
        return flags == {True}

    if _NEMOTRON_PARSER_MODEL_MARKER not in model.lower():
        return False
    from agent.model_metadata import is_local_endpoint

    return is_local_endpoint(base_url)


# ── the promoted text must look WHOLE ───────────────────────────────────────────────────────
# The promotion exists because some parsers file the WHOLE answer as reasoning when the model
# omits the closing delimiter; the two real cases it was written for were 38254 and 9890 chars.
# With no shape check, a reasoning field that was itself CUT OFF mid-token was promoted and
# delivered verbatim: on 2026-10-06 the digest cron emailed exactly 36 chars,
# ``The fix was already applied on 2026-``.
#
# The pre-existing ``looks_like_degenerate_final`` does not cover this: it is tuned for
# wrong-script / leading-punctuation collapses and explicitly does NOT match all-ASCII
# alphanumeric text (by design, so "42"/"SQLite" stay answers), and it is gated on
# ``intent_ack_continuation_mode() != "off"`` — which is "off" on chat_completions, so it is
# inert on that job's transport. This predicate is deliberately ORTHOGONAL: it asks "does this
# text look CUT OFF?", not "does it look like a wrong-language fragment?". A truncation is a
# terminal-token problem, so it is script- and length-independent.
#
# Trailing punctuation that legitimately ends a sentence — note "-" is NOT here: an em-dash
# mid-thought and a cut-off date both end in a hyphen-ish token, and a complete answer does not
# end in one.
_COMPLETE_TAIL_PUNCT = (".", "!", "?", "。", "！", "？", "…", '"', "'", ")", "]", "}", ":", ";")

# A dangling separator at the very end: the cut landed inside a token.
_DANGLING_SEPARATORS = ("-", "_", "/", "\\", "&", "=", "+", "@", "#", "$")

# An unclosed opener, or a trailing comma — both are mid-construct.
_MID_CONSTRUCT_TAILS = ("(", "[", "{", '"', "'", ",", "|", "<")

# A trailing function word: the sentence was going somewhere and stopped.
_DANGLING_WORDS = frozenset({
    "and", "or", "but", "the", "a", "an", "of", "to", "for", "with", "in", "on",
    "at", "by", "from", "as", "is", "was", "were", "be", "been", "that", "which",
    "so", "if", "when", "while", "because", "since", "than", "then",
})


def looks_like_truncated_promoted_reasoning(text: str) -> bool:
    """Whether a promoted reasoning field looks CUT OFF (so it must not be delivered).

    Orthogonal to ``looks_like_degenerate_final``: that one detects a wrong-script or
    leading-punctuation COLLAPSE, this one detects a TRUNCATION, which is a terminal-token
    problem and therefore script-independent. Deliberately conservative — it only fires on the
    shapes a provider cut produces and a finished answer almost never does:

      * ends on a dangling separator ("on 2026-", "see issue #", "path/to/"),
      * ends on an opening bracket/quote or a trailing comma,
      * ends on a bare conjunction/preposition/article ("...and", "...the", "...with").

    A complete short answer ("Done.", "42", "SQLite", "[SILENT]") never matches: the first two
    end in a terminal or contain no separator, and the sentinel is checked first so a cron
    silence marker always passes through untouched.
    """
    t = (text or "").strip()
    if not t:
        return False
    # Cron silence markers and explicit status sentinels are whole by construction.
    if t.strip("[]()").upper() in {"SILENT", "NO_REPLY", "NO REPLY"}:
        return False
    if t.endswith(_COMPLETE_TAIL_PUNCT):
        return False
    if t.endswith(_DANGLING_SEPARATORS) or t.endswith(_MID_CONSTRUCT_TAILS):
        return True
    # An UNBALANCED opener anywhere: the cut landed inside a parenthetical. Brackets only —
    # quotes are excluded because an apostrophe in ordinary prose ("don't", "the user's") is
    # not a delimiter.
    for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
        if t.count(opener) > t.count(closer):
            return True
    return t.rsplit(None, 1)[-1].lower().strip(".,;:!?") in _DANGLING_WORDS


def should_promote_reasoning_to_final(reasoning_text: str) -> bool:
    """Whether a reasoning-only clean stop may be promoted to the visible answer.

    Promotes when the reasoning is a plausible whole answer and DECLINES when it looks cut off:
    a truncated promotion is worse than the empty-response ladder, because the ladder would have
    nudge/continued and could still recover the real answer, whereas a delivered fragment is
    reported to the user as the completed result of an unattended job.
    """
    t = (reasoning_text or "").strip()
    if not t:
        return False
    return not looks_like_truncated_promoted_reasoning(t)


def promoted_final_text(
    agent: Any,
    assistant_message: Any,
    finish_reason: str | None,
    api_call_count: int,
    messages: list[dict[str, Any]],
) -> str | None:
    """The reasoning field as the final answer for a reasoning-only clean stop, or ``None``.

    ``None`` means "not a promotion candidate" — a wrong stop shape, tool calls present, private
    (signed) reasoning from another model, an untrusted route — OR "the reasoning looks CUT OFF".
    Declining a truncation hands the turn to the empty-response ladder, which can still
    nudge/continue and recover a real answer.

    The returned text is the ANSWER but never written into the assistant row's ``content``:
    chain-of-thought stored as ordinary content is indistinguishable from a real reply on every
    history surface (#111761). The caller keeps the row's ``content`` empty, with the text in its
    reasoning fields and the ``api_content`` sidecar, so the next turn replays it byte-identically.
    """
    if finish_reason != "stop" or assistant_message.tool_calls:
        return None
    content = assistant_message.content
    if content is not None and not (isinstance(content, str) and not content.strip()):
        return None
    # Anthropic thinking (signed ``thinking`` block, or a plugin's ``*.native_assistant`` carrier
    # of native Claude turns) is a summary written by a separate model, never the answer: it takes
    # the empty-response continuation instead.
    if any(
        isinstance(d, dict) and (
            (d.get("type") in ("thinking", "redacted_thinking") and (d.get("signature") or d.get("data")))
            or str(d.get("type") or "").endswith(".native_assistant")
        )
        for d in getattr(assistant_message, "reasoning_details", None) or ()
    ):
        return None
    if not answer_in_reasoning_capability(agent):
        return None
    promoted = agent._extract_reasoning(assistant_message) or None
    if not promoted or not should_promote_reasoning_to_final(promoted):
        if promoted:
            logger.warning(
                "Reasoning-only clean stop (%d chars) looks TRUNCATED (%r) — NOT promoting; "
                "handing off to the empty-response ladder (model=%s provider=%s)",
                len(promoted), promoted[:60], agent.model, agent.provider,
            )
        return None
    # WARNING, not INFO: a model that keeps ending turns this way is stalled (planning monologue,
    # zero tool calls) while the turn reports "complete".
    logger.warning(
        "Reasoning-only clean stop (%d chars) — returning the reasoning as the final "
        "response (model=%s provider=%s api_calls=%d tool_turns=%d)",
        len(promoted), agent.model, agent.provider, api_call_count,
        sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")),
    )
    return promoted
