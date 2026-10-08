# fmt: off
"""Discord adapter support: rich clarify choices + small shared helpers.

Kept out of ``adapter.py`` so the platform facade cannot outgrow its
code-health cap (``scripts/code_health/config.py``); ``adapter.py`` imports
from here. This module never imports ``adapter.py`` (no import cycle):
anything needing the adapter instance receives it as a parameter.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import discord

logger = logging.getLogger(__name__)

def resolve_exec_approval_admin_gate(config_extra: Optional[dict]) -> Tuple[bool, set]:
    """Resolve the exec-approval admin gate from ``extra``; returns ``(require_admin, admin_user_ids)``.
    Default OFF (user-scope buttons). When ``require_admin_for_exec_approval`` is true only
    ``allow_admin_from`` ids may click; on with no admins -> ``(True, set())`` (fail closed, log once).
    """
    extra = config_extra if isinstance(config_extra, dict) else {}
    raw_toggle = extra.get("require_admin_for_exec_approval", False)
    require_admin = str(raw_toggle).strip().lower() in {"true", "1", "yes"}
    if not require_admin:
        return (False, set())
    try:
        from gateway.slash_access import _coerce_id_list
        admin_ids = set(_coerce_id_list(extra.get("allow_admin_from")))
    except Exception:
        admin_ids = set()
    return (True, admin_ids)


# =========================================================================
# Clarify choice helpers -- normalize a raw choice (str OR structured option
# dict {label, brief, pros, cons}) into the text the different surfaces need.
# Structured option dicts power the #1 "real decision helper": the embed body
# shows each option's full brief plus pros/cons, while buttons stay short
# selectors. Bare strings keep the classic behavior end-to-end.
# =========================================================================


def _choice_text(c) -> str:
    """User-facing text for a raw choice (str or option dict) -- used to test
    existence and as a plain fallback. For a dict, prefers brief/label/
    description/text/title in that order (the canonical LLM tool-call keys)."""
    if isinstance(c, dict):
        for key in ("brief", "label", "description", "text", "title"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(c, (list, tuple)):
        return " ".join(x for x in (_choice_text(x) for x in c) if x).strip()
    if c is None:
        return ""
    return str(c).strip()


def _choice_button_text(c) -> str:
    """Short selector text for the button base -- the option's short label for a
    structured dict, else the full text for a bare string."""
    if isinstance(c, dict):
        for key in ("label", "brief", "description", "text", "title"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    return _choice_text(c)


def _choice_resolve_text(c) -> str:
    """The answer string resolved back to the agent when this choice is picked.
    For a structured option, prefer the full ``brief`` (then ``description``)
    so the agent receives the substance, falling back to the short label."""
    if isinstance(c, dict):
        for key in ("brief", "description"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for key in ("label", "text", "title"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    return _choice_text(c)


def _choice_label_line(i: int, c) -> str:
    """Native (non-code) option header: '**1. Label** -- brief'. The label is
    bold so the option stands out; a structured option's brief rides on the
    same line after ' -- ' (kept out of the mono box)."""
    label = _choice_button_text(c)
    if not label:
        return f"**{i}. (option)**"
    if isinstance(c, dict):
        brief = c.get("brief") or c.get("description")
        if isinstance(brief, str) and brief.strip() and brief.strip() != label:
            return f"**{i}. {label}** -- {brief.strip()}"
    return f"**{i}. {label}**"


def _choice_detail_block(c) -> Optional[str]:
    """Inner text of the indented code box for a structured option -- ONLY the
    pros/cons (the brief already lives on the option's label line). Returns
    None (no code box) for a bare-string option."""
    if not isinstance(c, dict):
        return None
    lines: list = []
    pros = c.get("pros") or []
    if pros:
        lines.append("   ✅ Pros:")
        for x in pros:
            s = str(x).strip()
            if s:
                lines.append(f"      • {s}")
    cons = c.get("cons") or []
    if cons:
        lines.append("")
        lines.append("   ❌ Cons:")
        for x in cons:
            s = str(x).strip()
            if s:
                lines.append(f"      • {s}")
    return "\n".join(lines) if lines else None



async def clarify_resend_mention(adapter, channel) -> Optional[str]:
    """Build the '@name ⬆️ still waiting on your decision' content for a
    re-posted clarify prompt.

    Mentions the ACTUAL participants of the thread/DM the prompt lives in,
    so an ignored prompt resurfaces in notifications even under allow-all
    config (where ``_allowed_user_ids`` is empty). Falls back to the
    allowlist, then gives up (returns None -> no content/mention)."""
    ids: set = set()
    bot_id = None
    try:
        bot_id = getattr(getattr(adapter._client, "user", None), "id", None)
    except Exception as e:  # best-effort; missing client/user is not fatal
        logger.debug("[%s] clarify bot_id lookup: %s", adapter.name, e, exc_info=True)
    # Thread members -- uses the thread-members API, no privileged intent
    try:
        if hasattr(channel, "fetch_members"):
            for m in await channel.fetch_members():
                uid = getattr(m, "id", None)
                if uid is not None:
                    ids.add(str(uid))
    except Exception as e:  # best-effort; thread-members API may be unsupported
        logger.debug("[%s] clarify thread members: %s", adapter.name, e, exc_info=True)
    # DM recipients
    try:
        for m in getattr(channel, "recipients", None) or []:
            uid = getattr(m, "id", None)
            if uid is not None:
                ids.add(str(uid))
    except Exception as e:  # best-effort; recipients may be unavailable
        logger.debug("[%s] clarify dm recipients: %s", adapter.name, e, exc_info=True)
    # Fall back to the allowlist if participants yielded nothing
    if not ids:
        ids = {str(u) for u in (adapter._allowed_user_ids or set())}
    if bot_id is not None:
        ids.discard(str(bot_id))
    if not ids:
        return None
    mention = "".join(f"<@{u}>" for u in sorted(ids))
    return f"{mention} ⬆️ still waiting on your decision"

def make_clarify_resend(adapter, chat_id, question, choices, clarify_id,
                         session_key, metadata):
    """Return an async closure that re-sends a pending clarify prompt.

    Used by ``ClarifyChoiceView.on_timeout``: when a multiple-choice
    prompt sits unanswered past the Discord button window, this deletes
    the expired message and sends a fresh copy (new button window) so the
    user can still answer — instead of the prompt going dead and leaving
    the agent thread wedged.  It reuses the SAME ``clarify_id`` and the
    same ``session_key`` so the gateway entry stays authoritative.

    If the clarify has already been resolved/cleared by the time the
    timeout fires, the resend no-ops (returns False) so we don't spam a
    fresh prompt for a question that's already been answered.
    """
    async def _resend(old_msg=None):
        try:
            from tools.clarify_gateway import get_pending_for_session, get_clarify_timeout
            import time as _t
            pending = None
            if session_key:
                # CRITICAL: a multi-choice clarify awaiting a button pick has
                # awaiting_text=False, and get_pending_for_session() by
                # default only returns free-text entries. Without
                # include_choice_prompts=True the resend closure saw None,
                # no-oped silently, and left a dead-looking prompt that
                # Discord's ~15-min button window then invalidated
                # ("interaction failed / didn't respond in time"). Choice
                # prompts MUST count as pending for the re-post loop.
                pending = get_pending_for_session(
                    session_key, include_choice_prompts=True,
                )
            if pending is None:
                return False
            # Intent (deliberate): re-post a fresh copy whenever the
            # clarify is still pending. There is intentionally NO budget
            # cap here -- the loop is bounded only by the entry's lifetime
            # (it no-ops once resolved/cleared or the 3h clarify_timeout
            # fires). A former count-based budget was dead code: every
            # resend rebuilds this closure fresh, so the counter reset
            # each cycle and never bounded anything. Removing it keeps the
            # real intent visible: time stays a blocker, the re-post is
            # the persistent nudge, and only the user's answer (or the
            # gateway clearing the entry) stops it.

            # #2 -- stamp each re-post with elapsed / time-left so the
            # "time is a blocker" model is visible at a glance.
            def _fmt_dur(secs: float) -> str:
                secs = max(0, int(secs))
                h, rem = divmod(secs, 3600)
                m = rem // 60
                return f"{h}h {m:02d}m" if h else f"{m}m"

            elapsed = max(0, int(_t.time() - getattr(pending, "asked_at", _t.time())))
            window = max(0, int(get_clarify_timeout()))
            left = max(0, window - elapsed)
            wait_hint = f"⏳ Waiting ~{_fmt_dur(elapsed)} · ~{_fmt_dur(left)} left"

            # Delete the old expired prompt if we can.
            if old_msg is not None:
                try:
                    await old_msg.delete()
                except Exception as e:  # stale message / already gone is fine
                    logger.debug("[%s] clarify old-prompt delete: %s", adapter.name, e, exc_info=True)
            # Send a fresh copy with the wait tag. The @mention (thread
            # participants) and its content string are computed inside
            # send_clarify from the live channel, because the allowlist
            # (`_allowed_user_ids`) may be empty under allow-all config.
            res_meta = dict(metadata or {})
            res_meta["_clarify_wait_hint"] = wait_hint
            res = await adapter.send_clarify(
                chat_id=chat_id,
                question=question,
                choices=choices,
                clarify_id=clarify_id,
                session_key=session_key,
                metadata=res_meta,
            )
            return bool(res.success)
        except Exception as e:
            # Fail-open nudge: any error means "don't nag", which is the
            # safe state — but keep the traceback so the why is recoverable.
            logger.debug("[%s] clarify resend failed: %s", adapter.name, e, exc_info=True)
            return False

    return _resend


def build_rich_choices_embed(question, choices, metadata):
    """Build the clarify embed: one numbered button per option plus "Other".

    Returns ``(embed, button_choices, clean_choices)``.  ``choices`` may contain
    bare strings OR structured option dicts ``{label, brief, pros, cons}`` (the
    #1 decision helper); structured dicts render their full brief plus
    pros/cons in the embed body with a short selector on the button.
    """
    max_desc = 4088
    body = str(question or "").strip()
    if len(body) > max_desc:
        body = body[: max_desc - 3] + "..."

    embed = discord.Embed(
        title="\u2753 Hermes needs your input",
        description=body,
        color=discord.Color.orange(),
    )

    # Normalise choices.  Garbage dicts with none of the canonical keys are dropped.
    clean_choices = [c for c in (choices or []) if _choice_text(c).strip()]
    # Discord allows up to 5 buttons per row, 5 rows per view = 25.  We reserve
    # one slot for the "Other" button, so cap at 24.
    clean_choices = clean_choices[:24]

    # Surface a live "waiting ~N / time left" tag on re-posts.
    if metadata is not None and metadata.get("_clarify_wait_hint"):
        embed.add_field(
            name="\u23f3",
            value=metadata["_clarify_wait_hint"],
            inline=False,
        )

    if clean_choices:
        # Full-text (+ pros/cons) decision brief in the body; buttons stay
        # short numbered selectors.  Numbers match across both.
        embed.add_field(
            name="How to answer",
            value="Pick a number below, or tap \u270f\ufe0f Other to type your own answer.",
            inline=False,
        )
        option_blocks = []
        for i, c in enumerate(clean_choices, 1):
            # Bold NATIVE label (stands out) + only the brief/pros/cons
            # inside a code box so the indentation survives.
            block = [_choice_label_line(i, c)]
            detail = _choice_detail_block(c)
            if detail:
                block.append(f"```\n{detail}\n```")
            option_blocks.append("\n".join(block))
        # Separate each option with a blank line so the brief reads as
        # distinct choices.  Chunk by whole options (never split one
        # mid-way) to stay under the 1000-char per-field limit.
        chunks = []
        cur_chunk, cur_chars = [], 0
        for block in option_blocks:
            cost = len(block) + (2 if cur_chunk else 0)  # "\n\n" join
            if cur_chunk and cur_chars + cost > 1000:
                chunks.append("\n\n".join(cur_chunk))
                cur_chunk, cur_chars = [block], len(block)
            else:
                cur_chunk.append(block)
                cur_chars += cost
        if cur_chunk:
            chunks.append("\n\n".join(cur_chunk))
        for idx, chunk in enumerate(chunks):
            embed.add_field(
                name="Options" if idx == 0 else "Options (cont.)",
                value=chunk,
                inline=False,
            )
        button_choices = [_choice_button_text(c) for c in clean_choices]
    else:
        embed.add_field(
            name="Reply",
            value="Reply in this channel with your answer.",
            inline=False,
        )
        button_choices = []
    return embed, button_choices, clean_choices
# fmt: on
