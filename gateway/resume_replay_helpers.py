"""Resume/recovery + replay helpers extracted from gateway/run.py (#54962).

Third slice of the gateway god-file unpacking: the resume-pending
recovery note builder and the assistant-message replay-entry builder,
plus the replay field whitelist they share. Pure functions — no module
state, no I/O — so they are directly unit-testable in isolation (which
is exactly why _build_replay_entry was originally lifted out of the
run_sync closure).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_resume_recovery_note(
    reason: Optional[str],
    message: str = "",
    *,
    interactive: bool = True,
) -> str:
    """Build the resume-pending recovery system note for an interrupted turn.

    ``reason`` is the session's ``resume_reason`` (``restart_timeout``,
    ``shutdown_timeout``, or anything else → generic interruption phrasing).
    ``message`` is the user's NEW message text; empty means this is the
    startup auto-resume turn synthesized by
    ``_schedule_resume_pending_sessions`` with no human message attached.

    ``interactive`` selects the empty-message guidance: on interactive
    platforms a human is present, so "report the restore and ask what next"
    is right.  On non-interactive event platforms (webhook, API server —
    adapters with ``interactive_resume = False``) nobody can answer; the
    resumed turn must instead complete the interrupted work, or the task is
    silently abandoned behind a "restored" acknowledgement that goes
    nowhere (#57056).
    """
    reason_phrase = (
        "a gateway restart"
        if reason == "restart_timeout"
        else "a gateway shutdown"
        if reason == "shutdown_timeout"
        else "a gateway interruption"
    )
    if message:
        resume_guidance = (
            "Address the user's NEW message below FIRST and focus "
            "on what the user is asking now."
        )
        tail_guidance = (
            "Do NOT re-execute old tool calls — skip any "
            "unfinished work from the conversation history."
        )
    elif interactive:
        resume_guidance = (
            "Report to the user that the session was restored "
            "successfully and ask what they would like to do next."
        )
        tail_guidance = (
            "Do NOT re-execute old tool calls — skip any "
            "unfinished work from the conversation history."
        )
    else:
        resume_guidance = (
            "No user is present on this non-interactive platform, "
            "so do NOT emit a 'session restored' acknowledgement "
            "or ask questions. Review the conversation history and "
            "CONTINUE the interrupted task to completion."
        )
        tail_guidance = (
            "Do NOT re-run tool calls whose results already "
            "appear in the history — resume from the first step "
            "that has no recorded result."
        )
    return (
        f"[System note: The previous turn was interrupted by "
        f"{reason_phrase}; the gateway is now back online. "
        f"Any restart/shutdown command in the history has already "
        f"run — do NOT re-execute or verify it. {resume_guidance} "
        f"{tail_guidance}]"
        + (f"\n\n{message}" if message else "")
    )


# Assistant-message fields that must survive transcript replay so multi-turn
# reasoning (thinking blocks, finish reasons) round-trips to the API builders.
_ASSISTANT_REPLAY_FIELDS: tuple[str, ...] = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "finish_reason",
)


def _build_replay_entry(
    role: str,
    content: Any,
    msg: Dict[str, Any],
    preserve_timestamp: bool = False,
) -> Dict[str, Any]:
    """Build a replay entry for a non-tool-calling message, preserving the
    assistant fields the agent's API builders rely on for multi-turn fidelity.

    Lifted out of the inline ``run_sync`` closure so the field whitelist can
    be unit-tested in isolation.  Mirrors the ``_ASSISTANT_REPLAY_FIELDS``
    contract above.

    ``preserve_timestamp``: when True, copy the source row's ``timestamp``
    onto the replay entry. Currently only user messages need this — the
    stale-dangerous-confirmation stripper in ``agent/replay_cleanup.py``
    reads the timestamp to decide whether a confirmation is too old to
    replay safely.  Assistant/tool messages are not timestamp-stripped in
    the same way, so we keep the existing default of dropping it.

    Empty values: most fields are dropped when falsy (matching the original
    PR #2974 behaviour) since an empty list/string for those carries no
    information.  The exception is ``reasoning_content``: DeepSeek/Kimi
    thinking-mode replay treats an empty string as a meaningful sentinel
    that ``_copy_reasoning_content_for_api`` upgrades to a single space.
    Dropping it here would make the gateway send no ``reasoning_content``
    at all on the next turn, which can cause HTTP 400 from strict thinking
    providers.
    """
    entry: Dict[str, Any] = {"role": role, "content": content}
    # api_content sidecar (persist-what-you-send, prompt-cache stability):
    # forward the exact bytes previously sent to the API for this message so
    # the agent's api_messages build can substitute them and keep the request
    # prefix byte-stable across turns. Forward ONLY when this replay pipeline
    # did not rewrite the content (timestamp injection, auto-continue strip,
    # mirror prefix): a rewritten clean content means the pipeline decided
    # different bytes must replay — resending the stored sidecar would
    # reintroduce exactly what was stripped. Dropping it costs one cache
    # boundary; resending stripped noise is a behavior regression.
    _sidecar = msg.get("api_content")
    if (
        role in ("user", "assistant")
        and isinstance(_sidecar, str)
        and _sidecar
        and content == msg.get("content")
    ):
        entry["api_content"] = _sidecar
    if role == "assistant":
        for _rkey in _ASSISTANT_REPLAY_FIELDS:
            if _rkey not in msg:
                continue
            _rval = msg.get(_rkey)
            if _rkey == "reasoning_content":
                # Preserve empty-string sentinel for thinking-mode replay.
                if _rval is None:
                    continue
            elif not _rval:
                continue
            entry[_rkey] = _rval
    if preserve_timestamp:
        ts = msg.get("timestamp")
        if ts:
            entry["timestamp"] = ts
    return entry
