"""agent:start / agent:end symmetry on the interrupt/drain follow-up turn.

Companion to ``test_drain_emits_agent_start.py``.  That suite proved the drain
path emits ``agent:start`` for the follow-up turn; this one is about the
matching ``agent:end``.

Ground truth (verified against commit 451386527, see FINDINGS.md): ``agent:end``
is emitted at exactly ONE site in ``gateway/run.py`` — the MAIN dispatch in
``_handle_message_with_agent`` — while ``agent:start`` is emitted at TWO sites
(main dispatch + the drain path in ``_run_agent``).  So a drained turn fires
``agent:start`` once more than ``agent:end``: the follow-up turn's start has no
matching end.  These tests drive the REAL ``_run_agent`` drain path (same seam
as the start suite) and assert the desired SYMMETRIC contract: every
``agent:start`` on the drain path is paired by an ``agent:end`` carrying the
same ``trigger``/``interrupt_depth`` discriminator, plus the follow-up turn's
response — and that NO end is emitted on the paths where no start is either.
"""

import pytest

from gateway.platforms.base import MessageEvent, MessageType

# Reuse the drain harness verbatim (DRY) — same fixtures the agent:start suite
# uses to drive the real _run_agent / interrupt-drain path.
from tests.gateway.test_drain_emits_agent_start import (
    SESSION_ID,
    _drive_drain,
    _source,
)


def _starts(hooks):
    return [ctx for (etype, ctx) in hooks.calls if etype == "agent:start"]


def _ends(hooks):
    return [ctx for (etype, ctx) in hooks.calls if etype == "agent:end"]


@pytest.mark.asyncio
async def test_drain_followup_start_and_end_are_balanced(monkeypatch, tmp_path):
    """One drained follow-up → equal agent:start and agent:end counts."""
    followup = MessageEvent(
        text="follow up text",
        message_type=MessageType.TEXT,
        source=_source(user_id="userB"),
        message_id="m2",
    )
    hooks = await _drive_drain(
        monkeypatch, tmp_path, followup, prepared_text="follow up text"
    )

    starts, ends = _starts(hooks), _ends(hooks)
    assert len(starts) == 1, f"expected one drain agent:start, got {len(starts)}"
    assert len(ends) == len(starts), (
        "drain follow-up must emit a matching agent:end for its agent:start "
        f"(start={len(starts)}, end={len(ends)})"
    )


@pytest.mark.asyncio
async def test_drain_end_mirrors_start_payload(monkeypatch, tmp_path):
    """The drain agent:end mirrors the start payload + carries the response."""
    followup = MessageEvent(
        text="follow up text",
        message_type=MessageType.TEXT,
        source=_source(user_id="userB"),
        message_id="m2",
    )
    hooks = await _drive_drain(
        monkeypatch, tmp_path, followup, prepared_text="follow up text"
    )

    ends = _ends(hooks)
    assert len(ends) == 1
    end = ends[0]
    # Same discriminator + identity fields as the start emit it pairs with.
    assert end["trigger"] == "interrupt"
    assert end["interrupt_depth"] == 1
    assert end["platform"] == "telegram"
    assert end["user_id"] == "userB"
    assert end["chat_id"] == "9001"
    assert end["session_id"] == SESSION_ID
    assert end["message"] == "follow up text"
    # Mirrors the main-path end (9428): a response field, capped at 500 chars.
    assert "response" in end
    assert isinstance(end["response"], str)
    assert len(end["response"]) <= 500


@pytest.mark.asyncio
async def test_nested_drain_end_increments_depth(monkeypatch, tmp_path):
    """An interrupt-of-an-interrupt pairs its start/end at depth 2."""
    followup = MessageEvent(
        text="deeper follow up",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="m8",
    )
    hooks = await _drive_drain(
        monkeypatch,
        tmp_path,
        followup,
        prepared_text="deeper follow up",
        interrupt_depth=1,
    )

    starts, ends = _starts(hooks), _ends(hooks)
    assert len(starts) == 1
    assert len(ends) == len(starts)
    assert ends[0]["trigger"] == "interrupt"
    assert ends[0]["interrupt_depth"] == 2


@pytest.mark.asyncio
async def test_no_end_when_followup_text_is_none(monkeypatch, tmp_path):
    """Follow-up dropped before _run_agent (transcription → None): neither a
    start NOR an end may fire — symmetric absence."""
    followup = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=_source(),
        media_urls=["/tmp/silent.ogg"],
        media_types=["audio/ogg"],
        message_id="m5",
    )
    hooks = await _drive_drain(monkeypatch, tmp_path, followup, prepared_text=None)

    assert _starts(hooks) == []
    assert _ends(hooks) == []


@pytest.mark.asyncio
async def test_no_end_at_max_interrupt_depth(monkeypatch, tmp_path):
    """At _MAX_INTERRUPT_DEPTH the drain re-queues instead of recursing and
    returns BEFORE the start emit — so neither a start NOR an end may fire.
    Guards against a future move of the emit above the depth cap."""
    followup = MessageEvent(
        text="too deep",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="m9",
    )
    hooks = await _drive_drain(
        monkeypatch,
        tmp_path,
        followup,
        prepared_text="too deep",
        interrupt_depth=3,  # == GatewayRunner._MAX_INTERRUPT_DEPTH
    )

    assert _starts(hooks) == []
    assert _ends(hooks) == []


@pytest.mark.asyncio
async def test_no_end_when_goal_continuation_inactive(monkeypatch, tmp_path):
    """A stale /goal continuation is discarded before _run_agent — neither a
    start NOR an end may fire."""
    followup = MessageEvent(
        text="[Continuing toward your standing goal]\nGoal: ship the thing",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="m6",
    )
    hooks = await _drive_drain(
        monkeypatch,
        tmp_path,
        followup,
        prepared_text="should never be used",
        goal_active=False,
    )

    assert _starts(hooks) == []
    assert _ends(hooks) == []
