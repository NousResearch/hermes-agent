"""Flood control must never turn a finalized Telegram reply into plain text.

``TelegramAdapter.edit_message`` finalizes a streamed reply by editing the preview
with ``parse_mode=MarkdownV2`` through ``_edit_markdown_or_plain``. When that edit
raises, the helper rewrites the message as stripped plain text. That rescue is
right for a MarkdownV2 parse failure, where the markup is what Telegram rejected.

It also fired on flood control, because the helper caught bare ``Exception``. A
``RetryAfter`` refusal says nothing about the markup, so a reply whose MarkdownV2
was perfectly valid arrived with its syntax showing: headings as a literal ``##``,
links as a literal ``[text](url)``. The outer handler in ``edit_message`` was
already written for flood control (short waits retry inline, long waits fail
closed so streaming falls back to a normal final send) but could never run while
the helper swallowed the exception first. And even when it did run, the inline
retry re-sent the RAW content, so the user saw the markdown syntax anyway.

The guard is deliberately narrow: timeouts and network blips keep the plain-text
rescue, because re-raising them would return ``retryable=True`` and the stream
consumer does not consult ``retryable`` on an edit failure. A timed-out finalize
edit would then have its tail sent again on top of the complete answer.

"""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    try:
        import telegram  # noqa: F401
        return
    except Exception:
        pass
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as adapter_mod  # noqa: E402
from plugins.platforms.telegram.adapter import (  # noqa: E402
    TelegramAdapter,
    _flood_wait_seconds,
    _looks_like_flood_error,
    _strip_mdv2,
)


CHAT_ID = "5230977008"
MESSAGE_ID = "4242"

# A reply shaped like the one that exposed this: headings plus bold and a link,
# all of which format_message converts into valid MarkdownV2.
CONTENT = (
    "## Best souvlaki choice\n"
    "\n"
    "Go to **[Athinaiko Souvlaki](https://maps.example/athinaiko)** at "
    "**Karolou Ntil 23**.\n"
    "\n"
    "- **4.8/5 from 1,025 reviews**\n"
    "- Open today **13:00 to 22:00**\n"
)

_PARSE_ERROR = "Bad Request: can't parse entities: Character '-' is reserved"


class _FloodError(Exception):
    """Mirrors python-telegram-bot's RetryAfter: carries ``retry_after``.

    ``retry_after`` may be a float or, under ``PTB_TIMEDELTA=1``, a
    ``datetime.timedelta``. Both shapes are exercised below.
    """

    def __init__(self, retry_after):
        super().__init__(f"Flood control exceeded. Retry in {retry_after} seconds")
        self.retry_after = retry_after


class _TextOnlyFloodError(Exception):
    """A flood refusal with no ``retry_after`` attribute at all.

    Not every raiser is PTB's RetryAfter. Detection and the wait have to come
    off Telegram's own wording, which says "Retry in Ns" rather than the
    "retry after" the outer handler historically looked for.
    """

    def __init__(self, seconds):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")


class _RecordingBot:
    """Records every Bot API call and replays a scripted failure sequence.

    ``script`` is consumed one entry per call (edits and sends share the same
    sequence): an exception is raised, ``None`` succeeds. Calls past the end of
    the script succeed, so a short script means "fail these, then work".
    """

    def __init__(self, script=()):
        self.script = list(script)
        self.calls: list[dict] = []
        self.sends: list[dict] = []
        self._next_id = int(MESSAGE_ID) + 1

    def _replay(self, kwargs):
        idx = len(self.calls) - 1
        if idx < len(self.script) and self.script[idx] is not None:
            raise self.script[idx]

    async def edit_message_text(self, **kwargs):
        kwargs = dict(kwargs, _op="edit")
        self.calls.append(kwargs)
        self._replay(kwargs)
        return MagicMock(message_id=int(MESSAGE_ID))

    async def send_message(self, **kwargs):
        kwargs = dict(kwargs, _op="send")
        self.calls.append(kwargs)
        self.sends.append(kwargs)
        self._replay(kwargs)
        self._next_id += 1
        return MagicMock(message_id=self._next_id)

    @property
    def formatted_calls(self) -> list[dict]:
        return [c for c in self.calls if c.get("parse_mode")]

    @property
    def plain_calls(self) -> list[dict]:
        return [c for c in self.calls if not c.get("parse_mode")]


def _adapter(bot) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = bot
    return adapter


def _edit(adapter, content: str = CONTENT):
    return asyncio.run(adapter.edit_message(CHAT_ID, MESSAGE_ID, content, finalize=True))


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Inline flood retries sleep for the requested wait; keep the suite fast but record the waits."""
    slept: list[float] = []

    async def _record(delay):
        slept.append(delay)

    monkeypatch.setattr(adapter_mod.asyncio, "sleep", _record)
    yield slept


# ---------------------------------------------------------------------------
# The regression: a flood refusal must not be mistaken for bad markup.
# ---------------------------------------------------------------------------

def test_over_cap_flood_does_not_rewrite_the_reply_as_plain_text():
    """An over-cap flood wait fails closed and leaves formatting alone."""
    bot = _RecordingBot([_FloodError(269.0)])
    result = _edit(_adapter(bot))

    assert bot.plain_calls == [], (
        "flood control is not a markup problem, so the reply must never be "
        f"re-sent unformatted; got {len(bot.plain_calls)} plain-text edit(s)"
    )
    assert result.success is False
    assert result.error == "flood_control:269.0"
    assert result.retry_after == pytest.approx(269.0)


def test_under_cap_flood_retries_with_the_formatted_content():
    """A short flood wait is retried inline, not downgraded to plain text."""
    bot = _RecordingBot([_FloodError(1.0)])
    result = _edit(_adapter(bot))

    assert result.success is True
    assert bot.plain_calls == [], "the inline flood retry must keep the markup"
    assert len(bot.calls) == 2, "expected the failed edit plus one inline retry"

    # The retry must carry the same MarkdownV2 render as the first attempt,
    # not the raw markdown the user would otherwise see.
    first, retry = bot.calls
    assert retry["text"] == first["text"]
    assert retry["parse_mode"] == first["parse_mode"]
    assert "## Best souvlaki choice" not in retry["text"], (
        "the retry re-sent raw markdown, so the heading syntax would show"
    )


@pytest.mark.parametrize("retry_after", [3.0, timedelta(seconds=3)])
def test_flood_wait_is_normalised_whatever_shape_retry_after_has(retry_after, _no_real_sleep):
    """``retry_after`` is a timedelta under PTB_TIMEDELTA=1.

    Comparing that against the inline wait cap raises TypeError from inside the
    outer exception handler, which would escape ``edit_message`` without
    retrying or returning a SendResult.
    """
    bot = _RecordingBot([_FloodError(retry_after)])
    result = _edit(_adapter(bot))

    assert result.success is True
    assert len(bot.calls) == 2
    assert bot.plain_calls == []
    assert _no_real_sleep == [pytest.approx(3.0)], (
        f"expected a 3s wait to reach asyncio.sleep, got {_no_real_sleep}"
    )


# ---------------------------------------------------------------------------
# The rescue that must survive: a genuine parse failure still falls back.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    _PARSE_ERROR,
    "Bad Request: can't parse entities in message text: unexpected end",
    "Bad Request: unsupported markdown in message",
])
def test_genuine_parse_failure_still_falls_back_to_plain_text(message):
    """The markup really was the problem, so the plain-text rescue applies."""
    bot = _RecordingBot([Exception(message)])
    result = _edit(_adapter(bot))

    assert len(bot.plain_calls) == 1, "a parse failure must still be rescued as plain text"
    assert result.success is True


def test_flood_after_a_parse_fallback_retries_the_plain_text_not_the_markup():
    """The degraded payload must survive into the inline flood retry.

    Sequence: the MarkdownV2 edit hits a parse error, the plain-text rescue then
    hits flood control, and the inline retry runs. Retrying the original
    MarkdownV2 there would walk straight back into the parse error that caused
    the fallback in the first place.
    """
    bot = _RecordingBot([Exception(_PARSE_ERROR), _FloodError(1.0), None])
    result = _edit(_adapter(bot))

    assert result.success is True
    assert len(bot.calls) == 3, (
        f"expected markdown attempt, plain rescue, inline retry; got {len(bot.calls)}"
    )
    markdown_attempt, plain_rescue, retry = bot.calls
    assert markdown_attempt.get("parse_mode")
    assert not plain_rescue.get("parse_mode")
    assert not retry.get("parse_mode"), "the flood retry reinstated MarkdownV2 that Telegram just rejected"
    assert retry["text"] == plain_rescue["text"], (
        "the flood retry must re-send the degraded payload, not the markup"
    )


def test_flood_then_not_modified_on_the_retry_is_a_success():
    """The inline retry can hit the not-modified no-op; that is success, not failure."""
    bot = _RecordingBot([_FloodError(1.0), Exception("Bad Request: message is not modified")])
    result = _edit(_adapter(bot))

    assert result.success is True
    assert len(bot.calls) == 2
    assert bot.plain_calls == []


def test_flood_then_parse_error_on_the_retry_keeps_the_plain_rescue():
    """The retry now carries MarkdownV2, so it can meet the parse rejection the refused first
    attempt never reached; the plain-text rescue must still apply there."""
    bot = _RecordingBot([_FloodError(1.0), Exception(_PARSE_ERROR), None])
    result = _edit(_adapter(bot))

    assert result.success is True
    assert len(bot.calls) == 3, f"expected refused attempt, formatted retry, plain rescue; got {len(bot.calls)}"
    refused, retry, rescue = bot.calls
    assert refused.get("parse_mode") and retry.get("parse_mode")
    assert not rescue.get("parse_mode")
    assert rescue["text"] == _strip_mdv2(CONTENT), "the rescue must send the same stripped text the helper's fallback sends"


def test_flood_then_flood_on_the_retry_fails_closed_without_downgrading():
    """A second flood refusal on the retry is not waited on again and never becomes plain text."""
    bot = _RecordingBot([_FloodError(1.0), _FloodError(200.0)])
    result = _edit(_adapter(bot))

    assert result.success is False
    assert bot.plain_calls == []
    assert len(bot.calls) == 2


# ---------------------------------------------------------------------------
# The narrowness is deliberate, not an oversight.
# ---------------------------------------------------------------------------

def test_transient_network_error_keeps_the_plain_text_rescue():
    """A timeout keeps the existing behaviour on purpose (see module docstring)."""
    bot = _RecordingBot([Exception("httpx.ReadTimeout: read timed out")])
    result = _edit(_adapter(bot))

    assert len(bot.plain_calls) == 1
    assert result.success is True


def test_message_not_modified_is_still_a_no_op_success():
    """The existing not-modified shortcut is untouched by the guard."""
    bot = _RecordingBot([Exception("Bad Request: message is not modified")])
    result = _edit(_adapter(bot))

    assert bot.plain_calls == []
    assert result.success is True


# ---------------------------------------------------------------------------
# Detection and the wait must not depend on PTB's attribute alone.
# ---------------------------------------------------------------------------

def test_text_only_flood_refusal_is_recognised_and_fails_closed():
    """No ``retry_after`` attribute, so both facts come from the message.

    Defaulting to a one-second wait here would retry inside a 269-second
    window: another request spent during exactly the period Telegram asked us
    to stay quiet.
    """
    bot = _RecordingBot([_TextOnlyFloodError(269)])
    result = _edit(_adapter(bot))

    assert bot.plain_calls == []
    assert result.success is False
    assert result.error == "flood_control:269.0"
    assert len(bot.calls) == 1, "an over-cap wait must not retry inline"


@pytest.mark.parametrize("error, expected", [
    (_FloodError(4.0), True),
    (_FloodError(timedelta(seconds=4)), True),
    (_TextOnlyFloodError(30), True),
    (Exception("Too Many Requests: retry after 12"), True),
    (Exception(_PARSE_ERROR), False),
    (Exception("httpx.ReadTimeout: read timed out"), False),
    (Exception("Bad Request: message to edit not found"), False),
])
def test_looks_like_flood_error_classifies_every_raiser_shape(error, expected):
    assert _looks_like_flood_error(error) is expected


@pytest.mark.parametrize("error, expected", [
    (_FloodError(7.5), 7.5),
    (_FloodError(timedelta(seconds=90)), 90.0),
    (_TextOnlyFloodError(269), 269.0),
    (Exception("Too Many Requests: retry after 12"), 12.0),
    (Exception("Flood control exceeded"), 1.0),
])
def test_flood_wait_seconds_reads_attribute_then_message_then_default(error, expected):
    assert _flood_wait_seconds(error) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The same rule has to hold once the reply overflows into a split.
# ---------------------------------------------------------------------------

def _long_content() -> str:
    long_content = CONTENT + ("\nfiller line to push this over the cap. " * 200)
    assert len(long_content) > 4096, "test content must cross the split threshold"
    return long_content


def test_overflowing_reply_does_not_downgrade_first_chunk_on_flood():
    """A reply past the length cap takes _edit_overflow_split, whose first-chunk edit
    goes through the same helper and used to fall back to plain text on any error."""
    bot = _RecordingBot([_FloodError(269.0)])
    result = _edit(_adapter(bot), _long_content())

    assert bot.plain_calls == [], (
        "the overflow path downgraded a flood-refused chunk to plain text; "
        f"{len(bot.plain_calls)} unformatted edit(s)"
    )
    assert result.success is False
    assert result.error == "flood_control:269.0"


def test_overflowing_reply_does_not_downgrade_a_continuation_on_flood():
    """Continuation chunks had their own MarkdownV2-then-plain loop; a flood refusal on
    the MarkdownV2 send must stop there instead of resending the chunk unformatted."""
    # first-chunk edit succeeds, the first continuation send is flood-refused
    bot = _RecordingBot([None, _FloodError(200.0)])
    result = _edit(_adapter(bot), _long_content())

    assert len(bot.sends) == 1, f"expected exactly one (refused) continuation send, got {len(bot.sends)}"
    assert bot.sends[0].get("parse_mode"), "the refused send was the MarkdownV2 attempt"
    assert [c for c in bot.sends if not c.get("parse_mode")] == [], (
        "a flood-refused continuation was resent as plain text"
    )
    assert result.success is False
    assert result.error == "overflow_continuation_failed"
    assert result.retryable is True


def test_overflowing_reply_still_falls_back_on_a_genuine_parse_error():
    """The plain-text rescue for real markup failures survives on the overflow path."""
    bot = _RecordingBot([Exception(_PARSE_ERROR)])
    result = _edit(_adapter(bot), _long_content())

    assert len(bot.plain_calls) >= 1, "a parse failure on the first chunk must still be rescued"
    assert result.success is True
