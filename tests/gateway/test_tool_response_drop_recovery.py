"""Regression tests for tool-using response silent drop (issue #29346).

When the agent returns a non-empty response that the extract pipeline
(extract_media / extract_images / extract_local_files / inline directive
strips) happens to reduce to an empty string, the ``if text_content:`` guard
in ``BasePlatformAdapter._process_message_background`` previously bypassed
the send entirely. The symptom was a ``response ready`` log followed by
silence — no ``Sending response`` line, no error — and the final answer
never reaching the channel.

The fix (A2/A3 of the silent-response-loss plan) preserves the pre-extract
response and, when no native attachment was produced to deliver in its
place, sanitizes the original text and sends it as a fallback on ALL
platforms (a ``response_delivery_recovered`` WARNING marks the recovery so
the silent-drop pattern is observable).  When even the sanitized recovery
yields nothing deliverable, a ``response_delivery_dropped`` ERROR fires so a
genuinely-lost response is never silent.

Salvaged and de-scoped from the superseded Discord-only PR #33842.
"""

import asyncio
import logging

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _DummyAdapter(BasePlatformAdapter):
    """Minimal BasePlatformAdapter for dispatch tests on any platform."""

    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.sent: list[dict] = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="msg-1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_event(platform: Platform, chat_id: str = "111", message_id: str = "m1") -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(platform=platform, chat_id=chat_id, chat_type="dm"),
        message_id=message_id,
    )


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()
    else:
        await asyncio.Event().wait()


def _strip_everything(adapter, monkeypatch):
    """Force the extract pipeline to reduce text_content to "" with no
    attachments — the exact failure mode that made the drop invisible."""
    monkeypatch.setattr(
        type(adapter), "extract_media", staticmethod(lambda content: ([], content))
    )
    monkeypatch.setattr(
        type(adapter), "extract_images", staticmethod(lambda content: ([], ""))
    )
    monkeypatch.setattr(
        type(adapter), "extract_local_files", staticmethod(lambda content: ([], ""))
    )


@pytest.mark.parametrize("platform", [Platform.DISCORD, Platform.TELEGRAM])
class TestExtractStripRecoveryAllPlatforms:
    """A non-empty response stripped to empty must be recovered on EVERY
    platform (the fix de-scopes the recovery from Discord-only)."""

    @pytest.mark.asyncio
    async def test_response_reduced_to_empty_is_recovered_and_sent(
        self, platform, monkeypatch, caplog
    ):
        adapter = _DummyAdapter(platform)
        adapter._keep_typing = _hold_typing

        tool_response = (
            "Based on my search, the cheapest TPE-PAR flight on Dec 14 is $632 "
            "via Saudia. Here are the top options sorted by price... "
        ) * 5
        assert len(tool_response) > 500

        async def handler(_event):
            return tool_response

        adapter.set_message_handler(handler)
        _strip_everything(adapter, monkeypatch)

        event = _make_event(platform)
        with caplog.at_level(logging.WARNING, logger="gateway.platforms.base"):
            await adapter._process_message_background(
                event, build_session_key(event.source)
            )

        # The response WAS delivered, not silently dropped.
        assert len(adapter.sent) == 1, f"expected 1 send, got {adapter.sent}"
        assert adapter.sent[0]["content"] == tool_response.strip()
        # And the recovery is observable via the stable event key.
        assert any(
            "response_delivery_recovered" in r.getMessage()
            for r in caplog.records
        ), [r.getMessage() for r in caplog.records]

    @pytest.mark.asyncio
    async def test_directives_stripped_from_fallback_text(self, platform, monkeypatch):
        adapter = _DummyAdapter(platform)
        adapter._keep_typing = _hold_typing

        raw = (
            "[[audio_as_voice]]\n[[as_document]]\nMEDIA: /tmp/nope.ogg\n"
            "The real answer the user should see."
        )

        async def handler(_event):
            return raw

        adapter.set_message_handler(handler)
        _strip_everything(adapter, monkeypatch)

        event = _make_event(platform)
        await adapter._process_message_background(event, build_session_key(event.source))

        assert len(adapter.sent) == 1
        delivered = adapter.sent[0]["content"]
        assert "[[audio_as_voice]]" not in delivered
        assert "[[as_document]]" not in delivered
        assert "MEDIA:" not in delivered
        assert "The real answer the user should see." in delivered

    @pytest.mark.asyncio
    async def test_no_fallback_when_attachment_produced(self, platform, monkeypatch):
        """When an image attachment IS extracted, the empty text_content is
        intentional — recovery must NOT re-send the original markdown and
        duplicate the attachment's content."""
        adapter = _DummyAdapter(platform)
        adapter._keep_typing = _hold_typing

        async def handler(_event):
            return "![chart](https://example.com/chart.png)"

        adapter.set_message_handler(handler)
        monkeypatch.setattr(
            type(adapter), "extract_media", staticmethod(lambda content: ([], content))
        )
        monkeypatch.setattr(
            type(adapter), "extract_images",
            staticmethod(lambda content: ([("https://example.com/chart.png", "chart")], "")),
        )
        monkeypatch.setattr(
            type(adapter), "extract_local_files", staticmethod(lambda content: ([], ""))
        )
        adapter.send_multiple_images = lambda *a, **kw: asyncio.sleep(0, result=None)

        event = _make_event(platform)
        await adapter._process_message_background(event, build_session_key(event.source))

        assert adapter.sent == [], f"expected no text echo, got {adapter.sent}"


class TestRecoveryDoesNotLeakMediaFragments:
    """The A2 recovery must not leak fragments of a MEDIA: path to the user.

    extract_media's real regex matches paths WITH SPACES; if the recovery
    sanitizes the raw pre-extract snapshot with a weaker MEDIA regex (one that
    stops at the first space), a spaced path whose file gets filtered out leaks
    a fragment like 'vacation photo.png'.  The recovery must instead use the
    post-extract_media `response`, which the strong regex already cleaned.
    """

    @pytest.mark.asyncio
    async def test_spaced_media_path_does_not_leak_fragment(self, monkeypatch, caplog):
        adapter = _DummyAdapter(Platform.DISCORD)
        adapter._keep_typing = _hold_typing

        async def handler(_event):
            # Spaced path with a valid extension — matched in full by the real
            # extract_media regex, then removed from the body.
            return "MEDIA: /tmp/nope_dir_zzz/my vacation photo.png"

        adapter.set_message_handler(handler)
        # Use the REAL extract_media (so the strong regex cleans `response`),
        # but force the path to be filtered out (unsafe/nonexistent) so we hit
        # the empty-text + no-attachment recovery branch deterministically.
        monkeypatch.setattr(
            type(adapter), "filter_media_delivery_paths", staticmethod(lambda m: [])
        )

        event = _make_event(Platform.DISCORD)
        with caplog.at_level(logging.ERROR, logger="gateway.platforms.base"):
            await adapter._process_message_background(
                event, build_session_key(event.source)
            )

        # No fragment of the media path may reach the user.
        leaked = [
            s for s in adapter.sent
            if "vacation" in s["content"] or "photo" in s["content"] or "MEDIA" in s["content"]
        ]
        assert leaked == [], f"media-path fragment leaked to user: {leaked}"
        # The genuinely-undeliverable response is logged loudly, not silent.
        assert any(
            "response_delivery_dropped" in r.getMessage()
            for r in caplog.records if r.levelno == logging.ERROR
        ), [r.getMessage() for r in caplog.records]


class TestUnrecoverableDropIsLoud:
    """A non-empty response that produces NOTHING deliverable (sanitizes to
    empty, no attachment) must log a response_delivery_dropped ERROR rather
    than vanishing silently."""

    @pytest.mark.asyncio
    async def test_directive_only_response_logs_dropped(self, monkeypatch, caplog):
        adapter = _DummyAdapter(Platform.DISCORD)
        adapter._keep_typing = _hold_typing

        async def handler(_event):
            return "[[audio_as_voice]]\nMEDIA: /tmp/missing.ogg"  # only directives

        adapter.set_message_handler(handler)
        # Extraction strips to empty AND the media path filtered out (no file).
        _strip_everything(adapter, monkeypatch)

        event = _make_event(Platform.DISCORD)
        with caplog.at_level(logging.ERROR, logger="gateway.platforms.base"):
            await adapter._process_message_background(
                event, build_session_key(event.source)
            )

        assert adapter.sent == []
        assert any(
            "response_delivery_dropped" in r.getMessage()
            for r in caplog.records if r.levelno == logging.ERROR
        ), [r.getMessage() for r in caplog.records]


# ===========================================================================
# Issue #44212: post-/stop stale interrupt silently swallows the next message
# ===========================================================================

class TestPostStopInterruptSwallow:
    """A `/stop` sets ``_interrupt_requested`` on the session's cached agent,
    but the flag is only cleared by the turn finalizer.  When the stopped run
    is hung or still draining, the flag survives the lock release and the
    session's NEXT message is killed at the top of the tool loop —
    ``interrupted=True, api_calls=0, final_response=""`` — which
    ``_normalize_empty_agent_response`` used to pass through as pure silence.

    Two-layer fix: ``_interrupt_and_clear_session`` evicts the cached agent
    (root cause), and the normalizer surfaces a notice for interrupted runs
    that never made an API call (a swallowed user turn, not a drain)."""

    def test_interrupted_zero_api_calls_surfaces_notice(self):
        """Interrupted before the first API call → the user's message was
        never processed; silence here swallows it (the #44212 malign shape:
        ``response ready ... api_calls=0 response=0 chars``)."""
        from gateway.run import _normalize_empty_agent_response

        agent_result = {
            "final_response": None,
            "api_calls": 0,
            "partial": False,
            "interrupted": True,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert response != "", "A turn killed before doing any work must not be silent"
        assert "send it again" in response.lower()

    def test_interrupted_after_work_stays_silent(self):
        """Interrupted mid-work → this is the drain of a run the user
        deliberately stopped/steered; its silence is intentional (any
        queued/interrupting message is delivered by the recursive drain
        inside _run_agent)."""
        from gateway.run import _normalize_empty_agent_response

        agent_result = {
            "final_response": None,
            "api_calls": 3,
            "partial": False,
            "interrupted": True,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert response == ""

    def test_uninterrupted_zero_api_calls_stays_silent(self):
        """No interrupt and no work — unchanged behavior."""
        from gateway.run import _normalize_empty_agent_response

        agent_result = {
            "final_response": None,
            "api_calls": 0,
            "partial": False,
            "interrupted": False,
        }

        response = _normalize_empty_agent_response(agent_result, "", history_len=10)

        assert response == ""

    @pytest.mark.asyncio
    async def test_interrupt_and_clear_session_evicts_cached_agent(self):
        """The control-interrupt path must evict the session's cached agent
        so its ``_interrupt_requested`` flag cannot leak into the next turn."""
        import threading

        from gateway.run import GatewayRunner, _INTERRUPT_REASON_STOP

        class _RecordingAgent:
            def __init__(self):
                self.interrupt_reasons = []

            def interrupt(self, reason=None):
                self.interrupt_reasons.append(reason)

        agent = _RecordingAgent()
        session_key = "agent:main:telegram:dm:12345"
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"
        )

        runner = object.__new__(GatewayRunner)
        runner._running_agents = {session_key: agent}
        runner._agent_cache = {session_key: (agent, "config-sig")}
        runner._agent_cache_lock = threading.Lock()
        runner.adapters = {}
        runner._pending_messages = {}

        invalidated = []
        runner._invalidate_session_run_generation = (
            lambda key, reason=None: invalidated.append((key, reason))
        )
        released = []
        runner._release_running_agent_state = (
            lambda key, **kw: released.append(key)
        )

        await runner._interrupt_and_clear_session(
            session_key,
            source,
            interrupt_reason=_INTERRUPT_REASON_STOP,
            invalidation_reason="stop_command",
        )

        assert agent.interrupt_reasons == [_INTERRUPT_REASON_STOP]
        assert released == [session_key]
        assert session_key not in runner._agent_cache, (
            "Cached agent with a set interrupt flag must be evicted on /stop "
            "so the flag cannot kill the session's next message (#44212)"
        )
