"""Observed (unmentioned) group images must reach the model on a later turn.

Regression coverage for the asymmetry reported in #47415: the @mention path
dispatches with ``media_urls`` and the model sees the photo natively, while the
observed path only ever persisted a text note pointing at the cached file.  A
follow-up "what was in that image?" therefore had no image to look at.
"""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_: "native"
    return runner


def _group_source(chat_id: str = "group-1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="group",
        user_name="alice",
    )


def _observed_row(path: str, *, mime: str = "image/png") -> dict:
    return {
        "role": "user",
        "observed": True,
        "content": (
            "[Alice Example|111]\nveja esta foto\n\n"
            f"[image 'observed.png' saved at: {path}]"
        ),
        "media_urls": [path],
        "media_types": [mime],
    }


@pytest.mark.asyncio
async def test_observed_group_image_is_attached_to_next_addressed_turn():
    """An addressed turn carrying no media of its own still sees the observed photo."""
    runner = _make_runner()
    source = _group_source()

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text="@bot what was in that image?", source=source),
        source=source,
        history=[_observed_row("/tmp/observed.png")],
    )

    assert runner._consume_pending_native_image_paths(build_session_key(source)) == [
        "/tmp/observed.png"
    ]


@pytest.mark.asyncio
async def test_observed_image_precedes_current_turn_image():
    """Observed history attaches before the current message's own attachment."""
    runner = _make_runner()
    source = _group_source()

    await runner._prepare_inbound_message_text(
        event=MessageEvent(
            text="@bot compare this with the earlier one",
            message_type=MessageType.PHOTO,
            source=source,
            media_urls=["/tmp/current.png"],
            media_types=["image/png"],
        ),
        source=source,
        history=[_observed_row("/tmp/observed.png")],
    )

    assert runner._consume_pending_native_image_paths(build_session_key(source)) == [
        "/tmp/observed.png",
        "/tmp/current.png",
    ]


@pytest.mark.asyncio
async def test_observed_non_image_media_is_not_attached():
    """Only images are attachable; observed documents stay as their text note."""
    runner = _make_runner()
    source = _group_source()

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text="@bot anything useful there?", source=source),
        source=source,
        history=[_observed_row("/tmp/observed.pdf", mime="application/pdf")],
    )

    assert runner._consume_pending_native_image_paths(build_session_key(source)) == []


@pytest.mark.asyncio
async def test_observed_images_are_capped_to_most_recent():
    """A busy group must not replay every cached photo on every addressed turn."""
    runner = _make_runner()
    source = _group_source()
    history = [_observed_row(f"/tmp/observed-{i}.png") for i in range(10)]

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text="@bot what did you see?", source=source),
        source=source,
        history=history,
    )

    attached = runner._consume_pending_native_image_paths(build_session_key(source))
    assert attached == [
        "/tmp/observed-6.png",
        "/tmp/observed-7.png",
        "/tmp/observed-8.png",
        "/tmp/observed-9.png",
    ]


@pytest.mark.asyncio
async def test_observed_image_is_not_reattached_after_an_addressed_exchange():
    """Attach on the turn that follows the photo, not on every later reply.

    Re-uploading the same pixels on every subsequent group message would put
    vision cost on the whole conversation; the text note keeps carrying it.
    """
    runner = _make_runner()
    source = _group_source()
    history = [
        _observed_row("/tmp/observed.png"),
        {"role": "user", "content": "@bot what was in that image?"},
        {"role": "assistant", "content": "A desk sheet."},
    ]

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text="@bot thanks, and what about the totals?", source=source),
        source=source,
        history=history,
    )

    assert runner._consume_pending_native_image_paths(build_session_key(source)) == []


@pytest.mark.asyncio
async def test_observed_image_after_an_exchange_is_attached_again():
    """A freshly observed photo still attaches even with older turns behind it."""
    runner = _make_runner()
    source = _group_source()
    history = [
        {"role": "user", "content": "@bot hello"},
        {"role": "assistant", "content": "Hi!"},
        _observed_row("/tmp/newer.png"),
    ]

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text="@bot what did Alice just post?", source=source),
        source=source,
        history=history,
    )

    assert runner._consume_pending_native_image_paths(build_session_key(source)) == [
        "/tmp/newer.png"
    ]


@pytest.mark.asyncio
async def test_legacy_observed_rows_without_media_urls_attach_nothing():
    """Transcripts written before this fix carry only the text note — no crash."""
    runner = _make_runner()
    source = _group_source()
    legacy = {
        "role": "user",
        "observed": True,
        "content": "[Alice Example|111]\nveja esta foto\n\n[image 'observed.png' saved at: /tmp/old.png]",
    }

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text="@bot what was in that image?", source=source),
        source=source,
        history=[legacy],
    )

    assert runner._consume_pending_native_image_paths(build_session_key(source)) == []
