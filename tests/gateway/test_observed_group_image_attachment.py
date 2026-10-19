"""Observed Telegram photos reach the next addressed turn through real SQLite (#47415)."""

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _build_gateway_agent_history
from gateway.session import SessionStore, build_session_key
from tests.gateway.test_telegram_group_gating import _group_message, _make_adapter


def _make_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_: "native"
    return runner


def _observe_photo(store, path):
    adapter = _make_adapter(observe_unmentioned_group_messages=True)
    adapter._session_store = store
    message = _group_message("look at this")
    event = adapter._build_message_event(message, MessageType.PHOTO)
    event.media_urls, event.media_types = [str(path)], ["image/png"]
    adapter._observe_unmentioned_group_message(message, MessageType.PHOTO, event=event)
    return adapter._telegram_group_observe_shared_source(event.source)


@pytest.mark.asyncio
async def test_observed_photo_survives_sqlite_and_attaches_once(tmp_path):
    from gateway.platforms.base import get_image_cache_dir

    image = get_image_cache_dir() / "observed.png"
    image.write_bytes(b"cached pixels")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        source = _observe_photo(store, image)
        entry = store.get_or_create_session(source)
        history = store.load_transcript(entry.session_id)
        runner = _make_runner()
        event = MessageEvent(text="@bot describe the photo", source=source)
        await runner._prepare_inbound_message_text(event=event, source=source, history=history)
        assert runner._consume_pending_native_image_paths(build_session_key(source)) == [str(image)]
        replay, context = _build_gateway_agent_history(
            history, channel_prompt="observed Telegram group context",
        )
        assert replay == []
        assert "look at this" in context
        assert "image_url" not in context
        # Without the observation prompt, historical pixels still stay out of replay.
        replay, _ = _build_gateway_agent_history(history)
        assert all(isinstance(row["content"], str) for row in replay)
        history += [{"role": "user", "content": event.text}, {"role": "assistant", "content": "A desk."}]
        await runner._prepare_inbound_message_text(event=event, source=source, history=history)
        assert runner._consume_pending_native_image_paths(build_session_key(source)) == []
    finally:
        store.close_all_db_handles()


def test_observed_attachment_bounds_and_legacy_rows(tmp_path):
    from gateway.observed_images import collect_observed_image_paths, observed_image_content
    from gateway.platforms.base import get_image_cache_dir

    cache = get_image_cache_dir()
    rows = []
    for i in range(8):
        image = cache / f"observed-{i}.png"
        image.write_bytes(b"cached pixels")
        rows.append({"role": "user", "observed": True,
                     "content": observed_image_content("photo", [str(image)], ["image/png"])})
    outside = tmp_path / "private.png"
    outside.write_bytes(b"private pixels")
    for raw in (str(outside), str(cache / "missing.png"), "https://example.com/image.png"):
        rows.append({"role": "user", "observed": True,
                     "content": [{"type": "image_url", "image_url": {"url": raw}}]})
    rows.append({"role": "user", "observed": True, "content": f"image saved at {outside}"})
    assert collect_observed_image_paths(rows) == [str(cache / f"observed-{i}.png") for i in range(4, 8)]
    assert observed_image_content("pdf", [str(outside)], ["application/pdf"]) == "pdf"


@pytest.mark.platforms("posix")
def test_observed_image_cache_symlink_cannot_escape_profile(tmp_path):
    from gateway.observed_images import collect_observed_image_paths, observed_image_content
    from gateway.platforms.base import get_image_cache_dir

    outside = tmp_path / "private.png"
    outside.write_bytes(b"private pixels")
    alias = get_image_cache_dir() / "alias.png"
    alias.symlink_to(outside)
    row = {"role": "user", "observed": True,
           "content": observed_image_content("photo", [str(alias)], ["image/png"])}
    assert collect_observed_image_paths([row]) == []
