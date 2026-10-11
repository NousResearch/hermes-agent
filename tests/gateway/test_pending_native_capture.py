"""Native identity and cached bytes remain attributable after a pending round trip."""

import hashlib
from typing import Any

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageType
from gateway.run import GatewayRunner
from gateway.shutdown_pending import PendingQueueSnapshot
from gateway.shutdown_pending_codec import decode_pending_event
from hermes_constants import get_hermes_home
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["m.text", "m.image", "m.sticker", "m.audio"])
async def test_actual_matrix_producer_preserves_native_identity_and_cached_bytes(kind):
    runner = GatewayRunner(GatewayConfig())
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@hermes:example.org",
            },
        )
    )
    adapter.gateway_runner = runner
    runner.adapters[Platform.MATRIX] = adapter
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(
        chat_id="!room:example.org", user_id="@alice:example.org", chat_type="group"
    )
    content: dict[str, Any] = {"msgtype": kind, "body": "authored"}
    paths, types = [], []
    extras: dict[str, Any] = {}
    if kind != "m.text":
        path = get_hermes_home() / "cache" / "native.dat"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"original native attachment")
        content.update({
            "url": "mxc://example.org/original",
            "info": {"mimetype": "audio/ogg" if kind == "m.audio" else "image/png"},
        })
        paths, types = [str(path)], [content["info"]["mimetype"]]
        extras = {
            "media_msgtype": kind,
            "media_urls": paths,
            "media_types": types,
            "message_type": MessageType.VOICE
            if kind == "m.audio"
            else MessageType.PHOTO,
        }
    event = await adapter._build_inbound_event(
        "!room:example.org",
        "@alice:example.org",
        "$original",
        "authored",
        content,
        {},
        ctx=("authored", False, "group", None, "Alice", False, source),
        **extras,
    )
    assert event is not None
    assert adapter._canonicalize(event.source) is not None
    snapshot = PendingQueueSnapshot.capture(
        runner._session_key_for_source(source), [event]
    )
    restored = decode_pending_event(snapshot.events[0], adapter=adapter)
    native = getattr(restored, "_pending_native_input", None)
    assert native is not None
    assert native.to_payload() == {
        "platform": "matrix",
        "content": content,
        "attachments": [
            {
                "path": path,
                "digest": hashlib.sha256(b"original native attachment").hexdigest(),
            }
            for path in paths
        ],
    }
    assert native.attachments_available(restored.media_urls)
    if paths:
        from pathlib import Path

        Path(paths[0]).write_bytes(b"replacement with the same path")
        assert not native.attachments_available(restored.media_urls)
    from gateway.platforms.base_pending import (
        merge_recorded,
        reserve_pending_dispatch,
        release_pending_dispatch_record,
    )
    from gateway.platforms.base_pending_merge import _absorb_pending_text

    secondary_content = {"msgtype": "m.text", "body": "second"}
    secondary = await adapter._build_inbound_event(
        "!room:example.org",
        "@alice:example.org",
        "$second",
        "second",
        secondary_content,
        {},
        ctx=("second", False, "group", None, "Alice", False, source),
    )
    assert secondary is not None
    assert adapter._canonicalize(secondary.source) is not None
    merge_recorded(restored, secondary, _absorb_pending_text)
    combined = PendingQueueSnapshot.capture(snapshot.session_key, [restored])
    remaining = decode_pending_event(combined.events[0], adapter=adapter)
    reserved = reserve_pending_dispatch(adapter, snapshot.session_key, remaining)
    try:
        assert adapter.withdraw_pending_message(
            "$original", chat_id=source.chat_id, sender_id="@alice:example.org"
        )
        captured = PendingQueueSnapshot.capture(
            snapshot.session_key, [remaining]
        ).events[0]
        assert (
            captured["uid"],
            captured["input_owner"],
            captured["event"]["text"],
            captured["native"],
        ) == (
            snapshot.events[0]["uid"],
            snapshot.events[0]["input_owner"],
            "second",
            {"platform": "matrix", "content": secondary_content, "attachments": []},
        )
        decoded = decode_pending_event(captured, adapter=adapter)
        assert decoded._pending_native_input is not None
        assert decoded._pending_native_input.attachments_available(decoded.media_urls)
    finally:
        release_pending_dispatch_record(adapter, snapshot.session_key, reserved)
