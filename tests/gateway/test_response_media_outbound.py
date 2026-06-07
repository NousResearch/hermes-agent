"""Outbound media: worker mints refs from MEDIA: tags; front relays + uploads."""

import pytest

from gateway.media_spool import MediaSpool, mint_outbound, kind_for
from gateway.worker_client import WorkerClient


def test_kind_classification():
    assert kind_for("/x/a.png", False) == "image"
    assert kind_for("/x/a.ogg", True) == "voice"
    assert kind_for("/x/a.mp4", False) == "video"
    assert kind_for("/x/a.pdf", False) == "document"


def test_mint_outbound_produces_resolvable_refs(tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"IMG")
    spool = MediaSpool(tmp_path / "spool")
    refs = mint_outbound(spool, [(str(img), False)])
    assert len(refs) == 1
    assert refs[0]["kind"] == "image"
    assert spool.resolve(refs[0]["ref"]) == b"IMG"


def test_mint_outbound_marks_voice(tmp_path):
    voice = tmp_path / "v.ogg"
    voice.write_bytes(b"OGG")
    spool = MediaSpool(tmp_path / "spool")
    refs = mint_outbound(spool, [(str(voice), True)])
    assert refs[0]["is_voice"] is True
    assert refs[0]["kind"] == "voice"


@pytest.mark.asyncio
async def test_client_relays_response_media_to_handler():
    handled = []

    async def media_handler(event):
        handled.append(event["media"])

    events = [
        {"event": "response.media", "media": [{"ref": "r1", "kind": "image", "filename": "a.png", "mime": "image/png", "size": 3}]},
        {"event": "run.completed", "output": "see image", "usage": {}},
    ]

    async def fake_post(url, body):
        return {"run_id": "run_1"}

    async def fake_sse(url):
        for e in events:
            yield e

    client = WorkerClient("http://127.0.0.1:5000", "k", post=fake_post, sse=fake_sse)

    class C:
        def on_delta(self, t): ...

    result = await client.dispatch(input="x", consumer=C(), media_handler=media_handler)
    assert handled == [events[0]["media"]]
    assert result["output"] == "see image"


@pytest.mark.asyncio
async def test_front_delivers_media_via_send_methods(tmp_path, monkeypatch):
    import gateway.run as gateway_run
    from unittest.mock import AsyncMock, MagicMock
    from gateway.config import Platform
    from gateway.session import SessionSource

    monkeypatch.setenv("HERMES_MEDIA_SPOOL", str(tmp_path / "spool"))
    # Worker mints an image + a voice file into the shared spool.
    from gateway.media_spool import MediaSpool, mint_outbound, default_spool_root

    (tmp_path / "img.png").write_bytes(b"IMG")
    (tmp_path / "v.ogg").write_bytes(b"OGG")
    refs = mint_outbound(MediaSpool(default_spool_root()), [(str(tmp_path / "img.png"), False), (str(tmp_path / "v.ogg"), True)])

    adapter = MagicMock()
    adapter.send_multiple_images = AsyncMock()
    adapter.send_voice = AsyncMock()
    r = object.__new__(gateway_run.GatewayRunner)
    r.adapters = {Platform.TELEGRAM: adapter}
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", user_id="u1")

    await r._deliver_worker_media(MagicMock(), src, {"media": refs})

    adapter.send_voice.assert_awaited_once()
    adapter.send_multiple_images.assert_awaited_once()
