"""Registry routing preserves Photon attachments without an in-process gateway."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from plugins.platforms.photon import adapter as photon
from tools import send_message_tool as delivery


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["Report", ""])
async def test_registered_photon_sender_receives_media_without_live_adapter(monkeypatch, tmp_path, message):
    image = tmp_path / "image.png"
    image.write_bytes(b"photon attachment contents")
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "sidecar-token")
    registrations = []
    photon.register(SimpleNamespace(
        register_platform=lambda **kwargs: registrations.append(PlatformEntry(**kwargs)),
        register_cli_command=lambda **_kwargs: None))
    entry = registrations[0]
    get_entry = platform_registry.get
    monkeypatch.setattr(platform_registry, "get", lambda name: entry if name == "photon" else get_entry(name))
    monkeypatch.setattr(delivery, "_live_adapter", lambda _platform: (None, None))
    requests = []

    async def receive(request):
        payload = await request.json()
        contents = Path(payload["path"]).read_bytes() if request.path == "/send-attachment" else None
        requests.append((request.path, request.headers.get("X-Hermes-Sidecar-Token"), payload, contents))
        return web.json_response({"ok": True, "messageId": "uploaded-id"})

    app = web.Application()
    app.router.add_post("/send", receive)
    app.router.add_post("/send-attachment", receive)
    async with TestServer(app) as server:
        result = await delivery._send_to_platform(
            Platform("photon"), PlatformConfig(extra={"sidecar_port": server.port}),
            "destination-space", message, media_files=[(str(image), False)])

    assert result.get("success"), result
    assert "warnings" not in result
    assert [path for path, _token, _payload, _contents in requests] == (
        ["/send", "/send-attachment"] if message else ["/send-attachment"])
    assert all(token == "sidecar-token" for _path, token, _payload, _contents in requests)
    _path, _token, attachment, contents = requests[-1]
    assert attachment["spaceId"] == "destination-space"
    assert attachment["path"] == str(image.resolve())
    assert attachment["kind"] == "attachment"
    assert attachment["mimeType"] == "image/png"
    assert contents == b"photon attachment contents"
    assert result["media_delivered"] is True
