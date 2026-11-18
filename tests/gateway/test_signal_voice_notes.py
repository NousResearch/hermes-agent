"""What signal-cli receives when Hermes sends a synthesized voice reply."""
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from plugins.platforms.signal.standalone import standalone_send
from tests.gateway.test_signal import _make_signal_adapter


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="requires FFmpeg tools")
@pytest.mark.parametrize("sender", ["gateway", "standalone"])
@pytest.mark.asyncio
async def test_opus_voice_reply_at_signal_cli(sender, monkeypatch, tmp_path):
    original = tmp_path / "reply.ogg"
    await asyncio.to_thread(subprocess.run, ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=0.25",
                    "-c:a", "libopus", str(original)], check=True)
    delivered = []

    def capture(params):
        path = Path(params["attachments"][0])
        # Capture the bytes during the send, while the temporary file must still exist.
        delivered.append((params, path, path.read_bytes()))

    if sender == "gateway":
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        async def rpc(method, params, rpc_id=None):
            assert method == "send"
            capture(params)
            return {"timestamp": 123}

        adapter._rpc = rpc
        result = await adapter.send_voice("group:abc123==", str(original), caption="reply")
        assert result.success
    else:
        def post(request):
            capture(json.loads(request.content)["params"])
            return httpx.Response(200, json={"result": {"timestamp": 123}})

        client = httpx.AsyncClient
        transport = httpx.MockTransport(post)
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=transport, **kw))
        result = await standalone_send(SimpleNamespace(extra={"account": "+15550100000"}),
                                       "group:abc123==", "reply", media_files=[(str(original), True)])
        assert result["success"]

    (params, path, audio), = delivered
    assert params.get("voiceNote") is True
    assert params["groupId"] == "abc123==" and params["message"] == "reply"
    assert path.suffix == ".m4a" and not path.exists()
    assert original.read_bytes().startswith(b"OggS")
    probe = await asyncio.to_thread(subprocess.run, ["ffprobe", "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "format=format_name:stream=codec_name",
                            "-of", "json", "pipe:0"], input=audio, capture_output=True, check=True)
    media = json.loads(probe.stdout)
    assert "m4a" in media["format"]["format_name"].split(",")
    assert media["streams"][0]["codec_name"] == "aac"
