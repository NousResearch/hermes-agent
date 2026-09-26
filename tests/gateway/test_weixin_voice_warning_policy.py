"""Voice downgrade diagnostic is optional; the actual attachment is not."""
import json
from unittest.mock import AsyncMock

import pytest
from gateway.config import PlatformConfig
from gateway.platforms import weixin as wx


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
@pytest.mark.parametrize("caption", [None, "Warning: requested caption"])
async def test_real_voice_file_send_filters_only_generated_caption(tmp_path, monkeypatch, setting, caption):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    config = {} if setting is None else {"display": {"suppress_warning_notifications": setting}}
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    adapter = wx.WeixinAdapter(PlatformConfig(token="fixture-token", extra={"account_id": "fixture"}))
    adapter._send_session = object()
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"fixture bytes")
    posts = []
    async def post(*args, **kwargs):
        posts.append(kwargs["payload"])
        return {"ret": 0}
    monkeypatch.setattr(wx, "_get_upload_url", AsyncMock(return_value={"upload_full_url": "https://example.invalid/upload"}))
    upload = AsyncMock(return_value="upload-reference")
    monkeypatch.setattr(wx, "_upload_ciphertext", upload)
    monkeypatch.setattr(wx, "_api_post", post)
    result = await adapter.send_voice("recipient", str(audio), caption=caption)
    assert result.success and result.message_id
    upload.assert_awaited_once()
    items = [item for payload in posts for item in payload["msg"]["item_list"]]
    text = [item["text_item"]["text"] for item in items if "text_item" in item]
    assert text == ([caption] if caption else [] if setting is True else ["[voice message as attachment]"])
    assert len([item for item in items if "file_item" in item]) == 1
