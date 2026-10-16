"""Cloud reply quoting through the documented YAML configuration and Graph sends."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, load_gateway_config


@pytest.mark.parametrize(
    ("setting", "quoted_chunks"),
    [
        ("reply_to_mode: off", (False, False)),
        ('reply_to_mode: "off"', (False, False)),
        ("reply_to_mode: first", (True, False)),
        ("reply_to_mode: all", (True, True)),
        ("", (True, False)),
        ("reply_to_mode: null", (True, False)),
        ('reply_to_mode: ""', (True, False)),
        ('extra: {reply_to_mode: "off"}', (True, False)),
        ('reply_to_mode: off\n    extra: {reply_to_mode: all}', (False, False)),
    ],
    ids=["bare-off", "quoted-off", "first", "all", "default", "null", "empty",
         "extra-only", "typed-over-extra"],
)
@pytest.mark.parametrize("send_kind", ["text", "interactive", "media"])
@pytest.mark.asyncio
async def test_yaml_reply_mode_graph_payloads(tmp_path, monkeypatch, setting, quoted_chunks, send_kind):
    from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / "config.yaml").write_text(
        f"platforms:\n  whatsapp_cloud:\n    enabled: true\n    {setting}\n", encoding="utf-8",
    )
    config = load_gateway_config()
    adapter = WhatsAppCloudAdapter(config.platforms[Platform.WHATSAPP_CLOUD])
    response = MagicMock(status_code=200)
    response.json.return_value = {"messages": [{"id": "wamid.OUTBOUND"}]}
    adapter._http_client = MagicMock()
    adapter._http_client.post = AsyncMock(return_value=response)
    chat_id, anchor = "15551234567", "wamid.INBOUND"

    if send_kind == "text":
        # Exercise real formatting and chunking with more than one Graph message.
        result = await adapter.send(chat_id, "x" * 5000, reply_to=anchor)
        expected_quotes = quoted_chunks
        expected_type = "text"
    elif send_kind == "interactive":
        result = await adapter.send_clarify(
            chat_id, "Continue?", ["Yes", "No"], "clarify-id", "session-key",
            metadata={"reply_to_message_id": anchor},
        )
        expected_quotes = quoted_chunks[:1]
        expected_type = "interactive"
    else:
        result = await adapter.send_image(chat_id, "https://example.com/image.jpg", reply_to=anchor)
        expected_quotes = quoted_chunks[:1]
        expected_type = "image"

    assert result.success
    payloads = [call.kwargs["json"] for call in adapter._http_client.post.call_args_list]
    assert len(payloads) == len(expected_quotes)
    for payload, quoted in zip(payloads, expected_quotes, strict=True):
        assert payload["type"] == expected_type
        assert payload["to"] == chat_id
        if quoted:
            assert payload["context"] == {"message_id": anchor}
        else:
            assert "context" not in payload
