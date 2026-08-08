"""Route/payload-level tests for the Signal v0.99 REST adapter mappings.

Unlike the callers-with-stubbed-``_rpc`` tests in test_signal.py, these
exercise ``_rpc`` itself against a fake HTTP client and assert the actual
method, URL, query params, and JSON body sent to signal-cli-rest-api.
"""
import base64
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.signal_rate_limit import SignalRateLimitError


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", text=""):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


class FakeClient:
    """Captures every HTTP call the adapter makes."""

    def __init__(self, response=None):
        self.response = response if response is not None else FakeResponse(json_data={})
        self.calls = []

    async def post(self, url, json=None, timeout=None, **kwargs):
        self.calls.append({"method": "POST", "url": url, "json": json, "params": None})
        return self.response

    async def get(self, url, params=None, timeout=None, **kwargs):
        self.calls.append({"method": "GET", "url": url, "json": None, "params": params})
        return self.response

    async def request(self, method, url, json=None, timeout=None, **kwargs):
        self.calls.append({"method": method, "url": url, "json": json, "params": None})
        return self.response


def _make_adapter(monkeypatch, response=None, account="+15551234567"):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    from gateway.platforms.signal import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": account}
    adapter = SignalAdapter(config)
    adapter.client = FakeClient(response)
    return adapter


# ---------------------------------------------------------------------------
# send → POST /v2/send
# ---------------------------------------------------------------------------

class TestSendRoute:
    @pytest.mark.asyncio
    async def test_send_posts_v2_send_with_number_and_recipients(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch, FakeResponse(status_code=201, json_data={"timestamp": 1234567890123})
        )
        result = await adapter._rpc(
            "send", {"recipient": ["+15559998888"], "message": "hello"}
        )

        call = adapter.client.calls[0]
        assert call["method"] == "POST"
        assert call["url"] == "http://localhost:8080/v2/send"
        assert call["json"]["number"] == "+15551234567"
        assert call["json"]["recipients"] == ["+15559998888"]
        assert call["json"]["message"] == "hello"
        assert "text_mode" not in call["json"]
        # REST timestamp is mapped back onto the JSON-RPC-shaped result
        assert result == {"results": [{"type": "SUCCESS"}], "timestamp": 1234567890123}

    @pytest.mark.asyncio
    async def test_send_group_recipient_is_prefixed(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=201, json_data={}))
        await adapter._rpc("send", {"groupId": "abc123==", "message": "hi"})

        body = adapter.client.calls[0]["json"]
        assert body["recipients"] == ["group.abc123=="]

    @pytest.mark.asyncio
    async def test_send_group_prefix_not_duplicated(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=201, json_data={}))
        await adapter._rpc("send", {"groupId": "group.abc123==", "message": "hi"})

        assert adapter.client.calls[0]["json"]["recipients"] == ["group.abc123=="]

    @pytest.mark.asyncio
    async def test_send_text_styles_set_styled_text_mode(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=201, json_data={}))
        await adapter._rpc(
            "send",
            {"recipient": ["+15559998888"], "message": "plain", "textStyles": ["0:5:BOLD"]},
        )

        body = adapter.client.calls[0]["json"]
        assert body["text_mode"] == "styled"
        assert body["message"] == "plain"

    @pytest.mark.asyncio
    async def test_send_styled_message_overrides_plain_body(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=201, json_data={}))
        await adapter._rpc(
            "send",
            {
                "recipient": ["+15559998888"],
                "message": "plain",
                "styled_message": "**bold**",
            },
        )

        body = adapter.client.calls[0]["json"]
        assert body["message"] == "**bold**"
        assert body["text_mode"] == "styled"

    @pytest.mark.asyncio
    async def test_send_attachments_become_base64_data_uris(self, monkeypatch, tmp_path):
        payload = b"\x89PNG\r\n\x1a\nfakepng"
        att = tmp_path / "pic.png"
        att.write_bytes(payload)

        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=201, json_data={}))
        await adapter._rpc(
            "send",
            {"recipient": ["+15559998888"], "message": "", "attachments": [str(att)]},
        )

        body = adapter.client.calls[0]["json"]
        (encoded,) = body["base64_attachments"]
        b64 = base64.b64encode(payload).decode("ascii")
        assert encoded == f"data:image/png;filename=pic.png;base64,{b64}"

    @pytest.mark.asyncio
    async def test_send_error_returns_none(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            FakeResponse(status_code=400, json_data={"error": "Invalid recipient"}),
        )
        result = await adapter._rpc(
            "send", {"recipient": ["+15559998888"], "message": "hello"}
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_send_rate_limit_raises_when_opted_in(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            FakeResponse(
                status_code=429,
                json_data={
                    "error": "Failed to send: [429] RateLimitException. Retry after 30 seconds."
                },
            ),
        )
        with pytest.raises(SignalRateLimitError) as exc_info:
            await adapter._rpc(
                "send",
                {"recipient": ["+15559998888"], "message": "hello"},
                raise_on_rate_limit=True,
            )
        assert exc_info.value.retry_after == 30

    @pytest.mark.asyncio
    async def test_send_rate_limit_swallowed_by_default(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            FakeResponse(status_code=429, json_data={"error": "[429] RateLimitException"}),
        )
        result = await adapter._rpc(
            "send", {"recipient": ["+15559998888"], "message": "hello"}
        )
        assert result is None


# ---------------------------------------------------------------------------
# sendTyping → PUT/DELETE /v1/typing-indicator/{account}
# ---------------------------------------------------------------------------

class TestTypingIndicatorRoute:
    @pytest.mark.asyncio
    async def test_start_typing_is_put(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=204))
        result = await adapter._rpc("sendTyping", {"recipient": ["+15559998888"]})

        call = adapter.client.calls[0]
        assert call["method"] == "PUT"
        assert call["url"] == "http://localhost:8080/v1/typing-indicator/%2B15551234567"
        assert call["json"] == {"recipient": "+15559998888"}
        assert result == {}

    @pytest.mark.asyncio
    async def test_stop_typing_is_delete(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=204))
        await adapter._rpc(
            "sendTyping", {"recipient": ["+15559998888"], "stop": True}
        )

        call = adapter.client.calls[0]
        assert call["method"] == "DELETE"
        assert call["url"] == "http://localhost:8080/v1/typing-indicator/%2B15551234567"

    @pytest.mark.asyncio
    async def test_typing_group_recipient(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=204))
        await adapter._rpc("sendTyping", {"groupId": "xyz=="})

        assert adapter.client.calls[0]["json"] == {"recipient": "group.xyz=="}

    @pytest.mark.asyncio
    async def test_typing_without_recipient_makes_no_request(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        result = await adapter._rpc("sendTyping", {})
        assert result is None
        assert adapter.client.calls == []


# ---------------------------------------------------------------------------
# sendReaction → POST/DELETE /v1/reactions/{account}
# ---------------------------------------------------------------------------

class TestReactionRoute:
    @pytest.mark.asyncio
    async def test_add_reaction_is_post_with_payload(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=204))
        result = await adapter._rpc(
            "sendReaction",
            {
                "recipient": ["+15559998888"],
                "emoji": "👍",
                "targetAuthor": "+15557776666",
                "targetTimestamp": 1234567890123,
            },
        )

        call = adapter.client.calls[0]
        assert call["method"] == "POST"
        assert call["url"] == "http://localhost:8080/v1/reactions/%2B15551234567"
        assert call["json"] == {
            "reaction": "👍",
            "recipient": "+15559998888",
            "target_author": "+15557776666",
            "timestamp": 1234567890123,
        }
        assert result == {}

    @pytest.mark.asyncio
    async def test_remove_reaction_is_delete(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=204))
        await adapter._rpc(
            "sendReaction",
            {
                "recipient": ["+15559998888"],
                "emoji": "👍",
                "targetAuthor": "+15557776666",
                "targetTimestamp": 1234567890123,
                "remove": True,
            },
        )

        call = adapter.client.calls[0]
        assert call["method"] == "DELETE"
        assert call["url"] == "http://localhost:8080/v1/reactions/%2B15551234567"
        assert call["json"]["reaction"] == "👍"


# ---------------------------------------------------------------------------
# listContacts / getContact → GET /v1/contacts/{account}
# ---------------------------------------------------------------------------

class TestContactsRoute:
    @pytest.mark.asyncio
    async def test_list_contacts_url_and_normalization(self, monkeypatch):
        rest_contact = {
            "number": "+15559998888",
            "uuid": "abcd-1234",
            "name": "Ada",
            "profile_name": "Ada L",
            "profile": {"given_name": "Ada"},
        }
        adapter = _make_adapter(
            monkeypatch, FakeResponse(status_code=200, json_data=[rest_contact])
        )
        contacts = await adapter._rpc("listContacts", {"account": "+15551234567"})

        call = adapter.client.calls[0]
        assert call["method"] == "GET"
        assert call["url"] == "http://localhost:8080/v1/contacts/%2B15551234567"
        assert call["params"] is None

        (contact,) = contacts
        # REST fields are normalized to the JSON-RPC shape callers expect
        assert contact["number"] == "+15559998888"
        assert contact["recipient"] == "+15559998888"
        assert contact["uuid"] == "abcd-1234"
        assert contact["serviceId"] == "abcd-1234"
        assert contact["name"] == "Ada"
        assert contact["profileName"] == "Ada L"

    @pytest.mark.asyncio
    async def test_list_contacts_all_recipients_query_param(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FakeResponse(status_code=200, json_data=[]))
        await adapter._rpc(
            "listContacts", {"account": "+15551234567", "allRecipients": True}
        )
        assert adapter.client.calls[0]["params"] == {"all_recipients": "true"}

    @pytest.mark.asyncio
    async def test_get_contact_matches_by_number_or_uuid(self, monkeypatch):
        contacts = [
            {"number": "+15550001111", "uuid": "uuid-a", "name": "A"},
            {"number": "+15559998888", "uuid": "uuid-b", "name": "B"},
        ]
        adapter = _make_adapter(
            monkeypatch, FakeResponse(status_code=200, json_data=contacts)
        )
        by_uuid = await adapter._rpc("getContact", {"contactAddress": "uuid-b"})
        assert by_uuid["name"] == "B"
        # getContact is implemented via the contacts listing route
        assert adapter.client.calls[0]["url"].endswith("/v1/contacts/%2B15551234567")
        assert adapter.client.calls[0]["params"] == {"all_recipients": "true"}


# ---------------------------------------------------------------------------
# getAttachment → GET /v1/attachments/{id}
# ---------------------------------------------------------------------------

class TestAttachmentRoute:
    @pytest.mark.asyncio
    async def test_get_attachment_url_encodes_id(self, monkeypatch):
        raw = b"attachment-bytes"
        adapter = _make_adapter(
            monkeypatch, FakeResponse(status_code=200, content=raw)
        )
        result = await adapter._rpc("getAttachment", {"id": "dir/file name.bin"})

        call = adapter.client.calls[0]
        assert call["method"] == "GET"
        assert call["url"] == "http://localhost:8080/v1/attachments/dir%2Ffile%20name.bin"
        assert result == {"data": base64.b64encode(raw).decode("ascii")}

    @pytest.mark.asyncio
    async def test_get_attachment_missing_id_makes_no_request(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert await adapter._rpc("getAttachment", {}) is None
        assert adapter.client.calls == []


# ---------------------------------------------------------------------------
# Dispatcher guardrails
# ---------------------------------------------------------------------------

class TestDispatcherGuardrails:
    @pytest.mark.asyncio
    async def test_unsupported_method_returns_none_without_request(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert await adapter._rpc("updateGroup", {}) is None
        assert adapter.client.calls == []

    @pytest.mark.asyncio
    async def test_rpc_without_client_returns_none(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.client = None
        assert await adapter._rpc("send", {"message": "x"}) is None
