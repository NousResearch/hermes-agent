import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from plugins.elevenlabs_calls import tools as call_tool


BASE_ARGS = {
    "agent_id": "agent-1",
    "agent_phone_number_id": "phone-1",
    "to_number": "+12025550123",
    "authorization_text": "Call +1 202 555 0123",
    "call_purpose": "Ask whether the shop is open tomorrow.",
}
USER_TASK = "Call +1 202 555 0123 and ask whether the shop is open tomorrow."


def _result(args=None, *, user_task=USER_TASK):
    return json.loads(
        call_tool._handle_outbound_call(args or BASE_ARGS, user_task=user_task)
    )


@pytest.mark.parametrize(
    ("args", "user_task", "reason_fragment"),
    [
        (BASE_ARGS, "", "unavailable"),
        ({**BASE_ARGS, "authorization_text": ""}, USER_TASK, "must quote"),
        (
            {**BASE_ARGS, "authorization_text": "Call somebody"},
            USER_TASK,
            "not found verbatim",
        ),
        (
            {**BASE_ARGS, "authorization_text": "Message +1 202 555 0123"},
            "Message +1 202 555 0123",
            "does not explicitly",
        ),
        ({**BASE_ARGS, "to_number": "2025550123"}, USER_TASK, "valid E.164"),
        (
            {**BASE_ARGS, "authorization_text": "Call the shop"},
            "Call the shop",
            "does not contain",
        ),
        (
            {**BASE_ARGS, "authorization_text": "Call me"},
            "Call me",
            "does not contain",
        ),
    ],
)
def test_blocks_when_latest_message_does_not_authorize_exact_call(
    args, user_task, reason_fragment
):
    result = _result(args, user_task=user_task)
    assert result["ok"] is False
    assert result["blocked"] is True
    assert reason_fragment in result["reason"]


@pytest.mark.parametrize(
    ("to_number", "authorization_text"),
    [
        ("+12025550123", "Call 202-555-0123"),
        ("+441632960000", "Ring 01632 960000"),
    ],
)
def test_accepts_supported_national_number_format(
    monkeypatch, to_number, authorization_text
):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret")
    args = {
        **BASE_ARGS,
        "authorization_text": authorization_text,
        "to_number": to_number,
    }
    with patch.object(
        call_tool.urllib.request, "urlopen", return_value=_response()
    ) as opener:
        result = _result(args, user_task=f"{authorization_text} and read the update.")
    assert result["ok"] is True
    assert opener.call_count == 1


def test_requires_api_key_after_authorization(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    result = _result()
    assert result == {"ok": False, "error": "ELEVENLABS_API_KEY is not configured."}


class _response:
    def __init__(self, body=b'{"conversation_id":"conv-1"}'):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def test_success_sends_expected_payload_once(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret")
    args = {
        **BASE_ARGS,
        "phone_number_provider": "twilio",
        "recipient_name": "Reception",
        "conversation_initiation_client_data": {
            "dynamic_variables": {"existing": "kept"}
        },
    }
    with patch.object(
        call_tool.urllib.request, "urlopen", return_value=_response()
    ) as opener:
        result = _result(args)

    assert result["ok"] is True
    assert result["provider"] == "twilio"
    assert opener.call_count == 1
    request = opener.call_args.args[0]
    assert request.full_url.endswith("/twilio/outbound-call")
    assert request.headers["Xi-api-key"] == "secret"
    payload = json.loads(request.data)
    client_data = payload["conversation_initiation_client_data"]
    assert client_data["dynamic_variables"]["existing"] == "kept"
    assert client_data["dynamic_variables"]["outbound_recipient_name"] == "Reception"
    prompt = client_data["conversation_config_override"]["agent"]["prompt"]["prompt"]
    assert "Never claim to be human" in prompt
    assert "John" not in prompt


def test_http_error_does_not_try_another_provider(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret")
    error = urllib.error.HTTPError(
        "https://example.test", 422, "bad request", {}, io.BytesIO(b'{"detail":"bad"}')
    )
    with patch.object(call_tool.urllib.request, "urlopen", side_effect=error) as opener:
        result = _result()
    assert result["ok"] is False
    assert result["status"] == 422
    assert result["retry_requires_new_user_authorization"] is True
    assert opener.call_count == 1


def test_network_error_does_not_retry(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "secret")
    with patch.object(
        call_tool.urllib.request,
        "urlopen",
        side_effect=urllib.error.URLError("offline"),
    ) as opener:
        result = _result()
    assert result["ok"] is False
    assert result["retry_requires_new_user_authorization"] is True
    assert opener.call_count == 1


def test_plugin_registers_guarded_tool():
    from plugins import elevenlabs_calls

    registrations = []

    class Context:
        def register_tool(self, **kwargs):
            registrations.append(kwargs)

    elevenlabs_calls.register(Context())

    assert len(registrations) == 1
    registration = registrations[0]
    assert registration["name"] == "elevenlabs_outbound_call"
    assert registration["toolset"] == "elevenlabs_calls"
    assert registration["handler"] is call_tool._handle_outbound_call
    assert registration["check_fn"] is call_tool._check_available
    assert registration["requires_env"] == ["ELEVENLABS_API_KEY"]
