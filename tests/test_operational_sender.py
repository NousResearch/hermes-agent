import pytest

from tools import operational_sender


def test_sender_requires_dedicated_token(monkeypatch):
    monkeypatch.delenv("SOLO_HERMES_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="SOLO_HERMES_BOT_TOKEN"):
        operational_sender.send_operational_message("Updated: test")


def test_sender_verifies_identity_before_send(monkeypatch):
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "secret")
    calls = []

    def fake_api(token, method, data):
        calls.append(method)
        return {"result": {"username": "solovision_halo_bot"}}

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)
    with pytest.raises(RuntimeError, match="identity"):
        operational_sender.send_operational_message("Changed: test")
    assert calls == ["getMe"]


def test_sender_sends_only_after_exact_identity(monkeypatch):
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "secret")
    calls = []

    def fake_api(token, method, data):
        calls.append((method, data))
        if method == "getMe":
            return {"result": {"username": "solo_hermes_bot"}}
        return {"ok": True, "result": {"message_id": 7}}

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)
    result = operational_sender.send_operational_message("Updated: test")
    assert result["result"]["message_id"] == 7
    assert [method for method, _ in calls] == ["getMe", "sendMessage"]
    assert calls[1][1]["chat_id"] == "8148316720"