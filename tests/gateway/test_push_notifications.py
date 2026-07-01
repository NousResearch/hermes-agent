import json

import pytest

from gateway import push_notifications as push


def _cfg(tmp_path, **overrides):
    data = {
        "enabled": True,
        "relay_url": "https://relay.example.test/v1/intents",
        "relay_token_env": "HERMES_PUSH_RELAY_TOKEN",
        "registration_store": str(tmp_path / "push_devices.json"),
        "redact_body": True,
        "events": list(push.DEFAULT_EVENTS),
    }
    data.update(overrides)
    return {"push_notifications": data}


def test_register_device_persists_without_echoing_token(tmp_path):
    result = push.register_device(
        {
            "device_id": "ios-device-1",
            "platform": "apns",
            "device_token": "apns-token-value",
            "events": ["approval.request", "clarify.request"],
        },
        session_key="stored-session",
        live_session_id="live-session",
        raw_config=_cfg(tmp_path),
    )

    registration = result["registration"]
    assert registration["device_id"] == "ios-device-1"
    assert registration["session_key"] == "stored-session"
    assert registration["last_live_session_id"] == "live-session"
    assert registration["has_device_token"] is True
    assert "device_token" not in registration

    store = json.loads((tmp_path / "push_devices.json").read_text())
    assert store["devices"]["ios-device-1"]["device_token"] == "apns-token-value"


@pytest.mark.parametrize(
    "params,error",
    [
        ({"device_id": "../bad", "device_token": "token"}, "device_id"),
        ({"device_id": "ok"}, "device_token or endpoint_id"),
        ({"device_id": "ok", "endpoint_id": "bad\nvalue"}, "control characters"),
    ],
)
def test_register_device_rejects_malformed_input(tmp_path, params, error):
    with pytest.raises(push.PushRegistrationError, match=error):
        push.register_device(
            params,
            session_key="stored-session",
            live_session_id="live-session",
            raw_config=_cfg(tmp_path),
        )


def test_build_approval_intent_is_redacted_and_fifo_session_keyed():
    intent = push.build_intent(
        "approval.request",
        registration={
            "device_id": "ios-device-1",
            "platform": "apns",
            "endpoint_id": "relay-endpoint",
            "redact_body": True,
        },
        session_key="stored-session",
        live_session_id="live-session",
        payload={
            "command": "curl -H 'Authorization: token ghp_secret' https://example.test",
            "description": "Run secret command",
            "allow_permanent": False,
        },
    )

    assert intent["contract_version"] == 1
    assert intent["target"] == {
        "device_id": "ios-device-1",
        "platform": "apns",
        "endpoint_id": "relay-endpoint",
    }
    assert intent["session"]["stored_session_id"] == "stored-session"
    assert intent["session"]["live_session_id"] == "live-session"
    assert intent["event"]["category"] == "approval.request"
    assert intent["event"]["redacted"] is True
    assert "secret" not in intent["event"]["body"]
    assert intent["action_context"]["rpc_method"] == "approval.respond"
    assert intent["action_context"]["params_base"] == {"session_id": "live-session"}
    assert intent["action_context"]["choices"] == ["once", "session", "deny"]
    assert intent["action_context"]["fifo_session_keyed"] is True
    assert intent["action_context"]["request_id"] is None


def test_build_clarify_intent_carries_request_id_and_choices():
    intent = push.build_intent(
        "clarify.request",
        registration={
            "device_id": "ios-device-1",
            "platform": "apns",
            "device_token": "apns-token",
            "redact_body": False,
        },
        session_key="stored-session",
        live_session_id="live-session",
        payload={
            "request_id": "clarify-123",
            "question": "Which branch should I use?",
            "choices": ["main", "release"],
        },
    )

    assert intent["target"]["device_token"] == "apns-token"
    assert intent["event"]["row_id"] == "clarify-123"
    assert intent["event"]["body"] == "Which branch should I use?"
    assert intent["action_context"] == {
        "kind": "clarify",
        "rpc_method": "clarify.respond",
        "params_base": {"request_id": "clarify-123"},
        "answer_param": "answer",
        "choices": ["main", "release"],
        "request_id": "clarify-123",
    }


def test_notify_event_posts_only_registered_events(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PUSH_RELAY_TOKEN", "relay-secret")
    config = _cfg(tmp_path)
    push.register_device(
        {
            "device_id": "ios-device-1",
            "endpoint_id": "relay-endpoint",
            "events": ["approval.request"],
            "redact_body": False,
        },
        session_key="stored-session",
        live_session_id="live-session",
        raw_config=config,
    )
    sent = []

    def sender(push_config, intent):
        sent.append((push_config, intent))

    skipped = push.notify_event(
        "clarify.request",
        session_key="stored-session",
        live_session_id="live-session",
        payload={"request_id": "clarify-1"},
        raw_config=config,
        sender=sender,
        background=False,
    )
    queued = push.notify_event(
        "approval.request",
        session_key="stored-session",
        live_session_id="live-session",
        payload={"description": "Approve deploy"},
        raw_config=config,
        sender=sender,
        background=False,
    )

    assert skipped == {"queued": 0}
    assert queued == {"queued": 1}
    assert len(sent) == 1
    assert sent[0][0].relay_token == "relay-secret"
    assert sent[0][1]["event"]["body"] == "Approve deploy"


def test_tui_gateway_push_rpc_methods(tmp_path, monkeypatch):
    from tui_gateway import server

    config = _cfg(tmp_path, enabled=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: config)
    server._sessions["live-session"] = {"session_key": "stored-session"}
    try:
        registered = server.handle_request(
            {
                "id": "1",
                "method": "push.register",
                "params": {
                    "session_id": "live-session",
                    "device_id": "ios-device-1",
                    "endpoint_id": "relay-endpoint",
                },
            }
        )
        assert registered["result"]["registration"]["device_id"] == "ios-device-1"

        listed = server.handle_request(
            {
                "id": "2",
                "method": "push.list",
                "params": {"session_id": "live-session"},
            }
        )
        assert [r["device_id"] for r in listed["result"]["registrations"]] == [
            "ios-device-1"
        ]
        assert listed["result"]["relay"]["enabled"] is False

        unregistered = server.handle_request(
            {
                "id": "3",
                "method": "push.unregister",
                "params": {"session_id": "live-session", "device_id": "ios-device-1"},
            }
        )
        assert unregistered["result"] == {"removed": True, "device_id": "ios-device-1"}
    finally:
        server._sessions.pop("live-session", None)
