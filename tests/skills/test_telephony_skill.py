from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "productivity"
    / "telephony"
    / "scripts"
    / "telephony.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("telephony_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_save_twilio_writes_env_and_state(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    result = mod.save_twilio(
        "AC123",
        "secret-token",
        phone_number="+1 (702) 555-1234",
        phone_sid="PN123",
    )

    env_text = (tmp_path / ".hermes" / ".env").read_text(encoding="utf-8")
    state = json.loads((tmp_path / ".hermes" / "telephony_state.json").read_text(encoding="utf-8"))

    assert result["success"] is True
    assert "TWILIO_ACCOUNT_SID=AC123" in env_text
    assert "TWILIO_AUTH_TOKEN=secret-token" in env_text
    assert "TWILIO_PHONE_NUMBER=+17025551234" in env_text
    assert "TWILIO_PHONE_NUMBER_SID=PN123" in env_text
    assert state["twilio"]["default_phone_number"] == "+17025551234"
    assert state["twilio"]["default_phone_sid"] == "PN123"


def test_upsert_env_updates_existing_values(tmp_path: Path):
    mod = load_module()
    env_path = tmp_path / ".env"
    env_path.write_text("TWILIO_PHONE_NUMBER=+15550000000\nOTHER=keep\n", encoding="utf-8")

    mod._upsert_env_file(
        {
            "TWILIO_PHONE_NUMBER": "+15551112222",
            "TWILIO_PHONE_NUMBER_SID": "PN999",
        },
        env_path=env_path,
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert "TWILIO_PHONE_NUMBER=+15551112222" in env_text
    assert "TWILIO_PHONE_NUMBER_SID=PN999" in env_text
    assert "OTHER=keep" in env_text


def test_messages_after_checkpoint_returns_only_newer_items():
    mod = load_module()
    messages = [
        {"sid": "SM3", "body": "newest"},
        {"sid": "SM2", "body": "middle"},
        {"sid": "SM1", "body": "oldest"},
    ]

    assert mod._messages_after_checkpoint(messages, "") == messages
    assert mod._messages_after_checkpoint(messages, "SM2") == [{"sid": "SM3", "body": "newest"}]
    assert mod._messages_after_checkpoint(messages, "SM3") == []


def test_twilio_buy_number_saves_env_and_state(tmp_path: Path):
    mod = load_module()
    state_path = tmp_path / "telephony_state.json"
    env_path = tmp_path / ".env"

    mod._twilio_request = lambda method, path, params=None, form=None: {
        "sid": "PN111",
        "phone_number": "+17025550123",
        "friendly_name": "Test Number",
        "capabilities": {"voice": True, "sms": True},
    }

    result = mod._twilio_buy_number(
        "+17025550123",
        save_env=True,
        state_path=state_path,
        env_path=env_path,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    env_text = env_path.read_text(encoding="utf-8")

    assert result["phone_sid"] == "PN111"
    assert state["twilio"]["default_phone_number"] == "+17025550123"
    assert state["twilio"]["default_phone_sid"] == "PN111"
    assert "TWILIO_PHONE_NUMBER=+17025550123" in env_text
    assert "TWILIO_PHONE_NUMBER_SID=PN111" in env_text


def test_twilio_inbox_marks_seen_checkpoint(tmp_path: Path):
    mod = load_module()
    state_path = tmp_path / "telephony_state.json"
    mod._save_state(
        {
            "version": 1,
            "twilio": {
                "default_phone_number": "+17025550123",
                "default_phone_sid": "PN111",
                "last_inbound_message_sid": "SM1",
            },
        },
        state_path,
    )

    mod._twilio_owned_numbers = lambda limit=50: [
        mod.OwnedTwilioNumber(
            sid="PN111",
            phone_number="+17025550123",
            friendly_name="Main",
            capabilities={"voice": True, "sms": True},
        )
    ]
    mod._twilio_request = lambda method, path, params=None, form=None: {
        "messages": [
            {
                "sid": "SM3",
                "direction": "inbound",
                "status": "received",
                "from": "+15551230000",
                "to": "+17025550123",
                "date_sent": "Tue, 14 Mar 2026 09:00:00 +0000",
                "body": "new message",
                "num_media": "0",
            },
            {
                "sid": "SM1",
                "direction": "inbound",
                "status": "received",
                "from": "+15551110000",
                "to": "+17025550123",
                "date_sent": "Tue, 14 Mar 2026 08:00:00 +0000",
                "body": "old message",
                "num_media": "0",
            },
        ]
    }

    result = mod._twilio_inbox(limit=10, since_last=True, mark_seen=True, state_path=state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert result["count"] == 1
    assert result["messages"][0]["sid"] == "SM3"
    assert state["twilio"]["last_inbound_message_sid"] == "SM3"


def test_vapi_import_twilio_number_saves_phone_number_id(tmp_path: Path):
    mod = load_module()
    state_path = tmp_path / "telephony_state.json"
    env_path = tmp_path / ".env"

    mod._vapi_api_key = lambda: "vapi-key"
    mod._twilio_creds = lambda: ("AC123", "token123")
    mod._resolve_twilio_number = lambda identifier=None: mod.OwnedTwilioNumber(
        sid="PN111",
        phone_number="+17025550123",
        friendly_name="Main",
        capabilities={"voice": True, "sms": True},
    )
    mod._json_request = lambda method, url, headers=None, params=None, form=None, json_body=None: {
        "id": "vapi-phone-xyz"
    }

    result = mod._vapi_import_twilio_number(
        save_env=True,
        state_path=state_path,
        env_path=env_path,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    env_text = env_path.read_text(encoding="utf-8")

    assert result["phone_number_id"] == "vapi-phone-xyz"
    assert state["vapi"]["phone_number_id"] == "vapi-phone-xyz"
    assert "VAPI_PHONE_NUMBER_ID=vapi-phone-xyz" in env_text


def test_diagnose_includes_decision_tree_and_saved_state(tmp_path: Path, monkeypatch):
    mod = load_module()
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    mod._save_state(
        {
            "version": 1,
            "twilio": {
                "default_phone_number": "+17025550123",
                "last_inbound_message_sid": "SM123",
            },
            "vapi": {
                "phone_number_id": "vapi-abc",
            },
        },
        hermes_home / "telephony_state.json",
    )
    (hermes_home / ".env").parent.mkdir(parents=True, exist_ok=True)
    (hermes_home / ".env").write_text(
        "TWILIO_ACCOUNT_SID=AC123\nTWILIO_AUTH_TOKEN=token\nBLAND_API_KEY=bland\n",
        encoding="utf-8",
    )

    result = mod.diagnose()

    assert result["providers"]["twilio"]["default_phone_number"] == "+17025550123"
    assert result["providers"]["twilio"]["last_inbound_message_sid"] == "SM123"
    assert result["providers"]["bland"]["configured"] is True
    assert result["providers"]["vapi"]["phone_number_id"] == "vapi-abc"
    assert any(item["use"] == "Twilio" for item in result["decision_tree"])
