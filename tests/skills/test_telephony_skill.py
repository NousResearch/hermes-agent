from __future__ import annotations

import importlib.util
import json
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






def test_diagnose_includes_decision_tree_and_saved_state(tmp_path: Path, monkeypatch):
    mod = load_module()
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    mod._save_state(
        {
            "version": 1,
            "twilio": {
                "default_phone_number": "+155512300123",
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

    assert result["providers"]["twilio"]["default_phone_number"] == "+155512300123"
    assert result["providers"]["twilio"]["last_inbound_message_sid"] == "SM123"
    assert result["providers"]["bland"]["configured"] is True
    assert result["providers"]["vapi"]["phone_number_id"] == "vapi-abc"
    assert any(item["use"] == "Twilio" for item in result["decision_tree"])


def _owned_number(mod):
    return mod.OwnedTwilioNumber(
        sid="PN1",
        phone_number="+15550000000",
        friendly_name="Test",
        capabilities={"sms": True},
    )


def test_twilio_inbox_mint_code_returns_handle_not_body(tmp_path: Path, monkeypatch):
    import json as _json

    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "_resolve_twilio_number", lambda ident=None: _owned_number(mod))
    monkeypatch.setattr(
        mod,
        "_twilio_request",
        lambda method, path, params=None: {
            "messages": [
                {
                    "sid": "SM1",
                    "direction": "inbound",
                    "status": "received",
                    "from": "+15559876543",
                    "to": "+15550000000",
                    "date_sent": "2026-09-23T00:00:00Z",
                    "body": "Your verification code is 123456",
                    "num_media": 0,
                }
            ]
        },
    )

    result = mod._twilio_inbox(limit=5, mint_code=True)

    assert result["success"] is True
    assert result["code_handle"].startswith("otp_")
    dump = _json.dumps(result)
    assert "123456" not in dump
    assert "verification code" not in dump
    assert "body" not in result
    assert result["message_sid"] == "SM1"

    from agent import code_registry

    assert code_registry.consume(result["code_handle"]) == "123456"


def test_twilio_inbox_mint_code_no_otp_returns_error_without_bodies(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "_resolve_twilio_number", lambda ident=None: _owned_number(mod))
    monkeypatch.setattr(
        mod,
        "_twilio_request",
        lambda method, path, params=None: {
            "messages": [
                {
                    "sid": "SM2",
                    "direction": "inbound",
                    "status": "received",
                    "from": "+15559876543",
                    "to": "+15550000000",
                    "date_sent": "2026-09-23T00:00:00Z",
                    "body": "Lunch at noon?",
                    "num_media": 0,
                }
            ]
        },
    )

    result = mod._twilio_inbox(limit=5, mint_code=True)

    assert result["success"] is False
    assert result["error"] == "no_verification_code_found"
    assert "messages" not in result
    assert "Lunch" not in str(result)
