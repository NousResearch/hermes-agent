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


def test_save_speko_writes_env_and_state(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    result = mod.save_speko(
        "sk-speko-test",
        phone_number="+1 (415) 555-0123",
        agent_id="agent_abc123",
        language="es",
    )

    env_text = (tmp_path / ".hermes" / ".env").read_text(encoding="utf-8")
    state = json.loads((tmp_path / ".hermes" / "telephony_state.json").read_text(encoding="utf-8"))

    assert result["success"] is True
    assert "SPEKO_API_KEY=sk-speko-test" in env_text
    assert "SPEKO_PHONE_NUMBER=+14155550123" in env_text
    assert "SPEKO_AGENT_ID=agent_abc123" in env_text
    assert "SPEKO_LANGUAGE=es" in env_text
    assert "PHONE_PROVIDER=speko" in env_text
    assert state["speko"]["phone_number"] == "+14155550123"
    assert state["speko"]["agent_id"] == "agent_abc123"


def _capture_speko(mod, response):
    calls = []

    def fake_request(method, path, *, json_body=None):
        calls.append({"method": method, "path": path, "json_body": json_body})
        return response(path) if callable(response) else response

    mod._speko_request = fake_request
    return calls


def test_speko_call_is_agentless_when_no_agent_saved(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("SPEKO_PHONE_NUMBER", "+14155550123")
    calls = _capture_speko(mod, {"sessionId": "sess_1", "status": "dialing"})

    result = mod._speko_call(
        "+1 555 123 0000",
        "Confirm the Tuesday appointment.",
        first_sentence="Hi, this is Ava.",
        language="es",
    )

    body = calls[0]["json_body"]
    assert calls[0]["path"] == "/sessions/phone"
    assert body["to"] == "+15551230000"
    assert body["from"] == "+14155550123"
    assert body["systemPrompt"] == "Confirm the Tuesday appointment."
    assert body["firstMessage"] == "Hi, this is Ava."
    assert body["intent"] == {"language": "es"}
    assert "agentId" not in body
    assert result["call_id"] == "sess_1"
    assert result["provider"] == "speko"
    assert "5551230000" not in json.dumps(result)


def test_speko_call_uses_saved_agent_instead_of_intent(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("SPEKO_PHONE_NUMBER", "+14155550123")
    monkeypatch.setenv("SPEKO_AGENT_ID", "agent_abc123")
    calls = _capture_speko(mod, {"sessionId": "sess_2", "status": "dialing"})

    mod._speko_call("+15551230000", "Ask about Wednesday instead.")

    body = calls[0]["json_body"]
    assert body["agentId"] == "agent_abc123"
    assert "intent" not in body


def test_speko_call_resolves_the_only_outbound_number(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    calls = _capture_speko(
        mod,
        lambda path: (
            [{"e164": "+14155550123", "direction": "both"}]
            if path == "/phone-numbers"
            else {"sessionId": "sess_3", "status": "dialing"}
        ),
    )

    mod._speko_call("+15551230000", "Say hello.")

    assert calls[0]["path"] == "/phone-numbers"
    assert calls[1]["json_body"]["from"] == "+14155550123"


def test_speko_call_refuses_to_guess_between_numbers(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _capture_speko(
        mod,
        [
            {"e164": "+14155550123", "direction": "both"},
            {"e164": "+14155550124", "direction": "outbound"},
        ],
    )

    try:
        mod._speko_call("+15551230000", "Say hello.")
    except mod.TelephonyError as exc:
        assert "several outbound numbers" in str(exc)
    else:  # pragma: no cover - the call must not proceed
        raise AssertionError("expected TelephonyError when several numbers are owned")


def test_speko_status_flattens_report_transcript(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("SPEKO_API_KEY", "sk-speko-test")
    _capture_speko(
        mod,
        lambda path: (
            {
                "session_id": "sess_4",
                "summary": "Appointment confirmed for Tuesday.",
                "outcome": "completed",
                "transcript": {
                    "entries": [
                        {"source": "agent", "text": "Hi, calling to confirm Tuesday."},
                        {"source": "user", "text": "Tuesday works."},
                        {"source": "agent", "text": "   "},
                    ]
                },
            }
            if path.endswith("/report")
            else {"status": "ended", "duration_seconds": 42, "language": "en"}
        ),
    )

    result = mod._speko_status("sess_4")

    assert result["status"] == "ended"
    assert result["duration_seconds"] == 42
    assert result["outcome"] == "completed"
    assert result["transcript"] == (
        "agent: Hi, calling to confirm Tuesday.\nuser: Tuesday works."
    )


def test_speko_status_explains_a_call_that_cannot_be_read_yet(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    def fake_request(method, path, *, json_body=None):
        raise mod.TelephonyError("HTTP 500 from /calls/sess_5")

    mod._speko_request = fake_request

    try:
        mod._speko_status("sess_5")
    except mod.TelephonyError as exc:
        assert "retry in a few seconds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected TelephonyError for an unreadable call")


def test_diagnose_reports_speko_readiness(tmp_path: Path, monkeypatch):
    mod = load_module()
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / ".env").write_text(
        "SPEKO_API_KEY=sk-speko-test\nSPEKO_PHONE_NUMBER=+14155550123\nPHONE_PROVIDER=speko\n",
        encoding="utf-8",
    )

    result = mod.diagnose()

    assert result["ai_call_provider"] == "speko"
    assert result["providers"]["speko"]["configured"] is True
    assert result["providers"]["speko"]["phone_number"] == "+14155550123"
    assert result["providers"]["speko"]["language"] == "en"
    assert any(item["use"] == "Speko" for item in result["decision_tree"])
