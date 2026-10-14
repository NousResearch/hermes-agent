import json
from pathlib import Path
from unittest.mock import patch

from agent.provider_problem_sensors import record_provider_request_result


def test_provider_problem_sensor_turns_on_after_five_of_last_eight_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HASS_URL", raising=False)
    monkeypatch.delenv("HASS_TOKEN", raising=False)

    for i in range(4):
        record_provider_request_result(provider="anthropic", success=False, model=f"m{i}")
    for i in range(3):
        record_provider_request_result(provider="anthropic", success=True, model=f"s{i}")

    result = record_provider_request_result(provider="anthropic", success=False, model="m4")

    assert result["total"] == 8
    assert result["errors"] == 5
    assert result["problem"] is True

    state = json.loads((tmp_path / "state/provider_problem_sensors.json").read_text())
    pdata = state["providers"]["anthropic"]
    assert pdata["entity_id"] == "binary_sensor.hermes_provider_anthropic_problem"
    assert pdata["problem"] is True
    assert [item["result"] for item in pdata["results"]].count("error") == 5


def test_provider_problem_sensors_are_independent_per_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HASS_URL", raising=False)
    monkeypatch.delenv("HASS_TOKEN", raising=False)

    for _ in range(8):
        record_provider_request_result(provider="anthropic", success=False)
    for _ in range(8):
        record_provider_request_result(provider="openai-codex", success=True)

    state = json.loads((tmp_path / "state/provider_problem_sensors.json").read_text())
    assert state["providers"]["anthropic"]["problem"] is True
    assert state["providers"]["openai-codex"]["problem"] is False
    assert state["providers"]["openai-codex"]["entity_id"] == "binary_sensor.hermes_provider_openai_codex_problem"


def test_homeassistant_publish_uses_problem_device_class(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HASS_URL", "http://ha.local:8123")
    monkeypatch.setenv("HASS_TOKEN", "token")

    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    with patch("agent.provider_problem_sensors.urllib.request.urlopen", fake_urlopen):
        for _ in range(8):
            record_provider_request_result(provider="custom:gpt-5.5", success=False)

    assert captured["url"].endswith("/api/states/binary_sensor.hermes_provider_custom_gpt_5_5_problem")
    assert captured["payload"]["state"] == "on"
    assert captured["payload"]["attributes"]["device_class"] == "problem"
    assert captured["payload"]["attributes"]["provider"] == "custom:gpt-5.5"
    assert captured["payload"]["attributes"]["error_count"] == 8
