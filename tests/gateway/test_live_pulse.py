from __future__ import annotations

import importlib
import json
import time
from pathlib import Path


class DummyAgent:
    session_id = "session-visible-id"
    model = "test-model"
    provider = "test-provider"

    def get_activity_summary(self) -> dict[str, object]:
        return {
            "current_tool": "terminal",
            "last_activity_desc": "running validation",
            "api_call_count": 2,
            "max_iterations": 90,
            "seconds_since_activity": 1,
        }


def reload_live_pulse(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import gateway.live_pulse as live_pulse

    return importlib.reload(live_pulse)


def test_build_live_pulse_reports_non_secret_active_agent_metadata(monkeypatch, tmp_path):
    live_pulse = reload_live_pulse(monkeypatch, tmp_path)
    monkeypatch.setattr(live_pulse, "_pending_approval_counts", lambda: {})
    monkeypatch.setattr(live_pulse, "_process_rows", lambda: [])

    pulse = live_pulse.build_live_pulse(
        running_agents={"telegram:chat:thread": DummyAgent()},
        running_started={"telegram:chat:thread": time.time() - 5},
        session_platform_resolver=lambda key: key.split(":", 1)[0],
    )

    assert pulse["schema_version"] == "hermes-live-pulse-v1"
    assert pulse["summary"]["active_agent_turns"] == 1
    assert pulse["summary"]["starting_turns"] == 0
    assert pulse["summary"]["pending_approval_count"] == 0
    assert pulse["summary"]["interruption_signal"] == "caution_active_workflow"
    assert pulse["sessions"][0]["session_key_hash"]
    assert "telegram:chat:thread" not in json.dumps(pulse)
    assert pulse["sessions"][0]["platform"] == "telegram"
    assert pulse["sessions"][0]["current_tool"] == "terminal"
    assert pulse["sessions"][0]["current_phase"] == "running validation"


def test_pending_sentinel_counts_as_starting_without_session_body(monkeypatch, tmp_path):
    live_pulse = reload_live_pulse(monkeypatch, tmp_path)
    sentinel = object()
    monkeypatch.setattr(live_pulse, "_pending_approval_counts", lambda: {})
    monkeypatch.setattr(live_pulse, "_process_rows", lambda: [])

    pulse = live_pulse.build_live_pulse(
        running_agents={"telegram:chat:thread": sentinel},
        running_started={"telegram:chat:thread": time.time() - 2},
        pending_sentinel=sentinel,
    )

    assert pulse["summary"]["active_agent_turns"] == 0
    assert pulse["summary"]["starting_turns"] == 1
    assert pulse["sessions"][0]["state"] == "starting"
    assert pulse["sessions"][0]["session_id"] == ""
    assert pulse["sessions"][0]["model"] == ""
    assert pulse["sessions"][0]["provider"] == ""


def test_pending_approval_counts_expose_counts_only(monkeypatch, tmp_path):
    live_pulse = reload_live_pulse(monkeypatch, tmp_path)
    monkeypatch.setattr(live_pulse, "_pending_approval_counts", lambda: {"telegram:chat:thread": 3})
    monkeypatch.setattr(live_pulse, "_process_rows", lambda: [])

    pulse = live_pulse.build_live_pulse(
        running_agents={"telegram:chat:thread": DummyAgent()},
        running_started={"telegram:chat:thread": time.time() - 5},
    )

    assert pulse["summary"]["pending_approval_count"] == 3
    assert pulse["sessions"][0]["state"] == "waiting_human_approval"
    assert pulse["sessions"][0]["pending_approval_count"] == 3
    serialized = json.dumps(pulse)
    assert "pattern_keys" not in serialized
    assert "rm -rf" not in serialized
    assert "No approval payload included" in serialized


def test_background_helpers_do_not_force_caution(monkeypatch, tmp_path):
    live_pulse = reload_live_pulse(monkeypatch, tmp_path)
    monkeypatch.setattr(live_pulse, "_pending_approval_counts", lambda: {})
    monkeypatch.setattr(
        live_pulse,
        "_process_rows",
        lambda: [
            {
                "process_id": "helper",
                "state": "background_running",
                "classification": "default_long_lived_helper",
                "interrupt_sensitive": False,
            }
        ],
    )

    pulse = live_pulse.build_live_pulse()

    assert pulse["summary"]["running_background_process_count"] == 0
    assert pulse["summary"]["background_helper_process_count"] == 1
    assert pulse["summary"]["background_process_total_count"] == 1
    assert pulse["summary"]["interruption_signal"] == "safe_to_message"


def test_interrupt_sensitive_background_work_forces_caution(monkeypatch, tmp_path):
    live_pulse = reload_live_pulse(monkeypatch, tmp_path)
    monkeypatch.setattr(live_pulse, "_pending_approval_counts", lambda: {})
    monkeypatch.setattr(
        live_pulse,
        "_process_rows",
        lambda: [
            {
                "process_id": "task-linked",
                "state": "background_running",
                "classification": "active_background_work",
                "interrupt_sensitive": True,
            }
        ],
    )

    pulse = live_pulse.build_live_pulse()

    assert pulse["summary"]["running_background_process_count"] == 1
    assert pulse["summary"]["background_helper_process_count"] == 0
    assert pulse["summary"]["interruption_signal"] == "caution_active_workflow"


def test_write_live_pulse_uses_hermes_home_profile_safe_path(monkeypatch, tmp_path):
    live_pulse = reload_live_pulse(monkeypatch, tmp_path)
    monkeypatch.setattr(live_pulse, "_pending_approval_counts", lambda: {})
    monkeypatch.setattr(live_pulse, "_process_rows", lambda: [])

    pulse = live_pulse.write_live_pulse()
    expected = tmp_path / "tools" / "agent_floor" / "hermes_live_pulse.json"

    assert expected.exists()
    written = json.loads(expected.read_text())
    assert written["schema_version"] == "hermes-live-pulse-v1"
    assert written["generated_at"] == pulse["generated_at"]
    source_limits = "\n".join(written["source_limits"])
    assert "No chat content included" in source_limits
    assert "No raw user prompt included" in source_limits
    assert "No raw tool output included" in source_limits
    assert "No sensitive command text included" in source_limits
    assert "No approval payload included" in source_limits
    assert "tokens" in source_limits
    assert "credentials" in source_limits
    assert "session bodies" in source_limits
