import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "memorymunch_operational_status.py"


def load_status_script():
    spec = importlib.util.spec_from_file_location("memorymunch_operational_status_under_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_capture_without_attempted_counts_as_recovered_firing():
    status = load_status_script()
    rows = [
        {"event": "turn_completed"},
        {"event": "live_capture_completed", "exchange_id": "conv::ok"},
    ]

    result = status.latest_capture_ok(rows)

    assert result["ok"] is True
    assert result["state"] == "completed_without_attempted"
    assert result["telemetry_warning"] == "missing_live_capture_attempted"


def test_later_failed_capture_overrides_older_completed_capture():
    status = load_status_script()
    rows = [
        {"event": "live_capture_completed", "exchange_id": "conv::old"},
        {"event": "live_capture_failed", "error": "boom"},
    ]

    result = status.latest_capture_ok(rows)

    assert result["ok"] is False
    assert result["state"] == "failed"


def test_runtime_plugin_drift_is_a_production_blocker(tmp_path):
    status = load_status_script()
    runtime = tmp_path / "runtime.py"
    vendored = tmp_path / "vendored.py"
    runtime.write_text("old sanitizer\n", encoding="utf-8")
    vendored.write_text("new sanitizer\n", encoding="utf-8")

    result = status.plugin_parity(runtime, vendored)

    assert result["ok"] is False
    assert result["runtime_sha256"] != result["vendored_sha256"]
    assert result["gap"] == "runtime_plugin_drift"


def test_live_briefing_contradictions_are_production_blockers():
    status = load_status_script()
    briefing = """
    <memorymunch-briefing>MemoryMunch audit query</memorymunch-briefing>
    <memorymunch-briefing>Kipbo Mortgage email and fulfilled eschatology theology atom</memorymunch-briefing>
    """

    result = status.detect_live_briefing_contradictions(briefing)

    assert result["ok"] is False
    assert "duplicate_memorymunch_briefing" in result["gaps"]
    assert "unrelated_activation_atom_in_technical_query" in result["gaps"]


def test_live_briefing_gate_fails_without_completed_turn():
    status = load_status_script()

    result = status.latest_turn_briefing_state([{"event": "turn_started"}])

    assert result["ok"] is False
    assert "latest_turn_missing" in result["gaps"]


def test_live_briefing_gate_scans_nested_latest_turn_rows():
    status = load_status_script()
    rows = [
        {"event": "turn_started"},
        {
            "event": "turn_completed",
            "nested": {
                "briefing": "<memorymunch-briefing>Hermes audit</memorymunch-briefing><memorymunch-briefing>theology</memorymunch-briefing>"
            },
        },
    ]

    result = status.latest_turn_briefing_state(rows)

    assert result["ok"] is False
    assert "duplicate_memorymunch_briefing" in result["gaps"]
    assert "unrelated_activation_atom_in_technical_query" in result["gaps"]
