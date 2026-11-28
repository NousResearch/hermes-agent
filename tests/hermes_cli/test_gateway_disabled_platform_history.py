"""Old disabled adapters must not look like failures of the current gateway."""

import copy
import pytest

from hermes_cli.gateway import _runtime_health_lines


@pytest.fixture
def history(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("platforms:\n  dingtalk:\n    enabled: false\n")
    monkeypatch.setattr("hermes_cli.gateway.get_hermes_home", lambda: tmp_path)
    record = {
        "gateway_state": "running", "pid": 178, "start_time": 200,
        "platforms": {"dingtalk": {
            "state": "fatal", "error_message": "old adapter unavailable",
            "writer_pid": 791, "writer_start_time": 100,
        }},
    }
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda: record)
    monkeypatch.setattr("gateway.status.runtime_status_is_stale", lambda _: False)
    return config, record


@pytest.mark.parametrize("writer_pid", [791, 178])
def test_disabled_previous_gateway_error_is_not_current_health(history, writer_pid):
    _, record = history
    record["platforms"]["dingtalk"]["writer_pid"] = writer_pid
    before = copy.deepcopy(record)

    assert not any("old adapter unavailable" in row for row in _runtime_health_lines())
    assert record == before


@pytest.mark.parametrize("case", ["enabled", "missing", "current", "unknown-writer", "unknown-owner", "secondary", "broken-config"])
def test_potentially_current_or_unknown_errors_stay_visible(history, case):
    config, record = history
    platform = record["platforms"]["dingtalk"]
    if case == "enabled":
        config.write_text("platforms:\n  dingtalk:\n    enabled: true\n")
    elif case == "missing":
        config.write_text("platforms: {}\n")
    elif case == "current":
        platform.update(writer_pid=178, writer_start_time=200)
    elif case == "unknown-writer":
        platform.pop("writer_pid")
    elif case == "unknown-owner":
        record.pop("start_time")
    elif case == "secondary":
        record["platforms"] = {"other:dingtalk": platform}
    else:
        config.write_text("platforms: [broken\n")

    assert any("old adapter unavailable" in row for row in _runtime_health_lines())
