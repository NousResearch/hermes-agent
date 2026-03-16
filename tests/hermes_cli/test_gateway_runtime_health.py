from hermes_cli.gateway import _runtime_health_lines
from gateway import status as gateway_status


def test_runtime_health_lines_include_fatal_platform_and_startup_reason(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "startup_failed",
            "exit_reason": "telegram conflict",
            "platforms": {
                "telegram": {
                    "state": "fatal",
                    "error_message": "another poller is active",
                }
            },
        },
    )

    lines = _runtime_health_lines()

    assert "⚠ telegram: another poller is active" in lines
    assert "⚠ Last startup issue: telegram conflict" in lines


def test_gateway_status_pid_path_uses_shared_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_status, "ensure_process_hermes_home_env", lambda: tmp_path)

    assert gateway_status._get_pid_path() == tmp_path / "gateway.pid"
