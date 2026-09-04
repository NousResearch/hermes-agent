from __future__ import annotations

import argparse
import json
from types import SimpleNamespace


def test_monitoring_status_parser_accepts_json_flag():
    from hermes_cli.subcommands.monitoring import build_monitoring_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_monitoring_parser(subparsers, cmd_monitoring=lambda args: None)

    args = parser.parse_args(["monitoring", "status", "--json"])

    assert args.monitoring_action == "status"
    assert args.json is True


def test_monitoring_status_payload_is_content_free(monkeypatch):
    from agent.monitoring.cron_health import CronHealthSnapshot
    from agent.monitoring.gateway_health import GatewayMetric
    from gateway import status as gateway_status
    from agent.monitoring import cron_health
    from hermes_cli.subcommands.monitoring import build_monitoring_status_payload

    runtime = {
        "gateway_state": "running",
        "active_agents": 2,
        "platforms": {
            "slack": {
                "state": "fatal",
                "error_message": "Bearer top-secret-token for alice@example.com",
            }
        },
    }
    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: runtime)
    monkeypatch.setattr(
        gateway_status,
        "resolve_gateway_liveness",
        lambda **kwargs: SimpleNamespace(running=True, probe_error=False),
    )
    monkeypatch.setattr(
        cron_health,
        "build_cron_health_snapshot",
        lambda: CronHealthSnapshot(
            metrics=[
                GatewayMetric(
                    "hermes.cron.jobs.enabled",
                    3,
                    {"job.name": "private payroll for alice@example.com"},
                )
            ],
            events=[],
        ),
    )

    payload = build_monitoring_status_payload({
        "monitoring": {
            "install_id": "private-install-id",
            "gateway_health_export": {"enabled": True},
            "export": {
                "otlp": {
                    "enabled": True,
                    "endpoint": "https://alice:secret@example.com/v1/metrics",
                }
            },
        }
    })

    assert payload["schema_version"] == 1
    assert payload["content_free"] is True
    assert payload["export"]["health_enabled"] is True
    assert payload["export"]["otlp_configured"] is True
    assert payload["health"]["gateway"]["status"] == "available"
    assert payload["health"]["cron"]["status"] == "available"
    assert {metric["name"] for metric in payload["health"]["gateway"]["metrics"]} >= {
        "hermes.gateway.up",
        "hermes.gateway.active_agents",
        "hermes.platform.up",
    }
    assert payload["health"]["cron"]["metrics"] == [
        {"name": "hermes.cron.jobs.enabled", "value": 3, "attributes": {}}
    ]

    rendered = json.dumps(payload)
    assert "private-install-id" not in rendered
    assert "alice@example.com" not in rendered
    assert "top-secret-token" not in rendered
    assert "alice:secret" not in rendered
    assert "events" not in payload["health"]["gateway"]
    assert "events" not in payload["health"]["cron"]
    assert "sha256:" in rendered
    assert {
        key
        for metric in payload["health"]["gateway"]["metrics"]
        for key in metric["attributes"]
    } <= {
        "service.instance.id",
        "service.version",
        "hermes.supervision_mode",
        "hermes.gateway.state",
        "hermes.platform",
        "hermes.platform.state",
        "hermes.error_code",
    }


def test_monitoring_status_payload_marks_failed_sections_unavailable(monkeypatch):
    from agent.monitoring import cron_health
    from gateway import status as gateway_status
    from hermes_cli.subcommands.monitoring import build_monitoring_status_payload

    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: {})
    monkeypatch.setattr(
        gateway_status,
        "resolve_gateway_liveness",
        lambda **kwargs: SimpleNamespace(running=False, probe_error=False),
    )

    def fail_cron_snapshot():
        raise RuntimeError("private path /Users/alice/jobs.json")

    monkeypatch.setattr(cron_health, "build_cron_health_snapshot", fail_cron_snapshot)

    payload = build_monitoring_status_payload({})

    assert payload["health"]["gateway"]["status"] == "available"
    assert payload["health"]["cron"] == {"status": "error", "metrics": []}
    assert "alice" not in json.dumps(payload)


def test_monitoring_status_payload_does_not_report_down_when_liveness_is_unknown(
    monkeypatch,
):
    from gateway import status as gateway_status
    from hermes_cli.subcommands.monitoring import build_monitoring_status_payload

    monkeypatch.setattr(gateway_status, "read_runtime_status", lambda: {})
    monkeypatch.setattr(
        gateway_status,
        "resolve_gateway_liveness",
        lambda **kwargs: SimpleNamespace(running=False, probe_error=True),
    )

    payload = build_monitoring_status_payload({})

    assert payload["health"]["gateway"] == {
        "status": "unavailable",
        "metrics": [],
    }


def test_monitoring_status_payload_marks_gateway_snapshot_errors(monkeypatch):
    from gateway import status as gateway_status
    from hermes_cli.subcommands.monitoring import build_monitoring_status_payload

    def fail_runtime_read():
        raise OSError("private path /Users/alice/gateway_state.json")

    monkeypatch.setattr(gateway_status, "read_runtime_status", fail_runtime_read)

    payload = build_monitoring_status_payload({})

    assert payload["health"]["gateway"] == {"status": "error", "metrics": []}
    assert "alice" not in json.dumps(payload)


def test_cmd_monitoring_prints_json_payload(monkeypatch, capsys):
    from hermes_cli import config as config_module
    from hermes_cli import main
    from hermes_cli.subcommands import monitoring

    expected = {
        "schema_version": 1,
        "content_free": True,
        "export": {},
        "health": {},
    }
    monkeypatch.setattr(config_module, "load_config", lambda: {})
    monkeypatch.setattr(
        monitoring, "build_monitoring_status_payload", lambda config: expected
    )

    main.cmd_monitoring(SimpleNamespace(monitoring_action="status", json=True))

    assert json.loads(capsys.readouterr().out) == expected
