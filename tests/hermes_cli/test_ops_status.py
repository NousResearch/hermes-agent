import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.ops_status as ops_status


def _secret() -> str:
    return "sk-" + ("x" * 30)


def _bearer() -> str:
    return "Bearer " + ("y" * 30)


def _fake_gateway_report(secret_path: Path | None = None) -> dict:
    wrapper = secret_path or Path("/tmp/Operator/scripts/hermes-gateway.sh")
    return {
        "schema_version": 1,
        "owner": "hermes-reliability-plane",
        "risk_tier": "R0",
        "read_only": True,
        "redacted": True,
        "overall_status": "pass",
        "summary": {"checks": 3, "errors": 0, "warnings": 0},
        "launchd": {
            "platform": "darwin",
            "active_label": "ai.hermes.gateway",
            "canonical_label": "ai.hermes.gateway",
            "legacy_label": "com.agent1.hermes.gateway",
            "plist_path": "/tmp/ai.hermes.gateway.plist",
            "expected_wrapper": str(wrapper),
            "active_label_state": {"loaded": True, "running": True},
            "legacy_label_state": {"loaded": False, "running": False},
            "program_summary": {
                "command_kind": "operator_wrapper",
                "uses_expected_wrapper": True,
                "first_argument": str(wrapper),
            },
        },
        "health": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8642,
            "auth_configured": True,
            "health_status": 200,
            "detailed_status": 401,
        },
        "checks": [],
    }


@pytest.fixture()
def isolated_ops_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    operator_root = tmp_path / "Operator"
    health_loop = operator_root / "health-loop"
    health_loop.mkdir(parents=True)
    logs_dir = hermes_home / "logs"
    logs_dir.mkdir()
    cron_dir = hermes_home / "cron"
    cron_dir.mkdir()

    secret = _secret()
    bearer = _bearer()
    (logs_dir / "gateway.log").write_text(
        f"WARNING auth header {bearer}\nERROR provider token {secret}\n",
        encoding="utf-8",
    )
    (logs_dir / "gateway.error.log").write_text(
        f"CRITICAL {secret}\n",
        encoding="utf-8",
    )
    (health_loop / "status.md").write_text(
        f"# Health\nprivate token {secret}\n",
        encoding="utf-8",
    )
    (health_loop / "status.json").write_text(
        json.dumps({"status": "ok", "token": secret}),
        encoding="utf-8",
    )
    (cron_dir / "jobs.json").write_text(
        json.dumps(
            [
                {
                    "id": "job-1",
                    "enabled": True,
                    "prompt": f"private prompt {secret}",
                    "last_status": "ok",
                },
                {
                    "id": "job-2",
                    "enabled": False,
                    "prompt": f"private bearer {bearer}",
                    "last_status": "error",
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_OPERATOR_ROOT", str(operator_root))
    monkeypatch.setattr(
        ops_status,
        "build_gateway_validation_report",
        lambda **kwargs: _fake_gateway_report(),
    )
    return {
        "home": hermes_home,
        "operator_root": operator_root,
        "secret": secret,
        "bearer": bearer,
    }


def test_ops_status_json_is_redacted_and_metadata_only(isolated_ops_home, capsys):
    ok = ops_status.run_ops_status(
        SimpleNamespace(json=True, no_health=True, launchctl_timeout=1, health_timeout=1)
    )

    assert ok is True
    output = capsys.readouterr().out
    assert isolated_ops_home["secret"] not in output
    assert isolated_ops_home["bearer"] not in output
    assert "private prompt" not in output
    assert "private token" not in output
    payload = json.loads(output)
    assert payload["read_only"] is True
    assert payload["redacted"] is True
    assert payload["risk_tier"] == "R0"
    assert payload["cron"]["content_included"] is False
    assert payload["cron"]["total_jobs"] == 2
    assert payload["cron"]["enabled_jobs"] == 1
    assert payload["cron"]["paused_jobs"] == 1
    assert payload["logs"]["content_included"] is False
    assert payload["logs"]["files"]["gateway"]["recent_warning_count"] >= 1
    assert payload["logs"]["files"]["gateway"]["recent_error_count"] >= 1
    assert payload["health_guardian"]["content_included"] is False
    assert payload["gateway"]["launchd"]["program_summary"]["uses_expected_wrapper"] is True


def test_ops_status_text_is_concise_and_redacted(isolated_ops_home, capsys):
    ok = ops_status.run_ops_status(
        SimpleNamespace(json=False, no_health=True, launchctl_timeout=1, health_timeout=1)
    )

    assert ok is True
    output = capsys.readouterr().out
    assert "Hermes ops status: PASS" in output
    assert "Gateway" in output
    assert "Cron" in output
    assert "Logs" in output
    assert "Next actions" in output
    assert isolated_ops_home["secret"] not in output
    assert isolated_ops_home["bearer"] not in output
    assert "private prompt" not in output
    assert "private token" not in output


def test_ops_status_markdown_is_redacted_handoff_receipt(isolated_ops_home, capsys):
    ok = ops_status.run_ops_status(
        SimpleNamespace(
            json=False,
            markdown=True,
            no_health=True,
            launchctl_timeout=1,
            health_timeout=1,
        )
    )

    assert ok is True
    output = capsys.readouterr().out
    assert output.startswith("# Hermes Ops Status\n")
    assert "- Status: `PASS`" in output
    assert "## Gateway" in output
    assert "## Logs" in output
    assert "| Log | Exists | Size | Warnings | Errors |" in output
    assert "Raw log lines are not included." in output
    assert "## Next Actions" in output
    assert isolated_ops_home["secret"] not in output
    assert isolated_ops_home["bearer"] not in output
    assert "private prompt" not in output
    assert "private token" not in output


def test_ops_status_cron_status_buckets_do_not_leak_arbitrary_values(
    isolated_ops_home, capsys
):
    secret = isolated_ops_home["secret"]
    jobs_path = isolated_ops_home["home"] / "cron" / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            [
                {
                    "id": "job-1",
                    "enabled": True,
                    "prompt": "redacted",
                    "last_status": f"private {secret}",
                }
            ]
        ),
        encoding="utf-8",
    )

    ok = ops_status.run_ops_status(
        SimpleNamespace(json=True, no_health=True, launchctl_timeout=1, health_timeout=1)
    )

    assert ok is True
    output = capsys.readouterr().out
    assert secret not in output
    assert "private " not in output
    payload = json.loads(output)
    assert payload["cron"]["last_status_counts"] == {"other": 1}


def test_ops_status_redacts_secret_like_gateway_paths(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    (hermes_home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    secret_path = tmp_path / ("sk-" + ("z" * 30)) / "hermes-gateway.sh"
    monkeypatch.setattr(
        ops_status,
        "build_gateway_validation_report",
        lambda **kwargs: _fake_gateway_report(secret_path),
    )

    ok = ops_status.run_ops_status(
        SimpleNamespace(json=True, no_health=True, launchctl_timeout=1, health_timeout=1)
    )

    assert ok is True
    output = capsys.readouterr().out
    assert str(secret_path) not in output
    assert "sk-" + ("z" * 30) not in output
    assert "hermes-gateway.sh" in output


def test_ops_status_rejects_invalid_timeouts(capsys):
    ok = ops_status.run_ops_status(
        SimpleNamespace(json=True, no_health=True, launchctl_timeout=0, health_timeout=1)
    )

    captured = capsys.readouterr()
    assert ok is False
    assert "--launchctl-timeout must be a positive number of seconds" in captured.err
    assert "Traceback" not in captured.err


def test_ops_status_cli_parser_defaults_to_status(monkeypatch):
    import hermes_cli.main as main

    captured = {}

    def fake_ops_command(args):
        captured["command"] = args.command
        captured["ops_command"] = getattr(args, "ops_command", None)

    monkeypatch.setattr(main, "cmd_ops", fake_ops_command)
    monkeypatch.setattr(main.sys, "argv", ["hermes", "ops"])

    main.main()

    assert captured == {"command": "ops", "ops_command": None}
