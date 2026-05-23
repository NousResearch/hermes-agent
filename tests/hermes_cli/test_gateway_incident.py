import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway_incident as incident


def _fake_validation(status: str = "pass") -> dict:
    return {
        "schema_version": 1,
        "owner": "hermes-reliability-plane",
        "risk_tier": "R0",
        "read_only": True,
        "redacted": True,
        "overall_status": status,
        "summary": {"checks": 3, "errors": 0 if status == "pass" else 1, "warnings": 1},
        "launchd": {},
        "health": {},
        "checks": [],
    }


def test_incident_bundle_writes_private_redacted_files(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    logs_root = hermes_home / "logs"
    logs_root.mkdir(parents=True)
    secret_canary = "THIS_CANARY_MUST_NOT_ENTER_THE_BUNDLE"
    (logs_root / "gateway.log").write_text(secret_canary, encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_HEALTH_LOOP_ROOT", str(tmp_path / "health-loop"))
    monkeypatch.setattr(incident, "build_gateway_validation_report", lambda **kwargs: _fake_validation())

    result = incident.build_gateway_incident_bundle(
        output_dir=tmp_path / "bundle",
        check_health=False,
    )

    bundle_dir = Path(result["output_dir"])
    assert stat.S_IMODE(bundle_dir.stat().st_mode) == 0o700
    assert result["bundle_created"] is True
    assert result["runtime_mutation"] is False
    assert result["private_memory_read"] is False
    assert result["raw_log_content_copied"] is False

    for filename in (
        "manifest.json",
        "gateway_validation.json",
        "artifact_metadata.json",
        "summary.md",
    ):
        path = bundle_dir / filename
        assert path.exists()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert secret_canary not in path.read_text(encoding="utf-8")

    metadata = json.loads((bundle_dir / "artifact_metadata.json").read_text())
    gateway_log = {
        item["label"]: item for item in metadata["artifacts"]
    }["hermes_home.logs.gateway.log"]
    assert gateway_log["exists"] is True
    assert gateway_log["size_bytes"] == len(secret_canary)
    assert gateway_log["content_copied"] is False


def test_incident_bundle_refuses_non_empty_output_without_force(tmp_path, monkeypatch):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(incident, "build_gateway_validation_report", lambda **kwargs: _fake_validation())

    with pytest.raises(incident.IncidentBundleError, match="not empty"):
        incident.build_gateway_incident_bundle(output_dir=output_dir)


def test_incident_bundle_force_overwrites_known_files_only(tmp_path, monkeypatch):
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    unrelated = output_dir / "operator-note.txt"
    unrelated.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(incident, "build_gateway_validation_report", lambda **kwargs: _fake_validation())

    result = incident.build_gateway_incident_bundle(output_dir=output_dir, force=True)

    assert result["bundle_created"] is True
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert (output_dir / "manifest.json").exists()


def test_incident_bundle_cli_outputs_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(incident, "build_gateway_validation_report", lambda **kwargs: _fake_validation())

    ok = incident.run_gateway_incident_bundle(
        SimpleNamespace(
            output=str(tmp_path / "bundle"),
            force=False,
            json=True,
            no_health=True,
            no_log_metadata=True,
            no_health_loop_metadata=True,
            launchctl_timeout=5,
            health_timeout=2,
        )
    )

    assert ok is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["bundle_created"] is True
    assert payload["redacted"] is True
    assert payload["runtime_mutation"] is False
    assert payload["validation_status"] == "pass"


def test_incident_bundle_invalid_timeout_returns_clean_error(tmp_path, capsys):
    ok = incident.run_gateway_incident_bundle(
        SimpleNamespace(
            output=str(tmp_path / "bundle"),
            force=False,
            json=True,
            no_health=True,
            no_log_metadata=True,
            no_health_loop_metadata=True,
            launchctl_timeout=0,
            health_timeout=2,
        )
    )

    captured = capsys.readouterr()
    assert ok is False
    assert "--launchctl-timeout must be a positive number of seconds" in captured.err
    assert "Traceback" not in captured.err


def test_incident_bundle_command_succeeds_when_validation_captures_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        incident,
        "build_gateway_validation_report",
        lambda **kwargs: _fake_validation("fail"),
    )

    result = incident.build_gateway_incident_bundle(
        output_dir=tmp_path / "bundle",
        check_health=False,
    )

    assert result["bundle_created"] is True
    assert result["validation_status"] == "fail"
