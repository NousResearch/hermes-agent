import json
import plistlib
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli
import hermes_cli.gateway_validation as validation


def _write_plist(path: Path, payload: dict) -> None:
    with path.open("wb") as fh:
        plistlib.dump(payload, fh)


def _disable_api_probe(monkeypatch):
    monkeypatch.setattr(validation, "read_raw_config", lambda: {})
    monkeypatch.setattr(validation, "get_env_value", lambda name: "")


def _fake_launchctl(
    active_loaded=True,
    legacy_loaded=False,
    *,
    active_pid="123",
    active_status="0",
):
    def run(cmd, capture_output=True, text=True, timeout=5.0):
        label = cmd[-1]
        if label == validation.CANONICAL_LAUNCHD_LABEL and active_loaded:
            return SimpleNamespace(
                returncode=0,
                stdout=f"{active_pid}\t{active_status}\t{validation.CANONICAL_LAUNCHD_LABEL}\n",
                stderr="",
            )
        if label == validation.LEGACY_LAUNCHD_LABEL and legacy_loaded:
            return SimpleNamespace(
                returncode=0,
                stdout=f"-\t0\t{validation.LEGACY_LAUNCHD_LABEL}\n",
                stderr="",
            )
        return SimpleNamespace(returncode=113, stdout="", stderr="Could not find service")

    return run


def test_launchd_validation_passes_with_wrapper_and_warns_on_legacy_label(
    tmp_path, monkeypatch
):
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    wrapper = tmp_path / "Operator" / "scripts" / "hermes-gateway.sh"
    _write_plist(
        plist_path,
        {
            "Label": validation.CANONICAL_LAUNCHD_LABEL,
            "ProgramArguments": [str(wrapper)],
            "StandardOutPath": "/tmp/gateway.log",
            "StandardErrorPath": "/tmp/gateway.error.log",
        },
    )

    _disable_api_probe(monkeypatch)
    monkeypatch.setattr(validation.sys, "platform", "darwin")
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: validation.CANONICAL_LAUNCHD_LABEL)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(validation.subprocess, "run", _fake_launchctl(legacy_loaded=True))

    report = validation.build_gateway_validation_report(expected_wrapper=wrapper)

    assert report["overall_status"] == "pass"
    checks = {item["id"]: item for item in report["checks"]}
    assert checks["launchd.wrapper"]["status"] == "pass"
    assert checks["launchd.legacy_label"]["status"] == "warn"
    assert report["launchd"]["program_summary"]["uses_expected_wrapper"] is True


def test_launchd_validation_fails_when_plist_bypasses_operator_wrapper(
    tmp_path, monkeypatch
):
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    wrapper = tmp_path / "Operator" / "scripts" / "hermes-gateway.sh"
    _write_plist(
        plist_path,
        {
            "Label": validation.CANONICAL_LAUNCHD_LABEL,
            "ProgramArguments": [
                "/tmp/venv/bin/python",
                "-m",
                "hermes_cli.main",
                "gateway",
                "run",
            ],
        },
    )

    _disable_api_probe(monkeypatch)
    monkeypatch.setattr(validation.sys, "platform", "darwin")
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: validation.CANONICAL_LAUNCHD_LABEL)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(validation.subprocess, "run", _fake_launchctl())

    report = validation.build_gateway_validation_report(expected_wrapper=wrapper)

    assert report["overall_status"] == "fail"
    checks = {item["id"]: item for item in report["checks"]}
    assert checks["launchd.wrapper"]["status"] == "fail"
    assert checks["launchd.wrapper"]["severity"] == "error"
    assert report["launchd"]["program_summary"]["command_kind"] == "python_module"


def test_launchd_validation_requires_wrapper_as_entrypoint(tmp_path, monkeypatch):
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    wrapper = tmp_path / "Operator" / "scripts" / "hermes-gateway.sh"
    _write_plist(
        plist_path,
        {
            "Label": validation.CANONICAL_LAUNCHD_LABEL,
            "ProgramArguments": ["/tmp/venv/bin/python", str(wrapper)],
        },
    )

    _disable_api_probe(monkeypatch)
    monkeypatch.setattr(validation.sys, "platform", "darwin")
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: validation.CANONICAL_LAUNCHD_LABEL)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(validation.subprocess, "run", _fake_launchctl())

    report = validation.build_gateway_validation_report(expected_wrapper=wrapper)

    checks = {item["id"]: item for item in report["checks"]}
    assert report["overall_status"] == "fail"
    assert checks["launchd.wrapper"]["status"] == "fail"


def test_launchd_validation_fails_when_active_label_loaded_but_not_running(
    tmp_path, monkeypatch
):
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    wrapper = tmp_path / "Operator" / "scripts" / "hermes-gateway.sh"
    _write_plist(
        plist_path,
        {
            "Label": validation.CANONICAL_LAUNCHD_LABEL,
            "ProgramArguments": [str(wrapper)],
        },
    )

    _disable_api_probe(monkeypatch)
    monkeypatch.setattr(validation.sys, "platform", "darwin")
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: validation.CANONICAL_LAUNCHD_LABEL)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(
        validation.subprocess,
        "run",
        _fake_launchctl(active_pid="-", active_status="256"),
    )

    report = validation.build_gateway_validation_report(
        expected_wrapper=wrapper,
        check_health=False,
    )

    checks = {item["id"]: item for item in report["checks"]}
    assert report["overall_status"] == "fail"
    assert checks["launchd.loaded"]["status"] == "fail"
    assert "not running" in checks["launchd.loaded"]["message"]


def test_launchd_validation_warns_when_running_with_nonzero_previous_exit(
    tmp_path, monkeypatch
):
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    wrapper = tmp_path / "Operator" / "scripts" / "hermes-gateway.sh"
    _write_plist(
        plist_path,
        {
            "Label": validation.CANONICAL_LAUNCHD_LABEL,
            "ProgramArguments": [str(wrapper)],
        },
    )

    _disable_api_probe(monkeypatch)
    monkeypatch.setattr(validation.sys, "platform", "darwin")
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: validation.CANONICAL_LAUNCHD_LABEL)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(
        validation.subprocess,
        "run",
        _fake_launchctl(active_pid="123", active_status="256"),
    )

    report = validation.build_gateway_validation_report(
        expected_wrapper=wrapper,
        check_health=False,
    )

    checks = {item["id"]: item for item in report["checks"]}
    assert report["overall_status"] == "pass"
    assert checks["launchd.loaded"]["status"] == "warn"
    assert "previous exit status was non-zero" in checks["launchd.loaded"]["message"]


class _FakeResponse:
    def __init__(self, status: int):
        self._status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getcode(self) -> int:
        return self._status


def test_health_detailed_unauthenticated_401_is_expected(monkeypatch):
    monkeypatch.setattr(validation.sys, "platform", "linux")
    monkeypatch.setattr(
        validation,
        "read_raw_config",
        lambda: {"platforms": {"api_server": {"enabled": True, "extra": {"port": 8642}}}},
    )
    monkeypatch.setattr(validation, "get_env_value", lambda name: "")

    def fake_urlopen(request, timeout=2.0):
        url = request.full_url
        if url.endswith("/health"):
            return _FakeResponse(200)
        if url.endswith("/health/detailed"):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", hdrs=None, fp=None)
        raise AssertionError(url)

    monkeypatch.setattr(validation.urllib.request, "urlopen", fake_urlopen)

    report = validation.build_gateway_validation_report()

    checks = {item["id"]: item for item in report["checks"]}
    assert checks["api_server.health"]["status"] == "pass"
    assert checks["api_server.health_detailed"]["status"] == "pass"
    assert report["overall_status"] == "pass"
    assert report["health"]["detailed_status"] == 401


def test_json_output_is_parseable_and_redacted(monkeypatch, capsys):
    monkeypatch.setattr(validation.sys, "platform", "linux")
    _disable_api_probe(monkeypatch)

    ok = validation.run_gateway_validation(
        SimpleNamespace(json=True, markdown=False, no_health=True)
    )

    assert ok is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["read_only"] is True
    assert payload["redacted"] is True
    assert "checks" in payload


def test_invalid_timeout_returns_clean_error(capsys):
    cases = [
        ("--health-timeout", {"launchctl_timeout": 5, "health_timeout": -1}),
        ("--health-timeout", {"launchctl_timeout": 5, "health_timeout": 0}),
        ("--launchctl-timeout", {"launchctl_timeout": 0, "health_timeout": 2}),
    ]

    for option, timeout_kwargs in cases:
        ok = validation.run_gateway_validation(
            SimpleNamespace(
                json=True,
                markdown=False,
                no_health=True,
                **timeout_kwargs,
            )
        )

        captured = capsys.readouterr()
        assert ok is False
        assert f"{option} must be a positive number of seconds" in captured.err
        assert "Traceback" not in captured.err
