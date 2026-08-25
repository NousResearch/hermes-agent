"""Regression coverage for fresh-process recovery after an update restart abort.

The updater may have loaded the pre-pull module graph when the checkout changes.
If the in-process gateway restart phase then raises, retrying through the same
interpreter cannot establish a coherent module generation.  Recovery must use a
new interpreter and must not invent a restart for manual gateways that have no
supervisor to bring them back.
"""

from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace

from hermes_cli import update_cmd


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _successful_recovery_result(profiles: list[str]) -> _Completed:
    return _Completed(
        0,
        stdout=json.dumps({"succeeded": profiles, "failed": []}),
    )


def _runtime(profile: str, supervisor: str, kind: str = "gateway"):
    return SimpleNamespace(
        profile=profile,
        supervisor=supervisor,
        kind=kind,
        pid=1234,
    )


def test_abort_recovery_hands_managed_profiles_to_a_fresh_process(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _successful_recovery_result(["coder", "default"])

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    plan = SimpleNamespace(
        runtimes=[
            _runtime("default", "systemd"),
            _runtime("coder", "launchd"),
            _runtime("manual-box", "manual"),
            _runtime("desktop", "desktop", kind="serve"),
        ]
    )

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    assert result == {
        "requested": ["coder", "default"],
        "succeeded": ["coder", "default"],
        "failed": [],
    }

    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-m", "hermes_cli.update_restart_recovery", "--stdin"]
    payload = json.loads(kwargs["input"])
    assert payload == {"profiles": ["coder", "default"]}
    assert kwargs["text"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False
    assert kwargs["env"]["HERMES_UPDATE_RESTART_RECOVERY"] == "1"


def test_abort_recovery_does_not_claim_success_when_fresh_process_fails(monkeypatch):
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(1),
    )
    plan = SimpleNamespace(runtimes=[_runtime("default", "systemd")])

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    assert result["requested"] == ["default"]
    assert result["succeeded"] == []
    assert result["failed"] == ["default"]


def test_abort_recovery_skips_profiles_already_restarted_by_the_phase(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _successful_recovery_result(["coder"])

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    plan = SimpleNamespace(
        runtimes=[_runtime("default", "systemd"), _runtime("coder", "systemd")]
    )

    result = update_cmd._recover_gateway_restart_after_abort(
        plan,
        gateway_mode=False,
        skip_profiles={"default"},
    )
    assert result == {"requested": ["coder"], "succeeded": ["coder"], "failed": []}
    assert json.loads(calls[0][1]["input"]) == {"profiles": ["coder"]}


def test_abort_recovery_rejects_partial_json_success(monkeypatch):
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(
            0,
            stdout=json.dumps({"succeeded": ["default"], "failed": ["coder"]}),
        ),
    )
    plan = SimpleNamespace(
        runtimes=[_runtime("default", "systemd"), _runtime("coder", "systemd")]
    )

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    assert result["requested"] == ["coder", "default"]
    assert result["succeeded"] == ["default"]
    assert result["failed"] == ["coder"]


def test_abort_recovery_rejects_malformed_json_success(monkeypatch):
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(0, stdout="not-json"),
    )
    plan = SimpleNamespace(runtimes=[_runtime("default", "systemd")])

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    assert result["requested"] == ["default"]
    assert result["succeeded"] == []
    assert result["failed"] == ["default"]


def test_abort_recovery_does_not_restart_manual_only_fleet(monkeypatch):
    calls = []
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    plan = SimpleNamespace(runtimes=[_runtime("manual-box", "manual")])

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    assert result == {"requested": [], "succeeded": [], "failed": []}
    assert calls == []


def test_service_matching_is_exact_for_overlapping_profile_names():
    assert update_cmd._gateway_service_matches_profile(
        "foo", "hermes-gateway-foo.service"
    )
    assert not update_cmd._gateway_service_matches_profile(
        "foo", "hermes-gateway-foobar.service"
    )
    assert update_cmd._gateway_service_matches_profile(
        "default", "ai.hermes.gateway"
    )
    assert not update_cmd._gateway_service_matches_profile(
        "default", "ai.hermes.gateway-foo"
    )


def test_recovery_child_restarts_each_profile_with_a_fresh_main(monkeypatch):
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Completed(0)

    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    result = recovery.restart_profiles(["default", "coder"], run=fake_run)

    assert result == {"succeeded": ["coder", "default"], "failed": []}
    assert [call[0] for call in calls] == [
        [sys.executable, "-m", "hermes_cli.main", "-p", "coder", "gateway", "restart"],
        [sys.executable, "-m", "hermes_cli.main", "-p", "default", "gateway", "restart"],
    ]
    for _, kwargs in calls:
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        assert kwargs["env"]["HERMES_UPDATE_RESTART_RECOVERY"] == "1"
        assert "_HERMES_GATEWAY" not in kwargs["env"]


def test_recovery_child_reports_failed_profile_without_losing_successes():
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")
    outcomes = iter((_Completed(1), _Completed(0)))

    result = recovery.restart_profiles(
        ["coder", "default"], run=lambda *args, **kwargs: next(outcomes)
    )

    assert result == {"succeeded": ["default"], "failed": ["coder"]}


def test_recovery_payload_rejects_path_like_profile_ids():
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")

    try:
        recovery._parse_payload(io.StringIO(json.dumps({"profiles": ["../other"]})))
    except ValueError as exc:
        assert "invalid profile" in str(exc)
    else:
        raise AssertionError("path-like profile id must be rejected")


def test_recovery_module_empty_payload_is_a_real_clean_process():
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.update_restart_recovery", "--stdin"],
        input=json.dumps({"profiles": []}),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {"succeeded": [], "failed": []}
