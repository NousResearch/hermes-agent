"""Regression coverage for fresh-process recovery after an update restart abort.

The updater may have loaded the pre-pull module graph when the checkout changes.
If the in-process gateway restart phase then raises, retrying through the same
interpreter cannot establish a coherent module generation.  Recovery must use a
new interpreter, must not invent a restart for manual gateways that have no
supervisor to bring them back, and must not claim supervisor coverage it never
observed: only a systemd-verified unit counts as ``verified``; a bare rc==0
relaunch is ``relaunch_attempted``.
"""

from __future__ import annotations

import importlib
import io
import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace

from hermes_cli import update_abort_recovery as abort_recovery
from hermes_cli import update_cmd


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _successful_recovery_result(
    verified: list[str] | None = None,
    relaunch_attempted: list[str] | None = None,
) -> _Completed:
    return _Completed(
        0,
        stdout=json.dumps(
            {
                "verified": verified or [],
                "relaunch_attempted": relaunch_attempted or [],
                "failed": [],
            }
        ),
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
        return _successful_recovery_result(verified=["coder", "default"])

    monkeypatch.setattr(abort_recovery.subprocess, "run", fake_run)
    plan = SimpleNamespace(
        runtimes=[
            _runtime("default", "systemd"),
            _runtime("coder", "launchd"),
            _runtime("manual-box", "manual"),
            _runtime("desktop", "desktop", kind="serve"),
        ]
    )

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    assert result["requested"] == ["coder", "default"]
    assert result["verified"] == ["coder", "default"]
    assert result["relaunch_attempted"] == []
    assert result["failed"] == []
    # Runtimes the pass does not own are recorded, not silently dropped.
    skipped = {(entry["profile"], entry["kind"]) for entry in result["skipped"]}
    assert skipped == {("manual-box", "gateway"), ("desktop", "serve")}
    assert all(entry["reason"] for entry in result["skipped"])

    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-m", "hermes_cli.update_restart_recovery", "--stdin"]
    payload = json.loads(kwargs["input"])
    assert payload["profiles"] == ["coder", "default"]
    assert payload["supervisors"] == {"coder": "launchd", "default": "systemd"}
    # Serve units travel in the same payload so one fresh child covers both
    # runtime families (#92145).
    assert set(payload["serve_units"]) == {"recover", "skip"}
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
    assert result["verified"] == []
    assert result["relaunch_attempted"] == []
    assert result["failed"] == ["default"]


def test_abort_recovery_reports_unverified_relaunch_conservatively(monkeypatch):
    """rc==0 without a systemd observation must not be reported as verified."""
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: _successful_recovery_result(
            relaunch_attempted=["default"]
        ),
    )
    plan = SimpleNamespace(runtimes=[_runtime("default", "systemd")])

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    assert result["verified"] == []
    assert result["relaunch_attempted"] == ["default"]
    assert result["failed"] == []


def test_abort_recovery_skips_profiles_already_restarted_by_the_phase(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _successful_recovery_result(verified=["coder"])

    monkeypatch.setattr(abort_recovery.subprocess, "run", fake_run)
    plan = SimpleNamespace(
        runtimes=[_runtime("default", "systemd"), _runtime("coder", "systemd")]
    )

    result = update_cmd._recover_gateway_restart_after_abort(
        plan,
        gateway_mode=False,
        skip_profiles={"default"},
    )
    assert result["requested"] == ["coder"]
    assert result["verified"] == ["coder"]
    assert result["failed"] == []
    assert json.loads(calls[0][1]["input"])["profiles"] == ["coder"]


def test_abort_recovery_rejects_partial_json_success(monkeypatch):
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(
            0,
            stdout=json.dumps(
                {"verified": ["default"], "relaunch_attempted": [], "failed": []}
            ),
        ),
    )
    plan = SimpleNamespace(
        runtimes=[_runtime("default", "systemd"), _runtime("coder", "systemd")]
    )

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    # "coder" is unaccounted for in the child's report: fail closed.
    assert result["requested"] == ["coder", "default"]
    assert result["verified"] == []
    assert result["failed"] == ["coder", "default"]


def test_abort_recovery_rejects_malformed_json_success(monkeypatch):
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(0, stdout="not-json"),
    )
    plan = SimpleNamespace(runtimes=[_runtime("default", "systemd")])

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    assert result["requested"] == ["default"]
    assert result["verified"] == []
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
    assert result["requested"] == []
    assert result["verified"] == []
    assert result["failed"] == []
    assert [entry["profile"] for entry in result["skipped"]] == ["manual-box"]
    assert calls == []


def test_abort_recovery_records_serve_runtimes_as_skipped_with_reason(monkeypatch):
    """Serve/dashboard ledger entries must not vanish from the recovery pass."""
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: _successful_recovery_result(verified=["default"]),
    )
    plan = SimpleNamespace(
        runtimes=[
            _runtime("default", "systemd"),
            _runtime("default", "desktop", kind="serve"),
            _runtime("ops", "manual-serve", kind="dashboard"),
        ]
    )

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)
    by_kind = {entry["kind"]: entry for entry in result["skipped"]}
    assert set(by_kind) == {"serve", "dashboard"}
    assert "desktop app" in by_kind["serve"]["reason"]
    assert by_kind["dashboard"]["profile"] == "ops"
    assert "relaunch" in by_kind["dashboard"]["reason"]


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

    # No supervisor observations were possible → conservative labels only.
    assert result == {
        "verified": [],
        "relaunch_attempted": ["coder", "default"],
        "failed": [],
    }
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


def test_recovery_child_verifies_systemd_profiles_via_is_active(monkeypatch):
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")
    monkeypatch.setattr(recovery.shutil, "which", lambda name: f"/bin/{name}")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[0].endswith("systemctl"):
            unit = argv[-1]
            active = unit == "hermes-gateway.service"
            return _Completed(0 if active else 3, stdout="active" if active else "inactive")
        return _Completed(0)

    result = recovery.restart_profiles(
        ["default", "coder"],
        supervisors={"default": "systemd", "coder": "launchd"},
        run=fake_run,
    )

    assert result == {
        "verified": ["default"],
        "relaunch_attempted": ["coder"],
        "failed": [],
    }
    # The launchd profile must never be probed with systemctl.
    systemctl_units = [argv[-1] for argv in calls if argv[0].endswith("systemctl")]
    assert all("coder" not in unit for unit in systemctl_units)


def test_recovery_child_treats_missing_systemctl_as_unverified(monkeypatch):
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")
    monkeypatch.setattr(recovery.shutil, "which", lambda name: None)

    result = recovery.restart_profiles(
        ["default"],
        supervisors={"default": "systemd"},
        run=lambda *args, **kwargs: _Completed(0),
    )

    assert result == {
        "verified": [],
        "relaunch_attempted": ["default"],
        "failed": [],
    }


def test_recovery_child_reports_failed_profile_without_losing_successes():
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")
    outcomes = iter((_Completed(1), _Completed(0)))

    result = recovery.restart_profiles(
        ["coder", "default"], run=lambda *args, **kwargs: next(outcomes)
    )

    assert result == {
        "verified": [],
        "relaunch_attempted": ["default"],
        "failed": ["coder"],
    }


def test_recovery_payload_rejects_path_like_profile_ids():
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")

    try:
        recovery._parse_payload(io.StringIO(json.dumps({"profiles": ["../other"]})))
    except ValueError as exc:
        assert "invalid profile" in str(exc)
    else:
        raise AssertionError("path-like profile id must be rejected")


def test_recovery_payload_rejects_malformed_supervisors_map():
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")

    try:
        recovery._parse_payload(
            io.StringIO(
                json.dumps(
                    {"profiles": ["default"], "supervisors": {"default": "sys/temd"}}
                )
            )
        )
    except ValueError as exc:
        assert "supervisors" in str(exc)
    else:
        raise AssertionError("malformed supervisors map must be rejected")


def test_recovery_module_empty_payload_is_a_real_clean_process():
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.update_restart_recovery", "--stdin"],
        input=json.dumps({"profiles": []}),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "failed": [],
        "relaunch_attempted": [],
        "verified": [],
        "serve_units": {"verified": [], "failed": []},
    }


def test_recovery_module_end_to_end_in_a_real_fresh_process(tmp_path):
    """E2E: the whole recovery protocol through a genuinely fresh interpreter.

    A ``sitecustomize`` shim in the child's ``PYTHONPATH`` intercepts the
    grandchild ``hermes_cli.main … gateway restart`` invocations (recording
    them and returning rc 0) and answers ``systemctl --user is-active`` with
    ``active`` only for the default profile's unit.  Everything else — stdin
    payload parsing, profile ordering, environment scrubbing, verification
    classification, JSON output, and exit code — runs the real module code in
    a real new process, exactly as the aborted updater would spawn it.
    """
    ledger = tmp_path / "grandchild_calls.jsonl"
    shim = textwrap.dedent(
        f"""
        import json
        import shutil
        import subprocess

        _real_run = subprocess.run
        _real_which = shutil.which
        _LEDGER = {str(ledger)!r}


        def _shim_which(name, *args, **kwargs):
            if name == "systemctl":
                return "/usr/bin/systemctl"
            return _real_which(name, *args, **kwargs)


        shutil.which = _shim_which


        def _shim_run(argv, *args, **kwargs):
            argv_list = list(argv)
            if "hermes_cli.main" in argv_list:
                with open(_LEDGER, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(argv_list) + "\\n")
                return subprocess.CompletedProcess(argv_list, 0, "", "")
            if argv_list and str(argv_list[0]).endswith("systemctl"):
                unit = argv_list[-1]
                if unit == "hermes-gateway.service":
                    return subprocess.CompletedProcess(argv_list, 0, "active\\n", "")
                return subprocess.CompletedProcess(argv_list, 3, "inactive\\n", "")
            return _real_run(argv, *args, **kwargs)


        subprocess.run = _shim_run
        """
    )
    (tmp_path / "sitecustomize.py").write_text(shim, encoding="utf-8")

    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    env["_HERMES_GATEWAY"] = "1"  # must be scrubbed before the grandchild runs

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.update_restart_recovery", "--stdin"],
        input=json.dumps(
            {
                "profiles": ["default", "coder"],
                "supervisors": {"default": "systemd", "coder": "launchd"},
            }
        ),
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "failed": [],
        "relaunch_attempted": ["coder"],
        "verified": ["default"],
        "serve_units": {"verified": [], "failed": []},
    }
    restarts = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [argv[argv.index("-p") + 1] for argv in restarts] == ["coder", "default"]
    for argv in restarts:
        assert argv[-2:] == ["gateway", "restart"]
