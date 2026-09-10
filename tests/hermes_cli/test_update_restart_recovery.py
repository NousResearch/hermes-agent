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
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

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
    """No gateway authority and no serve authority means no child at all.

    On a Linux host with systemctl the child is spawned anyway for the
    serve-unit pass (test_serve_only_fleet_still_spawns_the_recovery_child);
    this test pins the OTHER side of that contract, so the serve authority
    probe is explicitly disabled here.
    """
    calls = []
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        abort_recovery, "_serve_unit_recovery_available", lambda: False
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
    # Scope-qualified identities the restart phase may record.
    assert update_cmd._gateway_service_matches_profile(
        "default", "gui/501/ai.hermes.gateway"
    )
    assert update_cmd._gateway_service_matches_profile(
        "foo", "user/hermes-gateway-foo.service"
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


def _env_sensitive_systemctl(calls: list, restarted: list):
    """A ``systemctl`` faithful to the real one: every ``--user`` probe fails without a session bus.

    That is the observable that separates a reachable user manager from an absent one — the real
    command prints ``Failed to connect to user scope bus via local transport`` and exits 1.
    """

    def fake_run(argv, **kwargs):
        argv = list(argv)
        if "hermes_cli.main" in argv:  # the per-profile relaunch
            calls.append(argv)
            return _Completed(0)
        if argv and str(argv[0]).endswith("systemctl"):
            calls.append(argv)
            user_scope = "--user" in argv
            if user_scope and not (
                os.environ.get("XDG_RUNTIME_DIR") and os.environ.get("DBUS_SESSION_BUS_ADDRESS")
            ):
                return _Completed(1, stderr="Failed to connect to user scope bus via local transport")
            if "is-active" in argv:
                return _Completed(0, stdout="active\n")
            if "list-units" in argv:
                # The serve unit exists in the user manager only (system scope: nothing to recover).
                return _Completed(0, stdout="hermes-serve.service loaded active running\n" if user_scope else "")
            if "show" in argv:
                return _Completed(0, stdout="5151\n" if restarted else "4242\n")
            if "restart" in argv:
                restarted.append({"user" if user_scope else "system": argv[-1]})
                return _Completed(0)
            return _Completed(0)
        return _Completed(0)

    return fake_run


@pytest.mark.linux_only
def test_bus_less_recovery_child_reaches_our_user_manager(monkeypatch):
    """#107614: the recovery child must observe a user manager that IS on disk.

    ``update_abort_recovery`` spawns this module with ``os.environ.copy()``, so a bus-less
    dispatcher (``sudo -u <user>``, cron, a systemd service, an SSH wrapper) hands the child no
    session bus. Every ``systemctl --user`` probe then fails, and two things follow: the profile is
    reported ``relaunch_attempted`` rather than ``verified`` — which ``_abort_recovery_is_complete``
    reads as incomplete, so a healthy restart still fails the update — and user-scope
    ``hermes-serve*`` units are never listed, so they are neither restarted nor reported.
    """
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")
    runtime_dir = f"/run/user/{os.getuid()}"
    # The on-disk state ``loginctl enable-linger`` leaves behind: ours, and the bus socket exists.
    monkeypatch.setattr(recovery, "_runtime_dir_is_ours", lambda path: str(path) == runtime_dir)
    monkeypatch.setattr(recovery, "_path_exists", lambda path: str(path) == f"{runtime_dir}/bus")
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setattr(recovery.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list = []
    restarted: list = []
    run = _env_sensitive_systemctl(calls, restarted)

    profiles = recovery.restart_profiles(["default"], supervisors={"default": "systemd"}, run=run)
    serve = recovery.restart_serve_units(run=run)

    assert profiles == {"verified": ["default"], "relaunch_attempted": [], "failed": []}
    assert serve == {"verified": ["user/hermes-serve"], "failed": []}
    assert os.environ["XDG_RUNTIME_DIR"] == runtime_dir
    assert os.environ["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={runtime_dir}/bus"
    assert restarted == [{"user": "hermes-serve.service"}]


@pytest.mark.linux_only
def test_bus_less_recovery_child_never_fabricates_a_bus(monkeypatch):
    """Fail closed, never invent: with no user manager on disk the bare environment must survive.

    A fabricated bus address would turn an honest "could not verify" into a connect failure that
    looks like a broken host, and would let a later probe "reach" a manager that does not exist.
    """
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")
    monkeypatch.setattr(recovery, "_runtime_dir_is_ours", lambda path: False)
    monkeypatch.setattr(recovery, "_path_exists", lambda path: False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setattr(recovery.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list = []
    restarted: list = []
    run = _env_sensitive_systemctl(calls, restarted)

    profiles = recovery.restart_profiles(["default"], supervisors={"default": "systemd"}, run=run)
    serve = recovery.restart_serve_units(run=run)

    assert profiles == {"verified": [], "relaunch_attempted": ["default"], "failed": []}
    assert serve == {"verified": [], "failed": []}
    assert restarted == []
    assert "XDG_RUNTIME_DIR" not in os.environ
    assert "DBUS_SESSION_BUS_ADDRESS" not in os.environ


@pytest.mark.linux_only
def test_bus_less_recovery_child_result_completes_the_abort_path(monkeypatch):
    """#107614, consumer half: what the child prints is what decides the update's exit code.

    ``_abort_recovery_is_complete()`` requires ``not relaunch_attempted``, so a bus-less child's
    misclassification of a healthy profile makes completeness unprovable: the abort path keeps
    ``out.incomplete`` set and ``hermes update`` exits 1 for a gateway that WAS restarted onto the
    new code. Drives the real parent entry with the real child passes for a bus-less dispatch.
    """
    recovery = importlib.import_module("hermes_cli.update_restart_recovery")
    runtime_dir = f"/run/user/{os.getuid()}"
    monkeypatch.setattr(recovery, "_runtime_dir_is_ours", lambda path: str(path) == runtime_dir)
    monkeypatch.setattr(recovery, "_path_exists", lambda path: str(path) == f"{runtime_dir}/bus")
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setattr(recovery.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list = []
    restarted: list = []
    child_run = _env_sensitive_systemctl(calls, restarted)

    def fake_child(argv, **kwargs):
        """The fresh child's output: the production passes, in the environment it inherited."""
        payload = json.loads(kwargs["input"])
        result = recovery.restart_profiles(
            payload["profiles"], supervisors=payload["supervisors"], run=child_run
        )
        if payload["serve_units"]["recover"]:
            result["serve_units"] = recovery.restart_serve_units(run=child_run)
        return _Completed(0, stdout=json.dumps(result))

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_child)
    plan = SimpleNamespace(runtimes=[_runtime("default", "systemd")])

    result = update_cmd._recover_gateway_restart_after_abort(plan, gateway_mode=False)

    assert result["verified"] == ["default"]
    assert result["relaunch_attempted"] == []
    # The consequence the issue reports: the abort path may clear ``incomplete`` (exit 0) instead of
    # failing the whole update for a fleet that is already running the new code.
    assert update_cmd._abort_recovery_is_complete(
        planned_gateway_profiles={"default"},
        covered_gateway_profiles={"default"},
        recovery_result=result,
        stale_runtime_rows=[],
    )
