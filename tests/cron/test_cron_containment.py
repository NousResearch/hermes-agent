"""Per-occurrence containment (cron/containment.py) and its run_job teardown.

Invariant: no process spawned by a cron occurrence may
outlive the agent session. These tests cover the runtime side of that
invariant: registry lifecycle, synchronous unit teardown with survivor
verification, the stale sweep, host-unit detection, and the run_job wiring —
including the deferred-teardown path where the conversation thread outlives
run_job, which must tear containment down at the conversation's
real end, not at run_job's verdict.

systemd itself is faked with a stateful stub binary; a real user manager is
NOT required.
"""

from __future__ import annotations

# Collection-time settle, before the autouse home-IO guard installs: running the
# suite from a linked worktree makes ``_settle_interrupted_update()`` (module
# import side effect) stat the worktree's git-dir — which lives inside the MAIN
# checkout under the real Hermes home — and the guard would refuse that probe
# mid-test. Importing bootstrap here runs the probe once, unguarded and harmless
# (no interrupted update), so guarded tests never re-run it.
import hermes_bootstrap

import json
import os
import textwrap
import uuid
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fake systemctl: a stateful stub keyed by a JSON state file.
# ---------------------------------------------------------------------------

FAKE_SYSTEMCTL = textwrap.dedent("""\
    #!/usr/bin/env python3
    import json, os, sys
    state_path = os.environ["FAKE_SYSTEMCTL_STATE"]
    sysfs = os.environ["FAKE_SYSTEMCTL_SYSFS"]
    args = sys.argv[1:]
    if "--user" in args:
        args.remove("--user")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    units = state.setdefault("units", {})

    def _empty_cgroup(cgroup):
        # Real `systemctl stop/kill` empties the unit's cgroup; mirror that in
        # the fake sysfs tree so survivor assertions see the removal.
        if not cgroup:
            return
        procs = os.path.join(sysfs, cgroup.lstrip("/"), "cgroup.procs")
        if os.path.exists(procs):
            open(procs, "w").close()

    if args and args[0] == "show":
        unit = args[-1]
        info = units.get(unit, {"state": "not-found"})
        # Real `systemctl show` (no --value) emits key=value lines in no
        # guaranteed order; the module parses pairs, never positions.
        print(f"ActiveState={info['state']}")
        if info.get("cgroup"):
            print(f"ControlGroup={info['cgroup']}")
        sys.exit(0)
    if args and args[0] == "stop":
        unit = args[-1]
        info = units.get(unit)
        if info is None:
            print(f"Failed to stop {unit}: Unit {unit} not loaded.", file=sys.stderr)
            sys.exit(1)
        if info.get("stubborn"):
            sys.exit(0)  # refuse to stop: the escalation path must fire
        _empty_cgroup(info.get("cgroup"))
        info["state"] = "inactive"
        info["cgroup"] = ""
        json.dump(state, open(state_path, "w"))
        sys.exit(0)
    if args and args[:2] == ["kill", "--signal=SIGKILL"]:
        unit = args[-1]
        info = units.get(unit)
        if info is not None:
            if info.get("kill_resistant"):
                sys.exit(0)
            _empty_cgroup(info.get("cgroup"))
            info["state"] = "failed"
            info["cgroup"] = ""
            json.dump(state, open(state_path, "w"))
        sys.exit(0)
    sys.exit(2)
""")


@pytest.fixture()
def fake_systemctl(tmp_path, monkeypatch):
    """Install the stub binary + state file; point the cgroup root at tmp.

    The stub is a test-local script, but its argv shape (``systemctl --user
    stop <hermes-cron-*.scope>``) matches what the live-system guard must
    refuse for the REAL systemctl — so every test that lets the module exec
    it opts out explicitly below.
    """
    from cron import containment

    script = tmp_path / "fake-systemctl"
    script.write_text(FAKE_SYSTEMCTL)
    script.chmod(0o755)
    state_path = tmp_path / "systemctl-state.json"
    state_path.write_text("{}")
    sysfs = tmp_path / "sysfs"
    sysfs.mkdir()
    monkeypatch.setattr(containment, "SYSTEMCTL", str(script))
    monkeypatch.setattr(containment, "_PROC_CGROUP_ROOT", sysfs)
    monkeypatch.setenv("FAKE_SYSTEMCTL_STATE", str(state_path))
    monkeypatch.setenv("FAKE_SYSTEMCTL_SYSFS", str(sysfs))
    return {"state": state_path, "sysfs": sysfs}


def _set_unit(fake_systemctl, unit: str, state: str, cgroup: str = "", *,
              stubborn: bool = False, kill_resistant: bool = False):
    state_data = json.loads(fake_systemctl["state"].read_text())
    units = state_data.setdefault("units", {})
    units[unit] = {"state": state, "cgroup": cgroup, "stubborn": stubborn,
                   "kill_resistant": kill_resistant}
    fake_systemctl["state"].write_text(json.dumps(state_data))


def _make_cgroup(fake_systemctl, cgroup: str, pids=("4321",)):
    """Create a fake cgroup.procs file the module will read."""
    procs = fake_systemctl["sysfs"] / cgroup.lstrip("/")
    procs.mkdir(parents=True, exist_ok=True)
    (procs / "cgroup.procs").write_text("\n".join(pids) + "\n")


# ---------------------------------------------------------------------------
# Registry lifecycle
# ---------------------------------------------------------------------------

def test_begin_creates_registry_and_reports_host_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    info = containment.begin("exec-1")
    assert info is not None
    assert Path(info["dir"]) == tmp_path / "home" / "cron" / "containment" / "exec-1"
    assert isinstance(info["host_unit"], str)
    assert (Path(info["dir"]) / "units").exists() is False  # empty until registered


def test_begin_with_invalid_id_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    assert containment.begin("../escape") is None
    assert containment.begin("") is None
    # And nothing escaped the containment root.
    assert not (tmp_path / "escape").exists()


def test_register_unit_roundtrip_dedupes_and_rejects_malformed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    exec_id = "exec-reg"
    assert containment.begin(exec_id) is not None
    unit = containment.new_unit_name(exec_id)
    assert unit.startswith("hermes-cron-") and unit.endswith(".scope")

    assert containment.register_unit(exec_id, unit) is True
    assert containment.register_unit(exec_id, unit) is True  # idempotent
    assert containment.register_unit(exec_id, "evil.service") is False
    assert containment.register_unit(exec_id, "rm -rf") is False
    assert containment.register_unit(exec_id, "../../etc/passwd.scope") is False
    assert containment.register_unit(exec_id, "not-a-hermes-unit.scope") is False
    assert containment.register_unit(exec_id, "hermes-cron-x.service") is False
    assert containment.register_unit(exec_id, "unregistered.scope") is False
    assert containment.registered_units(exec_id) == [unit]


# The "systemctl" these tests exec is the fixture's stateful stub, never the
# real binary — the guard's argv pattern cannot tell them apart, so each test
# that reaches stop_unit/kill opts out explicitly.
_systemctl_stub = pytest.mark.live_system_guard_bypass


@_systemctl_stub
def test_end_stops_registered_units_and_removes_registry(fake_systemctl, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    exec_id = "exec-clean"
    containment.begin(exec_id)
    unit = containment.new_unit_name(exec_id)
    containment.register_unit(exec_id, unit)
    _set_unit(fake_systemctl, unit, "active", "/user.slice/u.service/a.slice/x.scope")
    _make_cgroup(fake_systemctl, "/user.slice/u.service/a.slice/x.scope")

    report = containment.end(exec_id)

    assert report["stopped"] == [unit]
    assert report["leftover"] == []
    assert not (tmp_path / "home" / "cron" / "containment" / exec_id).exists()


@_systemctl_stub
def test_end_reports_leftover_and_keeps_registry_for_retry(fake_systemctl, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    exec_id = "exec-stuck"
    containment.begin(exec_id)
    unit = containment.new_unit_name(exec_id)
    containment.register_unit(exec_id, unit)
    cgroup = "/user.slice/u.service/a.slice/stuck.scope"
    _set_unit(fake_systemctl, unit, "active", cgroup, stubborn=True, kill_resistant=True)
    _make_cgroup(fake_systemctl, cgroup)

    report = containment.end(exec_id)

    assert report["stopped"] == []
    assert len(report["leftover"]) == 1 and unit in report["leftover"][0]
    # Registry kept: the next fire's sweep can retry.
    assert containment.registered_units(exec_id) == [unit]


@_systemctl_stub
def test_stop_unit_escalates_to_sigkill_when_stop_refused(fake_systemctl, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    exec_id = "exec-escalate"
    containment.begin(exec_id)
    unit = containment.new_unit_name(exec_id)
    cgroup = "/user.slice/u.service/a.slice/esc.scope"
    _set_unit(fake_systemctl, unit, "active", cgroup, stubborn=True)  # stop no-ops
    _make_cgroup(fake_systemctl, cgroup)

    assert containment.stop_unit(unit) == "clean"
    assert not report_survivors(fake_systemctl, cgroup)


def report_survivors(fake_systemctl, cgroup):
    from cron import containment
    _status, pids = containment.cgroup_procs(cgroup)
    return pids


# ---------------------------------------------------------------------------
# Fail-closed cgroup verification
# ---------------------------------------------------------------------------

def test_cgroup_procs_distinguishes_empty_absent_and_pids(fake_systemctl):
    from cron import containment

    # Verified empty: readable file, zero PIDs.
    _make_cgroup(fake_systemctl, "/empty.scope", pids=())
    assert containment.cgroup_procs("/empty.scope") == ("verified-empty", [])

    # Has PIDs: readable file, live PIDs.
    _make_cgroup(fake_systemctl, "/busy.scope", pids=("11", "22"))
    assert containment.cgroup_procs("/busy.scope") == ("has-pids", [11, 22])

    # Verified absent: cgroup dir (and its procs file) genuinely gone.
    assert containment.cgroup_procs("/never-existed.scope") == ("verified-absent", [])

    # Unverifiable: procs file missing but the cgroup dir still exists.
    (fake_systemctl["sysfs"] / "dangling.scope").mkdir()
    assert containment.cgroup_procs("/dangling.scope") == ("unverifiable", [])

    # Unverifiable: empty/None cgroup path cannot be validated here.
    assert containment.cgroup_procs(None) == ("unverifiable", [])
    assert containment.cgroup_procs("") == ("unverifiable", [])


def test_cgroup_procs_fails_closed_on_permission_error(fake_systemctl):
    from cron import containment

    procs = fake_systemctl["sysfs"] / "denied.scope"
    procs.mkdir()
    (procs / "cgroup.procs").write_text("4321\n")
    (procs / "cgroup.procs").chmod(0o000)
    try:
        assert containment.cgroup_procs("/denied.scope") == ("unverifiable", [])
    finally:
        (procs / "cgroup.procs").chmod(0o644)


def test_cgroup_procs_fails_closed_on_malformed_content(fake_systemctl):
    from cron import containment

    _make_cgroup(fake_systemctl, "/garbage.scope", pids=("4321", "not-a-pid"))
    assert containment.cgroup_procs("/garbage.scope") == ("unverifiable", [])


@_systemctl_stub
def test_stop_unit_fails_closed_on_permission_error(fake_systemctl, tmp_path, monkeypatch):
    """An unreadable cgroup.procs is NOT a verified-empty cgroup: stop_unit
    must return an error status, never clean/gone."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    unit = "hermes-cron-x-perm.scope"
    cgroup = "/user.slice/u.service/a.slice/perm.scope"
    _set_unit(fake_systemctl, unit, "inactive", cgroup)  # dead, but cgroup unreadable
    _make_cgroup(fake_systemctl, cgroup)
    procs = fake_systemctl["sysfs"] / cgroup.lstrip("/") / "cgroup.procs"
    procs.chmod(0o000)
    try:
        assert containment.stop_unit(unit) == "error:unverifiable-cgroup"
    finally:
        procs.chmod(0o644)


@_systemctl_stub
def test_stop_unit_fails_closed_on_malformed_procs(fake_systemctl, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    unit = "hermes-cron-x-garbage.scope"
    cgroup = "/user.slice/u.service/a.slice/garbage.scope"
    _set_unit(fake_systemctl, unit, "inactive", cgroup)
    _make_cgroup(fake_systemctl, cgroup, pids=("4321", "oops"))
    assert containment.stop_unit(unit) == "error:unverifiable-cgroup"


@_systemctl_stub
def test_stop_unit_fails_closed_post_kill_on_unverifiable(fake_systemctl, tmp_path, monkeypatch):
    """SIGKILL escalation whose re-verification is unreadable must fail closed,
    not report clean."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    unit = "hermes-cron-x-postkill.scope"
    cgroup = "/user.slice/u.service/a.slice/postkill.scope"
    _set_unit(fake_systemctl, unit, "active", cgroup, stubborn=True)
    _make_cgroup(fake_systemctl, cgroup)
    procs = fake_systemctl["sysfs"] / cgroup.lstrip("/") / "cgroup.procs"
    monkeypatch.setattr(containment, "_KILL_SETTLE_SECONDS", 0.0)
    # chmod AFTER the first (readable) post-stop snapshot: the post-kill
    # re-verification is then unverifiable, which must fail closed.
    _original_run = containment.subprocess.run

    def _deny_after_stop(cmd, **kw):
        result = _original_run(cmd, **kw)
        if cmd[1:3] == ["--user", "stop"]:
            procs.chmod(0o000)
        return result

    monkeypatch.setattr(containment.subprocess, "run", _deny_after_stop)
    try:
        assert containment.stop_unit(unit) == "error:unverifiable-cgroup"
    finally:
        procs.chmod(0o644)


@_systemctl_stub
def test_end_keeps_registry_on_unverifiable_unit(fake_systemctl, tmp_path, monkeypatch):
    """PermissionError reading cgroup.procs: end() must report the unit as
    leftover and PRESERVE the registry dir for retry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    exec_id = "exec-unverifiable"
    containment.begin(exec_id)
    unit = containment.new_unit_name(exec_id)
    containment.register_unit(exec_id, unit)
    cgroup = "/user.slice/u.service/a.slice/denied.scope"
    _set_unit(fake_systemctl, unit, "inactive", cgroup)
    _make_cgroup(fake_systemctl, cgroup)
    procs = fake_systemctl["sysfs"] / cgroup.lstrip("/") / "cgroup.procs"
    procs.chmod(0o000)
    try:
        report = containment.end(exec_id)
    finally:
        procs.chmod(0o644)
    assert report["stopped"] == []
    assert len(report["leftover"]) == 1 and "unverifiable" in report["leftover"][0]
    assert containment.registered_units(exec_id) == [unit], "registry must be kept"


@_systemctl_stub
def test_end_mixed_clean_and_unverifiable_preserves_registry(fake_systemctl, tmp_path, monkeypatch):
    """One clean unit must not let partial teardown delete the registry while
    another unit is unverifiable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    exec_id = "exec-mixed"
    containment.begin(exec_id)
    clean_unit = containment.new_unit_name(exec_id)
    bad_unit = containment.new_unit_name(exec_id)
    containment.register_unit(exec_id, clean_unit)
    containment.register_unit(exec_id, bad_unit)
    _set_unit(fake_systemctl, clean_unit, "active", "/user.slice/u.service/a.slice/clean.scope")
    _make_cgroup(fake_systemctl, "/user.slice/u.service/a.slice/clean.scope", pids=())
    bad_cgroup = "/user.slice/u.service/a.slice/bad.scope"
    _set_unit(fake_systemctl, bad_unit, "inactive", bad_cgroup)
    _make_cgroup(fake_systemctl, bad_cgroup)
    procs = fake_systemctl["sysfs"] / bad_cgroup.lstrip("/") / "cgroup.procs"
    procs.chmod(0o000)
    try:
        report = containment.end(exec_id)
    finally:
        procs.chmod(0o644)
    assert clean_unit in report["stopped"]
    assert len(report["leftover"]) == 1
    assert containment.registered_units(exec_id), "registry must survive mixed teardown"


@_systemctl_stub
def test_end_verified_absent_cgroup_is_gone_and_registry_removed(fake_systemctl, tmp_path, monkeypatch):
    """Unit inactive and the cgroup dir genuinely gone: verified absent ->
    gone, and the registry is still removable."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    exec_id = "exec-absent"
    containment.begin(exec_id)
    unit = containment.new_unit_name(exec_id)
    containment.register_unit(exec_id, unit)
    # inactive unit pointing at a cgroup that does not exist at all.
    _set_unit(fake_systemctl, unit, "inactive",
              "/user.slice/u.service/a.slice/vanished.scope")

    report = containment.end(exec_id)

    assert report == {"stopped": [unit], "leftover": []}
    assert not (tmp_path / "home" / "cron" / "containment" / exec_id).exists()


@_systemctl_stub
def test_sweep_stale_keeps_registry_on_unverifiable_end(fake_systemctl, tmp_path, monkeypatch):
    """An unverifiable end() result during the stale sweep keeps the registry
    and is logged as a survivor, not counted as swept."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    exec_id = "exec-sweep-unverifiable"
    containment.begin(exec_id)
    unit = containment.new_unit_name(exec_id)
    containment.register_unit(exec_id, unit)
    cgroup = "/user.slice/u.service/a.slice/sweep-denied.scope"
    _set_unit(fake_systemctl, unit, "inactive", cgroup)
    _make_cgroup(fake_systemctl, cgroup)
    procs = fake_systemctl["sysfs"] / cgroup.lstrip("/") / "cgroup.procs"
    procs.chmod(0o000)
    try:
        swept = containment.sweep_stale(lambda _exec_id: False)
    finally:
        procs.chmod(0o644)
    assert swept == 0, "unverifiable teardown must not count as swept"
    assert containment.registered_units(exec_id) == [unit], "registry must be kept"


def test_stop_unit_reports_gone_for_unknown_unit(fake_systemctl, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    assert containment.stop_unit("hermes-cron-x-00000000.scope") in ("gone", "clean")


def test_end_on_never_begun_execution_is_a_clean_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    report = containment.end("exec-never-begun")
    assert report == {"stopped": [], "leftover": []}


@_systemctl_stub
def test_sweep_stale_only_touches_finished_executions(fake_systemctl, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    dead_id, live_id = "exec-dead", "exec-live"
    for exec_id in (dead_id, live_id):
        containment.begin(exec_id)
        unit = containment.new_unit_name(exec_id)
        containment.register_unit(exec_id, unit)
        _set_unit(fake_systemctl, unit, "active")

    swept = containment.sweep_stale(lambda exec_id: exec_id == live_id)

    assert swept == 1
    assert containment.registered_units(live_id), "live execution must keep its registry"
    assert not (tmp_path / "home" / "cron" / "containment" / dead_id).exists()


def test_sweep_survives_liveness_probe_failure(fake_systemctl, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment

    exec_id = "exec-probe-blows-up"
    containment.begin(exec_id)
    unit = containment.new_unit_name(exec_id)
    containment.register_unit(exec_id, unit)  # something real for the sweep to leave alone

    def _boom(_exec_id):
        raise RuntimeError("store unavailable")

    assert containment.sweep_stale(_boom) == 0
    assert containment.registered_units(exec_id)  # untouched


# ---------------------------------------------------------------------------
# Host unit detection
# ---------------------------------------------------------------------------

def test_parse_host_unit_detects_user_service():
    from cron.containment import _parse_host_unit

    line = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/hermes-gateway.service"
    assert _parse_host_unit(line) == "hermes-gateway.service"


def test_parse_host_unit_ignores_non_service_and_system_paths():
    from cron.containment import _parse_host_unit

    assert _parse_host_unit("0::/system.slice/sshd.service") == ""
    assert _parse_host_unit("0::/user.slice/user-1000.slice/user@1000.service/app.slice") == ""
    assert _parse_host_unit("") == ""
    assert _parse_host_unit("0::/user.slice/user-1000.slice/app.slice/session-2.scope") == ""


# ---------------------------------------------------------------------------
# run_job wiring
# ---------------------------------------------------------------------------

def _install_run_job_stubs(monkeypatch, observed: dict, agent_cls=None):
    """Same harness as tests/cron/test_cron_kanban_env_isolation.py."""
    import sys

    import cron.scheduler as sched
    from cron import scheduler_delivery as sched_delivery

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]
            self.task_id = self.session_id

        def run_conversation(self, *_a, **_kw):
            observed["during_run"] = agent_during_run(observed)
            return {"final_response": "done", "messages": []}

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

    fake_mod = type(sys)("run_agent")
    fake_mod.AIAgent = agent_cls or FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

    from hermes_cli import runtime_provider as _rtp
    monkeypatch.setattr(_rtp, "resolve_runtime_provider", lambda **_kw: {
        "provider": "test", "api_key": "k", "base_url": "http://test.local",
        "api_mode": "chat_completions"})
    monkeypatch.setattr(sched, "_build_job_prompt", lambda job, prerun_script=None, **kw: "hi")
    monkeypatch.setattr(sched_delivery, "_resolve_origin", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
    monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)


def agent_during_run(observed):
    from gateway.session_context import _UNSET, _VAR_MAP
    return {
        "containment_dir": _VAR_MAP["HERMES_CRON_CONTAINMENT_DIR"].get(),
        "host_unit": _VAR_MAP["HERMES_CRON_HOST_UNIT"].get(),
        "unset_sentinel": _UNSET,
    }


def _job(job_id="containment-iso", execution_id=None):
    job = {
        "id": job_id, "name": "containment-job", "prompt": "hi", "model": "test-model",
        "workdir": None, "schedule_display": "manual",
    }
    if execution_id:
        job["execution_id"] = execution_id
    return job


@_systemctl_stub
@pytest.mark.usefixtures("cron_owner")
def test_run_job_publishes_containment_contextvars_and_cleans_registry(
        fake_systemctl, tmp_path, monkeypatch):
    """During the run the ContextVars carry the registry dir; after the run the
    registry is gone (teardown ran synchronously in run_job's finally)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment
    import cron.scheduler as sched

    exec_id = f"exec-{uuid.uuid4().hex[:12]}"
    observed: dict = {}
    _install_run_job_stubs(monkeypatch, observed)

    def _run_conversation_with_registration(self, *_a, **_kw):
        observed["during_run"] = agent_during_run(observed)
        # The driver discovers the registry dir from the published ContextVar
        # (the canonical owner generates its own execution id) and registers
        # its scope into THAT registry — never one it guessed.
        run_exec_id = Path(observed["during_run"]["containment_dir"]).name
        unit = containment.new_unit_name(run_exec_id)
        assert containment.register_unit(run_exec_id, unit), "driver registration must succeed"
        observed["registered_unit"] = unit
        _set_unit(fake_systemctl, unit, "active")
        return {"final_response": "done", "messages": []}

    import sys
    fake_mod = sys.modules["run_agent"]
    fake_mod.AIAgent = type("FakeAgent", (), {
        "__init__": lambda self, **kw: None,
        "run_conversation": _run_conversation_with_registration,
        "get_activity_summary": lambda self: {"seconds_since_activity": 0.0},
    })

    success, *_ = sched.run_job(_job(execution_id=exec_id))
    assert success is True

    during = observed["during_run"]
    # The canonical owner generated its own execution id; the published dir
    # must be that id's registry under the (isolated) containment root.
    assert during["containment_dir"].startswith(
        str(tmp_path / "home" / "cron" / "containment") + "/")
    assert isinstance(during["host_unit"], str)

    from gateway.session_context import _UNSET, _VAR_MAP
    assert _VAR_MAP["HERMES_CRON_CONTAINMENT_DIR"].get() is _UNSET, "ContextVar must reset after the run"
    assert _VAR_MAP["HERMES_CRON_HOST_UNIT"].get() is _UNSET
    # The registered unit was stopped and the registry removed.
    assert not Path(during["containment_dir"]).exists()
    assert observed["registered_unit"]


@pytest.mark.usefixtures("cron_owner")
def test_run_job_generated_execution_id_still_gets_containment(tmp_path, monkeypatch):
    """A fire whose execution id the canonical owner generates (not the job dict)
    still gets containment env + teardown: every real cron fire carries an
    execution id, so no occurrence may escape the containment contract."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    import cron.scheduler as sched

    observed: dict = {}
    _install_run_job_stubs(monkeypatch, observed)
    success, *_ = sched.run_job(_job(job_id="generated-exec-id"))
    assert success is True
    containment_dir = observed["during_run"]["containment_dir"]
    assert containment_dir, "canonical fires always carry an execution id"
    assert containment_dir.startswith(str(tmp_path / "home" / "cron" / "containment"))
    assert isinstance(observed["during_run"]["host_unit"], str)


def test_scope_without_execution_id_exports_no_containment_env(tmp_path, monkeypatch):
    """Direct scope contract: no execution id -> no containment env published."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    import cron.scheduler as sched

    job = _job(job_id="no-exec-id")
    scope = sched._CronRunScope(job, "no-exec-id", None)
    assert scope._containment_execution_id == ""
    assert scope._containment_info is None
    assert scope._containment_tokens == []


@_systemctl_stub
@pytest.mark.usefixtures("cron_owner")
def test_run_job_failure_path_still_tears_down_containment(
        fake_systemctl, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment
    import cron.scheduler as sched
    import sys

    exec_id = f"exec-{uuid.uuid4().hex[:12]}"
    observed: dict = {}
    _install_run_job_stubs(monkeypatch, observed)

    def _failing_run(self, *_a, **_kw):
        observed["during_run"] = agent_during_run(observed)
        # Register into the registry the run actually published (the canonical
        # owner generates its own execution id), like the driver would.
        run_exec_id = Path(observed["during_run"]["containment_dir"]).name
        unit = containment.new_unit_name(run_exec_id)
        containment.register_unit(run_exec_id, unit)
        _set_unit(fake_systemctl, unit, "active")
        observed["run_exec_id"] = run_exec_id
        raise RuntimeError("conversation blew up after spawning roles")

    fake_mod = sys.modules["run_agent"]
    fake_mod.AIAgent = type("FakeAgent", (), {
        "__init__": lambda self, **kw: None,
        "run_conversation": _failing_run,
        "get_activity_summary": lambda self: {"seconds_since_activity": 0.0},
    })

    success, _output, _response, error = sched.run_job(_job(execution_id=exec_id))
    assert success is False and error
    run_exec_id = observed["run_exec_id"]
    assert not (tmp_path / "home" / "cron" / "containment" / run_exec_id).exists()


@_systemctl_stub
@pytest.mark.usefixtures("cron_owner")
def test_run_job_deferred_teardown_waits_for_the_conversation_thread(
        fake_systemctl, tmp_path, monkeypatch):
    """Inactivity timeout abandons the thread while the conversation still runs.
    Containment teardown must NOT kill in-flight roles at run_job's verdict —
    it must fire when the conversation thread actually finishes."""
    import threading

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from cron import containment
    import cron.scheduler as sched
    import sys

    exec_id = f"exec-{uuid.uuid4().hex[:12]}"
    release = threading.Event()
    observed: dict = {"end_calls": []}
    _install_run_job_stubs(monkeypatch, observed)
    # The stub installer pins HERMES_CRON_TIMEOUT=0 (unlimited) — re-arm the
    # inactivity limit AFTER it so the watchdog fires ~immediately.
    monkeypatch.setattr(sched, "_cron_inactivity_seconds", lambda: 1.0)
    _real_end = containment.end

    def _observing_end(execution_id):
        observed["end_calls"].append(execution_id)
        return _real_end(execution_id)

    monkeypatch.setattr(containment, "end", _observing_end)

    def _blocking_run(self, *_a, **_kw):
        # Register a scope like the driver would (into the published registry),
        # then hang the conversation.
        observed["during_run"] = agent_during_run(observed)
        run_exec_id = Path(observed["during_run"]["containment_dir"]).name
        unit = containment.new_unit_name(run_exec_id)
        containment.register_unit(run_exec_id, unit)
        _set_unit(fake_systemctl, unit, "active")
        observed["registered_unit"] = unit
        observed["run_exec_id"] = run_exec_id
        release.wait(timeout=30)
        return {"final_response": "late completion", "messages": []}

    fake_mod = sys.modules["run_agent"]
    fake_mod.AIAgent = type("BlockingAgent", (), {
        "__init__": lambda self, **kw: None,
        "run_conversation": _blocking_run,
        "get_activity_summary": lambda self: {"seconds_since_activity": 999.0},
        "interrupt": lambda self, *a, **k: None,
    })

    success, _output, _response, error = sched.run_job(_job(execution_id=exec_id))
    assert success is False and "idle" in (error or "").lower()
    run_exec_id = observed["run_exec_id"]
    # run_job returned while the conversation still hangs: teardown must NOT have run.
    assert observed["end_calls"] == []
    assert (tmp_path / "home" / "cron" / "containment" / run_exec_id).exists()

    release.set()
    deadline = __import__("time").monotonic() + 20
    while not observed["end_calls"] and __import__("time").monotonic() < deadline:
        threading.Event().wait(0.2)
    assert observed["end_calls"] == [run_exec_id], "teardown must fire at the conversation's real end"
    assert not (tmp_path / "home" / "cron" / "containment" / run_exec_id).exists()
