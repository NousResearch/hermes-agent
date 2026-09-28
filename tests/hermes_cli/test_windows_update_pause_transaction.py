"""Windows pause failure contracts through real upstream module imports.

The host OS is not forged. Process/socket/SCM effects are fakes; these are
native-host unit/regression tests, not live-service end-to-end evidence.
"""
import copy
import types

import pytest


pytestmark = pytest.mark.platforms("windows")


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from gateway import status as status_mod, control_socket
    from hermes_cli import main as cli_main, gateway as gateway_mod, update_cmd_windows as windows
    ns = vars(windows)
    trace = []
    state = {"alpha": "running", "beta": "running", "unrelated": "running"}
    procs = {101: types.SimpleNamespace(pid=101, profile="alpha", path=tmp_path / "alpha"),
             202: types.SimpleNamespace(pid=202, profile="beta", path=tmp_path / "beta")}
    for proc in procs.values():
        proc.path.mkdir()
    r = types.SimpleNamespace(ns=ns, trace=trace, state=state, procs=procs,
                              marker_failure=None, socket_timeout=False, recovery_failure=False,
                              discover_failure=False, launcher_failure=False, wait_failure=False,
                              record_failure=False, kill_failure=False, services=[], service_pids=set(),
                              running=[101, 202], launchers=[], survivors=set(), tokens=[],
                              process_times={101: 1.01, 202: 2.02, 303: 3.03, 404: 4.04},
                              mutate_after_pause=None, passive_plan={})

    def discover():
        trace.append(("discover",))
        if r.discover_failure:
            raise RuntimeError("supervisor inspection denied")
        return procs, r.services, r.service_pids, r.running

    def marker(path, pid):
        trace.append(("marker", path.name, pid))
        if pid == r.marker_failure:
            raise RuntimeError("second marker failure")
        (path / ".gateway-planned-stop.json").write_text(str(pid), encoding="utf-8")
        return True

    def pause(path):
        trace.append(("pause", path.name))
        state[path.name] = "paused"
        if r.mutate_after_pause:
            r.mutate_after_pause()
        if r.socket_timeout:
            raise TimeoutError("ACK lost after pause")
        return {"pausing": True, "drain_timeout": 2}

    def launchers(pids):
        trace.append(("launchers", tuple(pids)))
        if r.launcher_failure:
            raise RuntimeError("launcher discovery failure")
        return list(r.launchers)

    def wait(pids, *, timeout):
        trace.append(("wait", tuple(pids), timeout))
        if r.wait_failure:
            raise RuntimeError("drain observation failure")
        return set(r.survivors)

    def resume(token):
        snapshot = copy.deepcopy(token)
        r.tokens.append(snapshot)
        trace.append(("recover", tuple(sorted((token.get("profiles") or {}).keys()))))
        if r.recovery_failure:
            raise RuntimeError("recovery also failed")
        for profile in (token.get("profiles") or {}):
            state[profile] = "running"
        token["resume_needed"] = False

    def terminate(pid, **kwargs):
        trace.append(("kill", pid, kwargs.get("expected_start_time")))
        if r.kill_failure:
            raise RuntimeError("termination failed")

    def record(token, running):
        trace.append(("cold-plan", tuple(sorted(running))))
        if r.passive_plan:
            token["cold_start_profiles"] = dict(r.passive_plan)
        if r.record_failure:
            raise RuntimeError("cold plan failure")

    def stop_services(services, token, profiles, unmapped):
        trace.append(("SCM", tuple(s.name for s in services)))
        return token

    monkeypatch.setattr(status_mod, "get_process_start_time", lambda pid: r.process_times.get(pid))
    monkeypatch.setattr(status_mod, "terminate_pid", terminate)
    monkeypatch.setattr(control_socket, "pause_gateway_for_update", pause)
    monkeypatch.setattr(gateway_mod, "_capture_gateway_argv", lambda pid: ["python", "-m", "hermes_cli.main", "gateway", "run"])
    monkeypatch.setattr(gateway_mod, "_get_restart_drain_timeout", lambda: 1.0)
    monkeypatch.setattr(cli_main, "_venv_launcher_ancestors", launchers)
    monkeypatch.setattr(cli_main, "_wait_for_windows_update_gateway_exit", wait)
    monkeypatch.setattr(cli_main, "_resume_windows_gateways_after_update", resume)
    for name, fn in {
        "_discover_windows_gateways": discover,
        "_write_update_planned_stop_marker": marker,
        "_record_attested_cold_start_profiles": record,
        "_pause_windows_gateway_services": stop_services,
        "_windows_cold_start_plan": lambda: None,
    }.items():
        monkeypatch.setattr(windows, name, fn)
    r.call = windows._pause_windows_gateways_for_update
    r.facade = cli_main
    return r


def test_launcher_failure_has_no_prior_pause(rig):
    rig.launcher_failure = True
    with pytest.raises(RuntimeError, match="launcher discovery failure"):
        rig.call()
    assert not any(e[0] in {"marker", "pause", "kill", "SCM"} for e in rig.trace)
    assert set(rig.state.values()) == {"running"}


def test_partial_pause_failure_recovers_already_paused_profile(rig):
    rig.marker_failure = 202
    with pytest.raises(RuntimeError, match="second marker failure"):
        rig.call()
    assert rig.state["alpha"] == "running", "alpha was paused and lost before the caller received a token"
    assert rig.tokens and rig.tokens[0]["profiles"]["alpha"] == 101
    assert "unrelated" not in rig.tokens[0]["profiles"]
    assert not any(e[0] in {"kill", "SCM", "checkout", "venv"} for e in rig.trace)


def test_lost_ack_is_retained_as_uncertain_on_abort(rig):
    rig.socket_timeout = True
    rig.wait_failure = True
    with pytest.raises(RuntimeError, match="drain observation failure"):
        rig.call()
    assert rig.tokens, "an uncertain pause must still leave a recovery obligation"
    attempts = rig.tokens[0].get("pause_attempts", {})
    assert attempts["alpha"]["state"] == "unknown"
    assert attempts["alpha"]["pid"] == 101
    assert rig.state["alpha"] == "running"


def test_primary_and_recovery_failures_both_remain_visible(rig):
    rig.wait_failure = True
    rig.recovery_failure = True
    with pytest.raises(RuntimeError) as got:
        rig.call()
    error = got.value
    assert "drain observation failure" in str(error)
    assert "recovery also failed" in str(error)
    assert error.__cause__ is not None
    assert error.windows_gateway_resume["profiles"]["alpha"] == 101
    assert error.windows_gateway_resume["resume_needed"] is True
    assert rig.state["unrelated"] == "running"


@pytest.mark.parametrize("stage", ["wait_failure", "kill_failure", "record_failure"])
def test_later_failure_recovers_only_attempted_profiles(rig, stage):
    rig.running = [101]
    rig.survivors = {101}
    setattr(rig, stage, True)
    with pytest.raises(RuntimeError):
        rig.call()
    assert rig.tokens and set(rig.tokens[0]["profiles"]) == {"alpha"}
    assert rig.state == {"alpha": "running", "beta": "running", "unrelated": "running"}


def test_service_gateway_never_enters_socket_or_launcher_set(rig):
    service = types.SimpleNamespace(name="HermesBeta", gateway_pid=202, service_pid=303,
                                    profile="beta", descendant_identities=((202, 2.02),))
    rig.services = [service]
    rig.service_pids = {202}
    result = rig.call()
    assert ("pause", "beta") not in rig.trace
    assert ("launchers", (101,)) in rig.trace
    assert set(result["profiles"]) == {"alpha"}
    assert not any(e[0] == "kill" and e[1] in {202, 303} for e in rig.trace)


def test_launcher_service_overlap_refuses_before_any_pause(rig):
    rig.services = [types.SimpleNamespace(name="HermesBeta", gateway_pid=202, service_pid=303,
                                          profile="beta", descendant_identities=((202, 2.02),))]
    rig.service_pids = {202}
    rig.launchers = [303]
    with pytest.raises(RuntimeError, match="(?i)(service|SCM|ownership)"):
        rig.call()
    assert not any(e[0] in {"pause", "marker", "kill", "SCM"} for e in rig.trace)


def test_force_stop_uses_the_pre_pause_process_incarnation(rig):
    rig.running = [101]
    rig.survivors = {101}
    rig.mutate_after_pause = lambda: rig.process_times.update({101: 99.0})
    rig.call()
    assert ("kill", 101, 1.01) in rig.trace
    assert ("kill", 101, 99.0) not in rig.trace


def test_supervisor_refusal_is_before_first_effect(rig):
    rig.discover_failure = True
    with pytest.raises(RuntimeError, match="supervisor inspection denied"):
        rig.call()
    assert rig.trace == [("discover",)]


def test_positive_ack_preserves_active_turn_drain_budget(rig):
    rig.call()
    waits = [e for e in rig.trace if e[0] == "wait"]
    assert waits[0][2] >= 12.0


def test_success_returns_profiles_for_the_existing_resume_owner(rig):
    result = rig.call()
    assert result["resume_needed"] is True
    assert result["profiles"] == {"alpha": 101, "beta": 202}
    assert not rig.tokens


def test_unknown_process_identity_refuses_before_first_effect(rig):
    rig.process_times[101] = None
    with pytest.raises(RuntimeError, match="identity is unavailable"):
        rig.call()
    assert not any(e[0] in {"pause", "marker", "kill", "SCM"} for e in rig.trace)


def test_unmapped_failure_does_not_register_a_future_kill_target(rig):
    rig.running = [101, 303, 404]
    rig.kill_failure = True
    with pytest.raises(RuntimeError, match="termination failed"):
        rig.call()
    assert rig.tokens
    assert [entry["pid"] for entry in rig.tokens[0]["unmapped"]] == [303]
    assert not any(e[0] == "kill" and e[1] == 404 for e in rig.trace)


def test_abort_does_not_convert_passive_attestation_into_a_start(rig):
    rig.running = [101]
    rig.passive_plan = {"unrelated": "not-this-update"}
    rig.record_failure = True
    with pytest.raises(RuntimeError, match="cold plan failure"):
        rig.call()
    assert rig.tokens
    assert "cold_start_profiles" not in rig.tokens[0]
    assert set(rig.tokens[0]["profiles"]) == {"alpha"}


def test_success_preserves_passive_attested_cold_start_feature(rig):
    rig.passive_plan = {"previously-crashed": "generation-proof"}
    token = rig.call()
    assert token["cold_start_profiles"] == rig.passive_plan
    assert not rig.tokens


def test_discovered_incarnation_change_refuses_before_pause(rig):
    rig.procs[101].create_time = 1.01
    rig.process_times[101] = 99.0
    with pytest.raises(RuntimeError, match="identity changed"):
        rig.call()
    assert not any(e[0] in {"pause", "marker", "kill", "SCM"} for e in rig.trace)


@pytest.fixture
def scm(monkeypatch):
    from gateway import status as status_mod
    from hermes_cli import update_cmd, update_cmd_windows as windows
    ns = vars(windows)
    r = types.SimpleNamespace(states={"HermesAlpha": "running", "HermesBeta": "stopped"}, trace=[],
                              inspect_error=False, uncertain_stop=False, exists=lambda pid: True)
    specs = {
        "HermesAlpha": types.SimpleNamespace(name="HermesAlpha", profile="alpha", service_pid=301,
             gateway_pid=101, service_create_time=3.01, gateway_create_time=1.01,
             descendant_identities=((101, 1.01),)),
        "HermesBeta": types.SimpleNamespace(name="HermesBeta", profile="beta", service_pid=302,
             gateway_pid=202, service_create_time=3.02, gateway_create_time=2.02,
             descendant_identities=((202, 2.02),)),
    }

    def win_service(name):
        if r.inspect_error:
            raise PermissionError("SCM access denied before stop")
        process = types.SimpleNamespace(parents=lambda: [types.SimpleNamespace(pid=301), types.SimpleNamespace(pid=302)])
        psutil = types.SimpleNamespace(Process=lambda pid: process)
        service = types.SimpleNamespace(status=lambda: r.states[name], pid=lambda: specs[name].service_pid)
        return psutil, service

    def control(verb, name, service, settled):
        r.trace.append(("control", verb, name))
        r.states[name] = settled
        if r.uncertain_stop:
            raise TimeoutError("SCM reply lost after stop")

    def restore(name):
        r.trace.append(("restore", name))
        r.states[name] = "running"

    for name, fn in {
        "_win_service": win_service,
        "_sc_exe": control,
        "_poll_until": lambda predicate, *args: predicate(),
        "_original_process_is_alive": lambda *args: False,
        "_process_create_time": lambda psutil, pid, label: {301:3.01,302:3.02,101:1.01,202:2.02}[pid],
        "_resume_windows_gateways_after_update": lambda token: r.trace.append(("ordinary",)),
    }.items():
        monkeypatch.setattr(windows, name, fn)
    monkeypatch.setattr(update_cmd, "_restore_windows_gateway_service", restore)
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: r.exists(pid))
    r.ns = ns
    r.specs = specs
    r.pause = lambda services, token=None: windows._pause_windows_gateway_services(
        [specs[name] for name in services], token or {"resume_needed":True, "profiles":{}, "unmapped":[]},
        (token or {}).get("profiles",{}), [])
    return r


def test_pre_stop_identity_refusal_never_starts_untouched_service(scm):
    with pytest.raises(RuntimeError, match="not stably running"):
        scm.pause(["HermesBeta"])
    assert scm.states["HermesBeta"] == "stopped"
    assert not scm.trace


def test_pre_stop_access_denied_never_restores_unattempted_service(scm):
    scm.inspect_error = True
    with pytest.raises(RuntimeError, match="access denied"):
        scm.pause(["HermesBeta"])
    assert not scm.trace


def test_refusal_on_second_service_restores_only_first_service(scm):
    with pytest.raises(RuntimeError, match="not stably running"):
        scm.pause(["HermesAlpha", "HermesBeta"])
    assert scm.trace == [("control","stop","HermesAlpha"), ("restore","HermesAlpha")]
    assert scm.states == {"HermesAlpha":"running", "HermesBeta":"stopped"}


def test_uncertain_actual_stop_still_restores_current_service(scm):
    scm.uncertain_stop = True
    with pytest.raises(RuntimeError, match="reply lost"):
        scm.pause(["HermesAlpha"])
    assert scm.trace == [("control","stop","HermesAlpha"), ("restore","HermesAlpha")]
    assert scm.states["HermesAlpha"] == "running"


def test_scm_refusal_still_recovers_previously_paused_ordinary_gateways_once(scm):
    token = {"resume_needed":True, "profiles":{"alpha":101}, "unmapped":[]}
    with pytest.raises(RuntimeError, match="not stably running"):
        scm.pause(["HermesBeta"], token)
    assert scm.trace.count(("ordinary",)) == 1


def test_malformed_service_arguments_never_start_unattempted_service(scm):
    scm.specs["HermesBeta"].service_pid = "unreadable"
    with pytest.raises(RuntimeError):
        scm.pause(["HermesBeta"])
    assert not scm.trace


def test_drain_inspection_denied_is_not_reported_dead(scm):
    def denied(pid):
        raise PermissionError("access denied")
    scm.exists = denied
    assert scm.ns["_wait_for_windows_update_gateway_exit"]([101], timeout=0) == {101}


def test_drain_known_dead_stays_excluded(scm):
    scm.exists = lambda pid: False
    assert scm.ns["_wait_for_windows_update_gateway_exit"]([101], timeout=0) == set()


def test_drain_known_live_remains_a_survivor(scm):
    assert scm.ns["_wait_for_windows_update_gateway_exit"]([101], timeout=0) == {101}
