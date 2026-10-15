"""Equal profile names in different Hermes roots must not confer process ownership."""

import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import status
from hermes_cli import dashboard_procs, gateway, main_dashboard


@pytest.fixture
def process_table(tmp_path, monkeypatch):
    own_root, foreign_root = tmp_path / "owned", tmp_path / "foreign"
    home = own_root / "profiles" / "ops"
    home.mkdir(parents=True)
    (foreign_root / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway, "_get_ancestor_pids", lambda: set())
    monkeypatch.setattr(gateway, "_get_service_pids", lambda **k: set())
    monkeypatch.setattr(gateway, "_reaper_candidate_is_supervisor_owned", lambda pid: False)
    monkeypatch.setattr(status, "get_running_pid", lambda **k: None)
    monkeypatch.setattr(status, "_read_pid_record", lambda *a, **k: None)
    monkeypatch.setattr(status, "_read_gateway_lock_record", lambda *a, **k: None)
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: 123456)
    monkeypatch.setattr(dashboard_procs, "_process_age_seconds", lambda pid: 600)
    monkeypatch.setattr(gateway, "_await_gateway_exit", lambda *a, **k: [])
    monkeypatch.setattr(gateway, "_force_kill_survivors", lambda pids: None)
    effects = []
    monkeypatch.setattr(status, "write_planned_stop_marker", lambda pid: effects.append((pid, "marker")))
    monkeypatch.setattr(gateway.os, "kill", lambda pid, sig: effects.append((pid, sig)))
    argv, environments = {}, {}
    monkeypatch.setattr(main_dashboard, "_dashboard_cmdline_for_pid", lambda pid: argv.get(pid))
    monkeypatch.setattr(dashboard_procs, "_pid_environ", lambda pid: environments.get(pid))

    def commands():
        return [(pid, " ".join(args)) for pid, args in argv.items()]

    monkeypatch.setattr(gateway, "_iter_proc_cmdlines", lambda excluded: iter(commands()))
    monkeypatch.setattr(gateway.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="\n".join(f"{pid} {cmd}" for pid, cmd in commands())
    ))
    return own_root, foreign_root, argv, environments, effects


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("flag", [["--profile", "ops"], ["--profile=ops"], ["-p", "ops"]])
def test_reap_requires_matching_home_for_same_named_profile(process_table, flag):
    own_root, foreign_root, argv, environments, effects = process_table
    for pid, root in ((900001, own_root), (900002, foreign_root)):
        argv[pid] = ["python", "-m", "hermes_cli.main", *flag, "gateway", "run"]
        environments[pid] = {"HERMES_HOME": str(root)}
    argv[900003] = ["python", "-m", "hermes_cli.main", *flag, "gateway", "run"]
    # The third process has an unreadable environment: it must be spared.

    assert gateway._scan_gateway_pids(set(), all_profiles=True) == [900001, 900002, 900003]
    if gateway.supports_systemd_services():
        assert not gateway._reap_unsupervised_gateway_orphans(min_age_s=180)
        assert effects == []
    else:
        assert gateway._reap_unsupervised_gateway_orphans(min_age_s=180)
        assert effects == [(900001, "marker"), (900001, signal.SIGTERM)]
    assert gateway._scan_gateway_pids(set()) == [900001]


@pytest.mark.platforms("posix")
def test_explicit_home_distinguishes_equal_profile_names(process_table):
    own_root, foreign_root, argv, _, _ = process_table
    for pid, root in ((900001, own_root), (900002, foreign_root)):
        argv[pid] = [f"HERMES_HOME={root / 'profiles' / 'ops'}", "python", "-m", "hermes_cli.main",
                     "--profile", "ops", "gateway", "run"]
    assert gateway._scan_gateway_pids(set()) == [900001]
    assert gateway._scan_gateway_pids({900001}) == []
