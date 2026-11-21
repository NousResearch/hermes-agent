"""POSIX pre-swap gateway pause: what survives a killed updater, and what a paused update asks.

Real processes and a real record; the live stop/restart of real gateways is proven by
``tests/e2e/core/upgrade/git/test_hostile_pause.py`` (a real ``hermes update`` in a sandbox).
"""

from __future__ import annotations

import json
import os
import plistlib
import signal
import subprocess
import sys
import threading
import time

import pytest

from tests.hermes_cli.test_update_pause_record import _child, _reap_children

_UNIT = {"kind": "systemd", "scope": "user", "unit": "hermes-gateway-p2probe.service", "pid": 4242}


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
def test_a_killed_updaters_supervised_units_are_adopted_by_the_next_update(tmp_path):
    """The updater stopped a systemd unit and was SIGKILLed: the next ``hermes update`` must own
    restarting that unit (the record's only debt), never drop it."""
    owner = _child(f"""
        import time
        from hermes_cli import update_pause_record as r
        r.write(r.stamp_tree({{"platform": "posix", "resume_needed": True, "posix_units": [{_UNIT!r}]}}),
                owner=r.identity())
        print("written", flush=True)
        time.sleep(120)
    """, env={"HERMES_HOME": str(tmp_path)})
    assert owner.stdout.readline().strip() == "written"
    owner.send_signal(signal.SIGKILL)  # windows-footgun: ok — module skips on Windows
    owner.wait(timeout=10)

    nxt = _child("""
        import json
        from hermes_cli import update_pause_record as r
        adopted, claims = r.adopt_orphans()
        token = r.record_pause({"platform": "posix", "resume_needed": True, "unmapped": []}, adopted, claims)
        print(json.dumps({"units": token.get("posix_units"), "platform": token.get("platform")}))
    """, env={"HERMES_HOME": str(tmp_path)})
    out, _ = nxt.communicate(timeout=60)
    got = json.loads(out.strip().splitlines()[-1])
    assert got == {"units": [_UNIT], "platform": "posix"}, f"the adopted unit was lost: {got}"


@pytest.mark.platforms("posix")
def test_a_prompt_while_the_gateways_are_paused_takes_its_default_at_once(tmp_path, monkeypatch):
    """``/update`` relays prompts through the gateway; while this update holds it stopped nobody can
    answer, so the prompt must not park the update (and the downtime) for its 300 s timeout."""
    from hermes_cli import update_cmd, update_cmd_posix_pause

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(update_cmd_posix_pause, "_RUN", {"platform": "posix", "posix_stopped": True, "resume_needed": True})
    got: list[str] = []
    t = threading.Thread(target=lambda: got.append(update_cmd._gateway_prompt("Restore local changes now?", "n")),
                         daemon=True)
    t.start()
    t.join(timeout=10)
    assert got == ["n"], "a prompt waited for a gateway this update had stopped"
    assert not (tmp_path / ".update_prompt.json").exists()


_KILLED_AFTER_RECORDING = """
    import os, signal, sys
    from hermes_cli import update_cmd_posix_pause as m, update_pause_record as r
    pid, kind, asked, home = int(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", sys.argv[4]
    entry = {"pid": pid, "home": home}
    if kind == "systemd":
        entry.update(kind="systemd", scope="user", unit="hermes-gateway-p2probe.service", cgroup=None)
    else:
        entry["argv"] = ["hermes", "gateway", "run"]
    # Discovery is the seam (no real unit or install here); recording is the code under test.
    m._discover_systemd = lambda: [entry] if kind == "systemd" else []
    m._discover_launchd = lambda: []
    m._left_running = lambda units, jobs: (units, jobs, [])
    m._discover_bare = lambda service_pids: ([] if kind == "systemd" else [entry], [])
    def killed(token, entries):  # the update dies mid-stop, after (or before) asking this gateway
        if asked:
            r.mark_stop_sent(token, pid)
        os.kill(os.getpid(), signal.SIGKILL)  # windows-footgun: ok — POSIX-only test
    m._stop_gateways = killed
    m._pause(m._empty_token())
"""


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the child imports the pause module; nothing runs `hermes update`
@pytest.mark.parametrize("asked", [False, True], ids=["never-asked", "asked"])
@pytest.mark.parametrize("kind", ["bare", "systemd"])
def test_a_killed_updates_record_owes_only_what_it_stopped(tmp_path, kind, asked):
    """The update recorded a live gateway, then died. One it never asked is still serving: the next
    launch must not relaunch a twin. One it asked may still be draining: held until it exits, never
    relaunched beside it. Both hold for bare and supervised entries alike, judged by the record's
    durable identity of the real process."""
    gateway = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                               stdin=subprocess.DEVNULL)
    try:
        owner = _child(_KILLED_AFTER_RECORDING, str(gateway.pid), kind, "1" if asked else "0", str(tmp_path),
                       env={"HERMES_HOME": str(tmp_path)})
        owner.wait(timeout=60)
        assert owner.returncode == -signal.SIGKILL, "premise: the update was not killed mid-stop"  # windows-footgun: ok — POSIX-only test
        probe = _child("""
            import json
            from hermes_cli import update_pause_record as r
            found = r.orphans()
            token = r.drop_never_stopped(dict(found[0][1]["token"])) if found else {}
            draining = r.split_draining(token) if found else {}
            pids = lambda t: sorted(e["pid"] for key in ("unmapped", "posix_units") for e in t.get(key) or [])
            print(json.dumps({"recorded": bool(found), "relaunched": pids(token), "held": pids(draining)}))
        """, env={"HERMES_HOME": str(tmp_path)})
        out, _ = probe.communicate(timeout=60)
        got = json.loads(out.strip().splitlines()[-1])
        assert got["recorded"], "premise: the killed update left no record"
        assert got["relaunched"] == [], f"a still-running gateway would be relaunched beside itself: {got}"
        assert got["held"] == ([gateway.pid] if asked else []), f"wrong restart debt: {got}"
    finally:
        gateway.kill()
        gateway.wait(timeout=10)


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes through a read-only dir")
@pytest.mark.parametrize("prior", [{}, {"pause_id": "adopted-1", "resume_needed": True}], ids=["fresh", "adopted"])
def test_a_record_that_cannot_be_written_leaves_the_update_unpaused_and_ungated(tmp_path, monkeypatch, prior):
    """No durable record, no stop. This attempt must leave nothing that arms the checkout's tree
    gate (its required record write would then refuse the update), and an adopted obligation must
    survive exactly as it was."""
    from hermes_cli import update_cmd_posix_pause as m
    gateway = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = {"pid": gateway.pid, "argv": ["hermes", "gateway", "run"], "home": str(home)}
    monkeypatch.setattr(m, "_discover_systemd", list)
    monkeypatch.setattr(m, "_discover_launchd", list)
    monkeypatch.setattr(m, "_discover_bare", lambda service_pids: ([dict(entry)], []))
    monkeypatch.setattr(m, "_stop_gateways", lambda *a: pytest.fail("a gateway was stopped without a record"))
    token = {**m._empty_token(), **prior}
    before = json.loads(json.dumps(token))
    home.chmod(0o500)
    try:
        assert m._pause(token) is None
    finally:
        home.chmod(0o700)
        gateway.kill()
        gateway.wait(timeout=10)
    assert token == before, f"the failed attempt left state behind: {token}"


@pytest.mark.platforms("posix")
def test_a_gateway_still_working_past_the_wait_is_left_running_with_its_restart_pending(tmp_path, monkeypatch,
                                                                                        capsys):
    """Owner ruling: a non-forced update never stops live work. Both gateways took the drain and are
    still busy when the wait runs out: neither is stopped, SIGKILLed or relaunched beside itself by
    the rollback, and the update continues unpaused naming what each is still running."""
    from hermes_cli import update_cmd_posix_pause as m
    gateways = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
                for _ in range(2)]
    home = tmp_path / "home"
    home.mkdir()
    (home / "gateway_state.json").write_text(json.dumps({"active_work": [
        {"kind": "chat", "session": "agent:main:telegram:dm:1", "elapsed_s": 75}]}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    unit = {"kind": "systemd", "scope": "user", "unit": "hermes-gateway-p2probe.service", "pid": gateways[0].pid,
            "cgroup": None, "home": str(home)}
    bare = {"pid": gateways[1].pid, "argv": ["hermes", "gateway", "run"], "home": str(home)}
    touched = []
    monkeypatch.setattr(m, "_discover_systemd", lambda: [dict(unit)])
    monkeypatch.setattr(m, "_discover_launchd", list)
    monkeypatch.setattr(m, "_left_running", lambda units, jobs: (units, jobs, []))
    monkeypatch.setattr(m, "_discover_bare", lambda service_pids: ([dict(bare)], []))
    monkeypatch.setattr(m, "_ask_to_drain", lambda entry: True)
    monkeypatch.setattr(m, "_drain_budget", lambda: 0.5)
    for name in ("_stop_unit", "_stop_job", "_start_unit", "_start_job", "_relaunch_bare"):
        monkeypatch.setattr(m, name, lambda *a, _n=name, **k: touched.append(_n) or 0)
    try:
        assert m._pause(m._empty_token()) is None
        assert [g.poll() for g in gateways] == [None, None], "a busy gateway was killed"
    finally:
        for g in gateways:
            g.kill()
            g.wait(timeout=10)
    assert touched == [], f"a busy gateway was stopped or a second copy started: {touched}"
    out = capsys.readouterr().out
    assert out.count("still running live work (chat turn agent:main:telegram:dm:1") == 2, out
    assert "`hermes update --force` stops it now" in out


@pytest.mark.platforms("posix")
def test_a_replayed_gateway_keeps_its_own_exported_token_and_no_one_elses(tmp_path):
    """A named profile's hand-started gateway, replayed by an update launched under that profile,
    starts as the restart watcher started it: with the launcher's environment (an exported bot
    token may be its only credential). Any other profile, and the multiplexing root, never get it."""
    root = tmp_path / ".hermes"
    for name in ("alpha", "beta"):
        (root / "profiles" / name).mkdir(parents=True)
    probe = _child("""
        import json, sys
        from hermes_cli.update_cmd_posix_pause import _replay_env
        root = sys.argv[1]
        print(json.dumps({who: _replay_env(home).get("TELEGRAM_BOT_TOKEN") for who, home in (
            ("alpha", root + "/profiles/alpha"), ("beta", root + "/profiles/beta"), ("root", root))}))
    """, str(root), env={"HOME": str(tmp_path), "HERMES_HOME": str(root / "profiles" / "alpha"),
                         "TELEGRAM_BOT_TOKEN": "exported-alpha-token"})
    out, _ = probe.communicate(timeout=60)
    got = json.loads(out.strip().splitlines()[-1])
    assert got == {"alpha": "exported-alpha-token", "beta": None, "root": None}, got


@pytest.mark.platforms("posix")
def test_unknown_cgroup_membership_never_authorizes_stopping_the_unit():
    """A unit whose stop could kill this update stays running unless the update is PROVEN outside
    its cgroup; unreadable membership is unknown, never outside."""
    from hermes_cli.update_cmd_posix_pause import _escape_cgroup
    assert _escape_cgroup({"unit": "hermes-gateway-p2probe.service", "pid": 4242, "cgroup": None}) is False


@pytest.mark.platforms("macos")
@pytest.mark.live_system_guard_bypass
def test_a_launchd_job_is_paused_through_launchd_and_restarted_through_it(tmp_path):
    """A KeepAlive job merely killed is respawned by launchd on the old code mid-update; the pause
    boots it out (no respawn) and the restart bootstraps the same plist (a fresh PID)."""
    from hermes_cli.gateway import _launchd_print_service_pid
    from hermes_cli.update_cmd_posix_pause import _alive, _start_job, _stop_job
    label = f"ai.hermes.p2probe-{os.getpid()}"
    plist = tmp_path / f"{label}.plist"
    plist.write_bytes(plistlib.dumps({"Label": label, "ProgramArguments": ["/bin/sleep", "600"],
                                      "RunAtLoad": True, "KeepAlive": True}))
    domain = f"gui/{os.getuid()}"  # windows-footgun: ok — macOS-only test (platforms("macos"))
    if subprocess.run(["launchctl", "print", domain], capture_output=True, check=False).returncode:
        domain = f"user/{os.getuid()}"  # windows-footgun: ok — macOS-only test (platforms("macos"))

    def pid() -> int | None:
        return _launchd_print_service_pid(domain, label)[1]

    subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True, timeout=30)
    try:
        deadline = time.monotonic() + 15
        while not pid() and time.monotonic() < deadline:
            time.sleep(0.2)
        first = pid()
        assert first, "premise: launchd never started the probe job"
        job = {"kind": "launchd", "label": label, "domain": domain, "plist": str(plist), "pid": first}
        _stop_job(job)
        time.sleep(3.0)  # KeepAlive would have respawned a merely-killed job by now
        assert not _alive(first, None) and not pid(), "the paused job is running again under launchd"
        _start_job(job)
        assert pid() and pid() != first, "the restart did not bring the job back under launchd"
    finally:
        subprocess.run(["launchctl", "bootout", f"{domain}/{label}"], capture_output=True, check=False)


def _discover_probe_job(monkeypatch, tmp_path, plist_body: dict, pid: int) -> dict:
    """``_discover_launchd`` over one job whose label, PID and plist are this test's."""
    from hermes_cli import gateway, update_fleet_scope
    from hermes_cli.update_cmd_posix_pause import _discover_launchd
    label = "ai.hermes.gateway-p2probe"
    (tmp_path / f"{label}.plist").write_bytes(plistlib.dumps({"Label": label, **plist_body}))
    monkeypatch.setattr(gateway, "get_launchd_plist_path", lambda: tmp_path / "ai.hermes.gateway.plist")
    monkeypatch.setattr(gateway, "launchd_gateway_labels_for_install", lambda: [label])
    monkeypatch.setattr(gateway, "legacy_launchd_labels_for_install", lambda exclude=(): [])
    monkeypatch.setattr(gateway, "_locate_launchd_gateway_service", lambda _label: ("gui/0", pid))
    monkeypatch.setattr(update_fleet_scope, "launchd_label_skip_notice", lambda _label, scope=None: None)
    jobs = _discover_launchd()
    assert len(jobs) == 1
    return jobs[0]


@pytest.mark.platforms("macos")
def test_a_profile_launchd_job_drains_on_the_home_its_plist_pins(monkeypatch, tmp_path):
    """The job's PID is the ``osascript`` wrapper, whose environment macOS reads back empty, so the
    PID alone cannot name the home: the drain request must go to the home the plist pins."""
    pinned = str(tmp_path / "profiles" / "p2probe")
    wrapper = subprocess.Popen(["/usr/bin/osascript", "-l", "JavaScript", "-e", "delay(600)"],
                               env={**os.environ, "HERMES_HOME": pinned})
    try:
        job = _discover_probe_job(monkeypatch, tmp_path, {"EnvironmentVariables": {"HERMES_HOME": pinned}}, wrapper.pid)
    finally:
        wrapper.kill()
        wrapper.wait()
    assert job["home"] == pinned


@pytest.mark.platforms("macos")
def test_a_launchd_job_with_no_pinned_home_falls_back_to_its_process(monkeypatch, tmp_path):
    """A hand-written plist may pin nothing: the running process is then the only evidence."""
    from hermes_cli import update_fleet_scope
    monkeypatch.setattr(update_fleet_scope, "gateway_pid_home", lambda pid: f"/homes/{pid}")
    assert _discover_probe_job(monkeypatch, tmp_path, {}, 4242)["home"] == "/homes/4242"


@pytest.mark.platforms("posix")
def test_a_gateway_left_pending_by_the_pause_stays_pending_in_the_completion_process(tmp_path, monkeypatch, capsys):
    """The fleet restart runs in a fresh completion interpreter: the gateways the pause left running
    must reach it through the request's token, or it waits the full drain on the same busy gateway
    again. Restored by (pid, start time): a PID now held by another process is dropped."""
    from gateway.status import get_process_start_time
    from hermes_cli import update_cmd_posix_pause as m
    busy, recycled = (subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
                      for _ in range(2))
    try:
        entries = [{"pid": p.pid, "ct": get_process_start_time(p.pid), "home": str(tmp_path)} for p in (busy, recycled)]
        entries[1]["ct"] += 1  # the pause saw an earlier process under this PID
        token = m._empty_token()
        m._leave_running(token, entries)
        request = json.dumps({"windows_resume": token})  # what the completion child reads
        probe = _child("""
            import json, sys
            from hermes_cli import update_cmd_posix_pause as m
            m.adopt_left_pending(json.loads(sys.argv[1])["windows_resume"])
            print(json.dumps(sorted(pid for pid in m.left_pending() if m.is_left_pending(pid))))
        """, request, env={"HERMES_HOME": str(tmp_path)})
        out, _ = probe.communicate(timeout=60)
        assert json.loads(out.strip().splitlines()[-1]) == [busy.pid], out
    finally:
        for p in (busy, recycled):
            p.kill()
            p.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_a_pid_recycled_during_the_drain_wait_is_never_a_pending_restart():
    """Liveness is (pid, start time): a PID that now belongs to another process held no restart, so
    the caller's escalation stands instead of leaving it running as "restart pending"."""
    from gateway.status import get_process_start_time
    from hermes_cli.update_cmd_drain_report import restart_left_pending
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
    try:
        born = get_process_start_time(child.pid)
        assert restart_left_pending(child.pid, start_time=born) is True, "premise: the signalled gateway is pending"
        assert restart_left_pending(child.pid, start_time=born + 1) is False
        assert restart_left_pending(child.pid, start_time=None) is False
    finally:
        child.kill()
        child.wait(timeout=10)
