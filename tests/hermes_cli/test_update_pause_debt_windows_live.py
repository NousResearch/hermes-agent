"""Native Windows restart-debt boundaries of the update pause (#132338 R8): real files held the way
Windows readers hold them, a real SCM service, real processes and taskkill. Nothing here is mocked.

- A completed obligation stays completed when Windows refuses to delete one of its copies
  (a reader without FILE_SHARE_DELETE): the copy must never execute again.
- An SCM service reporting ``running`` retires its debt only once the gateway it supervises is ready.
- The pause's force-kill is authorized by the birth discovered before the drain, not one re-read
  after it (a PID reused during the drain reads its replacement's own birth).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import uuid
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

from hermes_cli import update_pause_record as pause_record

REPO = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.platforms("windows"), pytest.mark.live_system_guard_bypass]


def _hold(path: Path) -> subprocess.Popen:
    """Another process reading *path* with a plain ``open`` (shares read/write, not delete)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; f = open(sys.argv[1], 'rb'); print('held', flush=True); sys.stdin.read()",
         str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    assert proc.stdout.readline().strip() == "held"
    return proc


def _release(proc: subprocess.Popen) -> None:
    proc.stdin.close()
    proc.wait(timeout=30)


def _record_files(home: Path) -> list[str]:
    return sorted(p.name for p in home.glob(pause_record.RECORD_STEM + "*") if p.suffix in (".json", ".claim"))


def _orphan(profiles: dict) -> dict:
    token = {"pause_id": uuid.uuid4().hex, "resume_needed": True, "profiles": profiles}
    pause_record.write(token, owner=pause_record.UNOWNED)
    return token


_UPDATER = """
import sys
from hermes_cli import update_pause_record as r
token = {"pause_id": sys.argv[1], "resume_needed": True, "profiles": {"default": 4242}}
r.write(token, owner=r.identity())
print("written", flush=True)
sys.stdin.readline()
r.discharge(token)
print("discharged", flush=True)
"""


@pytest.mark.parametrize("carrier", ["claim", "record"])
def test_a_completed_obligation_never_runs_again_from_a_copy_windows_kept(tmp_path, monkeypatch, carrier):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    src = pause_record.record_path()
    if carrier == "claim":  # a recovering launch claims an orphan and finishes its resume
        _orphan({"default": 4242, "work": 4343})
        holder = _hold(src)
        try:
            won = pause_record.claim(src)
            assert won is not None and src.exists(), "premise: Windows refused to delete the claimed source"
            pause_record._hand_back(won[0], won[1], {**won[1]["token"], "resume_needed": False, "profiles": {}},
                                    {"profiles": {}, "unmapped": []})
            assert pause_record.orphans() == [], "the completed set is owed again from the copy Windows kept"
        finally:
            _release(holder)
    else:  # the update that paused them resumes them all and discharges its record, then exits
        updater = subprocess.Popen([sys.executable, "-c", textwrap.dedent(_UPDATER), uuid.uuid4().hex],
                                   cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                                   env={**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": str(REPO)})
        assert updater.stdout.readline().strip() == "written"
        holder = _hold(src)
        try:
            out, _ = updater.communicate("go\n", timeout=60)
            assert "discharged" in out and src.exists(), "premise: Windows refused to delete the discharged record"
            assert pause_record.orphans() == [], "a dead updater's discharged record is owed again"
        finally:
            _release(holder)
    pause_record.retire_redundant()  # the next launch, once nothing holds the copy
    assert _record_files(tmp_path) == []
    assert pause_record.orphans() == []
    assert not src.with_suffix(".retired").exists(), "the retired list outlived every copy it guarded"


def test_a_partly_resumed_claim_is_what_the_next_launch_resumes_not_the_copy_windows_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    src = pause_record.record_path()
    _orphan({"default": 4242, "work": 4343})
    holder = _hold(src)
    try:
        won = pause_record.claim(src)
        assert won is not None and src.exists(), "premise: Windows refused to delete the claimed source"
        # default came back; work did not: the claim goes back unowned still owing work only.
        pause_record._hand_back(won[0], won[1], {**won[1]["token"], "profiles": {"work": 4343}},
                                {"profiles": {}, "unmapped": []})
        owed = [body["token"]["profiles"] for _src, body in pause_record.orphans()]
    finally:
        _release(holder)
    assert owed == [{"work": 4343}], "the next launch would restart a gateway this launch already restarted"


# --- SCM: a running wrapper is not a ready gateway ---------------------------------------------

_WRAPPER_CS = r"""
using System;
using System.Diagnostics;
using System.IO;
using System.ServiceProcess;

public class Wrapper : ServiceBase {
    Process child;
    protected override void OnStart(string[] args) {
        string[] spec = File.ReadAllLines(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "child.txt"));
        ProcessStartInfo psi = new ProcessStartInfo(spec[0], spec[1]);
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        psi.WorkingDirectory = spec[2];
        for (int i = 3; i < spec.Length; i++) {
            int eq = spec[i].IndexOf('=');
            if (eq > 0) psi.EnvironmentVariables[spec[i].Substring(0, eq)] = spec[i].Substring(eq + 1);
        }
        child = Process.Start(psi);
    }
    public static void Main() { ServiceBase.Run(new Wrapper()); }
}
"""


@pytest.fixture(scope="module")
def service_wrapper(tmp_path_factory) -> Path:
    """A real SCM service executable that starts one configured child process, like a service wrapper."""
    windir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Microsoft.NET"
    csc = next((c for c in sorted(windir.glob("Framework*/v4.*/csc.exe"), reverse=True)), None)
    assert csc is not None, f"no .NET Framework C# compiler under {windir}"
    out = tmp_path_factory.mktemp("scm-wrapper")
    (out / "wrapper.cs").write_text(_WRAPPER_CS, encoding="utf-8")
    subprocess.run([str(csc), "/nologo", "/target:exe", "/reference:System.ServiceProcess.dll",
                    f"/out:{out / 'wrapper.exe'}", str(out / "wrapper.cs")], check=True, capture_output=True)
    return out / "wrapper.exe"


def _sc(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["sc.exe", *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          check=False)


@pytest.mark.parametrize("gateway", ["absent", "ready"])
def test_a_running_service_owes_its_gateway_until_that_gateway_is_ready(tmp_path, monkeypatch, service_wrapper, gateway):
    from hermes_cli import update_cmd_windows as w

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    bin_dir = tmp_path / "svc"
    bin_dir.mkdir()
    exe = bin_dir / "wrapper.exe"
    exe.write_bytes(service_wrapper.read_bytes())
    child = ([sys.executable, "-c \"import time; time.sleep(900)\""] if gateway == "absent"
             else [sys.executable, "-m hermes_cli.main gateway run"])
    (bin_dir / "child.txt").write_text("\n".join([*child, str(REPO), f"HERMES_HOME={home}", f"PYTHONPATH={REPO}",
                                                  "PYTHONIOENCODING=utf-8"]) + "\n", encoding="utf-8")
    name = f"hermes-r8-debt-{gateway}-{os.getpid()}"
    created = _sc("create", name, "binPath=", f'"{exe}"', "start=", "demand")
    assert created.returncode == 0, created.stdout + created.stderr
    if gateway == "absent":  # the budget only bounds how long the miss takes to report
        monkeypatch.setattr(w, "_SERVICE_READY_TIMEOUT_S", 20.0, raising=False)
    token = {"services": [name], "expected_services": [name], "restarted_services": [], "service_profiles": {name: "default"}}
    try:
        if gateway == "absent":
            with pytest.raises(RuntimeError):
                w._resume_windows_services(token)
            assert token["services"] == [name], "SCM 'running' retired the debt of a gateway that never started"
        else:
            try:
                w._resume_windows_services(token)
            except RuntimeError as exc:
                pytest.fail(f"{exc}\n{_logs(home)}")
            assert token["services"] == [] and token["restarted_services"] == [name]
    finally:
        pid = 0
        with suppress(Exception):
            pid = int(psutil.win_service_get(name).pid() or 0)  # type: ignore[attr-defined]
        if pid:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False)
        _sc("delete", name)


_RECOVERING_LAUNCH = """
import json, time
from hermes_cli import update_pause_record as r
started = time.monotonic()
r.recover(["status"])
print(json.dumps({"seconds": time.monotonic() - started,
                  "owed": [body["token"].get("services") for _src, body in r.orphans()]}))
"""


def test_a_launch_recovering_an_unready_service_keeps_the_debt_without_stalling(tmp_path, monkeypatch, service_wrapper):
    """Two CLI launches in a row recover an orphaned pause whose SCM service runs but whose gateway
    never comes up: the debt stays owed, and neither launch waits out the update's full readiness budget."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    bin_dir = tmp_path / "svc"
    bin_dir.mkdir()
    exe = bin_dir / "wrapper.exe"
    exe.write_bytes(service_wrapper.read_bytes())
    (bin_dir / "child.txt").write_text("\n".join([sys.executable, "-c \"import time; time.sleep(900)\"", str(REPO)]) + "\n",
                                       encoding="utf-8")
    name = f"hermes-r9-recover-{os.getpid()}"
    created = _sc("create", name, "binPath=", f'"{exe}"', "start=", "demand")
    assert created.returncode == 0, created.stdout + created.stderr
    # The record a killed update left; the tree gate passes (HEAD and tracked changes as stamped).
    pause_record.write(pause_record.stamp_tree({
        "resume_needed": True, "services": [name], "expected_services": [name], "restarted_services": [],
        "service_profiles": {name: "default"}}), owner=pause_record.UNOWNED)
    try:
        launches = []
        for _ in range(2):
            done = subprocess.run([sys.executable, "-c", _RECOVERING_LAUNCH], cwd=REPO, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=300,
                                  env={**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO)}, check=False)
            assert done.returncode == 0, done.stdout + done.stderr
            launches.append(json.loads(done.stdout.strip().splitlines()[-1]))
    finally:
        pid = 0
        with suppress(Exception):
            pid = int(psutil.win_service_get(name).pid() or 0)  # type: ignore[attr-defined]
        if pid:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False)
        _sc("delete", name)
    assert [launch["owed"] for launch in launches] == [[[name]], [[name]]], f"the unready service's debt was lost: {launches}"
    first, second = (launch["seconds"] for launch in launches)
    assert first < 40, f"the first launch stalled {first:.0f}s on the service's readiness"
    assert second < 20, f"the next launch stalled {second:.0f}s on a service it had just found unready"


def _logs(home: Path) -> str:
    return "\n".join(f"--- {p}\n{p.read_text(encoding='utf-8-sig', errors='replace')[-3000:]}"
                     for p in sorted((home / "logs").glob("*.log")))


# --- force-kill identity ------------------------------------------------------------------------

@pytest.mark.parametrize("discovered", ["this process", "an earlier process at this pid"])
def test_a_pause_force_kills_only_the_process_it_discovered(tmp_path, monkeypatch, discovered):
    from hermes_cli import update_cmd_windows as w
    from hermes_cli.gateway import ProfileGatewayProcess

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RESTART_DRAIN_TIMEOUT", "1")
    # Never answers its pause, so it survives the drain and reaches the force-kill.
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], stdin=subprocess.DEVNULL)
    try:
        born = psutil.Process(victim.pid).create_time()
        # The gateway discovered at this PID: this very process, or one born earlier that exited
        # during the drain while Windows handed its PID to the process running now.
        seen = born if discovered == "this process" else born - 5.0
        found = ProfileGatewayProcess(profile="r8kill", path=home, pid=victim.pid, create_time=seen)
        w._stop_windows_gateways([victim.pid], {victim.pid: found}, set(), [], [])
        deadline = time.monotonic() + 15
        while victim.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if discovered == "this process":
            assert victim.poll() is not None, "the discovered gateway survived its force-kill"
        else:
            assert victim.poll() is None, "force-kill took a process born after discovery (a reused PID)"
    finally:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(victim.pid)], capture_output=True, check=False)
        victim.wait(timeout=30)
