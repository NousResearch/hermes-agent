"""Real local-control and subprocess boundaries, never real service mutations."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

import hermes_dir_acl
from hermes_agent.hermes_cli.gateway_runtime_service import remaining

_REAL_PROCESS_IS_ELEVATED = hermes_dir_acl.process_is_elevated


@pytest.fixture(autouse=True)
def _non_elevated_client(monkeypatch):
    """Every start here is a normal client's. An elevated one is refused before any spawn
    (client_elevated, test_gateway_runtime_elevated_client.py), and hosted Windows runners are elevated."""
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: False)


def test_unmanaged_runtime_drops_session_config_bypass_flags(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_agent.hermes_cli import gateway_runtime_start as start
    from hermes_agent.hermes_cli import gateway_windows

    flags = ('HERMES_SAFE_MODE', 'HERMES_IGNORE_USER_CONFIG')
    for key in flags:
        monkeypatch.setenv(key, '1')
    monkeypatch.setenv('RUNTIME_TEST_SENTINEL', 'retained')
    monkeypatch.setattr(gateway_windows, 'windowless_gateway_restart_spec',
                        lambda command: (command, None, {'RUNTIME_TEST_OVERLAY': '1'}))
    captured = {}
    child = SimpleNamespace(pid=123)
    def spawn(command, **kwargs):
        captured.update(kwargs)
        return child
    monkeypatch.setattr(start.subprocess, 'Popen', spawn)
    assert start.spawn_unmanaged_gateway(tmp_path, deadline=time.monotonic() + 5) is child
    assert all(key not in captured['env'] for key in flags)
    assert captured['env']['RUNTIME_TEST_SENTINEL'] == 'retained'
    assert all(os.environ[key] == '1' for key in flags)


@pytest.mark.platforms("linux")
def test_ensure_waits_for_real_control_owner_without_claiming_pending_is_ready(tmp_path):
    from hermes_agent.gateway.control_socket import GatewayControlServer
    from hermes_agent.hermes_cli import gateway_runtime as runtime

    assert callable(getattr(runtime, "ensure_gateway_runtime", None))

    async def probe():
        home = tmp_path / "profile"
        home.mkdir(mode=0o700)
        payload = {"runtime_protocol": 1, "state": "starting", "instance_id": "peer",
                   "served_profiles": [{"profile_id": str(home), "home": str(home)}]}
        stalled = False
        unstall = threading.Event()  # set at teardown so the stalled handler thread exits
        def identify():
            if stalled:
                unstall.wait(30)  # outlives the caller's deadline: a stalled owner is 'starting', never ready
            return payload
        server = GatewayControlServer(home, verb_handlers={"identify": identify})
        assert await server.start()
        try:
            before = time.monotonic()
            result = await asyncio.to_thread(runtime.ensure_gateway_runtime, home, timeout=3.0)
            assert result.state == "starting" and result.reason_code == "deadline"
            assert time.monotonic() - before < 6
            payload["state"] = "draining"
            result = await asyncio.to_thread(runtime.ensure_gateway_runtime, home, timeout=3.0)
            assert result.state == "draining"
            payload["served_profiles"] = []
            result = await asyncio.to_thread(runtime.ensure_gateway_runtime, home, timeout=3.0)
            assert result.state == "inaccessible" and result.reason_code == "profile_mismatch"
            stalled = True
            result = await asyncio.to_thread(runtime.ensure_gateway_runtime, home, timeout=3.0)
            assert result.state == "starting" and result.reason_code == "deadline"
            assert not (home / "logs").exists()
        finally:
            unstall.set()
            await server.stop()
    asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_installed_service_start_is_nonmutating_and_failed_manager_never_spawns(tmp_path, monkeypatch):
    from hermes_agent.hermes_cli import gateway as gw, gateway_runtime as runtime

    assert callable(getattr(runtime, "ensure_gateway_runtime", None))
    from hermes_agent.hermes_cli.gateway_runtime_service import service_suffix
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    unit = tmp_path / ".config/systemd/user" / f"hermes-gateway-{service_suffix(home)}.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\nExecStart=/owned/inert/gateway\n", encoding="utf-8")
    original = unit.read_bytes()
    calls = tmp_path / "calls.jsonl"
    helper = tmp_path / "inert-supervisor"
    from hermes_agent.hermes_cli.gateway_runtime_service_identity import SYSTEMD_IDENTITY_PROPERTIES
    identity = {key: "" for key in SYSTEMD_IDENTITY_PROPERTIES}
    identity.update(User=str(os.getuid()), DynamicUser="no",
                    Environment=f'"HERMES_HOME={home}"',
                    ExecStart=f'{{ path={sys.executable} ; argv[]={sys.executable} -m hermes_cli.main gateway run ; ignore_errors=no ; }}')
    effective = "\n".join(f"{key}={value}" for key, value in identity.items())
    helper.write_text('#!' + sys.executable + '\nimport json,sys,time\nfrom pathlib import Path\n'
                      + f'p=Path({str(calls)!r})\n'
                      + 'with p.open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                      + 'if (p.parent/"stall").exists(): time.sleep(5)\n'
                      + 'if (p.parent/"fail").exists(): print("private-supervisor-token"); sys.exit(3)\n'
                      + 'if "show" in sys.argv and "system" in sys.argv and (p.parent/"single").exists():\n print("LoadState=not-found")\n'
                      + 'elif "show-environment" in sys.argv: print("")\n'
                      + 'elif "show" in sys.argv:\n print("LoadState=loaded\\nActiveState=inactive\\nSubState=dead\\nUnitFileState=enabled"); print(' + repr(effective) + ')\n'
                      + 'elif "start" in sys.argv: sys.exit(0)\n'
                      + 'else: sys.exit(91)\n', encoding="utf-8")
    helper.chmod(0o700)
    monkeypatch.setattr(gw, "_systemctl_cmd", lambda system=False: [str(helper), "system" if system else "user"])
    result = runtime.ensure_gateway_runtime(home, timeout=3.0)
    # Both scopes claiming this name is a conflict, never permission to choose one.
    assert result.state == "conflict"
    assert unit.read_bytes() == original
    assert not (home / "logs").exists()
    assert all(any(arg in {"show", "show-environment"} for arg in json.loads(line)) for line in calls.read_text().splitlines())
    (tmp_path / "single").touch()
    result = runtime.ensure_gateway_runtime(home, timeout=3.0)
    assert result.state == "starting" and result.reason_code == "deadline"
    assert sum("start" in json.loads(line) for line in calls.read_text().splitlines()) == 1
    assert unit.read_bytes() == original
    assert not (home / "logs").exists()
    (tmp_path / "fail").touch()
    result = runtime.ensure_gateway_runtime(home, timeout=3.0)
    assert result.state == "inaccessible" and result.reason_code == "service_manager_unavailable"
    assert "private-supervisor-token" not in repr(result)
    assert sum("start" in json.loads(line) for line in calls.read_text().splitlines()) == 1
    assert not (home / "logs").exists()
    (tmp_path / "stall").touch()
    before = time.monotonic()
    result = runtime.ensure_gateway_runtime(home, timeout=3.0)
    assert result.reason_code == "deadline"
    assert time.monotonic() - before < 6
    assert sum("start" in json.loads(line) for line in calls.read_text().splitlines()) == 1


@pytest.mark.platforms("linux")
def test_public_ensure_json_deadline_and_invalid_invocation(tmp_path):
    from hermes_agent.hermes_cli import gateway_runtime as runtime
    assert callable(getattr(runtime, "ensure_gateway_runtime", None))
    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    # A LIVE claim (this test process, at its own incarnation) fences; a dead one would not.
    from hermes_agent.hermes_cli.update_lock import process_create_time
    (home / ".hermes-update-in-progress").write_text(
        f"{os.getpid()}\n{int(time.time())}\nct:{process_create_time():.3f}\nrun:private-token-not-for-stdout\n",
        encoding="utf-8")
    env = {**os.environ, "HERMES_HOME": str(home)}
    result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "gateway", "ensure", "--json"],
                            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 6, result.stderr
    value = json.loads(result.stdout)
    assert value["state"] == "draining" and value["reason_code"] == "update_paused"
    assert "private-token" not in result.stdout
    result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "gateway", "ensure", "--json", "--timeout", "nan"],
                            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 2
    assert json.loads(result.stdout)["reason_code"] == "invalid_invocation"
    result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "gateway", "ensure", "--json", "--unknown", "private-value"],
                            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 2
    assert json.loads(result.stdout)["reason_code"] == "invalid_invocation"
    assert "private-value" not in result.stdout


def test_update_fence_follows_update_lock_judgement(tmp_path, monkeypatch):
    """A killed update's dead or malformed marker never blocks a launch (main's launch contract);
    the same marker fences while the checkout lock is held, and a live claim always fences."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime, update_lock
    from hermes_agent.hermes_cli.update_lock import process_create_time

    marker = tmp_path / update_lock.MARKER_NAME
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_ct = process_create_time(dead.pid)
    dead.wait(timeout=30)
    held = []
    monkeypatch.setattr(update_lock, "checkout_lock_held", lambda *a, **k: bool(held))
    for body in (f"{dead.pid}\n{int(time.time())}\nct:{dead_ct or 1.0:.3f}\n", "not a marker"):
        marker.write_text(body, encoding="utf-8")
        assert not runtime._update_fenced({tmp_path}), body
        held.append(True)
        assert runtime._update_fenced({tmp_path}), body
        held.clear()
    marker.write_text(f"{os.getpid()}\n{int(time.time())}\nct:{process_create_time():.3f}\n", encoding="utf-8")
    assert runtime._update_fenced({tmp_path})
    assert marker.is_file()  # read-only: a client never clears the updater's fence


@pytest.mark.platforms("linux")
def test_unmanaged_child_uses_explicit_home_and_survives_launcher_exit(tmp_path, monkeypatch):
    from hermes_agent.hermes_cli import gateway_runtime_start as start
    assert callable(getattr(start, "spawn_unmanaged_gateway", None))
    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    witness = home / "child.json"
    gate = home / "exit"
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import json,os,sys,time\nfrom pathlib import Path\n"
        "from hermes_cli.gateway_runtime_start import spawn_unmanaged_gateway\n"
        "import hermes_cli.gateway_runtime_start as start\n"
        f"home=Path({str(home)!r})\n"
        "real=start.subprocess.Popen\n"
        "def boundary(argv, **kw):\n"
        " if 'hermes_agent.hermes_cli.main' not in argv: return real(argv, **kw)  # e.g. git from the env builder\n"
        " assert argv[-3:] == ['gateway','run','--quiet']\n"
        " assert '--replace' not in argv and '--force' not in argv\n"
        " assert kw['env']['HERMES_HOME'] == str(home)\n"
        " code=\"import os,json,time; from pathlib import Path; h=Path(os.environ['HERMES_HOME']); (h/'child.json').write_text(json.dumps({'pid':os.getpid(),'sid':os.getsid(0),'home':str(h)}));\\nwhile not (h/'exit').exists(): time.sleep(.02)\"\n"
        " return real([sys.executable,'-c',code], **kw)\n"
        "start.subprocess.Popen=boundary\n"
        "child=spawn_unmanaged_gateway(home, deadline=time.monotonic()+5)\n"
        "print(child.pid, flush=True)\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path.cwd()), "HERMES_HOME": str(tmp_path / "wrong")}
    result = subprocess.run([sys.executable, str(launcher)], env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=10, check=False)
    try:
        assert result.returncode == 0, result.stderr
        deadline = time.monotonic() + 5
        while not witness.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        value = json.loads(witness.read_text())
        assert value["pid"] == int(result.stdout) == value["sid"]
        assert value["home"] == str(home)
    finally:
        gate.touch()


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike  # stub interpreter records env then exits; reaped below
def test_unmanaged_runtime_does_not_inherit_client_launch_flags(tmp_path, monkeypatch):
    # --yolo / --ignore-rules are frozen on the launching session's policy; inherited by the
    # daemon they would bypass approvals or skip skills.auto_load for every later session.
    from hermes_agent.hermes_cli import gateway_runtime_start as start

    home = tmp_path / 'policy-home'
    home.mkdir(mode=0o700)
    witness = home / 'policy.json'
    executable = tmp_path / 'owned-interpreter'
    repo = Path(__file__).resolve().parents[2]
    executable.write_text(
        '#!' + sys.executable + '\nimport json, os, sys\nfrom pathlib import Path\n'
        + f'sys.path.insert(0, {str(repo)!r})\n'
        + 'from tools import approval\n'
        + f'Path({str(witness)!r}).write_text(json.dumps('
        + "{'yolo': approval._YOLO_MODE_FROZEN, 'ignore_rules': os.environ.get('HERMES_IGNORE_RULES'),"
        + " 'sentinel': os.environ.get('RUNTIME_TEST_SENTINEL')}))\n",
        encoding='utf-8',
    )
    executable.chmod(0o700)
    monkeypatch.setattr(sys, 'executable', str(executable))
    monkeypatch.setenv('HERMES_YOLO_MODE', '1')
    monkeypatch.setenv('HERMES_IGNORE_RULES', '1')
    monkeypatch.setenv('RUNTIME_TEST_SENTINEL', 'retained')
    child = start.spawn_unmanaged_gateway(home, deadline=time.monotonic() + 5)
    try:
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    assert json.loads(witness.read_text()) == {'yolo': False, 'ignore_rules': None, 'sentinel': 'retained'}
    assert os.environ['HERMES_YOLO_MODE'] == os.environ['HERMES_IGNORE_RULES'] == '1'


@pytest.mark.platforms("linux")
def test_reserved_profile_without_pid_or_control_is_never_absent(tmp_path):
    from hermes_agent.gateway.runtime_ownership import ProfileOwnership
    from hermes_agent.hermes_cli.gateway_runtime import discover_gateway_endpoint
    home = tmp_path / "reserved"
    home.mkdir(mode=0o700)
    owner = ProfileOwnership()
    owner.reserve([home])
    try:
        assert not (home / "gateway.pid").exists()
        assert discover_gateway_endpoint(home).state == "starting"
        (home / "gateway.lock").chmod(0)
        assert discover_gateway_endpoint(home).state == "inaccessible"
    finally:
        (home / "gateway.lock").chmod(0o600)
        owner.close()
    assert discover_gateway_endpoint(home).state == "absent"


@pytest.mark.platforms("windows")
def test_native_windows_discovery_uses_same_user_pipe(tmp_path, monkeypatch):
    # The pipe server below runs on this process's own token, so both sides of the identity check
    # read its real elevation: a pinned non-elevated client refuses its own elevated server.
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", _REAL_PROCESS_IS_ELEVATED)
    from hermes_agent.gateway.runtime_bootstrap_windows import NativeControlServer
    from hermes_agent.hermes_cli.gateway_runtime import discover_gateway_endpoint
    home = tmp_path.resolve()
    def handle(raw, subject):
        assert subject.startswith("sid:")
        request = json.loads(raw)
        assert request["verb"] == "identify"
        return json.dumps({"protocol": 1, "id": 1, "ok": True, "result": {
            "runtime_protocol": 1, "state": "starting", "instance_id": "native-peer",
            "served_profiles": [{"home": str(home), "profile_id": str(home)}]
        }}).encode() + b"\n"
    server = NativeControlServer(home, handle)
    server.start()
    try:
        assert discover_gateway_endpoint(home, timeout=2).state == "starting"
    finally:
        server.close()


@pytest.mark.platforms("windows")
def test_native_windows_spawn_never_retries_without_breakaway(tmp_path, monkeypatch):
    from hermes_agent.hermes_cli import gateway_runtime_start as start
    from hermes_agent.hermes_cli.gateway_runtime_service import RuntimeStartError
    from hermes_agent.hermes_cli._subprocess_compat import windows_detach_flags
    calls = []
    def denied(argv, **kwargs):
        calls.append(kwargs)
        raise PermissionError("job forbids breakaway")
    monkeypatch.setattr(start.subprocess, "Popen", denied)
    with pytest.raises(RuntimeStartError, match="windows_breakaway_unavailable"):
        start.spawn_unmanaged_gateway(tmp_path, deadline=time.monotonic()+5)
    assert len(calls) == 1
    assert calls[0]["creationflags"] == windows_detach_flags()


def _absent_owner_home(tmp_path, monkeypatch):
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: runtime.GatewayDiscovery("absent"))
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    return home


@pytest.mark.platforms("windows")
def test_breakaway_refusal_names_the_winerror(tmp_path, monkeypatch):
    """A job that forbids breakaway reaches the client with the OS's own refusal, not a bare code."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_start as start
    home = _absent_owner_home(tmp_path, monkeypatch)
    def denied(argv, **kwargs):
        raise OSError(13, "Access is denied", None, 5)  # CreateProcess in a job without BREAKAWAY_OK
    monkeypatch.setattr(start.subprocess, "Popen", denied)
    result = runtime.ensure_gateway_runtime(home, timeout=5)
    assert (result.state, result.reason_code, result.detail) == (
        "inaccessible", "windows_breakaway_unavailable", "WinError 5: Access is denied")


def test_unopenable_stdio_log_is_not_a_breakaway_refusal(tmp_path, monkeypatch):
    """Only CreateProcess's refusal is the job's verdict: a start that cannot open its log says so."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_start as start
    home = _absent_owner_home(tmp_path, monkeypatch)
    (home / "logs" / "gateway-stdio.log").mkdir(parents=True)  # a directory: open("ab") fails on every host
    spawned = []
    # macOS tries a launchd job first; this test is about the direct spawn on every host.
    monkeypatch.setattr(start, "start_gateway_job", lambda *a, **k: None)
    monkeypatch.setattr(start.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    result = runtime.ensure_gateway_runtime(home, timeout=5)
    assert (result.state, result.reason_code) == ("inaccessible", "stdio_log_unavailable")
    assert result.detail and not spawned


def test_detached_restart_reports_a_spent_budget(monkeypatch, capsys, tmp_path):
    """`gateway restart --detach` prints the deadline verdict, not a TimeoutError traceback, when its
    budget runs out mid-start (spawn_unmanaged_gateway lets TimeoutError through for ensure)."""
    from hermes_agent.hermes_cli import gateway_runtime_start as start
    def spent(*args, **kwargs):
        raise TimeoutError
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", spent)
    assert start.start_detached_replacement(tmp_path) == 1
    assert "Could not start the gateway in the background: deadline" in capsys.readouterr().err


@pytest.mark.platforms("macos")
def test_native_launchd_ambiguous_domains_cannot_start(tmp_path, monkeypatch):
    from hermes_agent.hermes_cli.gateway_runtime_service import discover_existing_gateway_service, RuntimeStartError
    peer = tmp_path / "inert_launchd.py"
    peer.write_text("print('state = not running')\n", encoding="utf-8")
    actual = subprocess.run
    calls = []
    def boundary(argv, **kwargs):
        assert argv[0] == "launchctl"
        calls.append(argv)
        return actual([sys.executable, str(peer), *argv[1:]], **kwargs)
    monkeypatch.setattr(subprocess, "run", boundary)
    with pytest.raises(RuntimeStartError, match="service_scope_conflict"):
        discover_existing_gateway_service(tmp_path.resolve(), deadline=time.monotonic()+5)
    assert len(calls) == 2 and all(argv[1] == "print" for argv in calls)


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike  # stub interpreter records the resolved home then exits; reaped below
def test_unmanaged_root_home_child_ignores_sticky_active_profile(tmp_path, monkeypatch):
    """Explicit default selection survives the CLI child's own profile bootstrap (F15): with
    active_profile=other sticky, the spawned `gateway run` must still resolve the root home."""
    from hermes_agent.hermes_cli import gateway_runtime_start as start
    root = tmp_path / ".hermes"
    (root / "profiles" / "other").mkdir(parents=True)
    (root / "active_profile").write_text("other", encoding="utf-8")
    witness = tmp_path / "resolved.json"
    real = start.subprocess.Popen

    def boundary(argv, **kwargs):
        assert argv[:3] == [sys.executable, "-m", "hermes_agent.hermes_cli.main"]
        # Same argv and env as the real child; the module-import bootstrap is what resolves the profile.
        code = ("import json, os, sys\nsys.argv = ['hermes', *sys.argv[1:]]\nimport hermes_agent.hermes_cli.main\n"
                f"open({str(witness)!r}, 'w').write(json.dumps({{'home': os.environ['HERMES_HOME'], 'argv': sys.argv}}))")
        return real([sys.executable, "-c", code, *argv[3:]], **kwargs)

    monkeypatch.setattr(start.subprocess, "Popen", boundary)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    child = start.spawn_unmanaged_gateway(root, deadline=time.monotonic() + 5)
    try:
        assert child.wait(timeout=30) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    resolved = json.loads(witness.read_text())
    assert resolved["home"] == str(root), resolved
    assert resolved["argv"] == ["hermes", "gateway", "run", "--quiet"], resolved


@pytest.mark.parametrize("standalone", [False, True])
def test_named_profile_without_multiplex_evidence_starts_the_host_gateway(tmp_path, monkeypatch, standalone):
    """`gateway run` refuses a profiles/<name> home a gateway of its own, so ensure must start the host."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

    root = tmp_path / ".hermes"
    home = root / "profiles" / "alpha"
    home.mkdir(parents=True, mode=0o700)
    if standalone:
        (home / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")
    spawned = []
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: runtime.GatewayDiscovery("absent"))
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda target, **k: spawned.append(Path(target)))

    runtime.ensure_gateway_runtime(home, timeout=0.3)

    assert spawned == [(home if standalone else root).resolve()]


@pytest.mark.parametrize("flag,standalone,owner", [("false", False, "root"), ("true", True, "home")])
def test_cold_start_target_follows_boot_multiplex_policy(tmp_path, monkeypatch, flag, standalone, owner):
    """Boot settles a retired `multiplex_profiles: false` like unset (the host still serves the
    profile) and never serves a `standalone: true` secondary even under an explicit `true`."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    root = tmp_path / ".hermes"
    home = root / "profiles" / "alpha"
    home.mkdir(parents=True, mode=0o700)
    (root / "config.yaml").write_text(f"gateway:\n  multiplex_profiles: {flag}\n", encoding="utf-8")
    if standalone:
        (home / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")
    spawned = []
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: runtime.GatewayDiscovery("absent"))
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda target, **k: spawned.append(Path(target)))

    runtime.ensure_gateway_runtime(home, timeout=0.3)

    assert spawned == [{"root": root, "home": home}[owner].resolve()]


@pytest.mark.platforms("linux", "macos")
def test_reserved_root_never_suppresses_a_standalone_profile_start(tmp_path, monkeypatch):
    """A held root reservation means "the multiplexer is coming" only for profiles its boot policy
    serves; a `gateway.standalone: true` secondary must still get its own owner started."""
    from hermes_agent.gateway.runtime_ownership import ProfileOwnership
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

    root = tmp_path / ".hermes"
    home = root / "profiles" / "alpha"
    home.mkdir(parents=True, mode=0o700)
    (root / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    (home / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")
    spawned = []
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda target, **k: spawned.append(Path(target)))
    owner = ProfileOwnership()
    owner.reserve([root])
    try:
        runtime.ensure_gateway_runtime(home, timeout=0.5)
    finally:
        owner.close()
    assert spawned == [home.resolve()]


def _standalone_host(tmp_path, monkeypatch, record):
    """A named profile whose root holds its gateway lock and has written *record* (or nothing)
    to the root's real gateway_state.json; returns (home, spawned targets)."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_discovery as discovery
    from hermes_agent.hermes_cli import gateway_runtime_service as service, gateway_runtime_start as start

    root = tmp_path / ".hermes"
    home = root / "profiles" / "alpha"
    home.mkdir(parents=True, mode=0o700)
    if record is not None:
        (root / "gateway_state.json").write_text(json.dumps(record), encoding="utf-8")
    spawned = []
    monkeypatch.setattr(runtime, "_update_fenced", lambda homes: False)
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda *a, **k: runtime.GatewayDiscovery("absent"))
    monkeypatch.setattr(discovery, "missing_owner_state", lambda path: "starting" if Path(path).resolve() == root.resolve() else "absent")
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda target, **k: spawned.append(Path(target)))
    return home, spawned


_REASON = "duplicate Telegram token; run hermes gateway migrate --multiplex"


def test_ensure_reports_a_standalone_host_reason_for_an_unserved_named_profile(tmp_path, monkeypatch):
    """A live standalone host cannot later add this profile, so do not wait on its root lock: its
    reason is read from the root's real gateway_state.json (path and key included)."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime

    home, spawned = _standalone_host(tmp_path, monkeypatch, {
        "gateway_state": "running", "pid": os.getpid(), "served_profiles": [],
        "multiplex_standalone_reason": _REASON})

    result = runtime.ensure_gateway_runtime(home, timeout=1)

    assert (result.state, result.reason_code, result.detail) == ("inaccessible", "profile_parked", _REASON)
    assert spawned == []


@pytest.mark.parametrize("record", [
    None,  # the host has not written its state yet
    {"gateway_state": "running", "pid": os.getpid()},  # a multiplexer: no refusal
    {"gateway_state": "stopped", "pid": os.getpid(), "multiplex_standalone_reason": _REASON},
    {"gateway_state": "running", "pid": 999_999_999, "multiplex_standalone_reason": _REASON},
], ids=["no-state", "no-reason", "stopped", "dead-pid"])
def test_a_reserved_root_without_a_live_refusal_is_still_awaited(tmp_path, monkeypatch, record):
    """A held root lock with no LIVE standalone refusal is a multiplexer still starting: the named
    profile waits for it (never parked on a stale reason, never a competing per-profile spawn)."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime

    home, spawned = _standalone_host(tmp_path, monkeypatch, record)

    result = runtime.ensure_gateway_runtime(home, timeout=0.5)

    assert (result.state, result.reason_code) == ("starting", "deadline")
    assert spawned == []


@pytest.mark.platforms("linux", "macos")
def test_dead_unmanaged_start_is_a_prompt_terminal_verdict(tmp_path, monkeypatch):
    """The one start this call requested exited without leaving an owner: report it with the
    redacted tail of that launch's own output (never an earlier run's), never `starting` until the
    deadline, and never launch a second daemon."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

    home = tmp_path / "profile"
    (home / "logs").mkdir(parents=True, mode=0o700)
    (home / "logs" / "gateway-stdio.log").write_text("OLD_RUN_TRACEBACK\n", encoding="utf-8")
    secret = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
    children = []

    def dead_start(target, **kwargs):
        log = (target / "logs" / "gateway-stdio.log").open("ab")
        code = ("import sys; print('Traceback (most recent call last):'); "
                f"print('RuntimeError: boot failed with key {secret}'); sys.exit(3)")
        children.append(subprocess.Popen([sys.executable, "-c", code], stdout=log, stderr=log))
        return children[-1]
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", dead_start)
    before = time.monotonic()
    result = runtime.ensure_gateway_runtime(home, timeout=20.0)
    assert time.monotonic() - before < 10
    assert (result.state, result.reason_code) == ("inaccessible", "runtime_exited")
    assert "status 3" in result.detail and len(children) == 1
    assert "RuntimeError: boot failed with key" in result.detail, result.detail
    assert secret not in result.detail and "OLD_RUN_TRACEBACK" not in result.detail
    assert "gateway-stdio.log" in result.detail



@pytest.fixture
def fast_runway(monkeypatch):
    """A real StartRunway on a sub-second scale: base 0.5 s, 0.2 s windows, cap 5 s."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime, gateway_runtime_progress as progress
    monkeypatch.setattr(runtime, "DEFAULT_ENSURE_TIMEOUT", 0.5)
    monkeypatch.setattr(progress, "PROGRESS_WINDOW_S", 0.2)
    monkeypatch.setattr(progress, "_SAMPLE_EVERY_S", 0.05)
    monkeypatch.setattr(progress, "_LEAD_S", 1.0)
    monkeypatch.setattr(progress, "MAX_ADAPTIVE_TIMEOUT", 5.0)
    # The owner is found only where the real lookup would look: a home this call waits on, or its child.
    # Like the real lookup, an owner only where this call waits: a home it names, or its own child.
    monkeypatch.setattr(progress, "_starting_owner_pid", lambda homes, child: 4242 if homes or child else None)
    monkeypatch.setattr(runtime, "_update_fenced", lambda homes: False)
    return runtime, progress


def _a_probe_spends_the_budget(runtime, ready_after, *, seen_starting=True):
    """A start this call waits on (first probe: ``starting``) whose next probe uses the whole budget
    on a starved control loop. ``seen_starting=False``: the owner is never seen starting."""
    probes = []
    start = time.monotonic()

    def observe(home, deadline, replaced):
        remaining(deadline)  # like the real probe: an exhausted budget is the deadline verdict
        probes.append(time.monotonic() - start)
        if len(probes) == 1 and seen_starting:
            return runtime.GatewayDiscovery("starting")
        if len(probes) <= 2:
            time.sleep(max(0.0, deadline - time.monotonic()))  # health: allow HX013 -- the probe using its whole budget IS the case under test
            return runtime.GatewayDiscovery("inaccessible", reason_code="control_timeout")
        return runtime.GatewayDiscovery("ready" if time.monotonic() - start >= ready_after else "starting")
    return observe


def test_a_progressing_start_whose_probe_spends_the_budget_is_still_awaited(fast_runway, tmp_path, monkeypatch):
    runtime, progress = fast_runway
    monkeypatch.setattr(runtime, "_observe_replacing_stale", _a_probe_spends_the_budget(runtime, ready_after=1.0))
    monkeypatch.setattr(progress, "_cpu_seconds", lambda pid: time.monotonic())  # always burning CPU
    assert runtime.ensure_gateway_runtime(tmp_path).state == "ready"


def test_a_stalled_start_whose_probe_times_out_is_still_a_deadline(fast_runway, tmp_path, monkeypatch):
    runtime, progress = fast_runway
    monkeypatch.setattr(runtime, "_observe_replacing_stale", _a_probe_spends_the_budget(runtime, ready_after=60.0))
    monkeypatch.setattr(progress, "_cpu_seconds", lambda pid: 2.0)  # stopped: no CPU progress
    before = time.monotonic()
    result = runtime.ensure_gateway_runtime(tmp_path)
    assert (result.state, result.reason_code) == ("starting", "deadline")
    assert time.monotonic() - before < 2.0


def test_a_wedged_owner_never_seen_starting_is_a_deadline_however_hard_it_spins(fast_runway, tmp_path,
                                                                               monkeypatch):
    """A READY gateway whose control loop stopped answering is not a start: its CPU (a spin loop)
    never extends the wait past the base deadline, as on main."""
    runtime, progress = fast_runway
    monkeypatch.setattr(runtime, "_observe_replacing_stale",
                        _a_probe_spends_the_budget(runtime, ready_after=60.0, seen_starting=False))
    monkeypatch.setattr(progress, "_cpu_seconds", lambda pid: time.monotonic())  # spinning
    before = time.monotonic()
    result = runtime.ensure_gateway_runtime(tmp_path)
    assert (result.state, result.reason_code) == ("starting", "deadline")
    assert time.monotonic() - before < 2.0


def _probes(runtime, *answers, ready_after):
    """Primary probes answering ``answers`` in order ("budget": spend the whole budget, then answer
    ``starting`` as a multiplexer's swallowed timeout does; "timeout": the same as control_timeout),
    then ``ready`` once ``ready_after`` seconds passed (``starting`` until then)."""
    start = time.monotonic()
    queue = list(answers)

    def observe(home, deadline, replaced):
        remaining(deadline)  # like the real probe: an exhausted budget is the deadline verdict
        answer = queue.pop(0) if queue else None
        if answer in ("budget", "timeout"):
            time.sleep(max(0.0, deadline - time.monotonic()))  # health: allow HX013 -- the probe using its whole budget IS the case under test
            return runtime.GatewayDiscovery("inaccessible", reason_code="control_timeout") if answer == "timeout" \
                else runtime.GatewayDiscovery("starting")
        if answer is not None:
            return runtime.GatewayDiscovery(answer)
        return runtime.GatewayDiscovery("ready" if time.monotonic() - start >= ready_after else "starting")
    return observe


def _no_spawn_targets(monkeypatch, runtime, *, reserved=None, second="absent"):
    from hermes_agent.hermes_cli import gateway_runtime_multiplex as multiplex, gateway_runtime_service as service
    monkeypatch.setattr(runtime, "_starting_multiplexer_root", lambda home: reserved)
    monkeypatch.setattr(runtime, "_standalone_multiplexer_reason", lambda root: None)
    monkeypatch.setattr(multiplex, "multiplexer_serves_home", lambda home: None)
    monkeypatch.setattr(multiplex, "implied_host_root", lambda home: None)
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "discover_gateway_endpoint", lambda home, timeout: runtime.GatewayDiscovery(second))
    monkeypatch.setattr(runtime, "_missing_owner_verdict", lambda home: runtime.GatewayDiscovery(second))


def test_a_start_seen_only_by_the_race_closing_probe_is_still_sampled(fast_runway, tmp_path, monkeypatch):
    runtime, progress = fast_runway
    _no_spawn_targets(monkeypatch, runtime, second="starting")
    monkeypatch.setattr(runtime, "_observe_replacing_stale", _probes(runtime, "absent", "timeout", ready_after=1.0))
    monkeypatch.setattr(progress, "_cpu_seconds", lambda pid: time.monotonic())
    assert runtime.ensure_gateway_runtime(tmp_path).state == "ready"


def test_a_profile_awaiting_its_multiplexer_samples_the_root_not_itself(fast_runway, tmp_path, monkeypatch):
    """A named profile waiting on its root's pending reservation: the ROOT is the progressing owner."""
    runtime, progress = fast_runway
    root = tmp_path / "root"
    _no_spawn_targets(monkeypatch, runtime, reserved=root)
    monkeypatch.setattr(progress, "_starting_owner_pid", lambda homes, child: 4242 if root in homes else None)
    monkeypatch.setattr(runtime, "_observe_replacing_stale", _probes(runtime, "absent", "timeout", ready_after=1.0))
    monkeypatch.setattr(progress, "_cpu_seconds", lambda pid: time.monotonic())
    assert runtime.ensure_gateway_runtime(tmp_path / "profile").state == "ready"


def test_a_probe_that_spends_the_budget_without_a_control_timeout_still_gets_a_verdict_sample(
        fast_runway, tmp_path, monkeypatch):
    """A multiplexer's control timeout reaches the loop as ``starting`` (its root timeout is swallowed):
    the exhausted budget still gets one progress sample before the deadline verdict, even when the
    last sample is younger than the sampling interval."""
    runtime, progress = fast_runway
    monkeypatch.setattr(progress, "_SAMPLE_EVERY_S", 10.0)
    monkeypatch.setattr(runtime, "_observe_replacing_stale", _probes(runtime, "starting", "budget", ready_after=1.0))
    monkeypatch.setattr(progress, "_cpu_seconds", lambda pid: time.monotonic())
    assert runtime.ensure_gateway_runtime(tmp_path).state == "ready"


def test_a_root_that_refuses_after_the_profile_joined_it_parks_the_profile(fast_runway, tmp_path, monkeypatch):
    """A root reserves before publishing its standalone refusal: the profile that joined it in between
    gets ``profile_parked`` when the refusal appears, not the root's (sampled, busy) adaptive wait."""
    runtime, progress = fast_runway
    root = tmp_path / "root"
    _no_spawn_targets(monkeypatch, runtime, reserved=root)
    joined = time.monotonic()
    monkeypatch.setattr(runtime, "_standalone_multiplexer_reason",
                        lambda r: "standalone" if r == root and time.monotonic() - joined > 0.3 else None)
    monkeypatch.setattr(runtime, "_observe_replacing_stale", _probes(runtime, "absent", ready_after=60.0))
    monkeypatch.setattr(progress, "_cpu_seconds", lambda pid: time.monotonic())  # the root keeps burning CPU
    result = runtime.ensure_gateway_runtime(tmp_path / "profile")
    assert (result.state, result.reason_code, result.detail) == ("inaccessible", "profile_parked", "standalone")
