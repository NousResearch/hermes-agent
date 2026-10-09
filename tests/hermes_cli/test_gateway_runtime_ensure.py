"""Real local-control and subprocess boundaries, never real service mutations."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

def test_unmanaged_runtime_drops_session_config_bypass_flags(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import gateway_runtime_start as start
    from hermes_cli import gateway_windows

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
    from gateway.control_socket import GatewayControlServer
    from hermes_cli import gateway_runtime as runtime

    assert callable(getattr(runtime, "ensure_gateway_runtime", None))

    async def probe():
        home = tmp_path / "profile"
        home.mkdir(mode=0o700)
        payload = {"runtime_protocol": 1, "state": "starting", "instance_id": "peer",
                   "served_profiles": [{"profile_id": str(home), "home": str(home)}]}
        stalled = False
        def identify():
            if stalled:
                time.sleep(4)  # longer than the caller's deadline: a stalled owner is 'starting', never ready
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
            await server.stop()
    asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_installed_service_start_is_nonmutating_and_failed_manager_never_spawns(tmp_path, monkeypatch):
    from hermes_cli import gateway as gw, gateway_runtime as runtime

    assert callable(getattr(runtime, "ensure_gateway_runtime", None))
    from hermes_cli.gateway_runtime_service import service_suffix
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
    from hermes_cli.gateway_runtime_service_identity import SYSTEMD_IDENTITY_PROPERTIES
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
    from hermes_cli import gateway_runtime as runtime
    assert callable(getattr(runtime, "ensure_gateway_runtime", None))
    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    # A LIVE claim (this test process, at its own incarnation) fences; a dead one would not.
    from hermes_cli.update_lock import process_create_time
    (home / ".hermes-update-in-progress").write_text(
        f"{os.getpid()}\n{int(time.time())}\nct:{process_create_time():.3f}\nrun:private-token-not-for-stdout\n",
        encoding="utf-8")
    env = {**os.environ, "HERMES_HOME": str(home)}
    result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "gateway", "ensure", "--json"],
                            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
    assert result.returncode == 6, result.stderr
    value = json.loads(result.stdout)
    assert value["state"] == "draining" and value["reason_code"] == "update_paused"
    assert "private-token" not in result.stdout
    result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "gateway", "ensure", "--json", "--timeout", "nan"],
                            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert json.loads(result.stdout)["reason_code"] == "invalid_invocation"
    result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "gateway", "ensure", "--json", "--unknown", "private-value"],
                            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert json.loads(result.stdout)["reason_code"] == "invalid_invocation"
    assert "private-value" not in result.stdout


def test_update_fence_follows_update_lock_judgement(tmp_path, monkeypatch):
    """A killed update's dead or malformed marker never blocks a launch (main's launch contract);
    the same marker fences while the checkout lock is held, and a live claim always fences."""
    from hermes_cli import gateway_runtime as runtime, update_lock
    from hermes_cli.update_lock import process_create_time

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
    from hermes_cli import gateway_runtime_start as start
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
                            capture_output=True, text=True, timeout=10)
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
    from hermes_cli import gateway_runtime_start as start

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
    from gateway.runtime_ownership import ProfileOwnership
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
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
def test_native_windows_discovery_uses_same_user_pipe(tmp_path):
    from gateway.runtime_bootstrap_windows import NativeControlServer
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
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
    from hermes_cli import gateway_runtime_start as start
    from hermes_cli.gateway_runtime_service import RuntimeStartError
    from runtime.subprocess_compat import windows_detach_flags
    calls = []
    def denied(argv, **kwargs):
        calls.append(kwargs)
        raise PermissionError("job forbids breakaway")
    monkeypatch.setattr(start.subprocess, "Popen", denied)
    with pytest.raises(RuntimeStartError, match="windows_breakaway_unavailable"):
        start.spawn_unmanaged_gateway(tmp_path, deadline=time.monotonic()+5)
    assert len(calls) == 1
    assert calls[0]["creationflags"] == windows_detach_flags()


@pytest.mark.platforms("macos")
def test_native_launchd_ambiguous_domains_cannot_start(tmp_path, monkeypatch):
    from hermes_cli.gateway_runtime_service import discover_existing_gateway_service, RuntimeStartError
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
    from hermes_cli import gateway_runtime_start as start
    root = tmp_path / ".hermes"
    (root / "profiles" / "other").mkdir(parents=True)
    (root / "active_profile").write_text("other", encoding="utf-8")
    witness = tmp_path / "resolved.json"
    real = start.subprocess.Popen

    def boundary(argv, **kwargs):
        assert argv[:3] == [sys.executable, "-m", "hermes_cli.main"]
        # Same argv and env as the real child; the module-import bootstrap is what resolves the profile.
        code = ("import json, os, sys\nsys.argv = ['hermes', *sys.argv[1:]]\nimport hermes_cli.main\n"
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
    from hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

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
    from hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

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
    from gateway.runtime_ownership import ProfileOwnership
    from hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

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


@pytest.mark.platforms("linux", "macos")
def test_dead_unmanaged_start_is_a_prompt_terminal_verdict(tmp_path, monkeypatch):
    """The one start this call requested exited without leaving an owner: report it, never
    `starting` until the deadline, and never launch a second daemon."""
    from hermes_cli import gateway_runtime as runtime, gateway_runtime_service as service, gateway_runtime_start as start

    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    children = []

    def dead_start(target, **kwargs):
        children.append(subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"]))
        return children[-1]
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda *a, **k: None)
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", dead_start)
    before = time.monotonic()
    result = runtime.ensure_gateway_runtime(home, timeout=20.0)
    assert time.monotonic() - before < 10
    assert (result.state, result.reason_code) == ("inaccessible", "runtime_exited")
    assert "status 3" in result.detail and len(children) == 1
