"""A checkout transition must not finish in the old interpreter's module graph."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

import pytest


@pytest.fixture
def transition(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    package = root / "hermes_cli"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (root / "pm").mkdir()
    (root / "pm/__init__.py").write_text("OLD_API = True\n")

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "Completion test")
    git("config", "user.email", "completion@example.invalid")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-m", "old incompatible runtime")
    old = git("rev-parse", "HEAD")
    # Deliberately incompatible: a cached OLD_API-only PM cannot prepare this tree.
    (root / "pm/__init__.py").write_text(
        "from hermes_cli.probe import event\n"
        "def sync_venv(*, explicit, project_root):\n"
        "    assert explicit\n"
        "    event('prepare')\n"
    )
    (root / "pm/receipt.py").write_text(
        "from contextlib import nullcontext\n"
        "worker_context = lambda update_id: nullcontext()\n"
        "last_for_update = lambda update_id: {'update_id': update_id, 'outcome': 'success'}\n"
        "def accept_worker_receipt(data, update_id):\n"
        "    assert data['update_id'] == update_id\n"
    )
    (package / "probe.py").write_text(
        "import json, os, pathlib, sys\n"
        "def event(name, **values):\n"
        "    with pathlib.Path('events.jsonl').open('a') as f:\n"
        "        f.write(json.dumps(dict(name=name, pid=os.getpid(), python=sys.executable, **values)) + '\\n')\n"
    )
    (package / "runtime_paths.py").write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "selected_venv = lambda root: Path(sys.executable).parent.parent\n"
        "activation_environment = lambda root: {**os.environ, 'PYTHONPATH': str(root)}\n"
        "def activate_dependencies(root):\n"
        "    from hermes_cli.probe import event\n"
        "    event('activate')\n"
    )
    selected = tmp_path / "selected-python"
    venv.EnvBuilder(with_pip=False).create(selected)
    selected_python = selected / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    (root / "hermes_constants.py").write_text(
        f"venv_python_path = lambda root: {str(selected_python)!r}\n"
    )
    (package / "venv_sync.py").write_text(
        "from hermes_cli.probe import event\n"
        "publish_launchers = lambda root: event('launchers')\n"
    )
    (package / "source_build.py").write_text(
        "from hermes_cli.probe import event\n"
        "def build_update_products(root, *, desktop): event('build', desktop=desktop)\n"
    )
    (package / "main.py").write_text("")
    (package / "update_cmd_config.py").write_text("_LAST_SIBLING_SNAPSHOTS = {}\n")
    (package / "update_inventory.py").write_text(
        "from types import SimpleNamespace\nRuntimeRecord = UpdatePlan = SimpleNamespace\n"
    )
    (package / "update_cmd_maint.py").write_text(
        "from hermes_cli.probe import event\n"
        "def _run_post_update_maintenance(**kwargs):\n"
        "    from hermes_cli.update_cmd_config import _LAST_SIBLING_SNAPSHOTS\n"
        "    event('maintenance', snapshots=_LAST_SIBLING_SNAPSHOTS, **kwargs)\n"
        "    return True\n"
    )
    (package / "update_cmd.py").write_text(
        "from hermes_cli.probe import event\n"
        "_invalidate_update_cache = lambda: event('cache')\n"
        "_sweep_bytecode_after_update = lambda branch: event('bytecode')\n"
        "_write_fleet_restart_pending_marker = lambda **kw: event('pending')\n"
        "_write_gateway_update_exit_code = lambda ok: event('exit_marker', ok=ok)\n"
        "def _restart_gateway_fleet_after_update(plan, gateway_mode):\n"
        "    event('restart', profiles=[r.profile for r in plan.runtimes])\n"
        "    return object()\n"
        "def _resume_windows_gateways_and_merge_outcome(out, token, gateway_mode):\n"
        "    token['resume_needed'] = False\n"
        "    event('resume')\n"
        "def _resume_windows_gateways_after_update(token):\n"
        "    if token and token.get('resume_needed'):\n"
        "        token['resume_needed'] = False\n"
        "        event('emergency_resume')\n"
        "def _verify_fleet_after_update(out, **kw):\n"
        "    from hermes_cli.update_receipt import finalize_pending_update_receipt\n"
        "    event('verify')\n"
        "    finalize_pending_update_receipt(0, 'verified')\n"
    )
    (package / "update_receipt.py").write_text(
        "import contextvars, json, os, pathlib\n"
        "_current = contextvars.ContextVar('receipt', default=None)\n"
        "class UpdateReceipt: pass\n"
        "def finalize_pending_update_receipt(code, reason):\n"
        "    r = _current.get()\n"
        "    if r is None: return\n"
        "    r.data.update(exit_code=code, outcome='success' if code == 0 else 'failed', finished_at='now')\n"
        "    path = pathlib.Path(os.environ['HERMES_HOME']) / 'logs/update_receipts'\n"
        "    path.mkdir(parents=True, exist_ok=True)\n"
        "    path = path / ('update_test_' + r.correlation_id + '.json')\n"
        "    path.write_text(json.dumps(r.data, ensure_ascii=False), encoding=r.data.get('encoding', 'utf-8'))\n"
        "    _current.set(None)\n"
        "    return path\n"
    )
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-m", "new incompatible runtime")
    new = git("rev-parse", "HEAD")
    request = {
        "schema": 1, "source": str(root), "home": str(home), "branch": "main",
        "desktop": True, "assume_yes": True, "gateway_mode": True,
        "pre_update_version": "old", "snapshot_id": "active-before",
        "sibling_snapshots": {"work": "work-before"},
        "plan": {"runtimes": [{"kind": "gateway", "profile": "work"}]},
        "receipt": {"update_id": "b" * 32, "outcome": "running", "steps": []},
        "windows_resume": {"resume_needed": True, "profiles": {"work": [123]}},
    }
    return root, git, old, new, request


@pytest.mark.parametrize("bom_boundary", [None, "request", "receipt"])
def test_old_process_new_git_tree_completes_in_fresh_python(transition, tmp_path, bom_boundary):
    from hermes_cli import update_completion

    root, git, old, new, request = transition
    request["pre_update_version"] = "日本 café"
    if bom_boundary == "receipt":
        request["receipt"]["encoding"] = "utf-8-sig"
    # Copy executable code, not its text shape: the process exercises the real transport.
    shutil.copy2(update_completion.__file__, root / "hermes_cli/update_completion.py")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-m", "completion entrypoint")
    new = git("rev-parse", "HEAD")
    git("checkout", old)
    driver = (
        "import importlib.util, json, os, subprocess, sys\n"
        "spec = importlib.util.spec_from_file_location('transport', sys.argv[1])\n"
        "transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)\n"
        f"if {bom_boundary == 'request'!r}:\n"
        "    transport._write_json = lambda path, data: path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8-sig')\n"
        "import pm\nassert pm.OLD_API\n"
        "subprocess.run(['git', 'checkout', sys.argv[2]], check=True)\n"
        "result = transport.run_completion(json.loads(sys.argv[3]))\n"
        "assert pm.OLD_API, 'transport mutated old module graph'\n"
        "print('RESULT=' + json.dumps(result))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", driver, update_completion.__file__, new, json.dumps(request)],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root), "HERMES_HOME": request["home"]},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    response = json.loads(result.stdout.split("RESULT=")[1])
    assert response["exit_code"] == 0
    assert response["receipt"]["update_id"] == request["receipt"]["update_id"]
    assert response["windows_resume"]["resume_needed"] is False
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    by_name = {event["name"]: event for event in events}
    assert by_name["activate"]["pid"] == by_name["build"]["pid"]
    assert by_name["prepare"]["pid"] != by_name["build"]["pid"]
    assert Path(by_name["build"]["python"]).is_relative_to(root.parent / "selected-python")
    assert by_name["build"]["pid"] == by_name["maintenance"]["pid"] == by_name["restart"]["pid"]
    assert by_name["maintenance"]["snapshots"] == {"work": "work-before"}
    assert by_name["maintenance"]["pre_update_snapshot_id"] == "active-before"
    assert by_name["maintenance"]["pre_update_version"] == "日本 café"
    assert by_name["restart"]["profiles"] == ["work"]
    assert [e["name"] for e in events].index("exit_marker") < [e["name"] for e in events].index("restart")


@pytest.mark.parametrize("code", [0, 23])
def test_missing_child_result_fails_boundary_receipt_and_releases_lock(transition, monkeypatch, code):
    from types import SimpleNamespace
    from hermes_cli import main, update_cmd, update_receipt, update_lock

    root, git, old, new, request = transition
    (root / "hermes_cli/update_completion.py").write_text(f"import os\nos._exit({code})\n")
    monkeypatch.setenv("HERMES_HOME", request["home"])
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kw: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda state: None)

    def complete(args, gateway_mode):
        update_receipt.begin_update_receipt()
        request["receipt"] = update_receipt._current.get().data
        update_cmd._complete_source_update(request)

    monkeypatch.setattr(update_cmd, "_cmd_update_impl", complete)
    with pytest.raises(SystemExit) as error:
        main.cmd_update(SimpleNamespace(gateway=True))
    assert error.value.code == (code or 1)
    receipt = update_receipt.read_latest_receipt()
    assert receipt["outcome"] == "failed"
    assert receipt["exit_code"] == (code or 1)
    assert receipt["update_id"] == request["receipt"]["update_id"]
    assert request["windows_resume"]["resume_needed"] is True
    assert (Path(request["home"]) / ".update_exit_code").read_text().strip() == "1"
    lock = update_lock.UpdateLock()
    assert lock.acquire()
    lock.release()


@pytest.mark.parametrize("cleanup_failure", [None, "kill", "wait"])
def test_interrupt_after_child_success_demotes_gateway_marker_at_boundary(transition, monkeypatch, cleanup_failure):
    import io
    from types import SimpleNamespace
    from hermes_cli import main, update_cmd, update_lock, update_receipt

    root, git, old, new, request = transition
    marker = Path(request["home"]) / ".update_exit_code"
    (root / "hermes_cli/update_completion.py").write_text(
        "import os, pathlib, time\n"
        "(pathlib.Path(os.environ['HERMES_HOME']) / '.update_exit_code').write_text('0\\n')\n"
        "print('SUCCESS_PUBLISHED', flush=True)\n"
        "time.sleep(30)\n"
    )
    monkeypatch.setenv("HERMES_HOME", request["home"])
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kw: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda state: None)
    interrupted = False
    cleanup_error = (OSError("retained-handle kill failed") if cleanup_failure == "kill"
                     else subprocess.TimeoutExpired("completion", 5))
    waits = []
    children = []
    popen = subprocess.Popen

    def capture_child(*args, **kwargs):
        proc = popen(*args, **kwargs)
        if not children:
            children.append(proc)
            wait = proc.wait
            kill = proc.kill

            def cleanup_kill():
                kill()
                if cleanup_failure == "kill":
                    raise cleanup_error

            def cleanup_wait(timeout=None):
                waits.append(timeout)
                # Inject faults after real cleanup so a regression cannot leak the child.
                result = wait(timeout=5)
                if cleanup_failure == "wait":
                    raise cleanup_error
                return result

            monkeypatch.setattr(proc, "kill", cleanup_kill)
            monkeypatch.setattr(proc, "wait", cleanup_wait)
        return proc

    class Interrupt(io.StringIO):
        def write(self, value):
            nonlocal interrupted
            result = super().write(value)
            if not interrupted and "SUCCESS_PUBLISHED" in self.getvalue():
                assert marker.read_text().strip() == "0"
                interrupted = True
                raise KeyboardInterrupt()
            return result

    def complete(args, gateway_mode):
        update_receipt.begin_update_receipt()
        request["receipt"] = update_receipt._current.get().data
        monkeypatch.setattr(subprocess, "Popen", capture_child)
        update_cmd._complete_source_update(request)

    monkeypatch.setattr(sys, "stdout", Interrupt())
    monkeypatch.setattr(update_cmd, "_cmd_update_impl", complete)
    with pytest.raises(KeyboardInterrupt) as error:
        main.cmd_update(SimpleNamespace(gateway=True))
    assert interrupted, "child never published success before cancellation"
    assert children[0].poll() is not None
    assert children[0].stdout.closed
    assert waits and all(timeout is not None and timeout > 0 for timeout in waits)
    if cleanup_failure:
        assert error.value.__cause__ is cleanup_error
    receipt = update_receipt.read_latest_receipt()
    assert receipt["update_id"] == request["receipt"]["update_id"]
    assert receipt["outcome"] == "failed"
    assert receipt["exit_code"] == 1
    assert receipt["stop_reason"].startswith("KeyboardInterrupt:")
    assert marker.read_text().strip() == "1"
    lock = update_lock.UpdateLock()
    assert lock.acquire()
    lock.release()


@pytest.mark.platforms("posix")
def test_killed_selected_python_returns_signal_exit_status(transition):
    from hermes_cli import update_completion

    root, git, old, new, request = transition
    shutil.copy2(update_completion.__file__, root / "hermes_cli/update_completion.py")
    (root / "hermes_cli/source_build.py").write_text(
        "import os, signal\n"
        "def build_update_products(*a, **kw): os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    result = update_completion.run_completion(request)
    assert result["exit_code"] == 137
    assert result["pm_receipt"]["update_id"] == request["receipt"]["update_id"]


def test_failed_build_preserves_exit_status_without_maintenance(transition):
    from hermes_cli import update_completion

    root, git, old, new, request = transition
    shutil.copy2(update_completion.__file__, root / "hermes_cli/update_completion.py")
    (root / "hermes_cli/source_build.py").write_text(
        "import subprocess\n"
        "def build_update_products(*a, **kw): raise subprocess.CalledProcessError(23, ['builder'])\n"
    )
    result = update_completion.run_completion(request)
    assert result["exit_code"] == 23
    assert result["receipt"]["outcome"] == "failed"
    events = [json.loads(line)["name"] for line in (root / "events.jsonl").read_text().splitlines()]
    assert "maintenance" not in events
    assert "restart" not in events
    assert "emergency_resume" in events


def test_prepare_failure_preserves_correlated_pm_receipt(transition, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import main, update_cmd, update_completion, update_receipt

    root, git, old, new, request = transition
    shutil.copy2(update_completion.__file__, root / "hermes_cli/update_completion.py")
    (root / "pm/__init__.py").write_text(
        "def sync_venv(**kw): raise RuntimeError('dependency refused')\n"
    )
    with (root / "pm/receipt.py").open("a") as stream:
        stream.write("last_for_update = lambda update_id: {'update_id': update_id, 'outcome': 'refused', 'refusal': {'reason': 'dependency refused'}}\n")
    monkeypatch.setenv("HERMES_HOME", request["home"])
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kw: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda state: None)

    def complete(args, gateway_mode):
        update_receipt.begin_update_receipt()
        request["receipt"] = update_receipt._current.get().data
        update_cmd._complete_source_update(request)

    monkeypatch.setattr(update_cmd, "_cmd_update_impl", complete)
    with pytest.raises(SystemExit) as error:
        main.cmd_update(SimpleNamespace(gateway=True))
    assert error.value.code == 1
    receipt = update_receipt.read_latest_receipt()
    assert receipt["update_id"] == request["receipt"]["update_id"]
    assert receipt["pm_sync_outcome"] == "refused"
    assert receipt["pm_refusal"] == {"reason": "dependency refused"}


def test_bootstrap_does_not_initialize_old_site_packages(transition, tmp_path, monkeypatch):
    from hermes_cli import update_completion

    root, git, old, new, request = transition
    shutil.copy2(update_completion.__file__, root / "hermes_cli/update_completion.py")
    obsolete = tmp_path / "obsolete-python"
    venv.EnvBuilder(with_pip=False).create(obsolete)
    site = obsolete / ("Lib/site-packages" if os.name == "nt" else
                       f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    trap = tmp_path / "old-site-loaded"
    (site / "application.pth").write_text(f"import pathlib; pathlib.Path({str(trap)!r}).touch()\n")
    monkeypatch.setattr(sys, "executable", str(obsolete / ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
    result = update_completion.run_completion(request)
    assert result["exit_code"] == 0
    assert not trap.exists(), "preparation initialized the old application's .pth graph"


@pytest.mark.parametrize("stage", ["prepare", "selected"])
def test_progress_is_forwarded_before_held_stage_is_released(transition, monkeypatch, stage):
    import io
    import threading
    from hermes_cli import update_completion

    root, git, old, new, request = transition
    shutil.copy2(update_completion.__file__, root / "hermes_cli/update_completion.py")
    release = root / "release"
    released = root / "released"
    held_body = (
        "    print('STAGE_READY')\n"  # Deliberately no flush: -I ignores PYTHONUNBUFFERED.
        "    deadline = time.monotonic() + 20\n"
        "    while not Path('release').exists():\n"
        "        if time.monotonic() >= deadline: raise TimeoutError('stage was not released')\n"
        "        time.sleep(0.02)\n"
        "    Path('released').touch()\n"
    )
    module, definition = (
        ("pm/__init__.py", "def sync_venv(**kw):\n") if stage == "prepare" else
        ("hermes_cli/source_build.py", "def build_update_products(*a, **kw):\n")
    )
    (root / module).write_text("import time\nfrom pathlib import Path\n" + definition + held_body)
    observed = threading.Event()

    class Output(io.StringIO):
        def write(self, value):
            result = super().write(value)
            if "STAGE_READY" in self.getvalue():
                observed.set()
            return result

    output = Output()
    monkeypatch.setattr(sys, "stdout", output)
    responses, errors = [], []

    def run():
        try:
            responses.append(update_completion.run_completion(request))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert observed.wait(10), f"{stage} progress remained buffered: {output.getvalue()}"
        assert worker.is_alive(), "completion finished instead of waiting for release"
        assert not released.exists(), "stage passed its gate before progress was observed"
    finally:
        release.touch()
        worker.join(timeout=30)
    assert not worker.is_alive(), "completion did not stop after release"
    assert not errors
    assert released.exists()
    assert responses[0]["exit_code"] == 0


@pytest.mark.platforms("posix")
def test_interactive_configuration_keeps_terminal_input(transition):
    import pty
    import select
    import signal
    import time
    from hermes_cli import update_completion

    root, git, old, new, request = transition
    shutil.copy2(update_completion.__file__, root / "hermes_cli/update_completion.py")
    (root / "hermes_cli/update_cmd_maint.py").write_text(
        "import sys\nfrom hermes_cli.probe import event\n"
        "def _run_post_update_maintenance(**kw):\n"
        "    assert sys.stdin.isatty() and sys.stdout.isatty()\n"
        "    event('answer', value=input('CONFIG? '))\n"
        "    return True\n"
    )
    master, slave = pty.openpty()
    driver = "import json,runpy,sys; m=runpy.run_path(sys.argv[1]); raise SystemExit(m['run_completion'](json.loads(sys.argv[2]))['exit_code'])"
    proc = subprocess.Popen([sys.executable, "-c", driver, update_completion.__file__, json.dumps(request)],
                            cwd=root, stdin=slave, stdout=slave, stderr=slave)
    os.close(slave)
    output = b""
    try:
        deadline = time.monotonic() + 20
        while b"CONFIG?" not in output:
            assert time.monotonic() < deadline, output.decode(errors="replace")
            if select.select([master], [], [], 0.1)[0]:
                output += os.read(master, 8192)
        os.write(master, b"yes\n")
        assert proc.wait(timeout=20) == 0
        events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        assert next(e for e in events if e["name"] == "answer")["value"] == "yes"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        os.close(master)


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
def test_interrupt_reaps_completion_descendants_before_return(transition, monkeypatch):
    import io
    import psutil
    import time
    from hermes_cli import update_completion

    root, git, old, new, request = transition
    (root / "hermes_cli/update_completion.py").write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "print('READY ' + str(child.pid), flush=True)\n"
        "time.sleep(600)\n"
    )
    pids = []

    class Interrupt(io.StringIO):
        def write(self, value):
            if 'READY ' in value:
                pids.append(int(value.split('READY ')[1].strip()))
                raise KeyboardInterrupt()
            return super().write(value)

    monkeypatch.setattr(sys, "stdout", Interrupt())
    try:
        with pytest.raises(KeyboardInterrupt):
            update_completion.run_completion(request)
        assert pids
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pids[0]) or psutil.Process(pids[0]).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.02)
        else:
            pytest.fail("completion descendant survived parent cancellation")
    finally:
        for pid in pids:
            if psutil.pid_exists(pid):
                psutil.Process(pid).kill()


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("failure", ["launch", "timeout", "nonzero"])
def test_taskkill_failure_still_reaps_child_and_preserves_interrupt(tmp_path, monkeypatch, failure):
    """Only native Windows exercises taskkill dispatch and retained-handle kill."""
    import io
    from hermes_cli import update_completion

    package = tmp_path / "hermes_cli"
    package.mkdir()
    (package / "update_completion.py").write_text(
        "import time\nprint('READY', flush=True)\ntime.sleep(60)\n"
    )
    request = {"source": str(tmp_path), "home": str(tmp_path), "receipt": {"update_id": "test"}}
    interrupted = KeyboardInterrupt("cancel completion")
    child = None
    waits = []
    popen, run = subprocess.Popen, subprocess.run

    def capture_child(*args, **kwargs):
        nonlocal child
        proc = popen(*args, **kwargs)
        # The nonzero fault uses a real command, but it is not the completion child.
        if child is None:
            child = proc
            wait = proc.wait

            def bounded_wait(timeout=None):
                waits.append(timeout)
                return wait(timeout=timeout)

            monkeypatch.setattr(proc, "wait", bounded_wait)
        return proc

    def failed_taskkill(command, **kwargs):
        assert child is not None
        assert command == ["taskkill", "/T", "/F", "/PID", str(child.pid)]
        if failure == "launch":
            raise FileNotFoundError("taskkill unavailable")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        # Keep subprocess.run's real nonzero/check behavior, without killing a tree.
        return run([sys.executable, "-c", "raise SystemExit(9)"], **kwargs)

    class Interrupt(io.StringIO):
        def write(self, value):
            result = super().write(value)
            if "READY" in self.getvalue():
                raise interrupted
            return result

    monkeypatch.setattr(subprocess, "Popen", capture_child)
    monkeypatch.setattr(subprocess, "run", failed_taskkill)
    monkeypatch.setattr(sys, "stdout", Interrupt())
    try:
        with pytest.raises(KeyboardInterrupt) as error:
            update_completion.run_completion(request)
        assert error.value is interrupted
        expected = {"launch": FileNotFoundError, "timeout": subprocess.TimeoutExpired,
                    "nonzero": subprocess.CalledProcessError}[failure]
        assert isinstance(error.value.__cause__, expected)
        assert child is not None
        assert child.poll() is not None, "retained completion child survived failed taskkill"
        assert child.stdout.closed
        assert waits and all(timeout is not None and timeout > 0 for timeout in waits)
    finally:
        # Only this disposable child is ours; no service or process scan is involved.
        if child is not None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("correlated", [False, True])
def test_only_correlated_terminal_receipt_can_acknowledge_success(transition, encoding, correlated):
    from hermes_cli.update_completion import run_completion

    root, git, old, new, request = transition
    receipt = {"update_id": request["receipt"]["update_id"] if correlated else "wrong",
               "outcome": "success", "finished_at": "now", "detail": "日本 café"}
    (root / "hermes_cli/update_completion.py").write_text(
        "import json, pathlib, sys\n"
        "request = json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(\n"
        "    schema=1, update_id=request['receipt']['update_id'], exit_code=0,\n"
        f"    windows_resume={{}}, receipt={receipt!r}), ensure_ascii=False), encoding={encoding!r})\n",
        encoding="utf-8",
    )
    response = run_completion(request)
    assert response["exit_code"] == (0 if correlated else 1)
    assert response["receipt"] == (receipt if correlated else None)
