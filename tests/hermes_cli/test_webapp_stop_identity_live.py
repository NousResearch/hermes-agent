"""Webapp stop must distinguish a real server from another Python program's argv."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

import psutil
import pytest


# Use the real server/lifespan/ledger, without renderer builds or external services.
_SERVER = '''
import os
import sys
sys.path.insert(0, os.getcwd())
def audit(event, args):
    if event == "socket.connect":
        address = args[1]
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError("external egress forbidden in test server")
    if event == "os.killpg":
        raise RuntimeError("process-group signals forbidden in test server")
    if event == "os.kill" and args[1] != 0:
        import psutil
        owned = {os.getpid(), *(p.pid for p in psutil.Process().children(recursive=True))}
        if args[0] not in owned:
            raise RuntimeError("signal outside test server tree")
sys.addaudithook(audit)
from hermes_cli.web_server import start_server
surface = sys.argv[1]
start_server(host="127.0.0.1", port=0, open_browser=False, ui_surface=surface)
'''


def _health(port):
    with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as response:
        assert response.status == 200
        assert json.loads(response.read())["ok"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("registered", [True, False], ids=["ledger", "argv-only"])
def test_webapp_stop_spares_argument_tail_and_stale_identity_but_stops_real_server(tmp_path, monkeypatch, registered):
    from hermes_cli import dashboard_procs, main_dashboard, process_identity
    from hermes_cli.update_cmd_windows import _hermes_holder_subcommand, _live_argv
    from hermes_constants import get_hermes_home

    home = tmp_path / "hermes-home"
    home.mkdir()
    os_home = tmp_path / "os-home"
    os_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HOME", str(os_home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    for key in ("HERMES_DESKTOP", "HERMES_DESKTOP_CHILD_PID", "HERMES_PARENT_PID", "HERMES_SPAWN"):
        monkeypatch.delenv(key, raising=False)
    assert get_hermes_home() == home
    assert process_identity._ledger_path().parent == home

    python_home = tmp_path / "Python Home"
    python_home.symlink_to(Path(sys.executable).parent.parent, target_is_directory=True)
    python = python_home / "bin" / Path(sys.executable).name
    launcher_dir = tmp_path / "Hermes App"
    launcher_dir.mkdir()
    launcher = launcher_dir / "hermes"
    launcher.write_text(_SERVER, encoding="utf-8")
    children, owned, ports = {}, {}, {}
    real_kill = os.kill
    signals = []

    def confined_kill(pid, sig):
        if sig:
            assert pid in owned, f"refusing signal outside test children: {pid}"
            assert psutil.Process(pid).create_time() == owned[pid]
            signals.append(pid)
        return real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", confined_kill)
    with contextlib.ExitStack() as stack:
        try:
            for surface in ("webapp", "serve", "unrelated"):
                env = os.environ.copy()
                ready = tmp_path / f"{surface}-ready.json"
                env["HERMES_DESKTOP_READY_FILE"] = str(ready)
                argv = [str(python), "-u", "-W", "ignore", "-X", "dev", str(launcher), surface, "--port", "0"]
                if surface == "unrelated":
                    argv = [sys.executable, "-c", "import time; time.sleep(120)", "hermes", "webapp"]
                log_path = tmp_path / f"{surface}.log"
                log = stack.enter_context(log_path.open("w", encoding="utf-8"))
                child = subprocess.Popen(argv, cwd=Path(__file__).resolve().parents[2], env=env,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                children[surface] = child
                owned[child.pid] = psutil.Process(child.pid).create_time()
                if surface == "unrelated":
                    continue
                deadline = time.monotonic() + 60
                while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                assert ready.exists(), f"{surface} failed to start: {log_path.read_text(encoding='utf-8-sig')}"
                ports[surface] = json.loads(ready.read_text(encoding="utf-8-sig"))["port"]
                _health(ports[surface])
                live_argv = _live_argv(psutil, child.pid, "")
                assert live_argv is not None
                assert _hermes_holder_subcommand(live_argv) == surface
                for descendant in psutil.Process(child.pid).children(recursive=True):
                    owned[descendant.pid] = descendant.create_time()

            entries = process_identity.ledger_entries(verified_only=True)
            for surface in ("webapp", "serve"):
                assert any(e["pid"] == children[surface].pid and e["purpose"] == surface
                           and e["create_time"] == owned[e["pid"]] for e in entries)
            unrelated = children["unrelated"]
            assert dashboard_procs._hermes_home_for_pid(unrelated.pid) == str(home)
            # A leftover record for an older incarnation cannot turn this child into a server.
            assert process_identity.register_child(unrelated.pid, "webapp")
            ledger_path = process_identity._ledger_path()
            records = json.loads(ledger_path.read_text(encoding="utf-8-sig"))
            for entry in records:
                if entry["pid"] == unrelated.pid:
                    entry["create_time"] -= 10
            ledger_path.write_text(json.dumps(records), encoding="utf-8")
            assert unrelated.pid not in {e["pid"] for e in process_identity.ledger_entries()}

            if not registered:
                ledger_path.write_text("[]", encoding="utf-8")
            monkeypatch.setenv("COLUMNS", "40")  # ps must not truncate a long interpreter path.
            scanned = dict(dashboard_procs._scan_dashboard_processes())
            for surface in ("webapp", "serve"):
                runtime = main_dashboard._parse_dashboard_runtime(scanned[children[surface].pid])
                assert runtime is not None and runtime[0] == surface
            if registered:
                # A live ledger still provides purpose if the process table is unavailable.
                with monkeypatch.context() as patch:
                    patch.setattr(dashboard_procs, "_iter_process_table", lambda: [])
                    ledger_scan = dict(dashboard_procs._scan_dashboard_processes())
                for surface in ("webapp", "serve"):
                    runtime = main_dashboard._parse_dashboard_runtime(ledger_scan[children[surface].pid])
                    assert runtime is not None and runtime[0] == surface

            with pytest.raises(SystemExit) as exc:
                main_dashboard.cmd_webapp(argparse.Namespace(status=False, stop=True))
            assert exc.value.code == 0
            assert children["webapp"].wait(timeout=15) in (0, -signal.SIGTERM)
            assert children["serve"].poll() is None
            _health(ports["serve"])
            assert unrelated.poll() is None, "webapp --stop killed an unrelated Python invocation"
            assert unrelated.pid not in signals
            assert children["webapp"].pid in signals
            assert unrelated.pid not in dict(dashboard_procs._scan_dashboard_processes())
        finally:
            for child in children.values():
                if child.poll() is None:
                    for descendant in psutil.Process(child.pid).children(recursive=True):
                        owned[descendant.pid] = descendant.create_time()
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                child.wait(timeout=5)
            for pid, created in owned.items():
                with contextlib.suppress(psutil.NoSuchProcess):
                    child = psutil.Process(pid)
                    if child.create_time() == created and child.status() != psutil.STATUS_ZOMBIE:
                        child.kill()
                        child.wait(timeout=5)
