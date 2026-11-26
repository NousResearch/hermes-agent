"""The stop and restart paths read every module they import from memory, never from disk.

A running gateway outlives an in-place upgrade (a package manager or a checkout move replaces the
files under it). A function-level import that first runs during the stop then loads the NEWER file
against the older modules already in memory, and a name that file imports from them may not exist
yet: the stop raises and its teardown is lost (#159067). So boot loads the modules the stop path
imports lazily (``preload_stop_path_modules``); this test boots a gateway in a child process, runs a
restart's idle wait and a full stop, and fails on any repo module imported for the first time after boot.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_CHILD = r'''
import asyncio, json, os, signal, sys, threading, time
from pathlib import Path
repo, stop = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(repo))
import hermes_cli.env_loader as env_loader
env_loader.load_hermes_dotenv = lambda **_kwargs: []
from gateway import restart_when_idle, run_api, run_bootstrap, run_runtime
from gateway.config import GatewayConfig
from gateway.run_shutdown import GatewayShutdownMixin
from gateway.status import write_planned_stop_marker
from hermes_constants import get_hermes_home
from tests.waiting import async_wait_until

late = []

class FirstImportAfterBoot:
    """Records repo modules imported for the first time; never loads anything itself."""
    armed = False

    def find_spec(self, name, path=None, target=None):
        base = repo.joinpath(*name.split("."))
        if self.armed and name not in sys.modules and (
                base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()):
            late.append(name)
        return None

recorder = FirstImportAfterBoot()
sys.meta_path.insert(0, recorder)
stopped = threading.Event()

async def serve_until_stop(runner):
    # Background boot work (the dashboard server, listeners) keeps importing for a moment after the
    # runner starts; arm once it settles, so only the restart wait and the stop are measured.
    quiet = {"count": len(sys.modules), "since": time.monotonic()}

    def settled():
        if len(sys.modules) != quiet["count"]:
            quiet.update(count=len(sys.modules), since=time.monotonic())
        return time.monotonic() - quiet["since"] >= 1.0

    await async_wait_until(settled, timeout=60, interval=0.1, what="boot imports to settle")
    recorder.armed = True
    await restart_when_idle.await_idle(runner)
    GatewayShutdownMixin._restart_watcher_env()  # a detached restart builds its helper's environment
    if stop == "planned_sigterm":  # the CLI's planned stop: the planned-stop marker, then SIGTERM
        write_planned_stop_marker(os.getpid())
        os.kill(os.getpid(), signal.SIGTERM)
        await runner.wait_for_shutdown()

run_bootstrap._start_gateway_start_cron_and_housekeeping = lambda runner: (
    stopped, None, threading.Thread(target=stopped.wait, daemon=True),
    threading.Thread(target=stopped.wait, daemon=True))
run_runtime.wait_gateway_runtime = serve_until_stop
# The dashboard mounts in the background after READY and imports for as long as that takes under
# load; it is not part of a stop, so it stays unmounted here.
run_api.warm_gateway_dashboard = lambda runner: None
get_hermes_home().chmod(0o700)
try:
    verdict = asyncio.run(asyncio.wait_for(run_bootstrap.start_gateway(GatewayConfig()), 120))
except SystemExit as exc:  # a failed stop ends here; report it so the test fails on it
    verdict = f"SystemExit({exc.code})"
finally:
    stopped.set()
print(json.dumps({"armed": recorder.armed, "verdict": verdict, "late": sorted(set(late))}))
sys.stdout.flush()
os._exit(0)
'''


@pytest.mark.asyncio
async def test_boot_runs_the_preload_before_the_runner_starts(monkeypatch):
    import asyncio
    import threading

    from gateway import run_bootstrap, run_runtime
    from gateway.config import GatewayConfig
    from hermes_constants import get_hermes_home

    get_hermes_home().chmod(0o700)
    order, stopped = [], threading.Event()
    real_start = run_bootstrap._start_gateway_run_runner

    async def start_runner(*args, **kwargs):
        order.append("runner")
        return await real_start(*args, **kwargs)

    async def serve_until_stop(runner):
        return None

    monkeypatch.setattr(run_bootstrap, "preload_stop_path_modules", lambda: order.append("preload"))
    monkeypatch.setattr(run_bootstrap, "_start_gateway_run_runner", start_runner)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_start_cron_and_housekeeping",
                        lambda runner: (stopped, None, threading.Thread(target=stopped.wait, daemon=True),
                                        threading.Thread(target=stopped.wait, daemon=True)))
    monkeypatch.setattr(run_runtime, "wait_gateway_runtime", serve_until_stop)
    try:
        await asyncio.wait_for(run_bootstrap.start_gateway(GatewayConfig()), 30)
    except SystemExit:
        pass
    finally:
        stopped.set()
    assert order[:2] == ["preload", "runner"]


def test_a_stop_path_module_that_no_longer_imports_is_reported_at_boot(monkeypatch, caplog):
    from gateway import run_bootstrap

    monkeypatch.setattr(run_bootstrap, "STOP_PATH_MODULES", ("gateway.no_such_stop_path_module",))
    with caplog.at_level("WARNING", logger="gateway.run"):
        run_bootstrap.preload_stop_path_modules()
    assert any("gateway.no_such_stop_path_module" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("stop", ["return", pytest.param("planned_sigterm", marks=pytest.mark.platforms("linux", "macos"))])
def test_a_stop_imports_no_repo_module_boot_did_not_load(tmp_path, stop):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")
    (tmp_path / "managed").mkdir()
    env = {key: value for key, value in os.environ.items() if key != "PYTEST_CURRENT_TEST"}
    env.update(HERMES_HOME=str(home), HERMES_MANAGED_DIR=str(tmp_path / "managed"),
               HERMES_GATEWAY_LOCK_DIR=str(tmp_path / "locks"))
    child = tmp_path / "child.py"
    child.write_text(_CHILD, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(child), str(REPO), stop], cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, encoding="utf-8", errors="replace", timeout=180, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["armed"], "the child gateway never finished booting:\n" + proc.stderr[-3000:]
    assert result["verdict"] is True, f"the stop did not complete cleanly: {result['verdict']}\n" + proc.stderr[-3000:]
    assert not result["late"], "imported for the first time during the stop (add them to STOP_PATH_MODULES):\n  " + \
        "\n  ".join(result["late"])
