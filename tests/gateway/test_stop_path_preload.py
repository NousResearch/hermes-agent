"""The stop and restart paths read every module they import from memory, never from disk.

A running gateway outlives an in-place upgrade (a package manager or a checkout move replaces the
files under it). A function-level import that first runs during the stop then loads the NEWER file
against the older modules already in memory, and a name that file imports from them may not exist
yet: the stop raises and its teardown is lost (#159067). So every repo module the stop path imports
inside a function is loaded at boot (``preload_stop_path_modules``); this test fails when a stop-path
function gains an import that boot does not load.
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Entry points of the stop and restart paths, per file. Functions they reach in the same file (by
# name or attribute) are followed; calls into other files are not.
STOP_PATH_ENTRIES = {
    "gateway/run_shutdown.py": {"stop", "request_restart", "_describe_active_work", "_await_active_work_before_restart"},
    "gateway/run_shutdown_home_notice.py": {"settle_home_online_notice_debt"},
    "gateway/run_runtime.py": {"drain_gateway_runtime", "settle_gateway_runtime"},
    "gateway/restart_when_idle.py": {"request_restart", "request_service_restart", "force_restart"},
    "gateway/shutdown_watchdog.py": {"arm_shutdown_watchdog", "disarm_after_teardown"},
    "gateway/run_bootstrap.py": {"_start_gateway_shutdown_tail"},
}

_CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import hermes_cli.env_loader as env_loader
env_loader.load_hermes_dotenv = lambda **_kwargs: []
import gateway.run, gateway.run_bootstrap
from gateway import shutdown_watchdog
getattr(shutdown_watchdog, "preload_stop_path_modules", lambda: None)()
print(json.dumps(sorted(sys.modules)))
'''


@pytest.mark.asyncio
async def test_boot_runs_the_preload_before_the_runner_starts(monkeypatch):
    import asyncio
    import threading

    from gateway import run_bootstrap, run_runtime, shutdown_watchdog
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

    monkeypatch.setattr(shutdown_watchdog, "preload_stop_path_modules", lambda: order.append("preload"))
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


def _run_child(tmp_path, src):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")
    (tmp_path / "managed").mkdir()
    env = {key: value for key, value in os.environ.items() if key != "PYTEST_CURRENT_TEST"}
    env.update(HERMES_HOME=str(home), HERMES_MANAGED_DIR=str(tmp_path / "managed"),
               HERMES_GATEWAY_LOCK_DIR=str(tmp_path / "locks"))
    child = tmp_path / "child.py"
    child.write_text(src, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(child), str(REPO)], cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, encoding="utf-8", errors="replace", timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _repo_module(name: str) -> bool:
    base = REPO.joinpath(*name.split("."))
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def _stop_path_imports() -> dict:
    """Repo modules imported inside the functions the stop path reaches, mapped to one import site."""
    found = {}
    for rel, entries in STOP_PATH_ENTRIES.items():
        defs = {}
        for node in ast.walk(ast.parse((REPO / rel).read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.setdefault(node.name, []).append(node)
        seen, todo = set(), list(entries)
        while todo:
            name = todo.pop()
            if name in seen or name not in defs:
                continue
            seen.add(name)
            for node in (n for fn in defs[name] for n in ast.walk(fn)):
                ref = node.id if isinstance(node, ast.Name) else getattr(node, "attr", None)
                if ref in defs:
                    todo.append(ref)
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                for module in filter(_repo_module, names):
                    found.setdefault(module, f"{rel}:{node.lineno} ({name})")
    return found


def test_every_stop_path_import_is_loaded_at_boot(tmp_path):
    loaded = set(_run_child(tmp_path, _CHILD))
    imports = _stop_path_imports()
    missing = {module: site for module, site in imports.items() if module not in loaded}
    assert len(imports) > 20  # the walk found the stop path
    assert not missing, "stop-path imports boot does not load (add them to STOP_PATH_MODULES):\n" + "\n".join(
        f"  {module}  <- {site}" for module, site in sorted(missing.items()))
