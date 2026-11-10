"""LIVE Windows E2E: ``hermes plugins uninstall`` while a real gateway has the plugin loaded.

A real ``hermes gateway run`` child (temp HOME / HERMES_HOME) loads three enabled user plugins:
pure Python, one that keeps a file in its own directory open, and one that imports a native
``.pyd`` shipped inside its directory. Windows refuses to delete a file another process holds
(WinError 32) or a mapped DLL (WinError 5), so removing those trees under the live gateway is the
symptom. Each case uninstalls from a separate CLI process and asserts the user-visible contract:
the command succeeds, no loadable copy stays under ``plugins/`` (the next process start does not
see it) and the plugin is gone from config.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.platforms("windows"), pytest.mark.spawns_gateway_lookalike]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_REGISTER = """
def _probe(args, **kw):
    return "ok"


def register(ctx):
    ctx.register_tool(name="{tool}", toolset="{name}",
                      schema={{"name": "{tool}", "description": "probe",
                               "parameters": {{"type": "object", "properties": {{}}}}}},
                      handler=_probe)
"""
_HEADS = {
    "uninstall-plain": "",
    # A plugin that keeps its own log/cache file open for the life of the process.
    "uninstall-handle": "import os\n_LOG = open(os.path.join(os.path.dirname(__file__), 'cache.log'), 'a', encoding='utf-8')\n",
    # A plugin that ships a native extension inside its directory.
    "uninstall-native": "from . import select as _native  # noqa: F401\n",
}


def _native_extension() -> Path:
    found = sorted((Path(sys.base_prefix) / "DLLs").glob("select*.pyd"))
    if not found:
        pytest.fail(f"no stdlib select*.pyd under {sys.base_prefix}\\DLLs to vendor into the plugin")
    return found[0]


def _child_env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HERMES_", "PYTEST_")) and not k.endswith(("_API_KEY", "_TOKEN"))}
    env.update(HOME=str(home.parent), USERPROFILE=str(home.parent), HERMES_HOME=str(home),
               HERMES_GATEWAY_LOCK_DIR=str(home.parent / "locks"), PYTHONPATH=str(PROJECT_ROOT),
               PYTHONUTF8="1", NO_COLOR="1", HERMES_ACCEPT_HOOKS="1")
    return env


def _hermes(home: Path, *argv: str, timeout: float = 180) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", *argv], cwd=str(home),
                          env=_child_env(home), stdin=subprocess.DEVNULL, capture_output=True,
                          encoding="utf-8", errors="replace", timeout=timeout, check=False)


def _wait(pred, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.5)
    return bool(pred())


@pytest.fixture(scope="module")
def live_gateway(tmp_path_factory):
    root = tmp_path_factory.mktemp("uninstall-live")
    home = root / "fakehome" / ".hermes"
    plugins = home / "plugins"
    plugins.mkdir(parents=True)
    (root / "fakehome" / "locks").mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled:\n" + "".join(f"    - {n}\n" for n in _HEADS),
                                      encoding="utf-8")
    for name, head in _HEADS.items():
        d = plugins / name
        d.mkdir()
        tool = name.replace("-", "_")
        (d / "plugin.yaml").write_text(f"name: {name}\nversion: '1.0'\nprovides_tools:\n  - {tool}\n",
                                       encoding="utf-8")
        (d / "__init__.py").write_text(head + _REGISTER.format(name=name, tool=tool), encoding="utf-8")
    shutil.copy2(_native_extension(), plugins / "uninstall-native" / _native_extension().name)
    # Installed plugins carry an install record, which routes removal through the metadata-consistent path.
    (plugins / ".install-metadata.json").write_text(
        "{" + ", ".join(f'"{n}": {{"source": "local", "sha": "0"}}' for n in _HEADS) + "}", encoding="utf-8")

    log = open(root / "gateway.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "hermes_cli.main", "gateway", "run", "--force"],
                            cwd=str(home), env=_child_env(home), stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    from gateway.control_socket import identify_gateway, reload_gateway_plugins
    try:
        if not _wait(lambda: proc.poll() is not None or identify_gateway(home, timeout=2.0), 180):
            pytest.fail("gateway never answered identify")
        assert proc.poll() is None, (root / "gateway.log").read_text(encoding="utf-8", errors="replace")[-3000:]
        answer = reload_gateway_plugins(home) or {}
        missing = set(_HEADS) - set(answer.get("plugins") or ())
        assert not missing, f"gateway did not load {missing}: {answer}"
        yield home, proc
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=30, check=False)
        proc.wait(timeout=30)
        log.close()


@pytest.mark.parametrize("name", list(_HEADS))
def test_uninstall_while_gateway_has_plugin_loaded(live_gateway, name):
    home, proc = live_gateway
    result = _hermes(home, "plugins", "uninstall", name)
    print(f"UNINSTALL {name}: rc={result.returncode}\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")

    assert proc.poll() is None, "the gateway died during uninstall"
    assert result.returncode == 0, f"uninstall failed: {result.stdout[-800:]} {result.stderr[-800:]}"
    leftovers = sorted(str(p.relative_to(home / "plugins")) for p in (home / "plugins").rglob("plugin.yaml"))
    assert not any(name in p for p in leftovers), f"a loadable copy stayed under plugins/: {leftovers}"
    listing = _hermes(home, "plugins", "list")
    assert name not in listing.stdout, f"the next process still sees {name}:\n{listing.stdout[-1500:]}"
    assert name not in (home / "config.yaml").read_text(encoding="utf-8")
