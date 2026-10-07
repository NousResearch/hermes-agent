"""Mutating plugin verbs refuse while a REAL gateway-shaped lock holder is live (#70473).

``hermes plugins update`` / ``install --force`` / ``remove`` (and the dashboard twins)
mutate an installed checkout in place. A running gateway has already loaded that
plugin's callbacks, so a pull that deletes a lazily-imported module breaks the next
tool call even though the command reported success. The verbs must fail BEFORE any
git operation when the gateway is live, and keep working when it is not.

The holder is a real child process holding ``gateway.lock`` through
``acquire_gateway_runtime_lock`` with a gateway-shaped command line, so the guard
exercises the platform's actual liveness probe (``resolve_gateway_liveness``), not a
stub — the same fixture shape as ``tests/gateway/test_status_strict_identity_live.py``.
"""

from __future__ import annotations

import queue
import os
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

from hermes_cli import plugins_cmd
from hermes_cli import plugins_cmd_update

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Blocks on stdin: exits as soon as the parent's pipe closes, even when pytest is
# killed and teardown never runs.
_HOLDER = """
import os, sys
sys.path.insert(0, {root!r})
from gateway import status
if not status.acquire_gateway_runtime_lock():
    sys.exit("could not acquire gateway.lock")
status.write_pid_file()
print(os.getpid(), flush=True)
sys.stdin.read()
"""


def _readline(stream, timeout: float = 20.0) -> str:
    got: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: got.put(stream.readline()), daemon=True).start()
    try:
        return got.get(timeout=timeout)
    except queue.Empty:
        return ""


@pytest.fixture(scope="module")
def live_gateway_home(tmp_path_factory):
    """A temp HERMES_HOME whose gateway identity files name a real live holder."""
    root = tmp_path_factory.mktemp("plugin-live-gateway")
    home = root / "home"
    (home / "plugins").mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    script = root / "bin" / "hermes"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_HOLDER.format(root=str(PROJECT_ROOT)), encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(script), "gateway", "run"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "HERMES_HOME": str(home)},
    )
    line = _readline(proc.stdout).strip()
    if not line.isdigit():
        with suppress(Exception):
            proc.kill()
        pytest.fail(f"lock holder failed to start: {line!r} {proc.communicate(timeout=10)[1]}")
    yield home
    holder = None
    with suppress(psutil.Error):
        holder = psutil.Process(int(line))
    for target in ([holder] if holder else []) + [psutil.Process(proc.pid)]:
        with suppress(psutil.Error):
            target.kill()
    with suppress(Exception):
        proc.communicate(timeout=10)


@pytest.fixture
def plugin_home(live_gateway_home, monkeypatch):
    """The holder's home with one git-installed plugin; HERMES_HOME points there."""
    monkeypatch.setenv("HERMES_HOME", str(live_gateway_home))
    target = live_gateway_home / "plugins" / "git-plugin"
    (target / ".git").mkdir(parents=True, exist_ok=True)
    (target / "plugin.yaml").write_text("name: git-plugin\nversion: '1.0'\n", encoding="utf-8")
    (target / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (live_gateway_home / "plugins" / ".install-metadata.json").write_text("{}", encoding="utf-8")
    return live_gateway_home


@pytest.fixture
def stopped_home(tmp_path, monkeypatch):
    """A home with no gateway at all — stopped-gateway behaviour must be unchanged."""
    home = tmp_path / "home"
    (home / "plugins").mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.mark.spawns_gateway_lookalike
class TestLiveGatewayGuard:
    def test_probe_reports_the_holder_live(self, plugin_home):
        """The integration fixture really is 'running' by the guard's own probe."""
        assert plugins_cmd._gateway_is_running() is True

    def test_cmd_update_refuses_before_git_with_a_live_gateway(self, plugin_home, monkeypatch):
        pulled = []
        monkeypatch.setattr(plugins_cmd_update, "_pull_plugin_update",
                            lambda *a, **k: pulled.append(1) or "Already up to date.")

        with pytest.raises(SystemExit) as exc:
            plugins_cmd.cmd_update("git-plugin")

        assert exc.value.code == 1
        assert pulled == []  # refused BEFORE any git mutation

    def test_cmd_update_override_flag_proceeds(self, plugin_home, monkeypatch):
        pulled = []
        monkeypatch.setattr(plugins_cmd_update, "_pull_plugin_update",
                            lambda *a, **k: pulled.append(1) or "Already up to date.")

        plugins_cmd.cmd_update("git-plugin", allow_live_gateway=True)

        assert pulled == [1]

    def test_dashboard_update_refuses_with_clear_message(self, plugin_home, monkeypatch):
        pulled = []
        monkeypatch.setattr(plugins_cmd_update, "_pull_plugin_update",
                            lambda *a, **k: pulled.append(1) or "Already up to date.")

        result = plugins_cmd.dashboard_update_user_plugin("git-plugin")

        assert result["ok"] is False
        assert "gateway is running" in result["error"]
        assert "hermes gateway stop" in result["error"]
        assert pulled == []

    def test_dashboard_remove_refuses_and_keeps_the_tree(self, plugin_home):
        result = plugins_cmd.dashboard_remove_user_plugin("git-plugin")

        assert result["ok"] is False
        assert "gateway is running" in result["error"]
        assert (plugin_home / "plugins" / "git-plugin" / "plugin.yaml").exists()

    def test_force_reinstall_refuses(self, plugin_home, monkeypatch):
        installed = []
        monkeypatch.setattr(plugins_cmd, "_install_plugin_core",
                           lambda *a, **k: installed.append(1) or (None, {}, "git-plugin"))

        with pytest.raises(SystemExit) as exc:
            plugins_cmd.cmd_install("some/plugin", force=True)

        assert exc.value.code == 1
        assert installed == []

    def test_plain_install_is_not_gated(self, plugin_home, monkeypatch):
        """A brand-new install creates no checkout a gateway could have loaded, so only the
        forced reinstall path is gated — the guard must not fire before `--force` does."""
        class _Sentinel(Exception):
            pass

        def _boom(*_a, **_k):
            raise _Sentinel

        monkeypatch.setattr(plugins_cmd, "_install_plugin_core", _boom)

        with pytest.raises(_Sentinel):
            plugins_cmd.cmd_install("some/plugin", force=False)  # reaches the clone — not gated
        with pytest.raises(SystemExit) as exc:
            plugins_cmd.cmd_install("some/plugin", force=True)   # gated before the clone

        assert exc.value.code == 1

    def test_stopped_gateway_update_is_unchanged(self, stopped_home, monkeypatch):
        target = stopped_home / "plugins" / "git-plugin"
        (target / ".git").mkdir(parents=True, exist_ok=True)
        (target / "plugin.yaml").write_text("name: git-plugin\nversion: '1.0'\n", encoding="utf-8")
        (stopped_home / "plugins" / ".install-metadata.json").write_text("{}", encoding="utf-8")
        pulled = []
        monkeypatch.setattr(plugins_cmd_update, "_pull_plugin_update",
                            lambda *a, **k: pulled.append(1) or "Updating 111..222")

        plugins_cmd.cmd_update("git-plugin")

        assert pulled == [1]
