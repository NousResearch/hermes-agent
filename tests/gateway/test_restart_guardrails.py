"""Guardrails against the 2026-10-25 incident: a Python-ABI mismatch (a PM generation built
for CPython 3.14 selected by a 3.11 launcher) killed every spawned gateway mid-boot, and the
~30s crash loop's ~975 exits each broadcast "gateway shutting down" to the home channel.

Layers asserted here:
- a committed PM generation built for another Python is never activated;
- a gateway that stops seconds after boot (restart churn) does not broadcast;
- a genuine long-running stop still broadcasts.
"""

import contextlib
import os
import sys
import time

import pytest

import pm.environments as pm_environments
from gateway.config import HomeChannel, Platform
from tests.gateway.restart_test_helpers import make_restart_runner


def _venv(path, version):
    (path / "Lib" / "site-packages").mkdir(parents=True)
    (path / "pyvenv.cfg").write_text(f"home = C:\\Python\nversion_info = {version}\n", encoding="utf-8")
    return path


_OTHER_MINOR = 11 if sys.version_info.minor != 11 else 12


def _stub_runtime_selection(monkeypatch, tmp_path, generation):
    """Point activate_dependencies at *generation* without touching real PM state."""
    state = tmp_path / "install-state"
    state.mkdir(exist_ok=True)
    monkeypatch.setattr(pm_environments, "install_state_dir", lambda _root: state)
    monkeypatch.setattr(pm_environments, "committed_venv", lambda _root: generation)
    monkeypatch.setattr("hermes_cli.runtime_state.runtime_lock", lambda _root: contextlib.nullcontext(True))
    monkeypatch.setattr("hermes_cli.runtime_state.recover_publication", lambda _root: None)
    monkeypatch.setattr("hermes_cli.runtime_state.lease_generation", lambda _env: (lambda: None))
    # activate_dependencies rewrites these; record values so teardown restores them.
    monkeypatch.setenv("PYTHONPATH", "sentinel")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setenv("VIRTUAL_ENV", "sentinel")


@pytest.mark.platforms("windows")
def test_generation_for_another_python_is_never_activated(tmp_path, monkeypatch):
    """A cp314 generation must not be selected by a 3.11 process (and vice versa): the swap
    dies at the first native import (pydantic_core) and used to kill the gateway mid-boot."""
    root = tmp_path / "root"
    root.mkdir()
    generation = _venv(tmp_path / "gen", f"{sys.version_info.major}.{_OTHER_MINOR}.0")
    _stub_runtime_selection(monkeypatch, tmp_path, generation)
    monkeypatch.setattr(sys, "path", list(sys.path))

    before = list(sys.path)
    pm_environments.activate_dependencies(root)

    assert sys.path == before, "a mismatched generation must not rewrite the import path"
    assert str(pm_environments.site_packages(generation)) not in sys.path


@pytest.mark.platforms("windows")
def test_generation_for_the_running_interpreter_is_activated(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    generation = _venv(tmp_path / "gen", f"{sys.version_info.major}.{sys.version_info.minor}.0")
    _stub_runtime_selection(monkeypatch, tmp_path, generation)
    monkeypatch.setattr(sys, "path", list(sys.path))

    pm_environments.activate_dependencies(root)

    assert sys.path[0] == str(root.resolve())
    assert sys.path[1] == str(pm_environments.site_packages(generation))


@pytest.mark.asyncio
async def test_short_uptime_shutdown_suppresses_home_broadcast():
    """A process that stops seconds after boot is restart churn: no home-channel broadcast.
    One broken spawn loop sent ~975 of these in a night (2026-10-25)."""
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-chat",
        name="Telegram Home",
    )
    runner._gateway_started_at = time.time() - 10.0

    await runner._notify_active_sessions_of_shutdown()

    assert adapter.sent_calls == []


@pytest.mark.asyncio
async def test_long_uptime_shutdown_still_broadcasts_home_channel():
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-chat",
        name="Telegram Home",
    )
    runner._gateway_started_at = time.time() - 3600.0

    await runner._notify_active_sessions_of_shutdown()

    assert [call[0] for call in adapter.sent_calls] == ["home-chat"]
