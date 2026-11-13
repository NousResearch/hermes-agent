"""Gateway startup remains fail-open when PM admission itself fails.

Healthy/damaged real-store activation belongs to tests/pm/test_startup_activation.py.
"""
from unittest.mock import AsyncMock

import pytest


def test_gateway_main_survives_pm_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("pm.paths.install_root", lambda: tmp_path)
    import pm
    import hermes_agent.gateway.run as gateway

    failed = []
    def broken():
        failed.append("activate")
        raise RuntimeError("PM unavailable")

    monkeypatch.setattr(pm, "activate", broken)
    started = AsyncMock(return_value=True)
    exited = []
    monkeypatch.setattr(gateway, "start_gateway", started)
    monkeypatch.setattr(gateway, "_exit_after_graceful_shutdown", exited.append)
    monkeypatch.setattr("hermes_cli.boot_bootstrap.maybe_run_boot_bootstrap", lambda root: None)
    monkeypatch.setattr("sys.argv", ["gateway"])
    gateway.main()
    started.assert_awaited_once()
    assert exited == [0]
    assert failed == ["activate"]


@pytest.mark.asyncio
async def test_start_gateway_never_prompts_for_a_lazy_install_on_an_inherited_tty(monkeypatch):
    """A tmux respawn loop / hidden Windows console hands the gateway a TTY; a missing extra
    must not block the event loop on input() (#125607). Drives the real ensure_import."""
    import builtins
    import sys
    from types import SimpleNamespace

    import hermes_agent.gateway.run_bootstrap as bootstrap
    from pm import client, extras

    prompts, synced = [], []
    monkeypatch.setattr(extras, "available", lambda _: False)
    monkeypatch.setattr(client, "sync_venv", lambda x=None: synced.append(list(x or [])))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(builtins, "input", lambda text: prompts.append(text) or "n")

    async def runtime(*args):
        extras.ensure_import("fal")
        return True

    monkeypatch.setattr(bootstrap, "_start_gateway_runtime", runtime)
    import hermes_agent.gateway.run as gateway
    assert await gateway.start_gateway() is True
    assert prompts == [] and synced == [["fal"]]
