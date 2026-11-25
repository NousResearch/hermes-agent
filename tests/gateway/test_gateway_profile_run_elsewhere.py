"""The gateway process refuses to run a profile a plugin runs elsewhere.

``hermes gateway run`` (and every service or client start that ends in it) reaches
``gateway.run_bootstrap.start_gateway``, which exits 78 before it reads the profile's config
when ``<root>/profiles_run_elsewhere/<profile>.json`` exists.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from hermes_cli import profiles_run_elsewhere as pre


@pytest.fixture
def root(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "ops").mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (home / "profiles" / "ops" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.mark.parametrize("profile", ["ops", "default"])
def test_start_gateway_exits_78_before_reading_config(root, monkeypatch, capsys, profile):
    from gateway import run_bootstrap
    from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE

    if profile != "default":
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / profile))
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setattr("gateway.run.load_gateway_config_for_runner",
                        lambda: pytest.fail("config read for a profile a plugin runs elsewhere"))
    pre.mark_profile_run_elsewhere(profile, "sandbox-plugin", root=root)

    with pytest.raises(SystemExit) as exited:
        asyncio.run(run_bootstrap.start_gateway())
    assert exited.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    out = capsys.readouterr().out
    assert f"'{profile}'" in out and "sandbox-plugin" in out
    assert "HERMES_EXEC_ASK" not in os.environ


@pytest.mark.parametrize("marked", ["ops", "default"])
def test_marked_siblings_keep_the_gateway_off_an_external_cron_registry(root, monkeypatch, marked):
    """An external provider reconciles one unscoped registry toward this home's jobs; a marked profile's
    backend elsewhere may arm its jobs there, so the gateway keeps the built-in ticker as for any host
    with more than one profile. With ``default`` marked the gateway is a standalone ``ops``: it ticks only
    ``ops`` and lists only ``default``, and the registry may still hold both profiles' jobs."""
    import threading
    import types
    from cron import scheduler_provider as sp
    from cron import scheduler_thread
    from gateway import run as gateway_run

    class _Ticker:
        def __init__(self, *_a, **_k):
            pass

        def start(self):
            return None

        def is_alive(self):
            return False

    external = types.SimpleNamespace(name="external", start=lambda *_a, **_k: None)
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: external)
    monkeypatch.setattr(scheduler_thread, "SupervisedTickerThread", _Ticker)
    monkeypatch.setattr(gateway_run, "_start_gateway_housekeeping", lambda *_a, **_k: None)
    launch = "default" if marked == "ops" else "ops"
    if launch == "ops":
        (root / "profiles" / "ops" / "config.yaml").write_text("gateway:\n  standalone: true\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    runner = types.SimpleNamespace(config=types.SimpleNamespace(multiplex_profiles=True), adapters={},
                                   _profile_adapters=None, _primary_profile_name=launch,
                                   _draining=False, _external_drain_active=False)
    pre.mark_profile_run_elsewhere(marked, "sandbox-plugin", root=root)

    async def _start():
        return gateway_run._start_gateway_start_cron_and_housekeeping(runner)

    cron_stop, provider, _thread, housekeeping = asyncio.run(_start())
    cron_stop.set()
    if isinstance(housekeeping, threading.Thread):
        housekeeping.join(timeout=5)
    assert isinstance(provider, sp.InProcessCronScheduler)


def test_an_unmarked_profile_passes_the_refusal(root):
    from gateway import run_bootstrap

    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin", root=root)
    assert run_bootstrap._refuse_profile_run_elsewhere() is None  # HERMES_HOME is the default
