"""The notifier watcher honours the ``kanban.notify_in_gateway`` config gate.

Sibling of the dispatcher's ``dispatch_in_gateway`` gate: both watchers are
spawned from the same startup tuple, and a home that has no notify
subscriptions (e.g. a secondary profile's gateway) previously paid a permanent
one-DEBUG-line-per-5s-tick with no config key to stop it.
"""

import asyncio
import logging
from unittest.mock import MagicMock, patch

from gateway.config import Platform
from gateway.run import GatewayRunner

import hermes_cli.config as _cfg_mod

LOGGER_NAME = "gateway.run"


def _make_runner(with_adapter=False):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: MagicMock()} if with_adapter else {}
    runner._kanban_sub_fail_counts = {}
    return runner


class _StopLoop(Exception):
    """Raised from the first pre-tick sleep to unwind the watcher deterministically."""


def _run_watcher(monkeypatch, runner, cfg=None, *, env=None, load_fn=None):
    """Run the watcher to (and just past) its gates; returns (passed_gates).

    ``passed_gates`` is non-empty when execution reached the pre-tick wiring
    delay — i.e. every gate (env override, config load, config flag, imports)
    let it through.
    """
    if env is not None:
        monkeypatch.setenv("HERMES_KANBAN_NOTIFY_IN_GATEWAY", env)
    else:
        monkeypatch.delenv("HERMES_KANBAN_NOTIFY_IN_GATEWAY", raising=False)

    if load_fn is None:

        def _default_load():
            return cfg if cfg is not None else {"kanban": {}}

        load_fn = _default_load

    monkeypatch.setattr(_cfg_mod, "load_config", load_fn)

    passed_gates = []

    async def fake_sleep(delay):
        passed_gates.append(delay)
        runner._running = False
        raise _StopLoop

    async def fake_to_thread(fn, *args, **kwargs):
        return []

    async def run():
        with (
            patch("asyncio.sleep", side_effect=fake_sleep),
            patch("asyncio.to_thread", side_effect=fake_to_thread),
        ):
            try:
                await runner._kanban_notifier_watcher()
            except _StopLoop:
                pass

    asyncio.run(run())
    return passed_gates


def test_notifier_watcher_disabled_by_config(monkeypatch, caplog):
    """``kanban.notify_in_gateway: false`` must stop the loop before tick 1."""
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        runner = _make_runner(with_adapter=True)
        passed = _run_watcher(
            monkeypatch, runner, cfg={"kanban": {"notify_in_gateway": False}}
        )

    assert not passed, "watcher must return before the first tick when gated off"
    assert "kanban.notify_in_gateway=false" in caplog.text, (
        "an explicit disable must be visible at INFO"
    )


def test_notifier_watcher_disabled_by_env(monkeypatch, caplog):
    """``HERMES_KANBAN_NOTIFY_IN_GATEWAY=0`` must short-circuit before config load."""
    loaded_config = []

    def _track_load():
        loaded_config.append(True)
        return {"kanban": {}}

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        runner = _make_runner(with_adapter=True)
        passed = _run_watcher(monkeypatch, runner, env="0", load_fn=_track_load)

    assert not passed, "env override must stop the loop before the first tick"
    assert not loaded_config, "env disable must win without reading config.yaml"
    assert "HERMES_KANBAN_NOTIFY_IN_GATEWAY" in caplog.text, (
        "env disable must be visible at INFO"
    )


def test_notifier_watcher_enabled_by_default(monkeypatch):
    """With the flag unset the watcher must reach its polling loop as before."""
    runner = _make_runner(with_adapter=True)

    passed = _run_watcher(monkeypatch, runner, cfg={"kanban": {}})

    assert passed, "default must keep notifier polling for this profile's subscriptions"


def test_notifier_watcher_env_truthy_value_does_not_disable(monkeypatch):
    """Only the documented falsy spellings disable; arbitrary values don't."""
    runner = _make_runner(with_adapter=True)

    passed = _run_watcher(monkeypatch, runner, cfg={"kanban": {}}, env="1")

    assert passed, "env value '1' must not disable the notifier"


def test_notifier_watcher_config_load_failure_disables(monkeypatch, caplog):
    """A config that cannot be loaded must fail closed: no notifier loop."""

    def _boom():
        raise RuntimeError("config unreadable")

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        runner = _make_runner(with_adapter=True)
        passed = _run_watcher(monkeypatch, runner, load_fn=_boom)

    assert not passed, "config-load failure must disable the notifier, not run ungated"
    assert "cannot load config" in caplog.text, (
        "a config-load failure must say why the notifier is off"
    )
