"""A hosted gateway boots READY with the shared-board features disabled.

The shared board refuses to open wherever the process serves hosted tenants, so
the gateway's kanban loops must not crash at boot (the dispatcher's
``kanban_home()`` used to raise there) and must not retry a refused board open
every tick (the notifier's enumeration): each disables itself with one log line
before it touches any board path, and the gateway still comes up READY. On a
single-user machine nothing changes — the same loops dispatch and notify.
"""
import asyncio
import sys
import types

import pytest


def _serve_hosted_tenants(monkeypatch):
    """This process reports itself hosted the way the boot profile does
    (``hermes_cli.hosted_mode.is_hosted`` — absent from a checkout without that module, which is
    why the shared-board boundary's guarded import answers False there)."""
    module = types.ModuleType("hermes_cli.hosted_mode")
    module.is_hosted = lambda: True
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.asyncio
async def test_hosted_gateway_boots_ready_with_kanban_disabled(tmp_path, monkeypatch, caplog):
    """A real gateway start in a hosted process: READY, no exception, and the
    kanban dispatcher and notifier each disabled with one log line instead of a
    crash or an endless refused-open retry loop."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_runtime import (initialize_gateway_runtime, start_gateway_runtime_api,
                                     publish_gateway_runtime_ready)
    from gateway.runtime_ownership import process_ownership
    from gateway.control_socket import GatewayControlServer
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "shared-board"))
    _serve_hosted_tenants(monkeypatch)
    caplog.set_level("INFO")
    process_ownership.reserve([tmp_path])
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    runner.session_control_server = GatewayControlServer()
    try:
        assert await runner.session_control_server.start()
        await initialize_gateway_runtime(runner)
        await start_gateway_runtime_api(runner)
        started = await runner.start()
        publish_gateway_runtime_ready(runner)
        assert started is True, "the hosted gateway must come up READY, kanban disabled"
        assert "kanban dispatcher: disabled in a hosted process" in caplog.text
        assert "kanban notifier: disabled in a hosted process" in caplog.text
        assert "LookupError" not in caplog.text
    finally:
        await runner.stop()
        await runner.session_control_server.stop()
        process_ownership.close()


@pytest.mark.asyncio
async def test_local_gateway_boot_unchanged_still_runs_the_kanban_loops(tmp_path, monkeypatch):
    """The control: not hosted, the same boot keeps the loops armed — the
    dispatcher's boot resolves the board root instead of disabling."""
    from hermes_cli import config
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin
    from gateway.kanban_watchers_common import _release_singleton_lock
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "shared-board"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)
    monkeypatch.setattr(config, "load_config", dict)
    runner = GatewayKanbanWatchersMixin()
    try:
        boot = runner._kanban_dispatcher_boot()
        assert boot is not None, "locally the dispatcher boot must still resolve the board"
        _load_config, _kb, kanban_cfg = boot
        assert kanban_cfg == {}
    finally:
        _release_singleton_lock(getattr(runner, "_kanban_dispatcher_lock_handle", None))


def test_hosted_dispatcher_boot_disables_before_resolving_the_board_root(tmp_path, monkeypatch):
    """The dispatcher's boot path alone: with the process hosted it returns the
    disabled answer (None) before ``kanban_home()`` — whose hosted refusal is
    what used to raise here — is ever reached."""
    from hermes_cli import config
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "shared-board"))
    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)
    monkeypatch.setattr(config, "load_config", dict)
    _serve_hosted_tenants(monkeypatch)
    assert GatewayKanbanWatchersMixin()._kanban_dispatcher_boot() is None


def test_hosted_notifier_watcher_exits_before_any_board_open(tmp_path, monkeypatch):
    """The notifier's loop alone: hosted, it returns before its first tick —
    before any board enumeration or subscription read, with no retry."""
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "shared-board"))
    _serve_hosted_tenants(monkeypatch)

    class _Stub(GatewayKanbanWatchersMixin):
        _running = False  # the loop would exit after its first tick anyway

    async def _probe():
        import gateway.kanban_watchers_notifier as notifier
        calls = []

        def _collect(*args, **kwargs):
            calls.append((args, kwargs))
            return []

        monkeypatch.setattr(notifier, "_notifier_collect", _collect)
        await _Stub()._kanban_notifier_watcher()
        assert not calls, "a hosted notifier tick must never enumerate the board"

    asyncio.run(_probe())
