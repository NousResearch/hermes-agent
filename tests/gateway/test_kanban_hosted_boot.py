"""A hosted gateway boots READY with the shared-board features disabled.

The shared board refuses to open wherever the process serves hosted tenants, so
the gateway's kanban loops must not crash at boot (the dispatcher's
``kanban_home()`` used to raise there) and must not retry a refused board open
every tick (the notifier's enumeration, the idle-exit probe's board busy leg):
each disables itself with one log line before it touches any board path, and
the gateway still comes up READY. The boot test enumerates every kanban
background part a GatewayRunner starts and asserts each is disabled or never
started, with no refused-board LookupError logged across a few ticks. On a
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


#: The census: every kanban background part a GatewayRunner starts, pinned at
#: the entry point that would run its first board-touching work. The dispatcher
#: and the notifier are the two watchers ``start()`` spawns; the dispatcher's
#: tick is what spawns kanban workers, and a spawned worker is what binds the
#: worker context (which starts the submitter watcher) and what per-turn
#: recovery and the worker-loss recorder serve; the idle-exit probe's kanban
#: busy leg asks the board directly. None of them may run in a hosted process.
_KANBAN_BACKGROUND_PARTS = (
    ("notifier tick", "gateway.kanban_watchers", "_notifier_collect"),
    ("dispatcher tick", "gateway.kanban_watchers_dispatcher", "_KanbanDispatcher.tick_once"),
    ("dispatcher zombie reaper", "hermes_cli.kanban_db_dispatch", "reap_worker_zombies"),
    ("worker binding", "gateway.session_kanban", "bind_worker_context"),
    ("submitter watcher", "gateway.session_kanban", "_watch_submitter"),
    ("worker-loss recorder", "gateway.session_kanban", "record_worker_loss"),
    ("turn recovery", "agent.kanban_turn_recovery", "recover_failed_kanban_turns"),
    ("idle-exit board busy leg", "gateway.run_idle_exit", "_kanban_busy"),
)


def _spy_on_every_kanban_background_part(monkeypatch) -> list:
    """Wrap each census entry point to record its name when it runs; the wrapped
    originals stay in place so a part the gateway legitimately reaches still
    behaves (and its own hosted refusal, if any, still holds)."""
    import importlib

    started = []

    def _wrap(label, module_name, attribute):
        module = importlib.import_module(module_name)
        if "." in attribute:  # a method: patch the class, keep the binding
            cls_name, method = attribute.split(".", 1)
            target = getattr(module, cls_name)
        else:
            target, method = module, attribute
        original = getattr(target, method)

        def record(*args, **kwargs):
            started.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(target, method, record)

    for label, module_name, attribute in _KANBAN_BACKGROUND_PARTS:
        _wrap(label, module_name, attribute)
    return started


@pytest.mark.asyncio
async def test_hosted_gateway_boots_ready_with_kanban_disabled(tmp_path, monkeypatch, caplog):
    """A real gateway start in a hosted process: READY, no exception, and every
    kanban background part — the dispatcher and the notifier (each disabled with
    one log line instead of a crash or an endless refused-open retry loop), the
    dispatcher's workers, the submitter watcher they bind, the worker-loss
    recorder, the per-turn recovery and the idle-exit board busy leg — disabled
    or never started, with no refused-board LookupError across a few ticks."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_runtime import (initialize_gateway_runtime, start_gateway_runtime_api,
                                     publish_gateway_runtime_ready)
    from gateway.runtime_ownership import process_ownership
    from gateway.control_socket import GatewayControlServer
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "shared-board"))
    _serve_hosted_tenants(monkeypatch)
    started_parts = _spy_on_every_kanban_background_part(monkeypatch)
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
        # A few ticks of the gateway's loops: nothing kanban ran, nothing refused.
        for _ in range(3):
            # health: allow HX013 -- the ticks are the behaviour under test: the loops get real time to (wrongly) touch kanban
            await asyncio.sleep(0.1)
            assert "LookupError" not in caplog.text, "a refused board surfaced in the gateway log"
        assert "idle probe failed" not in caplog.text
        assert not started_parts, (
            f"a hosted gateway started kanban background work: {sorted(set(started_parts))}")
        # The disabled dispatcher never resolved the board root, so it never
        # took the machine-wide singleton dispatcher lock either.
        assert getattr(runner, "_kanban_dispatcher_lock_handle", None) is None
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


@pytest.fixture
def board_with_due_work(tmp_path, monkeypatch):
    """One due task in the machine's shared board, seeded while the process is
    not hosted: the dispatcher's reason to keep a local gateway up, and what a
    hosted process must never be asked about."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "shared-board"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    with kbc.connect_closing() as conn:
        kb.create_task(conn, title="due card", assignee="default")
    return kb.kanban_db_path()


def test_hosted_idle_probe_skips_the_shared_board(board_with_due_work, monkeypatch):
    """The idle/restart probe's kanban busy leg is disabled in a hosted process
    by the same gate as the dispatcher and the notifier. Without the gate the
    probe's board enumeration refuses, ``busy_reason`` fails closed with the
    refusal's name, and an otherwise idle hosted gateway can never retire: the
    watcher would retry the refused probe every tick forever."""
    from types import SimpleNamespace

    from gateway import run_idle_exit
    _serve_hosted_tenants(monkeypatch)
    for name in ("_messaging_busy", "_ledger_busy", "_profile_start_busy",
                 "_cron_busy", "_store_gates_busy"):
        monkeypatch.setattr(run_idle_exit, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(run_idle_exit, "_served_homes", lambda runner: [])
    runner = SimpleNamespace(
        _draining=False, _running=True, _active_work_count=lambda: 0,
        _scale_to_zero_has_live_background_work=lambda: False)
    # Three probes, the way the watcher re-asks across its ticks.
    for _ in range(3):
        assert run_idle_exit.busy_reason(runner, window=1, count_clients=False) is None


def test_locally_the_idle_probe_still_counts_board_work(board_with_due_work):
    """The control: not hosted, the same leg reads the board and reports the
    due task, so an idle exit still waits for the dispatcher's work."""
    from gateway import run_idle_exit
    assert run_idle_exit._kanban_busy() == "kanban work on board default"
