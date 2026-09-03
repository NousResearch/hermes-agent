import asyncio
import shutil
import subprocess
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
from agent.i18n import t
from gateway.platforms.base import MessageEvent, MessageType
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT,
)
from gateway.session import SessionEntry, build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.mark.asyncio
async def test_restart_command_while_busy_requests_drain_without_interrupt(monkeypatch):
    # Ensure INVOCATION_ID is NOT set — systemd sets this in service mode,
    # which changes the restart call signature.
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_EXTERNAL_SUPERVISOR", raising=False)
    # Hermeticity: neutralize the real container probe (see
    # test_restart_service_detection.py) — /.dockerenv on a containerized CI
    # runner would otherwise route via_service=True under this test.
    monkeypatch.setattr(
        "gateway.restart.is_container_restart_context", lambda: False
    )
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
    )
    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    result = await runner._handle_message(event)

    expected = t("gateway.draining", count=1)
    assert result == expected
    # Guard against the silent-degradation regression in #22266: if the i18n
    # catalog cannot be resolved (e.g. xdist workers losing the locales path)
    # then ``t("gateway.draining", count=1)`` returns the bare key
    # ``"gateway.draining"`` instead of the formatted English string, and both
    # sides of the equality above would still match. Assert on the catalog
    # output explicitly so a broken locale resolution fails loudly here.
    assert expected != "gateway.draining"
    assert "Draining" in expected and "1" in expected
    running_agent.interrupt.assert_not_called()
    runner.request_restart.assert_called_once_with(detached=True, via_service=False)


def test_load_busy_text_mode_follows_input_mode_and_honors_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_TEXT_MODE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_INPUT_MODE", raising=False)

    # No knobs set → follows busy_input_mode, which defaults to interrupt.
    assert gateway_run.GatewayRunner._load_busy_text_mode() == "interrupt"

    # busy_input_mode=queue propagates to text handling (single source of truth).
    (tmp_path / "config.yaml").write_text(
        "display:\n  busy_input_mode: queue\n", encoding="utf-8"
    )
    assert gateway_run.GatewayRunner._load_busy_text_mode() == "queue"

    # Legacy explicit busy_text_mode still wins for backward compat.
    (tmp_path / "config.yaml").write_text(
        "display:\n  busy_input_mode: interrupt\n  busy_text_mode: queue\n",
        encoding="utf-8",
    )
    assert gateway_run.GatewayRunner._load_busy_text_mode() == "queue"

    # Legacy env override wins too.
    (tmp_path / "config.yaml").write_text(
        "display:\n  busy_input_mode: interrupt\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_TEXT_MODE", "queue")
    assert gateway_run.GatewayRunner._load_busy_text_mode() == "queue"

    # Bogus legacy value is ignored → falls through to busy_input_mode (interrupt).
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_TEXT_MODE", "bogus")
    assert gateway_run.GatewayRunner._load_busy_text_mode() == "interrupt"


def test_load_signal_interrupt_grace_timeout_from_typed_config(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    assert (
        gateway_run.GatewayRunner._load_signal_interrupt_grace_timeout()
        == DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
    )

    (tmp_path / "config.yaml").write_text(
        "gateway:\n  signal_interrupt_grace_timeout: 0.25\n",
        encoding="utf-8",
    )
    assert gateway_run.GatewayRunner._load_signal_interrupt_grace_timeout() == 0.25

    (tmp_path / "config.yaml").write_text(
        "gateway:\n  signal_interrupt_grace_timeout: 0\n",
        encoding="utf-8",
    )
    assert gateway_run.GatewayRunner._load_signal_interrupt_grace_timeout() == 0.0

    (tmp_path / "config.yaml").write_text(
        "gateway:\n  signal_interrupt_grace_timeout: .inf\n",
        encoding="utf-8",
    )
    assert (
        gateway_run.GatewayRunner._load_signal_interrupt_grace_timeout()
        == DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
    )

    (tmp_path / "config.yaml").write_text(
        "gateway:\n  signal_interrupt_grace_timeout: invalid\n",
        encoding="utf-8",
    )
    assert (
        gateway_run.GatewayRunner._load_signal_interrupt_grace_timeout()
        == DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT
    )
    assert "Invalid signal_interrupt_grace_timeout" in caplog.text


@pytest.mark.asyncio
async def test_request_restart_is_idempotent():
    runner, _adapter = make_restart_runner()
    runner.stop = AsyncMock()
    runner._launch_detached_restart_command = AsyncMock()

    # _run_restart is held on self._restart_task and is intentionally NOT in
    # _background_tasks, so _stop_impl's cancel loop can't abort it mid-await
    # (see #12875).
    assert runner.request_restart(detached=True, via_service=False) is True
    assert runner._restart_task is not None
    assert runner._restart_task not in runner._background_tasks
    assert runner.request_restart(detached=True, via_service=False) is False
    # In-band restart marks draining immediately so new turns are refused
    # while any after-turn wait runs (#77184).
    assert runner._draining is True

    await runner._restart_task

    runner._launch_detached_restart_command.assert_awaited_once_with()
    runner.stop.assert_awaited_once_with(
        restart=True, detached_restart=True, service_restart=False
    )


@pytest.mark.asyncio
async def test_request_restart_defers_stop_until_active_turn_finishes():
    """Regression for #77184: requesting turn must not enter the drain set."""
    runner, _adapter = make_restart_runner()
    runner.stop = AsyncMock()
    runner._launch_detached_restart_command = AsyncMock()
    runner._restart_after_turn_timeout = 5.0
    session_key = "agent:main:telegram:dm:123"
    runner._running_agents[session_key] = MagicMock()

    assert runner.request_restart(detached=False, via_service=True) is True
    assert runner._draining is True

    # While the requesting turn is still active, stop() must not run.
    await asyncio.sleep(0.25)
    runner.stop.assert_not_awaited()
    assert session_key in runner._running_agents

    # Turn finishes → restart proceeds immediately (drain set empty).
    del runner._running_agents[session_key]
    await runner._restart_task

    runner.stop.assert_awaited_once_with(
        restart=True, detached_restart=False, service_restart=True
    )
    # Detached helper is only for the non-service path.
    runner._launch_detached_restart_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_restart_after_turn_timeout_zero_enters_stop_immediately():
    """restart_after_turn_timeout=0 preserves legacy immediate drain."""
    runner, _adapter = make_restart_runner()
    runner.stop = AsyncMock()
    runner._restart_after_turn_timeout = 0.0
    runner._running_agents["agent:main:telegram:dm:1"] = MagicMock()

    assert runner.request_restart(detached=False, via_service=True) is True
    await runner._restart_task

    runner.stop.assert_awaited_once_with(
        restart=True, detached_restart=False, service_restart=True
    )


@pytest.mark.asyncio
async def test_request_restart_after_turn_cap_elapsed_still_calls_stop():
    """Safety valve: wedged turns cannot pin the gateway forever."""
    runner, _adapter = make_restart_runner()
    runner.stop = AsyncMock()
    runner._restart_after_turn_timeout = 0.2
    runner._running_agents["agent:main:telegram:dm:1"] = MagicMock()

    assert runner.request_restart(detached=False, via_service=True) is True
    await runner._restart_task

    runner.stop.assert_awaited_once_with(
        restart=True, detached_restart=False, service_restart=True
    )
    # Agent was still present — stop() owns the interrupt path from here.
    assert runner._running_agents


@pytest.mark.asyncio
async def test_run_restart_excluded_from_stop_cancel_loop():
    """Regression for #12875: _run_restart is held on self._restart_task and
    kept OUT of _background_tasks, and the _stop_impl cancel loop explicitly
    skips it. If it were in _background_tasks, the cancel loop (which fires
    while _run_restart is awaiting _stop_task) would propagate CancelledError
    into _stop_impl and skip _shutdown_event.set() / _exit_code = 75."""
    runner, _adapter = make_restart_runner()
    runner.stop = AsyncMock()

    # A decoy background task that SHOULD be cancelled, plus the restart task
    # that must NOT be.
    async def _decoy():
        await asyncio.sleep(0.2)

    decoy = asyncio.create_task(_decoy())
    runner._background_tasks.add(decoy)
    decoy.add_done_callback(runner._background_tasks.discard)

    assert runner.request_restart(detached=False, via_service=True) is True
    restart_task = runner._restart_task
    assert restart_task is not None
    assert restart_task not in runner._background_tasks

    # Run the real cancel loop body in isolation (mirrors _stop_impl:7234).
    runner._stop_task = None
    for _task in list(runner._background_tasks):
        if _task is runner._stop_task:
            continue
        if _task is runner._restart_task:
            continue
        _task.cancel()

    await asyncio.sleep(0)  # let cancellation settle
    assert decoy.cancelled()
    assert not restart_task.cancelled()

    await restart_task
    runner.stop.assert_awaited_once_with(
        restart=True, detached_restart=False, service_restart=True
    )


@pytest.mark.windows_only
@pytest.mark.asyncio
async def test_windows_detached_restart_scrubs_gateway_marker(monkeypatch, tmp_path):
    """Faking sys.platform="win32" on Linux could not reach the real Windows
    detach branch (msvcrt/creationflags spawn, Lib/site-packages venv layout);
    this runs on the Windows CI job instead."""
    runner, _adapter = make_restart_runner()
    popen_calls = []
    venv_dir = tmp_path / "venv"
    site_packages = venv_dir / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)

    monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: ["hermes"])
    monkeypatch.setattr(gateway_run.os, "getpid", lambda: 321)
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))

    import hermes_cli._subprocess_compat as subprocess_compat

    monkeypatch.setattr(
        subprocess_compat,
        "windows_detach_popen_kwargs",
        lambda: {},
    )

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return MagicMock()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    await runner._launch_detached_restart_command()

    assert len(popen_calls) == 1
    cmd, kwargs = popen_calls[0]
    assert cmd[-3:] == ["hermes", "gateway", "restart"]
    assert kwargs["env"].get("_HERMES_GATEWAY") is None
    assert kwargs["env"].get(DELEGATED_CHILD_ENV_MARKER) is None
    assert kwargs["env"]["VIRTUAL_ENV"] == str(venv_dir)
    assert str(site_packages) in kwargs["env"]["PYTHONPATH"].split(gateway_run.os.pathsep)
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL


@pytest.mark.windows_only
@pytest.mark.asyncio
async def test_windows_detached_restart_watcher_keeps_console_python(monkeypatch, tmp_path):
    """The restart watcher must run sys.executable (console python) under the
    hidden-console detach kwargs — NOT swap in GUI-subsystem pythonw.exe,
    which would leave the watcher console-less and make its descendants
    flash visible conhosts (#54220/#56747).

    Faking sys.platform on Linux could not enter the Windows-only watcher
    spawn branch this asserts on, so it runs on the Windows CI job.
    """
    runner, _adapter = make_restart_runner()
    popen_calls = []
    venv_dir = tmp_path / "venv"
    site_packages = venv_dir / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)

    monkeypatch.setattr(gateway_run.sys, "executable", r"C:\venv\Scripts\python.exe")
    monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: ["hermes"])
    monkeypatch.setattr(gateway_run.os, "getpid", lambda: 321)
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))

    import hermes_cli._subprocess_compat as subprocess_compat

    monkeypatch.setattr(
        subprocess_compat,
        "windows_detach_popen_kwargs",
        lambda: {"creationflags": 0x08000200},
    )

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return MagicMock()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    await runner._launch_detached_restart_command()

    assert len(popen_calls) == 1
    cmd, kwargs = popen_calls[0]
    assert cmd[0] == r"C:\venv\Scripts\python.exe"
    assert cmd[-3:] == ["hermes", "gateway", "restart"]
    assert kwargs["creationflags"] == 0x08000200


@pytest.mark.skipif(gateway_run.sys.platform == "win32", reason="POSIX watcher path")
@pytest.mark.asyncio
async def test_posix_detached_restart_scrubs_process_role_markers(monkeypatch):
    runner, _adapter = make_restart_runner()
    popen_calls = []

    monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: ["hermes"])
    monkeypatch.setattr(gateway_run.os, "getpid", lambda: 321)
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/usr/bin/setsid" if name == "setsid" else None,
    )

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return MagicMock()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    await runner._launch_detached_restart_command()

    assert len(popen_calls) == 1
    cmd, kwargs = popen_calls[0]
    assert cmd[:2] == ["/usr/bin/setsid", "bash"]
    assert kwargs["env"].get("_HERMES_GATEWAY") is None
    assert kwargs["env"].get(DELEGATED_CHILD_ENV_MARKER) is None


# ── Shutdown notification tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_shutdown_notification_uses_persisted_origin_for_colon_ids():
    """Shutdown notifications should route from persisted origin, not reparsed keys."""
    runner, adapter = make_restart_runner()
    adapter.send = AsyncMock()
    source = make_restart_source(chat_id="!room123:example.org", chat_type="group")
    source.platform = gateway_run.Platform.MATRIX
    session_key = build_session_key(source)
    runner._running_agents[session_key] = MagicMock()
    runner.session_store._entries = {
        session_key: SessionEntry(
            session_key=session_key,
            session_id="sess-1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            origin=source,
            platform=source.platform,
            chat_type=source.chat_type,
        )
    }
    runner.adapters = {gateway_run.Platform.MATRIX: adapter}

    await runner._notify_active_sessions_of_shutdown()

    assert adapter.send.await_count == 1


@pytest.mark.asyncio
async def test_drain_suppress_skips_home_channel_keeps_session_ping(tmp_path, monkeypatch):
    """A suppress_notification drain marker mutes ONLY the home-channel broadcast.

    The per-active-session interrupt ping MUST still fire (it carries the
    "your task was interrupted, message me to resume" hint). This is the core
    drain-notification-suppression contract.
    """
    from gateway.config import HomeChannel, Platform
    import gateway.drain_control as dc

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner, adapter = make_restart_runner()
    # A home channel distinct from the active session's chat.
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-42",
        name="Ops Home",
    )
    # One active session in a different chat.
    runner._running_agents["agent:main:telegram:dm:999"] = MagicMock()

    # NAS auto-update drain: marker present with suppress_notification=True.
    dc.write_drain_request(principal="nas", suppress_notification=True)

    await runner._notify_active_sessions_of_shutdown()

    # Exactly one send — the active-session ping to chat 999. The home-channel
    # broadcast to home-42 was suppressed.
    assert len(adapter.sent_calls) == 1
    sent_chat_ids = {chat_id for chat_id, _content, _meta in adapter.sent_calls}
    assert "999" in sent_chat_ids
    assert "home-42" not in sent_chat_ids
    assert "shutting down" in adapter.sent[0]




def _wedged_agent(idle_seconds: float = 4000.0) -> MagicMock:
    """Agent double whose activity summary reports it idle past the timeout."""
    agent = MagicMock()
    agent.get_activity_summary = MagicMock(
        return_value={"seconds_since_activity": idle_seconds}
    )
    return agent


def _live_agent(idle_seconds: float = 1.0) -> MagicMock:
    agent = MagicMock()
    agent.get_activity_summary = MagicMock(
        return_value={"seconds_since_activity": idle_seconds}
    )
    return agent


@pytest.mark.asyncio
async def test_request_restart_skips_wait_when_only_wedged_turns(monkeypatch):
    """A turn idle past agent.gateway_timeout must not defer the restart.

    Regression: a WhatsApp turn wedged for 30+ min pinned `hermes update`
    in "draining" for the full restart_after_turn_timeout cap — the
    after-turn wait counted the wedged agent as active work even though
    the inactivity watchdog had already declared it dead (Aug 2026).
    """
    monkeypatch.delenv("HERMES_AGENT_TIMEOUT", raising=False)
    runner, _adapter = make_restart_runner()
    runner.stop = AsyncMock()
    # A cap long enough that the test would hang without the wedge bypass.
    runner._restart_after_turn_timeout = 300.0
    runner._running_agents["agent:main:whatsapp:dm:1"] = _wedged_agent()

    assert runner.request_restart(detached=False, via_service=True) is True
    await asyncio.wait_for(runner._restart_task, timeout=5.0)

    runner.stop.assert_awaited_once_with(
        restart=True, detached_restart=False, service_restart=True
    )
    # Wedged agent stays in the map — stop() owns the interrupt from here.
    assert runner._running_agents


@pytest.mark.asyncio
async def test_request_restart_still_waits_for_live_turn_alongside_wedged(monkeypatch):
    """Mixed live + wedged: the live turn is honored, the wedged one ignored."""
    monkeypatch.delenv("HERMES_AGENT_TIMEOUT", raising=False)
    runner, _adapter = make_restart_runner()
    runner.stop = AsyncMock()
    runner._launch_detached_restart_command = AsyncMock()
    runner._restart_after_turn_timeout = 300.0
    live_key = "agent:main:telegram:dm:2"
    runner._running_agents["agent:main:whatsapp:dm:1"] = _wedged_agent()
    runner._running_agents[live_key] = _live_agent()

    assert runner.request_restart(detached=False, via_service=True) is True

    # Live turn active → stop() must not run yet, wedged turn notwithstanding.
    await asyncio.sleep(0.25)
    runner.stop.assert_not_awaited()

    # Live turn finishes → restart proceeds without waiting on the wedged one.
    del runner._running_agents[live_key]
    await asyncio.wait_for(runner._restart_task, timeout=5.0)
    runner.stop.assert_awaited_once()


def test_wedged_agent_count_disabled_timeout_counts_nothing(monkeypatch):
    """gateway_timeout=0 (unbounded turns) disables wedge detection."""
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    runner, _adapter = make_restart_runner()
    runner._running_agents["agent:main:telegram:dm:1"] = _wedged_agent(10**6)
    assert runner._wedged_agent_count() == 0


def test_wedged_agent_count_ignores_sentinels_and_bad_summaries(monkeypatch):
    monkeypatch.delenv("HERMES_AGENT_TIMEOUT", raising=False)
    runner, _adapter = make_restart_runner()
    broken = MagicMock()
    broken.get_activity_summary = MagicMock(side_effect=RuntimeError("boom"))
    non_dict = MagicMock()  # auto-attr summary returns a MagicMock, not a dict
    runner._running_agents.update(
        {
            "pending": gateway_run._AGENT_PENDING_SENTINEL,
            "broken": broken,
            "non_dict": non_dict,
            "wedged": _wedged_agent(),
            "live": _live_agent(),
        }
    )
    assert runner._wedged_agent_count() == 1
