import asyncio
import shutil
import subprocess
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from agent.i18n import t
from gateway.platforms.base import MessageEvent, MessageType
from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
from gateway.session import SessionEntry, build_session_key
from gateway.slash_commands import (
    _RestartTransaction,
    _TransitionResult,
)
from tests.gateway.restart_test_helpers import (
    attach_real_launcher_under_mocked_popen,
    make_restart_runner,
    make_restart_source,
)


# ---------------------------------------------------------------------------
# Module-level Popen hard block
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _forbid_real_popen(monkeypatch):
    """Hard module-level guard: every test in this file runs with
    ``subprocess.Popen`` replaced by an assertion-raising stub.

    The four Popen-inspection tests in this file (``test_launch_detached_*``,
    ``test_windows_detached_*``) MUST replace ``subprocess.Popen`` with a
    controlled fake inside their own test body BEFORE invoking the
    launcher-under-test.

    This fixture catches the regression that produced PID 5896 — a
    detached restart watcher leaked from a test that did not mock
    ``subprocess.Popen``.
    """

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError(
            "Real subprocess.Popen is forbidden in drain/restart tests. "
            "If this test intentionally inspects Popen arguments, replace "
            "subprocess.Popen with a controlled fake via monkeypatch.setattr "
            "BEFORE invoking the launcher-under-test."
        )

    monkeypatch.setattr(subprocess, "Popen", _unexpected_popen)
    yield


@pytest.mark.asyncio
async def test_restart_command_while_busy_requests_drain_without_interrupt(monkeypatch, tmp_path):
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
    # Fail-closed safety: launcher and stop are AsyncMocks from fixture.
    assert isinstance(runner._launch_detached_restart_command, AsyncMock)
    assert isinstance(runner.stop, AsyncMock)

    # Safe fake protocol: create a minimal stub that mimics the
    # request_restart side effects without touching the OS.
    captured: dict = {}

    def safe_request_restart(
        *,
        detached: bool,
        via_service: bool,
        transaction: _RestartTransaction | None = None,
    ) -> bool:
        captured["detached"] = detached
        captured["via_service"] = via_service
        captured["transaction"] = transaction

        # Set the runner's restart flags (done by real request_restart).
        runner._restart_requested = True
        runner._restart_detached = detached
        runner._restart_via_service = via_service
        runner._restart_task_started = True

        async def _run_restart() -> None:
            # Complete the handoff protocol so the ack is ready.
            claimed = await transaction.claim_handoff()  # type: ignore[union-attr]
            assert claimed is True, "claim_handoff must succeed before complete_started"
            transition = await transaction.complete_started()  # type: ignore[union-attr]
            assert transition in (
                _TransitionResult.TRANSITIONED,
                _TransitionResult.ALREADY_COMPLETE,
            ), (
                f"complete_started must produce TRANSITIONED or ALREADY_COMPLETE, "
                f"got {transition!r}"
            )

            await runner.stop(
                restart=True,
                detached_restart=detached,
                service_restart=via_service,
            )

        task = asyncio.create_task(_run_restart())
        transaction.restart_task = task  # type: ignore[union-attr]
        return True

    runner.request_restart = safe_request_restart  # type: ignore[method-assign]

    event = MessageEvent(
        text="/restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
    )
    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    popen_calls: list = []

    def _unexpected_popen(*a, **k):
        popen_calls.append((a, k))
        raise AssertionError("real Popen in drain test — must not happen")

    with patch("gateway.run._hermes_home", tmp_path), patch(
        "subprocess.Popen", side_effect=_unexpected_popen
    ):
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
    # request_restart was called once via the safe protocol.
    assert captured["detached"] is True
    assert captured["via_service"] is False
    assert captured["transaction"] is not None
    assert isinstance(captured["transaction"], _RestartTransaction)
    assert captured["transaction"].request_id.startswith("req-")
    assert popen_calls == [], f"real Popen called: {popen_calls}"


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
async def test_launch_detached_restart_command_uses_setsid(monkeypatch):
    runner, _adapter = make_restart_runner()
    # This test inspects Popen arguments — opt into real launcher.
    # Caller MUST mock subprocess.Popen before invoking the launcher.
    attach_real_launcher_under_mocked_popen(runner)
    popen_calls = []

    monkeypatch.setattr(gateway_run.sys, "platform", "linux")
    monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: ["/usr/bin/hermes"])
    monkeypatch.setattr(gateway_run.os, "getpid", lambda: 321)
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/setsid" if cmd == "setsid" else None)

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return MagicMock()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    await runner._launch_detached_restart_command()

    assert len(popen_calls) == 1
    cmd, kwargs = popen_calls[0]
    assert cmd[:2] == ["/usr/bin/setsid", "bash"]
    assert "gateway restart" in cmd[-1]
    assert "kill -0 321" in cmd[-1]
    assert "deadline=$(( $(date +%s) +" in cmd[-1]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    # The watcher must NOT inherit the gateway marker, or the CLI's
    # self-restart loop guard refuses to run `hermes gateway restart`.
    assert kwargs["env"].get("_HERMES_GATEWAY") is None


@pytest.mark.asyncio
async def test_detached_restart_helper_is_idempotent(monkeypatch):
    runner, _adapter = make_restart_runner()
    # This test inspects Popen arguments — opt into real launcher.
    # Caller MUST mock subprocess.Popen before invoking the launcher.
    attach_real_launcher_under_mocked_popen(runner)
    popen_calls = []

    monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: ["/usr/bin/hermes"])
    monkeypatch.setattr(gateway_run.os, "getpid", lambda: 321)
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: popen_calls.append((a, k)))

    await runner._launch_detached_restart_command()
    await runner._launch_detached_restart_command()

    assert len(popen_calls) == 1


def test_windows_gateway_venv_imports_add_site_packages(monkeypatch, tmp_path):
    venv_dir = tmp_path / "venv"
    site_packages = venv_dir / "Lib" / "site-packages"
    pth_extra = tmp_path / "pywin32_system32"
    site_packages.mkdir(parents=True)
    pth_extra.mkdir()
    (site_packages / "pywin32.pth").write_text(str(pth_extra), encoding="utf-8")
    project_root = str(gateway_run.Path(gateway_run.__file__).resolve().parent.parent)

    monkeypatch.setattr(gateway_run.sys, "platform", "win32")
    monkeypatch.setattr(gateway_run.sys, "path", ["existing"])
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))
    monkeypatch.setenv("PYTHONPATH", "already-there")

    gateway_run._ensure_windows_gateway_venv_imports()

    assert gateway_run.sys.path[:2] == [project_root, str(site_packages)]
    assert str(pth_extra) in gateway_run.sys.path
    assert gateway_run.os.environ["VIRTUAL_ENV"] == str(venv_dir.resolve())
    pythonpath = gateway_run.os.environ["PYTHONPATH"].split(gateway_run.os.pathsep)
    assert pythonpath[:3] == [project_root, str(site_packages), "already-there"]


@pytest.mark.asyncio
async def test_windows_detached_restart_scrubs_gateway_marker(monkeypatch, tmp_path):
    """Faking sys.platform="win32" on Linux could not reach the real Windows
    detach branch (msvcrt/creationflags spawn, Lib/site-packages venv layout);
    this runs on the Windows CI job instead."""
    runner, _adapter = make_restart_runner()
    # This test inspects Popen arguments — opt into real launcher.
    # Caller MUST mock subprocess.Popen before invoking the launcher.
    attach_real_launcher_under_mocked_popen(runner)
    popen_calls = []
    venv_dir = tmp_path / "venv"
    site_packages = venv_dir / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)

    monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: ["hermes"])
    monkeypatch.setattr(gateway_run.os, "getpid", lambda: 321)
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
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
    # This test inspects Popen arguments — opt into real launcher.
    # Caller MUST mock subprocess.Popen before invoking the launcher.
    attach_real_launcher_under_mocked_popen(runner)
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


