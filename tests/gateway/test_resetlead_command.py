from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import EphemeralReply, MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


class _FakeProc:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr


def _make_source(platform: Platform = Platform.WHATSAPP) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="34600111222",
        chat_id="120363409910080294@g.us",
        user_name="Lead Tester",
        chat_type="group",
    )


def _make_event(text: str, platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(platform=platform), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.WHATSAPP,
        chat_type="group",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._active_session_locks = {}
    runner._destructive_slash_confirm_sessions = {}
    from gateway.run import GatewayRunner as _GR

    runner._session_key_for_source = _GR._session_key_for_source.__get__(runner, _GR)
    return runner


@pytest.mark.asyncio
async def test_dispatcher_routes_resetlead(monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner()
    sentinel = "resetlead handler reached"
    runner._maybe_confirm_destructive_slash = AsyncMock(return_value=sentinel)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/resetlead"))
    assert result == sentinel
    runner._maybe_confirm_destructive_slash.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_running_agent_bypasses_guard_for_resetlead(monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner()
    session_key = runner._session_key_for_source(_make_source())
    runner._running_agents[session_key] = object()
    runner._interrupt_and_clear_session = AsyncMock()  # type: ignore[attr-defined]
    runner._handle_resetlead_command = AsyncMock(return_value="ok")  # type: ignore[attr-defined]

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/resetlead"))
    assert result == "ok"
    runner._interrupt_and_clear_session.assert_awaited_once()  # type: ignore[attr-defined]
    _, kwargs = runner._interrupt_and_clear_session.await_args  # type: ignore[attr-defined]
    assert kwargs["invalidation_reason"] == "resetlead_command"


@pytest.mark.asyncio
async def test_resetlead_handler_resets_persistent_lead_and_session(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    runner = _make_runner()
    script_path = tmp_path / "skyops_leads.py"
    script_path.write_text("# helper stub\n", encoding="utf-8")
    runner._handle_reset_command = AsyncMock(  # type: ignore[attr-defined]
        return_value=EphemeralReply("Started a new conversation.", ttl_seconds=45)
    )

    calls = []

    async def _fake_create_subprocess_exec(*args, **kwargs):
        calls.append((args, kwargs))
        payload = (
            '{"ok": true, "lead_id": "lead-123", "reset": true, '
            '"previous_state": "qualified"}'
        ).encode("utf-8")
        return _FakeProc(returncode=0, stdout=payload, stderr=b"")

    monkeypatch.setattr(gateway_run.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(runner, "_skyops_leads_script_path", lambda: Path(script_path))

    reply = await runner._handle_resetlead_command(_make_event("/resetlead"))

    assert isinstance(reply, EphemeralReply)
    assert reply.ttl_seconds == 45
    assert "Lead lead-123 reset (previous state: qualified)." in str(reply)
    assert "Started a new conversation." in str(reply)
    runner._handle_reset_command.assert_awaited_once()  # type: ignore[attr-defined]

    args, kwargs = calls[0]
    assert args[0].endswith("python") or "python" in Path(args[0]).name.lower()
    assert args[1] == str(script_path)
    assert args[2:4] == ("reset", "--allow-missing")
    assert "--whatsapp-id" in args
    assert "34600111222" in args
    assert "--display-name" in args
    assert "Lead Tester" in args
    assert kwargs["stdout"] is gateway_run.asyncio.subprocess.PIPE
    assert kwargs["stderr"] is gateway_run.asyncio.subprocess.PIPE
