"""A plugin slash command asks its user to pick an option (``PluginContext.ask_choice``) and gets the
picked value back, on the surfaces the gateway serves:

* a messaging chat gets the clarify prompt (here the base adapter's numbered text) and the typed
  number answers it while the command is still running;
* a local session (classic CLI, Ink TUI, Desktop over ``hermes-gateway-v1``) gets the ``clarify``
  server request on its own socket, and the viewer's response frame settles it although that
  socket's ``slash.exec`` is still waiting for the handler.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.waiting import async_wait_until

_CHOICES = [{"value": "default", "label": "Default", "description": "delegation.model / parent model"},
            {"value": "model_map", "label": "Model routes"}, "profile_routes"]
_LABELS = ["Default — delegation.model / parent model", "Model routes", "profile_routes"]


class _TextAdapter(BasePlatformAdapter):
    """A chat with no native card: the base ``send_clarify`` numbered list is what the user sees."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1")


def _runner(adapter):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")})
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False)
    entry = SessionEntry(session_key=build_session_key(_source()), session_id="sess-1", created_at=datetime.now(),
                         updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm")
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents, runner._pending_messages, runner._pending_approvals = {}, {}, {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_a_chat_answers_the_plugin_command_choice_with_a_typed_number(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugin_choices import ask_choice
    from tools import clarify_gateway

    adapter = _TextAdapter()
    runner = _runner(adapter)
    adapter.set_message_handler(runner._handle_message)
    monkeypatch.setattr(plugins_mod, "get_plugin_command_handler", lambda name: (
        (lambda arg: f"MODE={ask_choice('Delegation mode', _CHOICES)}") if name == "delegation" else None))
    monkeypatch.setattr(clarify_gateway, "get_clarify_timeout", lambda: 20)

    def event(text):
        return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source(), message_id=text)

    await adapter.handle_message(event("/delegation"))
    await async_wait_until(lambda: any("Model routes" in s for s in adapter.sent), timeout=10)
    prompt = next(s for s in adapter.sent if "Model routes" in s)
    assert "Delegation mode" in prompt and "delegation.model / parent model" in prompt, prompt
    await adapter.handle_message(event("2"))  # the command still runs: this must answer it, not queue
    await async_wait_until(lambda: any(s.startswith("MODE=") for s in adapter.sent), timeout=10)
    assert [s for s in adapter.sent if s.startswith("MODE=")] == ["MODE=model_map"], adapter.sent


_PLUGIN = f'''
def register(ctx):
    choices = {json.dumps(_CHOICES)}
    ctx.register_command("delegation", lambda arg: "MODE=" + str(ctx.ask_choice("Delegation mode", choices)),
                         description="Pick the delegation mode")
'''


@pytest.mark.platforms("linux")
def test_a_local_viewer_answers_the_choice_while_its_slash_exec_waits(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir(mode=0o700)
    user.mkdir()
    plugin = home / "plugins" / "pick"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: pick\nversion: '1.0'\n")
    (plugin / "__init__.py").write_text(_PLUGIN)
    (home / "config.yaml").write_text(json.dumps({
        "gateway": {"multiplex_profiles": False}, "clarify": {"timeout": 15},
        "model": {"provider": "custom", "default": "m", "base_url": "http://127.0.0.1:9/v1"},
        "platform_toolsets": {"cli": []}, "plugins": {"enabled": ["pick"]}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY="loopback-only", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")

    async def viewer(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, "session.create", request_id="pick", source="cli", toolsets=[],
                                cwd=str(tmp_path))
            sid = created["result"]["session_id"]
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": "exec", "method": "slash.exec",
                                      "params": {"session_id": sid, "command": "delegation"}}))
            asked = None
            async with asyncio.timeout(40):
                while True:
                    frame = json.loads(await ws.recv())
                    if frame.get("method") == "clarify":  # answered as Desktop's and Ink's panels do
                        asked = frame["params"]["questions"]
                        await ws.send(json.dumps({"jsonrpc": "2.0", "id": "lock", "method": "clarify.lock", "params": {
                            "request_id": frame["id"], "question_id": asked[0]["qid"],
                            "answer": asked[0]["choices"][2]}}))
                    elif frame.get("id") == "exec":
                        return asked, frame

    with daemon(root, home, env, barrier=False) as (proc, desc):
        asked, reply = asyncio.run(viewer(desc))
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=20)
    assert asked and asked[0]["question"] == "Delegation mode" and asked[0]["choices"] == _LABELS, asked
    assert reply["result"]["output"] == "MODE=profile_routes", reply
