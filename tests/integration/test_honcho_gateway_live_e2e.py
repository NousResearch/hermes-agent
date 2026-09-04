"""Live gateway end-to-end for honcho identity mapping: a real GatewayRunner under a sandbox home, a
Telegram adapter whose only mock is the outbound send, a live model, and a throwaway Honcho workspace.

Two accounts message the gateway. The gateway itself stamps their rows in the sandbox state.db, so
``hermes honcho peers map`` sees exactly what it would see in production; it re-maps the stranger, and
the stranger's next turn builds a fresh agent (the identity keys bust the agent cache) and lands on the
new peer.

Opt-in only: ``HONCHO_E2E=1``, ``HERMES_LIVE_TESTS=1``, an OpenRouter key (env or ~/.hermes/.env), and a
honcho credential as in test_honcho_peer_mapping_live. ``HERMES_LIVE_MODEL`` picks the model.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.integration.test_honcho_peer_mapping_live import _credential, _delete_workspace

pytestmark = pytest.mark.integration


def _load_user_env() -> None:
    env_file = Path.home() / ".hermes" / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_user_env()
_LIVE = os.environ.get("HONCHO_E2E") == "1" and os.environ.get("HERMES_LIVE_TESTS") == "1"
_OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_MODEL = os.environ.get("HERMES_LIVE_MODEL", "google/gemini-2.5-flash")
_REPLY_WAIT = 180.0
_HONCHO_WAIT = 90.0

ALICE = SimpleNamespace(user_id="111000111", name="alice", chat_id="111000111")
BOB = SimpleNamespace(user_id="222000222", name="bob", chat_id="222000222")


@pytest.fixture(scope="module")
def live():
    if not _LIVE:
        pytest.skip("set HONCHO_E2E=1 and HERMES_LIVE_TESTS=1 to run the live gateway suite")
    if not _OR_KEY:
        pytest.skip("OPENROUTER_API_KEY not configured")
    key, base_url = _credential()
    if not key:
        pytest.skip("no honcho credential")
    from honcho import Honcho

    workspace = f"hermes-e2e-gw-{uuid.uuid4().hex[:8]}"
    kwargs = {"api_key": key, "workspace_id": workspace}
    if base_url:
        kwargs["base_url"] = base_url
    client = Honcho(**kwargs)
    yield _Live(key, base_url, workspace, client)
    _delete_workspace(client, workspace)


class _Live:
    """Live handles; the repr hides the credential so a failing fixture never prints it."""

    def __init__(self, key, base_url, workspace, client):
        self.key, self.base_url, self.workspace, self.client = key, base_url, workspace, client

    def __repr__(self) -> str:
        return f"<live workspace={self.workspace}>"


@pytest.fixture
def gateway(tmp_path, monkeypatch, live):
    """A real GatewayRunner in an isolated HERMES_HOME with honcho as the memory provider."""
    import gateway.run as gateway_run
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.memory.honcho.client import reset_honcho_client

    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"model:\n  default: {_MODEL}\n  provider: openrouter\n"
        "memory:\n  provider: honcho\n"
        "approvals:\n  destructive_slash_confirm: false\n"
    )
    honcho_cfg = {"apiKey": live.key, "hosts": {"hermes": {
        "enabled": True, "workspace": live.workspace, "peerName": "eri-e2e", "aiPeer": "hermes-e2e",
        "saveMessages": True, "writeFrequency": "turn", "recallMode": "context", "observationMode": "directional",
        "dialecticCadence": 999, "contextCadence": 999,
        "pinUserPeer": False, "runtimePeerPrefix": "tg_", "userPeerAliases": {ALICE.user_id: "eri-e2e"},
    }}}
    if live.base_url:
        honcho_cfg["baseUrl"] = live.base_url
    cfg_path = home / "honcho.json"
    cfg_path.write_text(json.dumps(honcho_cfg, indent=2))

    for var in ("HONCHO_API_KEY", "HONCHO_BASE_URL", "HONCHO_URL", "HONCHO_WORKSPACE", "HERMES_HONCHO_HOST",
                "GATEWAY_ALLOW_ALL_USERS", "TELEGRAM_ALLOW_ALL_USERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", _OR_KEY)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", f"{ALICE.user_id},{BOB.user_id}")
    token = set_hermes_home_override(home)
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    reset_honcho_client()

    runner = gateway_run.GatewayRunner(
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-token")}))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="e2e-token"))
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="e2e-resp"))
    adapter.send_typing = AsyncMock()
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[Platform.TELEGRAM] = adapter

    yield SimpleNamespace(runner=runner, adapter=adapter, home=home, cfg_path=cfg_path)

    reset_honcho_client()
    reset_hermes_home_override(token)


def _event(account, text: str):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    return MessageEvent(
        text=text, message_id=f"msg-{uuid.uuid4().hex[:8]}",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=account.chat_id, user_id=account.user_id,
                             user_name=account.name, chat_type="dm"),
    )


def _sent_text(call) -> str:
    """The message body of one adapter.send call, whichever way the gateway passed it."""
    if len(call.args) > 1 and isinstance(call.args[1], str):
        return call.args[1]
    for key in ("content", "text", "message"):
        value = call.kwargs.get(key)
        if isinstance(value, str):
            return value
    return ""


async def _say(gw, account, text: str) -> str:
    """Deliver one inbound message, wait for the turn to finish, and return the last text the adapter sent."""
    gw.adapter.send.reset_mock()
    await gw.adapter.handle_message(_event(account, text))
    deadline = time.monotonic() + _REPLY_WAIT
    while time.monotonic() < deadline:
        texts = [t for t in (_sent_text(c) for c in gw.adapter.send.await_args_list) if t.strip()]
        if texts and not gw.runner._running_agents:
            return texts[-1]
        await asyncio.sleep(0.5)
    pytest.fail(f"no reply for {account.name} within {_REPLY_WAIT:.0f}s")


def _authored_by(live, peer_id: str, text: str, timeout: float = _HONCHO_WAIT) -> bool:
    """True once a message containing ``text`` is stored under ``peer_id`` in any of its sessions."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            for session in live.client.peer(peer_id).sessions():
                for message in session.messages(size=50):
                    if str(message.peer_id) == peer_id and text in str(message.content):
                        return True
        except Exception:
            pass
        time.sleep(3)
    return False


@pytest.mark.asyncio
async def test_gateway_turns_land_on_mapped_peers_and_peers_map_remaps_live(gateway, live, monkeypatch, capsys):
    import plugins.memory.honcho.cli as honcho_cli
    from hermes_state import SessionDB

    gw = gateway
    reply = await _say(gw, ALICE, "Please remember that my favorite color is teal. Reply in one short sentence.")
    assert reply.strip()
    assert _authored_by(live, "eri-e2e", "favorite color is teal"), "alice's turn did not land on eri-e2e"

    reply = await _say(gw, BOB, "Please remember that I keep bees in Vermont. Reply in one short sentence.")
    assert reply.strip()
    assert _authored_by(live, f"tg_{BOB.user_id}", "bees in Vermont"), "bob's turn did not land on tg_<id>"

    # the gateway stamped both accounts into the sandbox state.db; peers map reads them from there
    rows = SessionDB(gw.home / "state.db", read_only=True)
    seen = honcho_cli._seen_gateway_accounts(gw.home / "state.db")
    assert {a["user_id"] for a in seen} >= {ALICE.user_id, BOB.user_id}, seen
    del rows

    answers = iter([BOB.user_id, "bob-e2e", ""])
    monkeypatch.setattr(honcho_cli, "_prompt", lambda label, default=None, secret=False: next(answers))
    honcho_cli.honcho_command(SimpleNamespace(honcho_command="peers", peers_action="map", target_profile=None))
    out = capsys.readouterr().out
    assert "bob-e2e" in out
    assert json.loads(gw.cfg_path.read_text())["hosts"]["hermes"]["userPeerAliases"][BOB.user_id] == "bob-e2e"

    # the identity keys are in the agent-cache signature, so bob's next turn builds a fresh agent
    reply = await _say(gw, BOB, "What do I keep, and where? Reply in one short sentence.")
    assert reply.strip()
    assert _authored_by(live, "bob-e2e", "What do I keep"), "bob's re-mapped turn did not land on bob-e2e"
    assert not _authored_by(live, f"tg_{BOB.user_id}", "What do I keep", timeout=5)
