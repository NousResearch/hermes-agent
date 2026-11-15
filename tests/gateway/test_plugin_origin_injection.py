"""``ctx.inject_message(origin=...)`` starts a gateway session in the plugin's OWN profile (#132609).

Real ``GatewayRunner`` + real ``BasePlatformAdapter`` ingress with a recording transport; only the
model call (``_run_agent_inner``) is replaced, and it records the home and secret scope the turn
actually ran under.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

PLUGIN = "desk"


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self, label: str):
        super().__init__(PlatformConfig(enabled=True, token=f"tok-{label}"), Platform.TELEGRAM)
        self.label = label
        self.sent: list[tuple[str, str]] = []
        self.send_scopes: list[tuple] = []  # (home, PROBE_MARKER) each send ran under
        self._running = True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from agent.secret_scope import get_secret
        from hermes_constants import get_hermes_home
        self.sent.append((str(chat_id), content))
        self.send_scopes.append((get_hermes_home(), get_secret("PROBE_MARKER")))
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


def _write_home(home, marker: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"entries": {PLUGIN: {"allow_gateway_injection": True}}},
    }), encoding="utf-8")
    (home / ".env").write_text(f"TELEGRAM_ALLOWED_USERS=u1\nPROBE_MARKER={marker}\n", encoding="utf-8")


def _context(home) -> tuple[PluginContext, PluginManager]:
    manager = PluginManager(scope_key=str(home))
    return PluginContext(PluginManifest(name=PLUGIN, key=PLUGIN, source="user"), manager), manager


def _origin(chat_id: str, **extra) -> dict:
    return {"platform": "telegram", "chat_id": chat_id, "chat_type": "group", "thread_id": "7",
            "user_id": "u1", "user_name": "Requester", **extra}


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """Multiplexed gateway serving ``default`` (bot P), ``alpha`` (its own bot A) and ``beta`` (its
    own bot B); a route sends chat 999 on bot A to ``beta``."""
    home = tmp_path / "hh"
    homes = {"default": home, "alpha": home / "profiles" / "alpha", "beta": home / "profiles" / "beta"}
    for name, profile_home in homes.items():
        _write_home(profile_home, marker=name)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)

    config = GatewayConfig(multiplex_profiles=True, sessions_dir=home / "sessions")
    config.profile_routes = parse_profile_routes([
        {"name": "beta-on-a", "platform": "telegram", "profile": "beta", "chat_id": "999", "bot_profile": "alpha"},
    ])
    runner = GatewayRunner(config)
    adapters = {name: _RecordingAdapter(name) for name in homes}
    runner.adapters = {Platform.TELEGRAM: adapters["default"]}
    runner._profile_adapters = {name: {Platform.TELEGRAM: adapters[name]} for name in ("alpha", "beta")}
    for name, adapter in adapters.items():
        adapter.gateway_runner = runner
        if name != "default":
            adapter.set_owner_profile(name)
            adapter.set_message_handler(runner._make_profile_message_handler(name))
        else:
            adapter.set_message_handler(runner._handle_message)

    turns = []

    async def _fake_turn(message, context_prompt, history, source, session_id, **_kwargs):
        from agent.secret_scope import get_secret
        from hermes_constants import get_hermes_home
        turns.append(SimpleNamespace(message=message, history=list(history), session_id=session_id,
                                     home=get_hermes_home(), marker=get_secret("PROBE_MARKER")))
        return {"final_response": f"reply-{len(turns)}", "messages": [], "tools": [],
                "history_offset": len(history), "last_prompt_tokens": 0}

    runner._run_agent_inner = _fake_turn
    served = list(homes.items())
    with patch("hermes_cli.profiles.profiles_to_serve", return_value=served), \
            patch("hermes_cli.profiles.get_profile_dir", side_effect=lambda n: homes[n]), \
            patch("hermes_cli.profiles.profile_exists", side_effect=lambda n: n in homes):
        yield SimpleNamespace(runner=runner, homes=homes, adapters=adapters, turns=turns)


async def _publish(runner):
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    runner._install_plugin_message_injector()


async def _settle(runner, adapter, rounds: int = 200):
    for _ in range(rounds):
        await asyncio.sleep(0.01)
        if not runner._background_tasks and not adapter._session_tasks and not adapter._active_sessions:
            return


@pytest.mark.asyncio
async def test_origin_creates_session_in_own_profile_and_replies_to_origin(gateway):
    """No session at the origin: the plugin's profile gets a fresh session, the turn runs under that
    profile's home and secrets, the reply goes back to the origin chat, and a human follow-up in the
    same chat continues the SAME session (A -> B -> A: beta in between, alpha's scope comes back)."""
    runner, alpha, beta = gateway.runner, gateway.adapters["alpha"], gateway.adapters["beta"]
    ctx_alpha, mgr_alpha = _context(gateway.homes["alpha"])
    ctx_beta, mgr_beta = _context(gateway.homes["beta"])
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[mgr_alpha, mgr_beta]):
        await _publish(runner)
        assert ctx_alpha.inject_message("Kick off the review", origin=_origin("100")) is True
        await _settle(runner, alpha)
        assert ctx_beta.inject_message("Beta's own post", origin=_origin("200")) is True
        await _settle(runner, beta)
        assert ctx_alpha.inject_message("Second post", origin=_origin("101")) is True
        await _settle(runner, alpha)

    first, middle, last = gateway.turns
    assert (first.home, first.marker) == (gateway.homes["alpha"], "alpha")
    assert (middle.home, middle.marker) == (gateway.homes["beta"], "beta")
    assert (last.home, last.marker) == (gateway.homes["alpha"], "alpha")
    assert first.message.endswith("Kick off the review") and first.history == []
    assert alpha.sent == [("100", "reply-1"), ("101", "reply-3")]
    assert beta.sent == [("200", "reply-2")]
    # The reply is sent after the handler returns: it must still run in the plugin's profile scope.
    assert alpha.send_scopes == [(gateway.homes["alpha"], "alpha")] * 2
    assert beta.send_scopes == [(gateway.homes["beta"], "beta")]
    assert gateway.adapters["default"].sent == []

    entry = runner.session_store._entries["agent:alpha:telegram:group:100:7"]
    assert entry.session_id == first.session_id
    assert not any(key.startswith(("agent:main:", "agent:beta:")) and ":100:" in key
                   for key in runner.session_store._entries)

    # A human reply in the same chat keys into the plugin-created session. A secondary adapter's
    # intake task runs inside its profile scope (it is spawned from connect() under that scope).
    reply = alpha.build_source(chat_id="100", chat_type="group", thread_id="7", user_id="u1")
    async with gateway_run._async_profile_runtime_scope(gateway.homes["alpha"]):
        await alpha.handle_message(MessageEvent(text="thanks", message_type=MessageType.TEXT, source=reply))
    await _settle(runner, alpha)
    assert gateway.turns[-1].session_id == first.session_id
    assert alpha.sent[-1] == ("100", "reply-4")


@pytest.mark.asyncio
async def test_origin_cannot_reach_another_profile(gateway):
    """A plugin in ``alpha`` cannot start a session in ``beta``: not by naming the profile, not
    through a chat the gateway routes to ``beta``. Refusal is synchronous and nothing is created."""
    runner, alpha = gateway.runner, gateway.adapters["alpha"]
    ctx_alpha, mgr_alpha = _context(gateway.homes["alpha"])
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[mgr_alpha]):
        await _publish(runner)
        assert ctx_alpha.inject_message("x", origin=_origin("100", profile="beta")) is False
        assert ctx_alpha.inject_message("x", origin=_origin("999")) is False  # routed to beta
        assert ctx_alpha.inject_message("x", origin={"platform": "telegram"}) is False  # malformed
        await _settle(runner, alpha)

    assert gateway.turns == [] and alpha.sent == [] and gateway.adapters["beta"].sent == []
    assert runner.session_store._entries == {}


@pytest.mark.asyncio
async def test_origin_without_connected_adapter_returns_false(gateway):
    runner = gateway.runner
    ctx_alpha, mgr_alpha = _context(gateway.homes["alpha"])
    runner._profile_adapters["alpha"] = {}  # alpha's bot is down; never borrow default's
    with patch("hermes_cli.plugins._known_plugin_managers", return_value=[mgr_alpha]):
        await _publish(runner)
        assert ctx_alpha.inject_message("x", origin=_origin("100")) is False
        assert ctx_alpha.inject_message("x", origin={**_origin("100"), "platform": "discord"}) is False
    assert runner._background_tasks == set()
    assert gateway.adapters["default"].sent == []
