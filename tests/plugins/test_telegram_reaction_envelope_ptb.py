"""Offline SDK-envelope regressions, not validation of Telegram wire JSON.

Keep this outside tests/gateway: that conftest installs a Telegram SDK mock.
Real PTB decoding/dispatch and disk-discovered plugins exercise the observer;
only the HTTP edge is replaced, with a labeled getMe fixture and no poller.
"""
import asyncio
import copy
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("telegram", reason="python-telegram-bot not installed")

from telegram import Update
from telegram.ext import Application, ExtBot
from telegram.request import BaseRequest

from agent import secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner, _profile_runtime_scope
from hermes_cli import plugins
from hermes_constants import get_hermes_home
from plugins.platforms.telegram.adapter import TelegramAdapter


class OfflineRequest(BaseRequest):
    def __init__(self):
        self.calls = []

    @property
    def read_timeout(self):
        return 5.0

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        self.calls.append(url.rsplit("/", 1)[-1])
        assert self.calls[-1] == "getMe", "Only the offline getMe fixture is allowed"
        return 200, json.dumps({"ok": True, "result": {
            "id": 900001, "is_bot": True, "first_name": "Offline fixture",
            "username": "offline_fixture_bot",
        }}).encode()


_PLUGIN = '''import json
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home
from .state import IMPORT_HOME, IMPORT_SECRET, SEEN

def observe(platform, event_type, payload):
    SEEN.append(payload["update_id"])
    row = {"event": {"platform": platform, "event_type": event_type, "payload": payload},
           "home": str(get_hermes_home()), "secret": get_secret("TELEGRAM_ALLOWED_USERS"),
           "import_home": IMPORT_HOME, "import_secret": IMPORT_SECRET, "sequence": len(SEEN)}
    with (get_hermes_home() / "observed.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\\n")

def register(ctx):
    ctx.register_hook("gateway_platform_event", observe)
'''
_STATE = '''from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home
IMPORT_HOME = str(get_hermes_home())
IMPORT_SECRET = get_secret("TELEGRAM_ALLOWED_USERS")
SEEN = []
'''


def _reaction_update(*, chat_id=321, actor_id=777, actor_kind="user", update_id=0):
    actor = {"id": actor_id, "is_bot": False, "first_name": "Alice"}
    if actor_kind == "actor_chat":
        actor = {"id": actor_id, "type": "channel", "title": "Moderators"}
    return {"update_id": update_id, "message_reaction": {
        "chat": {"id": chat_id, "type": "supergroup", "is_forum": True},
        "message_id": 456, actor_kind: actor, "date": 1786464000,
        "old_reaction": [{"type": "emoji", "emoji": "👍"},
                         {"type": "custom_emoji", "custom_emoji_id": "111"}],
        "new_reaction": [{"type": "emoji", "emoji": "🔥"},
                         {"type": "custom_emoji", "custom_emoji_id": "555"}],
    }}


@pytest.fixture
def observer(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Network forbidden in the offline observer fixture")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    home, work = tmp_path / "default", tmp_path / "work"
    for directory, allowed in ((home, "777,-100777"), (work, "888,-100888")):
        plugin = directory / "plugins/reaction-fixture"
        plugin.mkdir(parents=True)
        (directory / ".env").write_text(f"TELEGRAM_ALLOWED_USERS={allowed}\n", encoding="utf-8")
        (directory / "config.yaml").write_text("plugins:\n  enabled: [reaction-fixture]\n", encoding="utf-8")
        (plugin / "plugin.yaml").write_text(
            "name: reaction-fixture\nversion: 1.0.0\ndescription: Offline observer fixture\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
        (plugin / "state.py").write_text(_STATE, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Neither ambient value may leak into the receiving profile's admission check.
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: work if name == "work" else home)
    monkeypatch.setattr("gateway.run._multiplex_profile_homes", lambda _config: [("default", home), ("work", work)])

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True, profile_routes=[
        ProfileRoute(name="work-chat", platform="telegram", profile="work", chat_id="123"),
    ])
    runner._primary_profile_name = "default"
    adapter = TelegramAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    adapter.set_platform_event_handler(runner._make_default_profile_platform_event_handler())

    def events():
        rows = []
        for directory in (home, work):
            log = directory / "observed.jsonl"
            if log.exists():
                rows.extend(json.loads(line) for line in log.read_text(encoding="utf-8").splitlines())
        return sorted(rows, key=lambda row: int(row["event"]["payload"]["update_id"]))

    def dispatch(raw_updates):
        async def run():
            request, polling_request = OfflineRequest(), OfflineRequest()
            app = Application.builder().bot(ExtBot(
                token="900001:OFFLINE_FIXTURE", request=request, get_updates_request=polling_request,
            )).updater(None).build()
            adapter._register_handlers(app)
            errors, decoded = [], []

            async def on_error(update, context):
                errors.append(context.error)

            app.add_error_handler(on_error)
            async with app:
                for raw in raw_updates:
                    update = Update.de_json(copy.deepcopy(raw), bot=app.bot)
                    decoded.append(update)
                    await app.process_update(update)
            assert not errors
            assert request.calls == ["getMe"] and polling_request.calls == []
            return decoded

        return asyncio.run(run())

    plugins._reset_plugin_managers_for_tests()
    try:
        # The same slug and relative module must load independently in both homes.
        for directory in (home, work):
            with _profile_runtime_scope(directory):
                plugins.discover_plugins()
                assert plugins.has_hook("gateway_platform_event")
        yield SimpleNamespace(home=home, work=work, events=events, dispatch=dispatch)
    finally:
        plugins._reset_plugin_managers_for_tests()


@pytest.mark.parametrize("actor_kind", ["user", "actor_chat"])
def test_registered_observer_discovers_profile_local_plugins(observer, actor_kind):
    updates = []
    for update_id, (chat_id, actor_id) in enumerate(((321, 777), (123, 777), (123, 888), (321, 777))):
        if actor_kind == "actor_chat":
            actor_id = -100000 - actor_id
        raw = _reaction_update(chat_id=chat_id, actor_id=actor_id, actor_kind=actor_kind, update_id=update_id)
        anonymous = copy.deepcopy(raw)
        del anonymous["message_reaction"][actor_kind]
        updates.extend((raw, anonymous))
    observer.dispatch(updates)
    seen = observer.events()
    # Shared ingress authorizes against A, even when the callback runs in B.
    assert [row["home"] for row in seen] == [str(observer.home), str(observer.work), str(observer.home)]
    assert [row["secret"] for row in seen] == ["777,-100777", "888,-100888", "777,-100777"]
    assert [row["sequence"] for row in seen] == [1, 1, 2]
    assert [row["event"]["payload"]["update_id"] for row in seen] == ["0", "1", "3"]
    for row in seen:
        assert row["home"] == row["import_home"] and row["secret"] == row["import_secret"]
        event = row["event"]
        assert event["platform"] == "telegram" and event["event_type"] == "reaction"
        payload = event["payload"]
        assert payload["actor_id"] == ("777" if actor_kind == "user" else "-100777")
        assert payload["old_emojis"] == ["👍"] and payload["old_custom_emoji_ids"] == ["111"]
        assert payload["emojis"] == ["🔥"] and payload["custom_emoji_ids"] == ["555"]
        assert payload["chat_type"] == "forum" and payload["thread_id"] is None
    assert get_hermes_home() == observer.home


@pytest.mark.parametrize("field", ["old_reaction", "new_reaction"])
@pytest.mark.parametrize("wire_case", ["missing", "null", "empty_object", "empty_string", "empty_list"])
def test_decoded_empty_snapshot_does_not_prove_wire_presence(observer, field, wire_case):
    """PTB erases wire validity; the observer projects the decoded empty state.

    The empty-list controls are valid additions/removals. The other inputs are
    outside Telegram's required-fields contract, not newly accepted wire syntax.
    """
    control, raw = _reaction_update(), _reaction_update()
    control["message_reaction"][field] = []
    if wire_case == "missing":
        raw["message_reaction"].pop(field)
    else:
        raw["message_reaction"][field] = {
            "null": None, "empty_object": {}, "empty_string": "", "empty_list": [],
        }[wire_case]
    decoded = observer.dispatch([control, raw])
    assert all(getattr(update.message_reaction, field) == () for update in decoded)
    assert decoded[0].to_dict() == decoded[1].to_dict()
    seen = observer.events()
    assert len(seen) == 2
    assert seen[0]["event"] == seen[1]["event"]
    payload = seen[0]["event"]["payload"]
    assert payload["old_emojis"] == ([] if field == "old_reaction" else ["👍"])
    assert payload["old_custom_emoji_ids"] == ([] if field == "old_reaction" else ["111"])
    assert payload["emojis"] == ([] if field == "new_reaction" else ["🔥"])
    assert payload["custom_emoji_ids"] == ([] if field == "new_reaction" else ["555"])
    assert payload["thread_id"] is None


@pytest.mark.parametrize("field", ["old_reaction", "new_reaction"])
@pytest.mark.parametrize("reaction_type", ["paid", "future_unknown"])
def test_observable_unsupported_typed_states_are_not_removals(observer, field, reaction_type):
    raw = _reaction_update()
    raw["message_reaction"][field] = [{"type": reaction_type}]
    decoded, = observer.dispatch([raw])
    assert getattr(decoded.message_reaction, field)[0].type == reaction_type
    assert observer.events() == []
