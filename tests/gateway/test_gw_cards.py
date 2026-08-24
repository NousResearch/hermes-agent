"""The optional GW Cards bridge claims only a complete generated handler."""
from __future__ import annotations

import asyncio
import sys
from types import ModuleType

from gateway import gw_cards


def _write_handler(path, *, commands=("/gwtasks",), artifact=None,
                   complete=True, replies=False):
    """Write a generated-handler-shaped module at an explicit path."""
    lines = [
        f"GW_CARD_COMMANDS = {commands!r}",
        "def is_gw_card(_data): return False",
        "def is_gw_card_command(_text): return False",
        f"def _artifact(): return {artifact if artifact is not None else {}!r}",
    ]
    if complete:
        lines.extend([
            "async def handle_gw_card_callback(*_args): return None",
            "async def handle_gw_card_command(*_args): return None",
        ])
    if replies:
        lines.extend([
            "def is_gw_card_message(message): return bool(getattr(message, 'reply_to_message', None))",
            "def enable_gw_card_message_forwarding(): return None",
            "async def handle_gw_card_message(*_args): return True",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _install(monkeypatch, tmp_path, **kwargs):
    path = _write_handler(tmp_path / "gw_card_handler.py", **kwargs)
    monkeypatch.setenv(gw_cards.HANDLER_PATH_ENV, str(path))
    return path


def test_absent_handler_advertises_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv(gw_cards.HANDLER_PATH_ENV, str(tmp_path / "missing.py"))
    assert gw_cards.command_menu_entries() == []


def test_menu_is_bound_to_handler_commands_not_artifact(monkeypatch, tmp_path):
    _install(
        monkeypatch, tmp_path,
        commands=("/gwtasks", "/oldest", "/gwtasks"),
        artifact={"command_menu": [
            {"command": "gwtasks", "description": "Task backlog"},
            {"command": "foreign", "description": "Must not be exposed"},
        ]},
    )

    assert gw_cards.command_menu_entries() == [
        ("gwtasks", "Task backlog"),
        ("oldest", "GW task cards"),
    ]


def test_incomplete_handler_is_not_a_command_owner(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path, complete=False)

    assert gw_cards.handler() is None
    assert gw_cards.command_menu_entries() == []


def test_bad_artifact_keeps_only_dispatchable_generic_entries(monkeypatch, tmp_path):
    path = _install(monkeypatch, tmp_path, commands=("/gwtasks",))
    path.write_text(path.read_text(encoding="utf-8").replace(
        "def _artifact(): return {}",
        "def _artifact(): raise ValueError('bad artifact')",
    ), encoding="utf-8")

    assert gw_cards.command_menu_entries() == [("gwtasks", "GW task cards")]


def test_foreign_module_cannot_supply_an_absent_local_handler(monkeypatch, tmp_path):
    """A handler from another checkout must not make this checkout claim."""
    foreign = ModuleType("tools.gw_card_handler")
    foreign.GW_CARD_COMMANDS = ("/foreign",)
    monkeypatch.setitem(sys.modules, "tools.gw_card_handler", foreign)
    monkeypatch.setenv(gw_cards.HANDLER_PATH_ENV, str(tmp_path / "missing.py"))

    assert gw_cards.handler() is None
    assert gw_cards.command_menu_entries() == []


def test_default_path_is_anchored_to_this_checkout(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    gateway = checkout / "gateway"
    tools = checkout / "tools"
    gateway.mkdir(parents=True)
    tools.mkdir()
    _write_handler(
        tools / "gw_card_handler.py",
        commands=("/gwtasks",),
        artifact={"command_menu": [
            {"command": "gwtasks", "description": "Checkout handler"},
        ]},
    )
    foreign = ModuleType("tools.gw_card_handler")
    foreign.GW_CARD_COMMANDS = ("/foreign",)
    monkeypatch.setitem(sys.modules, "tools.gw_card_handler", foreign)
    monkeypatch.delenv(gw_cards.HANDLER_PATH_ENV, raising=False)
    monkeypatch.setattr(gw_cards, "__file__", str(gateway / "gw_cards.py"))

    assert gw_cards.command_menu_entries() == [("gwtasks", "Checkout handler")]


def test_local_file_wins_over_a_foreign_module(monkeypatch, tmp_path):
    foreign = ModuleType("tools.gw_card_handler")
    foreign.GW_CARD_COMMANDS = ("/foreign",)
    monkeypatch.setitem(sys.modules, "tools.gw_card_handler", foreign)
    _install(monkeypatch, tmp_path,
             commands=("/gwtasks",),
             artifact={"command_menu": [
                 {"command": "gwtasks", "description": "Local handler"},
             ]})

    assert gw_cards.command_menu_entries() == [("gwtasks", "Local handler")]


def test_reply_capability_requires_the_complete_generated_surface():
    partial = ModuleType("partial")
    partial.is_gw_card_message = lambda _message: True
    partial.handle_gw_card_message = lambda *_args: True
    assert gw_cards.enable_message_forwarding(partial) is False

    seen = []
    complete = ModuleType("complete")
    complete.is_gw_card_message = lambda _message: True

    async def message(*_args):
        return True

    def enable():
        seen.append("enabled")

    complete.handle_gw_card_message = message
    complete.enable_gw_card_message_forwarding = enable
    assert gw_cards.enable_message_forwarding(complete) is True
    assert seen == ["enabled"]


def test_handler_loader_accepts_old_surface_but_exposes_new_reply_surface(
    monkeypatch, tmp_path
):
    _install(monkeypatch, tmp_path, complete=True, replies=True)
    mod = gw_cards.handler()
    assert mod is not None
    assert gw_cards.enable_message_forwarding(mod) is True


def test_telegram_adapter_dispatches_the_same_generated_handler(monkeypatch):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from plugins.platforms.telegram import adapter as telegram_adapter

    seen = []
    mod = ModuleType("tools.gw_card_handler")
    mod.is_gw_card = lambda data: data == "card|1"
    mod.is_gw_card_command = lambda text: text == "/gwtasks"
    mod.is_gw_card_message = lambda message: bool(
        getattr(message, "reply_to_message", None)
    )

    async def callback(query, data, adapter_name):
        seen.append(("callback", query, data, adapter_name))

    async def command(message, text, adapter_name):
        seen.append(("command", message, text, adapter_name))

    async def card_message(message, adapter_name):
        seen.append(("message", message, adapter_name))
        return True

    def enable_replies():
        seen.append(("enabled",))

    mod.handle_gw_card_callback = callback
    mod.handle_gw_card_command = command
    mod.handle_gw_card_message = card_message
    mod.enable_gw_card_message_forwarding = enable_replies
    monkeypatch.setattr(telegram_adapter, "gw_cards_handler", lambda: mod)

    adapter = object.__new__(TelegramAdapter)
    monkeypatch.setattr(TelegramAdapter, "name", property(lambda _self: "telegram"))
    message = type(
        "Message", (), {"text": "/gwtasks", "reply_to_message": None}
    )()
    adapter._effective_update_message = lambda _update: message

    asyncio.run(adapter._handle_command(object(), None))
    query = type("Query", (), {"data": "card|1"})()
    asyncio.run(adapter._handle_callback_query(
        type("Update", (), {"callback_query": query})(), None
    ))

    assert seen[:3] == [
        ("command", message, "/gwtasks", "telegram"),
        ("enabled",),
        ("callback", query, "card|1", "telegram"),
    ]

    reply = type(
        "Message",
        (),
        {
            "text": "new wording",
            "reply_to_message": object(),
            "from_user": type("User", (), {"id": 7})(),
            "chat": type("Chat", (), {"id": 5})(),
        },
    )()
    adapter._effective_update_message = lambda _update: reply
    adapter._is_user_authorized_from_message = lambda _message: True
    asyncio.run(
        adapter._handle_text_message(
            type("Update", (), {"message": reply, "update_id": 1})(), None
        )
    )
    assert seen[-2:] == [("enabled",), ("message", reply, "telegram")]

    # An unowned reply continues down the normal Telegram path.
    async def unhandled(_message, _adapter_name):
        seen.append(("message-unhandled",))
        return False

    mod.handle_gw_card_message = unhandled
    adapter._should_process_message = lambda _message: False
    adapter._should_observe_unmentioned_group_message = lambda _message: False
    asyncio.run(
        adapter._handle_text_message(
            type("Update", (), {"message": reply, "update_id": 2})(), None
        )
    )
    assert seen[-1] == ("message-unhandled",)
