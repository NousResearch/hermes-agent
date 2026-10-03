"""Regression guard: plugin-owned inline buttons (samimizer approval cards).

`ok:` / `edit:` / `skip:` / `deck:` taps must be forwarded as a
``gateway_platform_event`` of type ``callback_query`` through the same auth
boundary as reactions, NOT handled inline by the adapter. The plugin answers
the tap itself (truthy hook result claims it). If nothing claims it inside
``_PLUGIN_CALLBACK_ANSWER_WINDOW``, the adapter answers so the button stops
spinning instead of hanging forever on the tapper's client.

Run: python3 -m pytest tests/test_telegram_plugin_callback_forward.py -q
"""

import asyncio
from types import SimpleNamespace

import pytest

from plugins.platforms.telegram import adapter as adapter_mod
from gateway.config import Platform


def _make_query(data="ok:card-1", user_id=111, chat_id=222, message_id=333, callback_id="cbq-1"):
    user = SimpleNamespace(id=user_id, username="sam", full_name="Sam")
    chat = SimpleNamespace(id=chat_id, type="private", is_forum=False)
    message = SimpleNamespace(chat=chat, message_id=message_id, message_thread_id=None, is_topic_message=False)
    answered = []

    async def _answer(text=None):
        answered.append(text)

    query = SimpleNamespace(
        data=data, id=callback_id, from_user=user, message=message, answer=_answer,
    )
    return query, answered


@pytest.fixture()
def adapter():
    class _Probe(adapter_mod.TelegramAdapter):
        @property
        def name(self) -> str:
            return "Telegram"

    obj = _Probe.__new__(_Probe)
    obj._background_tasks = set()
    obj.platform = Platform.TELEGRAM
    return obj


def test_plugin_prefix_is_dispatched_not_handled_inline(adapter, monkeypatch):
    """`_handle_callback_query` must route ok:/edit:/skip:/deck: to the forwarder and return."""
    query, _ = _make_query(data="ok:card-1")
    update = SimpleNamespace(callback_query=query)
    context = SimpleNamespace()

    calls = []

    def _fake_spawn(q, received_at):
        calls.append((q, received_at))

    monkeypatch.setattr(adapter, "_spawn_plugin_callback_forward", _fake_spawn)
    monkeypatch.setattr(adapter, "_accept_update", lambda: None)

    asyncio.run(adapter._handle_callback_query(update, context))

    assert len(calls) == 1, "an ok:/edit:/skip:/deck: tap must be forwarded, not handled inline"
    assert calls[0][0] is query


@pytest.mark.parametrize("prefix", ["ok:", "edit:", "skip:", "deck:"])
def test_all_four_samimizer_prefixes_are_recognized(adapter, monkeypatch, prefix):
    query, _ = _make_query(data=f"{prefix}card-9")
    update = SimpleNamespace(callback_query=query)
    context = SimpleNamespace()

    calls = []
    monkeypatch.setattr(adapter, "_spawn_plugin_callback_forward", lambda q, t: calls.append(q))
    monkeypatch.setattr(adapter, "_accept_update", lambda: None)

    asyncio.run(adapter._handle_callback_query(update, context))
    assert calls, f"prefix {prefix!r} was not routed to the plugin forwarder"


def test_plugin_claims_the_tap_adapter_does_not_answer(adapter, monkeypatch):
    """A truthy hook result means the plugin already answered; the adapter must stay silent."""
    query, answered = _make_query(data="ok:card-1")

    async def _handler(event, source):
        assert event["event_type"] == "callback_query"
        assert event["payload"]["data"] == "ok:card-1"
        return [True]  # plugin claimed it and answered itself; handler mirrors invoke_hook's List[Any]

    obj_handler = _handler
    adapter._platform_event_handler = obj_handler
    monkeypatch.setattr(adapter_mod, "has_hook", lambda name: True, raising=False)
    import hermes_cli.lifecycle as lifecycle_mod
    monkeypatch.setattr(lifecycle_mod, "has_hook", lambda name: True)

    handled = asyncio.run(adapter._forward_plugin_callback(query, 1234.0))

    assert handled is True
    assert answered == [], "the adapter must not answer a tap the plugin already claimed"


def test_unclaimed_tap_is_answered_by_adapter_after_window(adapter, monkeypatch):
    """No hook subscriber -> nothing claims it -> adapter answers so the spinner stops."""
    query, answered = _make_query(data="edit:card-2")

    adapter._platform_event_handler = None
    import hermes_cli.lifecycle as lifecycle_mod
    monkeypatch.setattr(lifecycle_mod, "has_hook", lambda name: False)

    handled = asyncio.run(adapter._forward_plugin_callback(query, 1234.0))

    assert handled is False
    assert answered == [adapter_mod._PLUGIN_CALLBACK_UNHANDLED_TEXT]


def test_falsy_hook_result_is_treated_as_unclaimed(adapter, monkeypatch):
    """A plugin that runs but returns False/None did not claim the tap; adapter must still answer."""
    query, answered = _make_query(data="skip:card-3")

    async def _handler(event, source):
        return False

    adapter._platform_event_handler = _handler
    import hermes_cli.lifecycle as lifecycle_mod
    monkeypatch.setattr(lifecycle_mod, "has_hook", lambda name: True)

    handled = asyncio.run(adapter._forward_plugin_callback(query, 1234.0))

    assert handled is False
    assert answered == [adapter_mod._PLUGIN_CALLBACK_UNHANDLED_TEXT]


def test_hook_timeout_falls_through_to_adapter_answer(adapter, monkeypatch):
    """A hook that never responds inside the window must not hang the button forever."""
    query, answered = _make_query(data="deck:card-4")

    async def _slow_handler(event, source):
        await asyncio.sleep(10)
        return True

    adapter._platform_event_handler = _slow_handler
    import hermes_cli.lifecycle as lifecycle_mod
    monkeypatch.setattr(lifecycle_mod, "has_hook", lambda name: True)
    monkeypatch.setattr(adapter_mod, "_PLUGIN_CALLBACK_ANSWER_WINDOW", 0.05)

    handled = asyncio.run(adapter._forward_plugin_callback(query, 1234.0))

    assert handled is False
    assert answered == [adapter_mod._PLUGIN_CALLBACK_UNHANDLED_TEXT]


def test_malformed_tap_missing_identity_falls_through_to_answer(adapter, monkeypatch):
    """No user_id -> _normalize_callback_query_event returns None -> adapter still answers."""
    query, answered = _make_query(data="ok:card-5", user_id=None)

    async def _handler(event, source):
        raise AssertionError("handler must not be invoked for a malformed event")

    adapter._platform_event_handler = _handler
    import hermes_cli.lifecycle as lifecycle_mod
    monkeypatch.setattr(lifecycle_mod, "has_hook", lambda name: True)

    handled = asyncio.run(adapter._forward_plugin_callback(query, 1234.0))

    assert handled is False
    assert answered == [adapter_mod._PLUGIN_CALLBACK_UNHANDLED_TEXT]
