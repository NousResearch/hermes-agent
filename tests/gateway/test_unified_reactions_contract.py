"""Unified cross-platform reaction contract (BasePlatformAdapter).

Pins the *invariants* of the reaction channel rather than any single platform's
wire format:

  * The base adapter always defines add_reaction / remove_reaction (so callers
    never have to probe for method existence), defaulting to a structured
    "not supported" result gated by SUPPORTS_REACTIONS=False.
  * Adapters that advertise SUPPORTS_REACTIONS=True override both coroutines and
    keep the unified add_reaction(self, chat_id, emoji, message_id=None)
    signature, returning a dict carrying ``success``.
  * Signal's composite "<author>:<timestamp_ms>" message-id parsing round-trips.
"""

import asyncio
import inspect

import pytest

from gateway.platforms.base import BasePlatformAdapter


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_base_defines_reaction_methods():
    assert inspect.iscoroutinefunction(BasePlatformAdapter.add_reaction)
    assert inspect.iscoroutinefunction(BasePlatformAdapter.remove_reaction)
    assert BasePlatformAdapter.SUPPORTS_REACTIONS is False


def test_default_reaction_is_structured_not_supported():
    # Call the base coroutines directly without instantiating the ABC — the
    # default implementation only reads ``self.name``.
    fake = type("X", (), {"name": "bare"})()
    add = _run(BasePlatformAdapter.add_reaction(fake, "chat", "👍", "m1"))
    rem = _run(BasePlatformAdapter.remove_reaction(fake, "chat", "m1"))
    for res in (add, rem):
        assert isinstance(res, dict)
        assert res["success"] is False
        assert "does not support" in res["error"]


def test_signal_message_id_round_trip():
    pytest.importorskip("aiohttp")
    from gateway.platforms.signal import SignalAdapter

    parse = SignalAdapter._parse_signal_message_id
    assert parse("uuid-abc:1718900000000") == ("uuid-abc", 1718900000000)
    # author may itself contain ':' — rpartition keeps the last segment as ts
    assert parse("a:b:1700000000000") == ("a:b", 1700000000000)
    # malformed inputs degrade to (None, None) rather than raising
    assert parse(None) == (None, None)
    assert parse("no-timestamp") == (None, None)
    assert parse("author:notanumber") == (None, None)


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("gateway.platforms.signal", "SignalAdapter"),
        ("plugins.platforms.telegram.adapter", "TelegramAdapter"),
        ("plugins.platforms.photon.adapter", "PhotonAdapter"),
        ("plugins.platforms.whatsapp.adapter", "WhatsAppAdapter"),
    ],
)
def test_reaction_capable_adapters_advertise_and_implement(module_path, class_name):
    """Each reaction-capable adapter sets the flag AND overrides both coroutines
    with the unified signature."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
    except Exception:  # optional deps (telegram/aiohttp) absent in this env
        pytest.skip(f"{module_path} not importable here")
        return
    cls = getattr(mod, class_name)
    assert cls.SUPPORTS_REACTIONS is True
    assert cls.add_reaction is not BasePlatformAdapter.add_reaction
    assert cls.remove_reaction is not BasePlatformAdapter.remove_reaction
    sig = inspect.signature(cls.add_reaction)
    params = list(sig.parameters)
    assert params[:3] == ["self", "chat_id", "emoji"]
    assert "message_id" in sig.parameters
