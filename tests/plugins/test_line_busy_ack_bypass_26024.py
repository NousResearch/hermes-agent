"""Integration test for #26024: configurable busy-ack templates must still
surface as visible LINE bubbles even when the template text is localized /
emoji-free (i.e. does not match the hardcoded ``_SYSTEM_BYPASS_PREFIXES``).

Regression guard for the hermes-sweeper finding on PR #26150: a busy-ack
template rendered by ``gateway/run.py`` routes through the LINE adapter's
``send()``. When the chat has an outstanding PENDING postback button, a
prefix-free template was being swallowed into the postback cache instead of
being pushed as a visible message. The fix carries a structured ``busy_ack``
delivery marker in metadata so LINE bypasses the cache regardless of the
operator-controlled template text.
"""

import asyncio
import types

import pytest

from plugins.platforms.line.adapter import LineAdapter


def _make_adapter() -> LineAdapter:
    cfg = types.SimpleNamespace(extra={})
    adapter = LineAdapter(cfg)

    class _FakeClient:
        def __init__(self) -> None:
            self.pushed: list = []
            self.replied: list = []

        async def push(self, chat_id, messages):
            self.pushed.append((chat_id, messages))

        async def reply(self, token, messages):
            self.replied.append((token, messages))

    adapter._client = _FakeClient()
    return adapter


def test_localized_busy_ack_with_pending_button_bypasses_cache_via_metadata():
    adapter = _make_adapter()
    chat_id = "Uabc123"

    # Simulate an outstanding PENDING postback button for this chat, which
    # normally diverts responses into the postback cache.
    rid = adapter._cache.register_pending(chat_id)
    adapter._pending_buttons[chat_id] = rid

    # A localized / emoji-free busy-ack template — does NOT match any
    # hardcoded _SYSTEM_BYPASS_PREFIXES.
    localized_ack = "Ich bearbeite gerade eine Aufgabe. Antwort folgt gleich."

    result = asyncio.run(
        adapter.send(chat_id, localized_ack, metadata={"busy_ack": True})
    )

    assert result.success
    # It must be pushed as a visible bubble, not cached.
    assert adapter._client.pushed, "busy-ack should be pushed to LINE, not cached"
    # The pending postback entry must be untouched (not flipped to READY).
    from plugins.platforms.line.adapter import State
    assert adapter._cache.get(rid).state is State.PENDING


def test_normal_response_with_pending_button_still_caches():
    adapter = _make_adapter()
    chat_id = "Uxyz789"

    rid = adapter._cache.register_pending(chat_id)
    adapter._pending_buttons[chat_id] = rid

    result = asyncio.run(
        adapter.send(chat_id, "Here is your normal answer.", metadata=None)
    )

    assert result.success
    # No metadata marker + non-system content → routed into the cache.
    assert not adapter._client.pushed, "normal response should be cached, not pushed"
    assert result.message_id == rid


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
