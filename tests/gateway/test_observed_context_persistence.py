"""Regression coverage for API-only Telegram observed context (#137627)."""

from types import SimpleNamespace

import pytest

from hermes_agent.gateway.run import _OBSERVED_GROUP_CONTEXT_HEADER
from hermes_agent.gateway.run_turn_runner import TurnRunner
from hermes_agent.gateway.turn_context import TurnContext


def test_observed_context_native_image_turn_persists_clean_multimodal_message(monkeypatch):
    """The observed wrapper is API-only even when native vision uses content parts: a str persist
    override cannot replace list content, so the WRAPPED list was persisted and re-wrapped every turn."""
    native_message = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    captured = {}

    class Agent:
        def run_conversation(self, message, **kwargs):
            captured["message"] = message
            captured["persist_user_message"] = kwargs.get("persist_user_message")
            return {"final_response": "done"}

    ctx = TurnContext(
        message="What is in this image?",
        session_id="session-1",
        session_key="telegram:session-1",
        source=SimpleNamespace(user_id="user-1", user_name="Example", is_bot=False),
    )
    runner = object.__new__(TurnRunner)
    runner._ctx = ctx
    monkeypatch.setattr(runner, "_native_image_run_message", lambda: native_message)

    # Production always hands a clean str override (the timestamp-strip path), so the native list must win.
    runner._run_conversation_with_approval(Agent(), [], "earlier group chatter", "What is in this image?", None)

    assert captured["message"][0]["text"].startswith(_OBSERVED_GROUP_CONTEXT_HEADER)
    assert captured["persist_user_message"] == native_message


def test_observed_context_native_image_turn_persists_clean_override_with_image_hints(monkeypatch):
    """The model-facing text carries a timestamp the stored row must not: the clean override replaces
    the leading copy of the live text while the native image hints and image parts survive."""
    live_text = "[Sat 2026-11-07 10:00] What is in this image?"
    hints = "\n\n[Image attached at: /cache/img/photo.png]"
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    native_message = [{"type": "text", "text": live_text + hints}, image_part]
    captured = {}

    class Agent:
        def run_conversation(self, message, **kwargs):
            captured["persist_user_message"] = kwargs.get("persist_user_message")
            return {"final_response": "done"}

    ctx = TurnContext(
        message=live_text,
        session_id="session-1",
        session_key="telegram:session-1",
        source=SimpleNamespace(user_id="user-1", user_name="Example", is_bot=False),
    )
    runner = object.__new__(TurnRunner)
    runner._ctx = ctx
    monkeypatch.setattr(runner, "_native_image_run_message", lambda: native_message)

    runner._run_conversation_with_approval(Agent(), [], "earlier group chatter", "What is in this image?", None)

    assert captured["persist_user_message"] == [
        {"type": "text", "text": "What is in this image?" + hints}, image_part,
    ]


@pytest.mark.parametrize(("override", "live_text"), [
    (None, "What is in this image?"),  # no clean override (#137632's case)
    ("What is in this image?", "[Sat 2026-11-07 10:00] Something else"),  # live text is not the part's prefix
])
def test_observed_context_native_image_turn_keeps_pre_wrap_list_without_matching_override(monkeypatch, override, live_text):
    """Without a clean override that maps onto the first text part, the pre-wrap list is persisted as is."""
    native_message = [
        {"type": "text", "text": "What is in this image?\n\n[Image attached at: /cache/img/photo.png]"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    captured = {}

    class Agent:
        def run_conversation(self, message, **kwargs):
            captured["persist_user_message"] = kwargs.get("persist_user_message")
            return {"final_response": "done"}

    ctx = TurnContext(
        message=live_text,
        session_id="session-1",
        session_key="telegram:session-1",
        source=SimpleNamespace(user_id="user-1", user_name="Example", is_bot=False),
    )
    runner = object.__new__(TurnRunner)
    runner._ctx = ctx
    monkeypatch.setattr(runner, "_native_image_run_message", lambda: native_message)

    runner._run_conversation_with_approval(Agent(), [], "earlier group chatter", override, None)

    assert captured["persist_user_message"] == native_message
