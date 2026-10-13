"""Regression coverage for API-only Telegram observed context (#137627)."""

from types import SimpleNamespace

from gateway.run import _OBSERVED_GROUP_CONTEXT_HEADER
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


def test_observed_context_native_image_turn_persists_clean_multimodal_message(monkeypatch):
    """The observed wrapper is API-only even when native vision uses content parts."""
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

    runner._run_conversation_with_approval(Agent(), [], "earlier group chatter", None, None)

    assert captured["message"][0]["text"].startswith(_OBSERVED_GROUP_CONTEXT_HEADER)
    assert captured["persist_user_message"] == native_message
