from hermes_cli.product_chat import (
    get_product_chat_session,
    product_chat_session_id,
    stream_product_chat_turn,
)


def test_product_chat_session_id_is_stable():
    user = {"preferred_username": "admin"}
    first = product_chat_session_id(user)
    second = product_chat_session_id(user)

    assert first == second
    assert first.startswith("product_admin_")


def test_get_product_chat_session_filters_visible_roles(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.product_chat.SessionDB",
        lambda: type(
            "_DB",
            (),
            {
                "get_session": lambda self, session_id: {"id": session_id},
                "get_messages_as_conversation": lambda self, session_id: [
                    {"role": "user", "content": "hello"},
                    {"role": "tool", "content": "hidden"},
                    {"role": "assistant", "content": "world"},
                ],
                "close": lambda self: None,
            },
        )(),
    )

    payload = get_product_chat_session({"preferred_username": "admin"})

    assert payload["session_id"].startswith("product_admin_")
    assert payload["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]


def test_stream_product_chat_turn_emits_reasoning_answer_and_final(monkeypatch):
    class _FakeAgent:
        def run_conversation(self, message, conversation_history=None, stream_callback=None, sync_honcho=None):
            assert message == "hello"
            assert conversation_history == [
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "done"},
            ]
            self.reasoning_callback("thinking")
            stream_callback("answer")
            return {"final_response": "done"}

        def __init__(self, reasoning_callback):
            self.reasoning_callback = reasoning_callback

    class _FakeDB:
        def close(self):
            return None

    messages = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "done"},
    ]

    monkeypatch.setattr(
        "hermes_cli.product_chat._build_agent",
        lambda **kwargs: (_FakeAgent(kwargs["reasoning_callback"]), _FakeDB()),
    )
    monkeypatch.setattr(
        "hermes_cli.product_chat._load_session_messages",
        lambda db, session_id: messages,
    )

    events = list(stream_product_chat_turn({"preferred_username": "admin"}, "hello"))

    assert any("event: start" in item for item in events)
    assert any('event: reasoning\ndata: {"delta": "thinking"}' in item for item in events)
    assert any('event: answer\ndata: {"delta": "answer"}' in item for item in events)
    assert any('"final_response": "done"' in item for item in events)
