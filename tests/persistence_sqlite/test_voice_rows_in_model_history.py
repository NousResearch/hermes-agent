"""Saved spoken rows remain available to both model and display histories."""

from persistence.sqlite.conversation._store.messages import (
    append_message,
    get_messages_as_conversation,
    get_resume_conversations,
)
from persistence.sqlite.conversation._store.sessions import create_session


def test_spoken_readback_reaches_model_and_display_history(backend, scope_a):
    holder = backend.open(scope_a).holder()
    create_session(holder, "voice", "system prompt")
    append_message(holder, "voice", "user", "Explain El Niño.")
    append_message(holder, "voice", "assistant", "El Niño warms the Pacific.")
    append_message(holder, "voice", "assistant", "Checking. The Pacific gets warmer.",
                   display_metadata={"voice_live": {}})

    expected = ["Explain El Niño.", "El Niño warms the Pacific.",
                "Checking. The Pacific gets warmer."]
    assert [m["content"] for m in get_messages_as_conversation(holder, "voice")] == expected
    model, display = get_resume_conversations(holder, "voice")
    assert [m["content"] for m in model] == [expected[0], "\n".join(expected[1:])]
    assert [m["content"] for m in display] == expected
