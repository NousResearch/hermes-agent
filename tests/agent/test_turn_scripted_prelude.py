"""A scripted prelude's tool call id is unique per chat: a rerun of onboarding plays the same name card at the
same place in a new chat, and the desktop keys tool rows by that id."""

from agent.turn_scripted_prelude import prelude_call_id


def test_the_same_card_at_the_same_place_gets_a_different_id_in_another_chat():
    first = prelude_call_id("chat-1", "setup_beat", {"beat": "name"}, 1)

    assert first == prelude_call_id("chat-1", "setup_beat", {"beat": "name"}, 1)
    assert first != prelude_call_id("chat-2", "setup_beat", {"beat": "name"}, 1)
