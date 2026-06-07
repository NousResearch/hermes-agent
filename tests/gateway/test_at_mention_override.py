"""@<profile> per-turn override — routes one turn without changing the binding."""

from gateway.chat_bindings import parse_profile_mention


def test_parses_leading_mention():
    assert parse_profile_mention("@coder fix this bug") == ("coder", "fix this bug")


def test_no_mention_returns_text_unchanged():
    assert parse_profile_mention("just a normal message") == (None, "just a normal message")


def test_mention_only_no_body():
    # A bare "@coder" with no message is not a routing override.
    assert parse_profile_mention("@coder") == (None, "@coder")


def test_email_like_text_is_not_a_mention():
    assert parse_profile_mention("ping bob@example.com please") == (None, "ping bob@example.com please")


def test_hyphenated_profile_name():
    assert parse_profile_mention("@code-helper do it") == ("code-helper", "do it")


def test_mention_does_not_mutate_binding(tmp_path):
    # A per-turn @mention must not change the persisted chat binding.
    from gateway.chat_bindings import ChatBindings, chat_binding_key
    from gateway.session import SessionSource
    from gateway.platforms.base import Platform

    src = SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", user_id="u1")
    b = ChatBindings(tmp_path / "b.json")
    b.set(chat_binding_key(src), "personal")
    # Parsing a mention is read-only — the store is untouched.
    parse_profile_mention("@coder hi")
    assert b.get(chat_binding_key(src)) == "personal"
