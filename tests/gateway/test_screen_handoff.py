import json
from gateway.screen_handoff import (
    ScreenHandoffStore,
    _reset_for_tests,
    has_screen_handoff_notify,
    notify_screen_handoff,
    register_screen_handoff_notify,
    unregister_screen_handoff_notify,
)


def _source():
    return json.dumps({
        "platform": "telegram", "chat_id": "42", "chat_type": "dm", "user_id": "42",
    })


def test_screen_handoff_is_idempotent_and_never_persists_tokens(tmp_path):
    store = ScreenHandoffStore(tmp_path / "profile", db_path=tmp_path / "handoff.db")
    first, created = store.create_or_get(session_id="session-1", source_json=_source(), reason="login")
    second, reused = store.create_or_get(session_id="session-1", source_json=_source(), reason="again")

    assert created is True
    assert reused is False
    assert first.request_id == second.request_id
    assert first.invite_token
    assert first.confirmation_code
    raw = (tmp_path / "handoff.db").read_bytes()
    assert first.invite_token.encode() not in raw
    assert first.confirmation_code.encode() not in raw


def test_screen_handoff_authorization_mints_cookie_secret_and_is_explicit(tmp_path):
    store = ScreenHandoffStore(tmp_path / "profile", db_path=tmp_path / "handoff.db")
    handoff, _ = store.create_or_get(session_id="session-1", source_json=_source(), reason="login")
    assert store.by_token(handoff.invite_token, mark_opened=True).state == "opened"
    assert store.authorize(handoff.invite_token, "wrong") is None
    authorized = store.authorize(handoff.invite_token, handoff.confirmation_code)
    assert authorized is not None
    assert authorized.web_session_token
    assert store.web_session(authorized.web_session_token).state == "authorized"
    assert store.take_over(authorized.web_session_token, "screen-viewer")
    active = store.web_session(authorized.web_session_token)
    assert active.state == "human"
    returned = store.return_to_agent(authorized.web_session_token, "screen-viewer")
    assert returned.state == "returned"
    assert store.return_to_agent(authorized.web_session_token, "screen-viewer") is None


def test_reissue_rotates_invitation_without_creating_a_second_request(tmp_path):
    store = ScreenHandoffStore(tmp_path / "profile", db_path=tmp_path / "handoff.db")
    first, _ = store.create_or_get(session_id="session-1", source_json=_source(), reason="login")
    second = store.reissue(session_id="session-1", source_json=_source(), reason="/screen")
    assert second is not None
    assert second.request_id == first.request_id
    assert second.invite_token != first.invite_token
    assert second.confirmation_code != first.confirmation_code
    assert store.by_token(first.invite_token) is None


def test_screen_handoff_callback_is_called_and_refusal_does_not_authorize(tmp_path):
    calls = []
    register_screen_handoff_notify("session-1", calls.append)
    assert has_screen_handoff_notify("session-1")
    assert notify_screen_handoff("session-1", {"invite_url": "private"})
    assert calls == [{"invite_url": "private"}]
    unregister_screen_handoff_notify("session-1")
    assert not notify_screen_handoff("session-1", {})
    _reset_for_tests()

    store = ScreenHandoffStore(tmp_path / "profile", db_path=tmp_path / "handoff.db")
    handoff, _ = store.create_or_get(session_id="session-1", source_json=_source(), reason="login")
    assert store.refuse(handoff.invite_token)
    assert store.by_token(handoff.invite_token) is None
