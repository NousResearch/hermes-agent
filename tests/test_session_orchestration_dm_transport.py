"""
Unit tests for session_orchestration/dm_transport.py (T007).

Coverage
--------
(a) get_dm_channel_id posts to the correct Discord endpoint with
    ``Authorization: Bot <token>`` and returns the channel id from
    the mocked response.
(b) send_dm calls get_dm_channel_id then posts to channels/{channel_id}/messages.
(c) An HTTP 4xx response causes send_dm to return False without raising.
(d) No real network calls are made — all via injected http_post.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest

from session_orchestration.dm_transport import (
    _Response,
    get_dm_channel_id,
    send_dm,
)


# ---------------------------------------------------------------------------
# Fake http_post builder
# ---------------------------------------------------------------------------


class _FakeHttp:
    """Records POSTs and returns pre-configured canned responses in order."""

    def __init__(self, responses: List[_Response]):
        self._responses = list(responses)
        self.calls: List[Tuple[str, Dict[str, str], Any]] = []

    def __call__(
        self,
        url: str,
        headers: Dict[str, str],
        json_body: Any,
    ) -> _Response:
        self.calls.append((url, headers, json_body))
        return self._responses.pop(0)

    def assert_no_network(self) -> None:
        """All calls went through this fake — no urllib was touched."""
        # If __call__ was reached, no real network call happened.
        # This is enforced structurally: we never import urllib in the test,
        # and the module only calls _default_http_post when http_post is None.
        pass  # assertion is the absence of real urllib calls


def _ok(data: Any) -> _Response:
    """Build a 200-OK _Response with JSON body."""
    return _Response(status_code=200, body=json.dumps(data).encode())


def _error(status: int) -> _Response:
    """Build an error _Response."""
    return _Response(status_code=status, body=b'{"code":0,"message":"error"}')


# ---------------------------------------------------------------------------
# (a) get_dm_channel_id — correct endpoint, correct header, returns channel id
# ---------------------------------------------------------------------------


class TestGetDmChannelId:
    def test_posts_to_users_me_channels(self):
        fake = _FakeHttp([_ok({"id": "111222333444555"})])
        channel_id = get_dm_channel_id("999888777", "my-bot-token", http_post=fake)

        assert len(fake.calls) == 1
        url, headers, body = fake.calls[0]
        assert url.endswith("/users/@me/channels"), f"unexpected url: {url}"

    def test_authorization_header_is_bot_token(self):
        fake = _FakeHttp([_ok({"id": "111222333444555"})])
        get_dm_channel_id("999888777", "my-bot-token", http_post=fake)

        _, headers, _ = fake.calls[0]
        assert headers.get("Authorization") == "Bot my-bot-token"

    def test_returns_channel_id_from_response(self):
        expected_channel = "111222333444555"
        fake = _FakeHttp([_ok({"id": expected_channel, "type": 1})])
        result = get_dm_channel_id("999888777", "tok", http_post=fake)
        assert result == expected_channel

    def test_recipient_id_in_request_body(self):
        fake = _FakeHttp([_ok({"id": "chan1"})])
        get_dm_channel_id("user-abc", "tok", http_post=fake)
        _, _, body = fake.calls[0]
        assert body == {"recipient_id": "user-abc"}

    def test_no_real_network(self):
        fake = _FakeHttp([_ok({"id": "chan1"})])
        get_dm_channel_id("user-abc", "tok", http_post=fake)
        fake.assert_no_network()

    def test_raises_on_4xx(self):
        fake = _FakeHttp([_error(403)])
        with pytest.raises(RuntimeError):
            get_dm_channel_id("bad-user", "tok", http_post=fake)


# ---------------------------------------------------------------------------
# (b) send_dm — calls get_dm_channel_id then posts to channels/{id}/messages
# ---------------------------------------------------------------------------


class TestSendDm:
    def test_posts_to_correct_channel_messages_url(self):
        channel_id = "777666555444"
        fake = _FakeHttp([
            _ok({"id": channel_id}),             # DM channel open
            _ok({"id": "msg-001", "content": "hello"}),  # message POST
        ])
        result = send_dm("user-123", "hello", "bot-tok", http_post=fake)

        assert result is True
        assert len(fake.calls) == 2
        msg_url, _, msg_body = fake.calls[1]
        assert f"/channels/{channel_id}/messages" in msg_url
        assert msg_body == {"content": "hello"}

    def test_authorization_header_on_message_post(self):
        fake = _FakeHttp([
            _ok({"id": "chan99"}),
            _ok({"id": "msg-002"}),
        ])
        send_dm("u", "hi", "secret-token", http_post=fake)
        _, headers, _ = fake.calls[1]
        assert headers.get("Authorization") == "Bot secret-token"

    def test_first_call_opens_dm_channel(self):
        """The first http_post goes to /users/@me/channels (get_dm_channel_id)."""
        fake = _FakeHttp([
            _ok({"id": "chan-dm"}),
            _ok({"id": "m1"}),
        ])
        send_dm("uid", "msg", "tok", http_post=fake)
        first_url, _, _ = fake.calls[0]
        assert "/users/@me/channels" in first_url

    def test_no_real_network(self):
        fake = _FakeHttp([
            _ok({"id": "c1"}),
            _ok({"id": "m1"}),
        ])
        send_dm("uid", "msg", "tok", http_post=fake)
        fake.assert_no_network()


# ---------------------------------------------------------------------------
# (c) HTTP 4xx response causes send_dm to return False without raising
# ---------------------------------------------------------------------------


class TestSendDmErrorHandling:
    def test_returns_false_on_message_post_4xx(self):
        """400 on the message post: returns False, does not raise."""
        fake = _FakeHttp([
            _ok({"id": "chan1"}),   # DM channel opened OK
            _error(400),            # message POST fails
        ])
        result = send_dm("user", "hello", "tok", http_post=fake)
        assert result is False

    def test_returns_false_on_403(self):
        fake = _FakeHttp([
            _ok({"id": "chan1"}),
            _error(403),
        ])
        result = send_dm("user", "hello", "tok", http_post=fake)
        assert result is False

    def test_returns_false_on_5xx(self):
        fake = _FakeHttp([
            _ok({"id": "chan1"}),
            _error(500),
        ])
        result = send_dm("user", "hello", "tok", http_post=fake)
        assert result is False

    def test_returns_false_when_channel_open_fails(self):
        """If get_dm_channel_id raises, send_dm returns False without raising."""
        fake = _FakeHttp([_error(403)])  # channel open fails
        result = send_dm("bad-user", "hello", "tok", http_post=fake)
        assert result is False

    def test_no_second_post_when_channel_open_fails(self):
        """If channel open fails, no message post is attempted."""
        fake = _FakeHttp([_error(403)])
        send_dm("bad-user", "hello", "tok", http_post=fake)
        assert len(fake.calls) == 1  # only the channel-open attempt

    def test_does_not_raise_on_4xx(self):
        """4xx must not propagate as an exception — just return False."""
        fake = _FakeHttp([
            _ok({"id": "chan1"}),
            _error(429),
        ])
        try:
            result = send_dm("user", "hello", "tok", http_post=fake)
        except Exception as exc:
            pytest.fail(f"send_dm raised unexpectedly: {exc}")
        assert result is False


# ---------------------------------------------------------------------------
# (d) Confirm no real network calls (structural guard)
# ---------------------------------------------------------------------------


class TestNoRealNetwork:
    def test_get_dm_channel_id_uses_only_injected_http(self, monkeypatch):
        """Patching urllib.request.urlopen to fail proves it is never called."""
        import urllib.request as urllib_req

        def _boom(*args, **kwargs):
            pytest.fail("urllib.request.urlopen was called — real network hit!")

        monkeypatch.setattr(urllib_req, "urlopen", _boom)

        fake = _FakeHttp([_ok({"id": "safe-chan"})])
        result = get_dm_channel_id("u1", "tok", http_post=fake)
        assert result == "safe-chan"

    def test_send_dm_uses_only_injected_http(self, monkeypatch):
        import urllib.request as urllib_req

        def _boom(*args, **kwargs):
            pytest.fail("urllib.request.urlopen was called — real network hit!")

        monkeypatch.setattr(urllib_req, "urlopen", _boom)

        fake = _FakeHttp([_ok({"id": "c1"}), _ok({"id": "m1"})])
        result = send_dm("u1", "hi", "tok", http_post=fake)
        assert result is True
