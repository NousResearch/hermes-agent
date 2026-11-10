"""A revoked Matrix access token (M_UNKNOWN_TOKEN / "Token is not active") must be replaced by a
fresh password login and the send retried exactly once. Regression: the bot kept receiving events
but every reply failed, silently, until an operator restarted the gateway."""

import pytest

from gateway.config import PlatformConfig


def _adapter(**extra):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    cfg = {"homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}
    cfg.update(extra)
    return MatrixAdapter(PlatformConfig(enabled=True, token="", extra=cfg))


class _Api:
    def __init__(self, token):
        self.token = token


class _LoginResponse:
    access_token = "syt_fresh"
    device_id = "HERMES"


class _Client:
    def __init__(self, token="syt_stale", fail_times=1):
        self.api = _Api(token)
        self.device_id = "HERMES"
        self.fail_times = fail_times
        self.login_calls = 0
        self.send_calls = 0

    async def login(self, **kwargs):
        self.login_calls += 1
        self.login_kwargs = kwargs
        return _LoginResponse()

    async def send_message_event(self, room_id, event_type, content):
        self.send_calls += 1
        if self.send_calls <= self.fail_times:
            raise RuntimeError("M_UNKNOWN_TOKEN: Token is not active")
        return "$event"


@pytest.mark.parametrize("message", [
    "M_UNKNOWN_TOKEN: Token is not active",
    "matrix: token is not active",
    "Unknown token",
])
def test_token_invalid_error_detection(message):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    assert MatrixAdapter._token_invalid_error(RuntimeError(message))


def test_token_invalid_error_ignores_other_failures():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    assert not MatrixAdapter._token_invalid_error(RuntimeError("M_FORBIDDEN"))


@pytest.mark.asyncio
async def test_send_relogins_and_retries_once():
    adapter = _adapter(password="pw", device_id="HERMES")
    client = _Client()
    adapter._client = client

    event_id = await adapter._send_room_message("!room:example.org", {"msgtype": "m.text", "body": "hi"})

    assert event_id == "$event"
    assert client.send_calls == 2
    assert client.login_calls == 1
    assert client.api.token == "syt_fresh"
    assert adapter._access_token == "syt_fresh"
    assert client.login_kwargs["identifier"] == "@bot:example.org"
    assert client.login_kwargs["device_id"] == "HERMES"


@pytest.mark.asyncio
async def test_repeated_failure_does_not_loop():
    adapter = _adapter(password="pw", device_id="HERMES")
    client = _Client(fail_times=5)
    adapter._client = client

    with pytest.raises(RuntimeError):
        await adapter._send_room_message("!room:example.org", {"msgtype": "m.text", "body": "hi"})

    assert client.send_calls == 2
    assert client.login_calls == 1


@pytest.mark.asyncio
async def test_static_token_profile_does_not_try_to_login():
    adapter = _adapter()
    adapter._access_token = "syt_static"
    client = _Client()
    adapter._client = client

    with pytest.raises(RuntimeError):
        await adapter._send_room_message("!room:example.org", {"msgtype": "m.text", "body": "hi"})

    assert client.login_calls == 0
    assert client.send_calls == 1
