"""Slack Web API responses must be read as real SDK responses, not dicts.

``slack_sdk`` returns ``SlackResponse``/``AsyncSlackResponse`` — mapping-like
objects that are **not** ``dict`` subclasses. Code that gated on
``isinstance(resp, dict)`` therefore took its "unexpected shape" branch on
every real call, collapsing user/channel names to raw IDs, treating every user
as a non-bot, and reporting successful sends as failures. Existing Slack tests
injected plain dicts, so the defect was invisible to them; every case here
exercises the SDK-shaped response as well.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


# Import the real response class first: the mock installer below fills
# sys.modules with MagicMocks, which would mask an installed slack_sdk.
try:
    from slack_sdk.web.async_slack_response import (  # noqa: E402
        AsyncSlackResponse as _AsyncSlackResponse,
    )
except Exception:  # pragma: no cover - slack extra not installed
    _AsyncSlackResponse = None


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same pattern as test_slack_mention.py)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import (  # noqa: E402
    SlackAdapter,
    _slack_response_payload,
    _standalone_upload_file,
)


class _SdkLikeResponse:
    """Stand-in for ``AsyncSlackResponse``: mapping-like, but not a ``dict``."""

    def __init__(self, data):
        self.data = data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __getitem__(self, key):
        return self.data.get(key)


def _real_sdk_response(data):
    """Build an actual ``AsyncSlackResponse`` — the shape production sees."""
    return _AsyncSlackResponse(
        client=None,
        http_verb="POST",
        api_url="https://slack.com/api/users.info",
        req_args={},
        data=data,
        headers={},
        status_code=200,
    )


# Every behavioral case runs against both the hand-rolled stand-in and, when
# installed, the real SDK class — mocks alone are what hid this bug.
_FACTORIES = [pytest.param(_SdkLikeResponse, id="sdk-like")]
if isinstance(_AsyncSlackResponse, type):
    _FACTORIES.append(pytest.param(_real_sdk_response, id="slack_sdk"))

response_shape = pytest.mark.parametrize("make_response", _FACTORIES)


def _make_adapter():
    # object.__new__ skips __init__ (heavy setup) — established slack-test pattern.
    adapter = object.__new__(SlackAdapter)
    adapter._app = MagicMock()
    adapter._user_name_cache = {}
    adapter._user_is_bot_cache = {}
    adapter._channel_name_cache = {}
    adapter._channel_team = {}
    adapter._USER_NAME_CACHE_MAX = 5000
    adapter._CHANNEL_NAME_CACHE_MAX = 5000
    return adapter


# ── the normalizer's contract ───────────────────────────────────────────────


class TestSlackResponsePayload:
    def test_plain_dict_passes_through(self):
        payload = {"ok": True}
        assert _slack_response_payload(payload) is payload

    @response_shape
    def test_sdk_response_yields_its_data(self, make_response):
        assert _slack_response_payload(make_response({"ok": True})) == {"ok": True}

    @response_shape
    def test_sdk_response_is_not_a_dict(self, make_response):
        """The premise of the bug: the runtime object fails an isinstance dict gate."""
        assert not isinstance(make_response({"ok": True}), dict)

    @response_shape
    def test_binary_response_is_not_mistaken_for_data(self, make_response):
        """``SlackResponse.data`` may be bytes; callers need their fallback then."""
        assert _slack_response_payload(make_response(b"\x89PNG")) == {}

    def test_unknown_shape_yields_empty(self):
        assert _slack_response_payload(object()) == {}
        assert _slack_response_payload(None) == {}


# ── the call sites that silently degraded ──────────────────────────────────


class TestIdentityResolution:
    @response_shape
    def test_user_name_resolves(self, make_response):
        """The reported symptom: names collapsed to the raw user id."""
        adapter = _make_adapter()
        adapter._app.client.users_info = AsyncMock(
            return_value=make_response(
                {"ok": True, "user": {"profile": {"display_name": "Nikita"}}}
            )
        )
        name = asyncio.run(adapter._resolve_user_name("U_HUMAN"))
        assert name == "Nikita"

    @response_shape
    def test_user_is_bot_resolves(self, make_response):
        """allow_bots policy depends on this; a wrong False re-opens routing loops."""
        adapter = _make_adapter()
        adapter._app.client.users_info = AsyncMock(
            return_value=make_response(
                {"ok": True, "user": {"is_bot": True, "profile": {}}}
            )
        )
        assert asyncio.run(adapter._resolve_user_is_bot("U_PEER_BOT")) is True

    @response_shape
    def test_channel_name_resolves(self, make_response):
        adapter = _make_adapter()
        client = MagicMock()
        client.conversations_info = AsyncMock(
            return_value=make_response({"ok": True, "channel": {"name": "general"}})
        )
        adapter._get_client = lambda *_a, **_kw: client
        assert asyncio.run(adapter._resolve_channel_name("C_GEN")) == "general"

    def test_unknown_shape_still_falls_back_to_the_id(self):
        """Degradation for genuinely unreadable responses must be preserved."""
        adapter = _make_adapter()
        adapter._app.client.users_info = AsyncMock(return_value=object())
        assert asyncio.run(adapter._resolve_user_name("U_HUMAN")) == "U_HUMAN"


class TestSendPaths:
    @response_shape
    def test_ephemeral_reply_is_reported_as_delivered(self, make_response):
        """Slash-command replies were reported as 'unexpected_response' failures."""
        adapter = _make_adapter()
        client = MagicMock()
        client.chat_postEphemeral = AsyncMock(
            return_value=make_response({"ok": True})
        )
        adapter._get_client = lambda *_a, **_kw: client
        result = asyncio.run(
            adapter._post_ephemeral_fallback("C_GEN", {"user_id": "U_HUMAN"}, "hi")
        )
        assert result.success is True

    @response_shape
    def test_ephemeral_error_is_surfaced(self, make_response):
        adapter = _make_adapter()
        client = MagicMock()
        client.chat_postEphemeral = AsyncMock(
            return_value=make_response({"ok": False, "error": "channel_not_found"})
        )
        adapter._get_client = lambda *_a, **_kw: client
        result = asyncio.run(
            adapter._post_ephemeral_fallback("C_GEN", {"user_id": "U_HUMAN"}, "hi")
        )
        assert result.success is False
        assert "channel_not_found" in result.error

    @response_shape
    def test_upload_returns_the_message_id(self, make_response, tmp_path):
        """A lost message_id breaks threading of follow-up sends."""
        media = tmp_path / "note.txt"
        media.write_text("x")
        client = MagicMock()
        client.files_upload_v2 = AsyncMock(
            return_value=make_response(
                {"ok": True, "file": {"timestamp": "123.456"}}
            )
        )
        result = asyncio.run(
            _standalone_upload_file(client, "C_GEN", str(media))
        )
        assert result == {
            "success": True,
            "message_id": "123.456",
            "raw": client.files_upload_v2.return_value,
        }

    @response_shape
    def test_upload_error_is_surfaced(self, make_response, tmp_path):
        media = tmp_path / "note.txt"
        media.write_text("x")
        client = MagicMock()
        client.files_upload_v2 = AsyncMock(
            return_value=make_response({"ok": False, "error": "not_in_channel"})
        )
        result = asyncio.run(
            _standalone_upload_file(client, "C_GEN", str(media))
        )
        assert "not_in_channel" in result["error"]
