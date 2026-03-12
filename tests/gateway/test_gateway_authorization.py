"""Tests for GatewayRunner user authorization behavior."""

from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def _make_discord_source(user_id: str = "507191144904654858", user_name: str = "treeindustries") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="1049724380118261770",
        chat_type="group",
        user_id=user_id,
        user_name=user_name,
    )


def test_authorizes_discord_user_by_id_allowlist():
    runner = _make_runner()
    source = _make_discord_source()

    with patch.dict("os.environ", {"DISCORD_ALLOWED_USERS": "507191144904654858"}, clear=True):
        assert runner._is_user_authorized(source) is True


def test_authorizes_discord_user_by_username_allowlist_case_insensitive():
    runner = _make_runner()
    source = _make_discord_source(user_name="TreeIndustries")

    with patch.dict("os.environ", {"DISCORD_ALLOWED_USERS": "JJTREE,@treeindustries"}, clear=True):
        assert runner._is_user_authorized(source) is True


def test_denies_discord_user_when_not_in_allowlist():
    runner = _make_runner()
    source = _make_discord_source(user_name="treeindustries")

    with patch.dict("os.environ", {"DISCORD_ALLOWED_USERS": "somebodyelse,123"}, clear=True):
        assert runner._is_user_authorized(source) is False
