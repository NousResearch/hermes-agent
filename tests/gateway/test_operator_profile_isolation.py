# Copyright (c) Nous Research
# Licensed under the MIT license.

"""Tests for operator profile isolation in multi-user / gateway sessions (#110686).

Validates that the operator's private USER PROFILE (USER.md) is not leaked to
non-operator callers or shared multi-user sessions.
"""

from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.authz_mixin import GatewayAuthorizationMixin
from run_agent import AIAgent


class DummyGateway(GatewayAuthorizationMixin):
    def __init__(self):
        self._pairing_store = None

    def _adapter_profile_for_source(self, source):
        return None

    def _pairing_store_for(self, source):
        return self._pairing_store


class TestOperatorSourceAuthz:
    """Verify operator source identification and profile inclusion rules."""

    def test_local_platform_is_operator(self):
        gw = DummyGateway()
        src = SessionSource(platform=Platform.LOCAL, chat_id="local")
        assert gw.is_operator_source(src) is True
        assert gw.should_include_operator_profile(src) is True

    def test_allowed_users_is_operator(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "123456789,987654321")
        gw = DummyGateway()
        operator_src = SessionSource(
            platform=Platform.DISCORD,
            chat_id="dm-1",
            chat_type="dm",
            user_id="123456789",
            user_name="operator",
        )
        non_operator_src = SessionSource(
            platform=Platform.DISCORD,
            chat_id="dm-2",
            chat_type="dm",
            user_id="555555555",
            user_name="random_guest",
        )

        assert gw.is_operator_source(operator_src) is True
        assert gw.should_include_operator_profile(operator_src) is True

        assert gw.is_operator_source(non_operator_src) is False
        assert gw.should_include_operator_profile(non_operator_src) is False

    def test_shared_multi_user_omits_operator_profile(self, monkeypatch):
        monkeypatch.setenv("DISCORD_ALLOWED_USERS", "123456789")
        gw = DummyGateway()
        shared_src = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chan-1",
            chat_type="channel",
            user_id="123456789",
            user_name="operator",
        )
        # In a non-DM channel, operator profile is omitted to prevent leaking to room
        assert gw.should_include_operator_profile(shared_src) is False

    def test_pairing_store_approved_is_operator(self):
        mock_pairing = MagicMock()
        mock_pairing.is_approved.side_effect = lambda plat, uid: uid == "paired_user"
        gw = DummyGateway()
        gw._pairing_store = mock_pairing

        paired_src = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="dm-paired",
            chat_type="dm",
            user_id="paired_user",
        )
        unpaired_src = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="dm-unpaired",
            chat_type="dm",
            user_id="stranger",
        )

        assert gw.is_operator_source(paired_src) is True
        assert gw.should_include_operator_profile(paired_src) is True

        assert gw.is_operator_source(unpaired_src) is False
        assert gw.should_include_operator_profile(unpaired_src) is False


class TestAgentUserProfileIsolation:
    """Verify AIAgent respects user_profile_enabled in prompt assembly."""

    def test_agent_user_profile_disabled(self):
        agent = AIAgent(
            model="test-model",
            base_url="http://127.0.0.1:11434/v1",
            api_key="test-key",
            quiet_mode=True,
            user_profile_enabled=False,
        )
        assert agent._user_profile_enabled is False
