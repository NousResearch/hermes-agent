"""Test that inbound messages with telegram_business_connection_id are authorized
in the gateway authz mixin.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.session import SessionSource
from gateway.platforms.base import Platform


class MockGateway(GatewayAuthorizationMixin):
    def __init__(self):
        self._bot_loop_guard = MagicMock()
        self._bot_loop_guard.admit.return_value = (True, "admitted")
        self._bot_loop_guard.blocked.return_value = False
        self.adapters = {}

    def _adapter_profile_for_source(self, source):
        return None

    def _chat_scoped_grant(self, source, profile, is_group, allow_delegation):
        return False

    def _pairing_store_for(self, source):
        return None

    def _authorization_adapter(self, platform, profile=None):
        return None

    def _delivery_adapter_for(self, source):
        return None


def test_telegram_business_connection_authorized():
    gw = MockGateway()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        user_id="customer_999",
        chat_type="private",
    )
    # Set telegram_business_connection_id attribute
    source.telegram_business_connection_id = "biz_conn_abc123"

    assert gw._is_user_authorized(source) is True


def test_regular_unlisted_telegram_user_denied():
    gw = MockGateway()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        user_id="random_stranger_111",
        chat_type="private",
    )
    assert gw._is_user_authorized(source) is False
