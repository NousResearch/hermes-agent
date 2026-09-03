"""SessionSource.platform must be a Platform at runtime.

The annotation was not enforced: a caller constructing SessionSource with a
plain-string platform crashed with AttributeError on ``platform.value`` access
inside the authz gate (gateway/authz_mixin.py) instead of getting a clean
deny/authorize decision. __post_init__ now coerces valid strings and fails
fast on unknown ones — matching the wire path (from_dict), which already
raised ValueError for unknown platforms."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_string_platform_is_coerced_and_authz_decides_cleanly(monkeypatch):
    """A string platform no longer crashes the authz gate — it coerces to the
    enum member, so ``platform.value`` access behaves like any other source."""
    _clear_auth_env(monkeypatch)

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {
        Platform.TELEGRAM: SimpleNamespace(
            send=AsyncMock(),
            authorization_is_upstream=False,
            enforces_own_access_policy=False,
        )
    }
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False

    src = SessionSource(
        platform="telegram", user_id="4", chat_id="12", chat_type="dm"
    )
    assert isinstance(src.platform, Platform)
    assert src.platform is Platform.TELEGRAM

    # No env allowlist configured and no pairing entry => clean deny, not
    # an AttributeError escaping the gate.
    assert runner._is_user_authorized(src) is False


def test_unknown_platform_string_fails_fast_at_construction():
    """Invalid values raise at construction like from_dict does, rather than
    surfacing later as an AttributeError mid-authorization."""
    with pytest.raises(ValueError):
        SessionSource(platform="not-a-platform", chat_id="12")


def test_enum_platform_passes_through_unchanged():
    src = SessionSource(platform=Platform.SLACK, chat_id="12")
    assert src.platform is Platform.SLACK
