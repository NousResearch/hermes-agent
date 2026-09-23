"""Allowlist matching compares qualified user_ids, not bare '@' localparts.

The generic ``user_id.split("@")[0]`` alias in ``_principal_matches_allowlist``
was added for WhatsApp JIDs (``<phone>@s.whatsapp.net``) before WhatsApp got a
dedicated alias expansion. Left unconditional, it makes a bare allowlist entry
``alice`` admit ``alice@<any domain>`` on every platform whose user_id is
'@'-shaped (email, Google Chat, iMessage handles) — domains the sender controls.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.authz_mixin import _principal_matches_allowlist
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _source(platform: Platform, user_id: str) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=user_id,
        user_name="tester",
        chat_type="dm",
    )


def _make_runner(platform: Platform):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True)})
    runner.adapters = {platform: SimpleNamespace(send=AsyncMock())}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._update_prompts = {}
    runner.hooks = SimpleNamespace(dispatch=AsyncMock(return_value=None))
    runner._sessions = {}
    return runner


# BlueBubbles registers no platform allowlist env and its adapter does no sender
# gate, so GATEWAY_ALLOWED_USERS is the only enforcement on that path.
@pytest.mark.parametrize(
    ("platform", "env_var"),
    [(Platform.EMAIL, "EMAIL_ALLOWED_USERS"), (Platform.BLUEBUBBLES, "GATEWAY_ALLOWED_USERS")],
)
def test_bare_localpart_entry_admits_no_foreign_domain(monkeypatch, platform, env_var):
    for key in ("EMAIL_ALLOWED_USERS", "EMAIL_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(env_var, "alice,bob@corp.example")
    runner = _make_runner(platform)

    assert runner._is_user_authorized(_source(platform, "alice@evil.example")) is False
    assert runner._is_user_authorized(_source(platform, "bob@evil.example")) is False
    # Positive control: a qualified entry still admits its exact sender.
    assert runner._is_user_authorized(_source(platform, "bob@corp.example")) is True


@pytest.mark.parametrize(
    ("platform", "user_id"),
    [
        (Platform.WHATSAPP, "15550000001@s.whatsapp.net"),
        (Platform.WHATSAPP, "15550000001:47@s.whatsapp.net"),
        (Platform.WHATSAPP_CLOUD, "15550000001@s.whatsapp.net"),
    ],
)
def test_whatsapp_bare_phone_entry_still_matches_jid(platform, user_id):
    """The original intent survives via the scoped WhatsApp expansion, not the generic split."""
    assert _principal_matches_allowlist(_source(platform, user_id), user_id, {"15550000001"}) is True
