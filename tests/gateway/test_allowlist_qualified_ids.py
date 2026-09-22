"""Allowlist matching compares qualified user_ids, not bare '@' localparts.

The generic ``user_id.split("@")[0]`` alias in ``_principal_matches_allowlist``
was added for WhatsApp JIDs (``<phone>@s.whatsapp.net``) before WhatsApp got a
dedicated alias expansion. Left unconditional, it makes a bare allowlist entry
``alice`` admit ``alice@<any domain>`` on every platform whose user_id is
'@'-shaped (email, Google Chat, iMessage handles) — domains the sender controls.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


def test_bare_localpart_entry_does_not_admit_foreign_domain():
    """alice@evil.example must not match an allowlist holding bare ``alice``."""
    source = _source(Platform.EMAIL, "alice@evil.example")
    assert _principal_matches_allowlist(source, "alice@evil.example", {"alice"}) is False


def test_qualified_entry_still_matches():
    source = _source(Platform.EMAIL, "alice@corp.example")
    assert _principal_matches_allowlist(
        source, "alice@corp.example", {"alice@corp.example"}
    ) is True


def test_whatsapp_bare_phone_entry_still_matches_jid():
    """The original intent survives via the scoped WhatsApp expansion, not the
    generic split: a bare phone allowlist entry matches the sender's JID."""
    source = _source(Platform.WHATSAPP, "15550000001@s.whatsapp.net")
    assert _principal_matches_allowlist(
        source, "15550000001@s.whatsapp.net", {"15550000001"}
    ) is True


def test_whatsapp_device_suffix_jid_matches_bare_phone():
    source = _source(Platform.WHATSAPP, "15550000001:47@s.whatsapp.net")
    assert _principal_matches_allowlist(
        source, "15550000001:47@s.whatsapp.net", {"15550000001"}
    ) is True


# ------------------------------------------------------------- full authz path

def _make_runner(platform: Platform, config: GatewayConfig):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._update_prompts = {}
    runner.hooks = SimpleNamespace(dispatch=AsyncMock(return_value=None))
    runner._sessions = {}
    return runner, adapter


def test_email_allowlist_bare_localpart_rejects_foreign_domain(monkeypatch):
    """End to end: EMAIL_ALLOWED_USERS=alice must not authorize alice@evil.example."""
    for key in ("EMAIL_ALLOWED_USERS", "EMAIL_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EMAIL_ALLOWED_USERS", "alice")

    runner, _adapter = _make_runner(
        Platform.EMAIL,
        GatewayConfig(platforms={Platform.EMAIL: PlatformConfig(enabled=True)}),
    )

    assert runner._is_user_authorized(_source(Platform.EMAIL, "alice@evil.example")) is False
    assert runner._is_user_authorized(_source(Platform.EMAIL, "mallory@other.example")) is False


def test_global_allowlist_bare_localpart_rejects_foreign_domain(monkeypatch):
    """The reachable widening: BlueBubbles registers no platform allowlist env,
    so GATEWAY_ALLOWED_USERS is the only gate and the adapter does no sender
    check of its own — an iMessage email handle is matched as-is. A bare
    ``alice`` entry must not admit alice@evil.example."""
    for key in ("GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "alice")

    runner, _adapter = _make_runner(
        Platform.BLUEBUBBLES,
        GatewayConfig(platforms={Platform.BLUEBUBBLES: PlatformConfig(enabled=True)}),
    )

    assert runner._is_user_authorized(
        _source(Platform.BLUEBUBBLES, "alice@evil.example")
    ) is False
    assert runner._is_user_authorized(
        _source(Platform.BLUEBUBBLES, "alice@alice.example")
    ) is False


def test_global_allowlist_qualified_entry_still_authorizes(monkeypatch):
    """Positive control on the same path: a fully-qualified entry still admits
    its exact sender, so the gate is not simply rejecting everything."""
    for key in ("GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "alice@corp.example")

    runner, _adapter = _make_runner(
        Platform.BLUEBUBBLES,
        GatewayConfig(platforms={Platform.BLUEBUBBLES: PlatformConfig(enabled=True)}),
    )

    assert runner._is_user_authorized(
        _source(Platform.BLUEBUBBLES, "alice@corp.example")
    ) is True
    assert runner._is_user_authorized(
        _source(Platform.BLUEBUBBLES, "alice@evil.example")
    ) is False
