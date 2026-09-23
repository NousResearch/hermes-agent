"""Allowlist principal matching must not widen on the '@' localpart (#119446).

A bare allowlist entry (``alice``) authorizes exactly that id.  It must NOT
authorize ``alice@<any domain>`` on platforms whose user_id is email- or
handle-shaped (google_chat, bluebubbles, email).  The WhatsApp bare-phone case
keeps working through the platform's own alias expansion.
"""
import types

from gateway.authz_mixin import Platform, _principal_matches_allowlist


def _source(platform, user_name=""):
    return types.SimpleNamespace(platform=platform, user_name=user_name)


def _enum(*names):
    for n in names:
        v = getattr(Platform, n, None)
        if v is not None:
            return v
    return None


GOOGLE_CHAT = _enum("GOOGLE_CHAT", "GOOGLECHAT")
BLUEBUBBLES = _enum("BLUEBUBBLES")
EMAIL = _enum("EMAIL")


def test_bare_entry_rejects_foreign_domain_localpart():
    src = _source(None)
    assert not _principal_matches_allowlist(src, "alice@evil.example", {"alice"})


def test_bare_entry_matches_exact_id():
    src = _source(None)
    assert _principal_matches_allowlist(src, "alice", {"alice"})


def test_qualified_entry_matches_exact_address():
    src = _source(None)
    assert _principal_matches_allowlist(src, "alice@example.com", {"alice@example.com"})


def test_qualified_entry_rejects_other_domain():
    src = _source(None)
    assert not _principal_matches_allowlist(src, "alice@evil.example", {"alice@example.com"})


def test_email_shaped_platforms_reject_foreign_domain():
    for platform in (GOOGLE_CHAT, BLUEBUBBLES, EMAIL):
        src = _source(platform)
        assert not _principal_matches_allowlist(src, "alice@evil.example", {"alice"}), platform


def test_whatsapp_jid_matches_bare_phone_entry():
    src = _source(Platform.WHATSAPP)
    assert _principal_matches_allowlist(src, "15551234567@s.whatsapp.net", {"15551234567"})


def test_whatsapp_device_suffix_jid_matches_bare_phone_entry():
    src = _source(Platform.WHATSAPP)
    assert _principal_matches_allowlist(src, "15551234567:12@s.whatsapp.net", {"15551234567"})


def test_whatsapp_cloud_jid_matches_bare_phone_entry():
    src = _source(Platform.WHATSAPP_CLOUD)
    assert _principal_matches_allowlist(src, "15551234567@s.whatsapp.net", {"15551234567"})


def test_wildcard_and_exact_still_work():
    src = _source(None)
    assert _principal_matches_allowlist(src, "alice", {"alice", "bob"})
    assert not _principal_matches_allowlist(src, "carol", {"alice", "bob"})
