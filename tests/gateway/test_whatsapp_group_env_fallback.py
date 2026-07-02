"""Verify that the WhatsApp group allowlist reads env vars as fallback.

Regression test for #56767: after the built-in → plugin migration,
``WHATSAPP_GROUP_ALLOW_FROM`` / ``WHATSAPP_GROUP_ALLOWED_USERS`` were no
longer read, causing group messages to be silently dropped when the user
configured the allowlist via ``.env`` instead of ``config.yaml``.
"""

from gateway.config import Platform, PlatformConfig

from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


# --- Env-only fallback (the core regression) ---


def test_group_allowlist_reads_whatsapp_group_allow_from_env(monkeypatch):
    """Env-only setup: ``WHATSAPP_GROUP_ALLOW_FROM`` must populate the
    group allowlist when ``config.yaml`` has no ``group_allow_from``."""
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOW_FROM", "12036300111@g.us, 12036300222@g.us")
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter._group_allow_from == {"12036300111@g.us", "12036300222@g.us"}


def test_group_allowlist_reads_whatsapp_group_allowed_users_env(monkeypatch):
    """Env-only setup: ``WHATSAPP_GROUP_ALLOWED_USERS`` (setup-wizard name)
    must also populate the group allowlist."""
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "12036300333@g.us")
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter._group_allow_from == {"12036300333@g.us"}


# --- Config takes precedence over env ---


def test_group_allowlist_config_extra_wins_over_env(monkeypatch):
    """Explicit ``group_allow_from`` in config must not be widened by a
    stale env var."""
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOW_FROM", "12036300999@g.us")
    adapter = WhatsAppAdapter(PlatformConfig(
        enabled=True, extra={"group_allow_from": ["12036300444@g.us"]},
    ))
    assert adapter._group_allow_from == {"12036300444@g.us"}


def test_group_allowlist_camelcase_config_wins_over_env(monkeypatch):
    """CamelCase ``groupAllowFrom`` in config extra also takes precedence."""
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "12036300999@g.us")
    adapter = WhatsAppAdapter(PlatformConfig(
        enabled=True, extra={"groupAllowFrom": ["12036300555@g.us"]},
    ))
    assert adapter._group_allow_from == {"12036300555@g.us"}


# --- Env var priority: GROUP_ALLOW_FROM checked first ---


def test_group_allow_from_checked_before_group_allowed_users(monkeypatch):
    """When both env vars are set, ``WHATSAPP_GROUP_ALLOW_FROM`` wins
    (checked first in the ``or`` chain)."""
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOW_FROM", "12036300666@g.us")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "12036300777@g.us")
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter._group_allow_from == {"12036300666@g.us"}


# --- Empty / missing env var ---


def test_group_allowlist_empty_when_no_config_no_env(monkeypatch):
    """No config and no env → empty allowlist (default)."""
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter._group_allow_from == set()


# --- End-to-end: env-only allowlist gates group messages ---


def test_env_only_allowlist_accepts_listed_group(monkeypatch):
    """Env-only ``group_policy=allowlist`` + ``group_allow_from`` lets a
    listed group through via ``_is_group_allowed``."""
    monkeypatch.setenv("WHATSAPP_GROUP_POLICY", "allowlist")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOW_FROM", "120363001234567890@g.us")
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter._group_allow_from == {"120363001234567890@g.us"}
    assert adapter._is_group_allowed("120363001234567890@g.us") is True
    assert adapter._is_group_allowed("999999999999@g.us") is False
