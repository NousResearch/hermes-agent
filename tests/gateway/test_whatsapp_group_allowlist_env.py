"""Regression test for #72529 — the WhatsApp adapter never read WHATSAPP_GROUP_ALLOWED_USERS.

The Node bridge receives WHATSAPP_GROUP_ALLOWED_USERS via _BRIDGE_PASSTHROUGH_ENV, and the
YAML bridge maps config.yaml ``whatsapp.group_allow_from`` onto the same env carrier — but
``WhatsAppAdapter.__init__`` seeded ``_group_allow_from`` from ``extra`` alone. An install
that configured the group allowlist only through the documented env var (or relied on the
YAML bridge to populate it) silently ran group gating with an empty allowlist: with
``group_policy: allowlist`` every group message was rejected, and with ``group_policy: open``
nothing changed — the triage on #72529 flagged exactly this half as "no PR yet".

The DM path already had the right shape (``_select_dm_allowlist`` reads config by key
*presence*, then the scoped env). This test pins the group path to the same precedence:

1. ``extra["group_allow_from"]`` (explicit list or CSV) wins, even when empty-ish forms differ;
2. the legacy ``groupAllowFrom`` key follows;
3. with neither config key present, WHATSAPP_GROUP_ALLOW_FROM / WHATSAPP_GROUP_ALLOWED_USERS
   (profile-scoped, multiplex-safe) seed the allowlist;
4. config wins over env — an env value must never broaden a config-seeded list.
"""
from unittest.mock import patch

import pytest

from gateway.config import Platform, PlatformConfig


def _adapter_with_extra(extra):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=extra))


class TestGroupAllowlistEnvFallback:
    """Env-only group allowlists must reach ``_group_allow_from``."""

    def test_env_only_group_allowlist_is_read(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "120363001234567890@g.us, 86316009876@s.whatsapp.net")
        monkeypatch.delenv("WHATSAPP_GROUP_ALLOW_FROM", raising=False)
        adapter = _adapter_with_extra({})
        assert adapter._group_allow_from == {"120363001234567890@g.us", "86316009876@s.whatsapp.net"}

    def test_group_allow_from_env_alias(self, monkeypatch):
        """WHATSAPP_GROUP_ALLOW_FROM (the allow_from-style spelling) is honoured too."""
        monkeypatch.setenv("WHATSAPP_GROUP_ALLOW_FROM", "86316009876@s.whatsapp.net")
        monkeypatch.delenv("WHATSAPP_GROUP_ALLOWED_USERS", raising=False)
        adapter = _adapter_with_extra({})
        assert adapter._group_allow_from == {"86316009876@s.whatsapp.net"}

    def test_config_group_allow_from_still_wins(self, monkeypatch):
        """An explicit config list stays authoritative — the env value must not broaden it."""
        monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "1111111111@g.us")
        adapter = _adapter_with_extra({"group_allow_from": ["120363001234567890@g.us"]})
        assert adapter._group_allow_from == {"120363001234567890@g.us"}

    def test_legacy_group_allow_from_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "1111111111@g.us")
        adapter = _adapter_with_extra({"groupAllowFrom": ["86316009876@s.whatsapp.net"]})
        assert adapter._group_allow_from == {"86316009876@s.whatsapp.net"}

    def test_empty_env_and_no_config_means_empty_allowlist(self, monkeypatch):
        monkeypatch.delenv("WHATSAPP_GROUP_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("WHATSAPP_GROUP_ALLOW_FROM", raising=False)
        adapter = _adapter_with_extra({})
        assert adapter._group_allow_from == set()

    def test_scoped_env_read_through_secret_scope(self, tmp_path, monkeypatch):
        """The env read is profile-scoped (multiplex-safe) like every other WHATSAPP_* read."""
        from agent import secret_scope as ss

        (tmp_path / ".env").write_text("WHATSAPP_GROUP_ALLOWED_USERS=86316009876@s.whatsapp.net\n")
        monkeypatch.delenv("WHATSAPP_GROUP_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("WHATSAPP_GROUP_ALLOW_FROM", raising=False)

        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
        try:
            adapter = _adapter_with_extra({})
            assert adapter._group_allow_from == {"86316009876@s.whatsapp.net"}
        finally:
            ss.reset_secret_scope(tok)
            ss.set_multiplex_active(False)


class TestGroupGatingUsesEnvSeededAllowlist:
    """#72529's symptom: authorized group senders rejected because the allowlist was empty."""

    def test_allowlisted_group_chat_passes_intake(self, monkeypatch):
        """group_policy=allowlist + env-only allowlist → the group chat is admitted."""
        monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "120363001234567890@g.us")
        monkeypatch.setenv("WHATSAPP_GROUP_POLICY", "allowlist")
        adapter = _adapter_with_extra({})
        assert adapter._group_policy == "allowlist"
        assert adapter._is_group_allowed("120363001234567890@g.us") is True
        assert adapter._is_group_allowed("99999999999@g.us") is False

    def test_open_group_policy_unaffected(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_GROUP_POLICY", "open")
        monkeypatch.delenv("WHATSAPP_GROUP_ALLOWED_USERS", raising=False)
        adapter = _adapter_with_extra({})
        assert adapter._is_group_allowed("120363001234567890@g.us") is True
