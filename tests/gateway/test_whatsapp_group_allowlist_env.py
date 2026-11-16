"""Regression for #72529 — the WhatsApp adapter never read WHATSAPP_GROUP_ALLOWED_USERS.

The Node bridge received the env carrier via _BRIDGE_PASSTHROUGH_ENV, but ``WhatsAppAdapter.__init__``
seeded ``_group_allow_from`` from ``extra`` alone, so an env-only install ran ``group_policy: allowlist``
against an empty allowlist and rejected every group. Precedence mirrors the DM path: config key presence
wins, then the profile-scoped env.
"""

from gateway.config import PlatformConfig


def _adapter_with_extra(extra):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=extra))


class TestGroupAllowlistEnv:
    def test_env_only_group_allowlist_gates_intake_through_the_secret_scope(self, tmp_path, monkeypatch):
        """No config key: the profile-scoped WHATSAPP_GROUP_ALLOWED_USERS seeds the allowlist ``_is_group_allowed``
        reads (multiplex-safe like every other WHATSAPP_* read)."""
        from agent import secret_scope as ss

        (tmp_path / ".env").write_text("WHATSAPP_GROUP_ALLOWED_USERS=120363001234567890@g.us\nWHATSAPP_GROUP_POLICY=allowlist\n", encoding="utf-8")
        monkeypatch.delenv("WHATSAPP_GROUP_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("WHATSAPP_GROUP_ALLOW_FROM", raising=False)
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
        try:
            adapter = _adapter_with_extra({})
        finally:
            ss.reset_secret_scope(tok)
            ss.set_multiplex_active(False)
        assert adapter._group_allow_from == {"120363001234567890@g.us"}
        assert adapter._is_group_allowed("120363001234567890@g.us") is True
        assert adapter._is_group_allowed("99999999999@g.us") is False

    def test_config_group_allow_from_wins_over_env(self, monkeypatch):
        """An explicit config list (either spelling) stays authoritative — env must not broaden it."""
        monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "1111111111@g.us")
        assert _adapter_with_extra({"group_allow_from": ["120363001234567890@g.us"]})._group_allow_from == {"120363001234567890@g.us"}
        assert _adapter_with_extra({"groupAllowFrom": []})._group_allow_from == set()
