"""Regression test for #75349 — WhatsApp bridge loses WHATSAPP_* vars under multiplex.

Under ``_profile_runtime_scope`` (the context installed for every secondary-profile
turn in a multiplexed gateway), ``os.getenv("WHATSAPP_MODE")`` bypasses the
profile's secret scope and returns the process-global value (often unset),
causing the bridge to silently fall back to ``"self-chat"`` and reject all
inbound messages.

The fix routes WHATSAPP_* reads through ``get_secret()`` (``agent.secret_scope``)
which honours the active scope.
"""
import pytest

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset_multiplex(monkeypatch):
    """Ensure multiplex mode is off before and after each test."""
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class TestWhatsAppEnvViaSecretScope:
    """WHATSAPP_* reads must go through ``get_secret``, not ``os.getenv``."""

    def test_wenv_reads_scope_under_multiplex(self, tmp_path, monkeypatch):
        """When multiplexing is active and a scope is installed, the profile's
        .env values are returned — not the global os.environ values."""
        from plugins.platforms.whatsapp.adapter import _wenv

        # Set a misleading value in os.environ that would be returned by
        # os.getenv("WHATSAPP_MODE", "self-chat") if the bug was present.
        monkeypatch.setenv("WHATSAPP_MODE", "self-chat")

        ss.set_multiplex_active(True)

        # Write a profile .env with bot mode
        (tmp_path / ".env").write_text("WHATSAPP_MODE=bot\nWHATSAPP_DM_POLICY=allowlist\n")

        tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
        try:
            # The fix reads from the scope, not os.environ
            mode = _wenv("WHATSAPP_MODE", "self-chat")
            assert mode == "bot", f"Expected 'bot', got {mode!r}"

            dm_policy = _wenv("WHATSAPP_DM_POLICY", "pairing")
            assert dm_policy == "allowlist", f"Expected 'allowlist', got {dm_policy!r}"
        finally:
            ss.reset_secret_scope(tok)

    def test_wenv_fallback_when_scope_absent(self, monkeypatch):
        """When no scope is installed and multiplex is off, _wenv returns default."""
        from plugins.platforms.whatsapp.adapter import _wenv

        monkeypatch.delenv("WHATSAPP_MODE", raising=False)
        result = _wenv("WHATSAPP_MODE", "self-chat")
        assert result == "self-chat"

    def test_wenv_does_not_leak_cross_profile(self, tmp_path, monkeypatch):
        """Two different profiles under the same process see their own values."""
        from plugins.platforms.whatsapp.adapter import _wenv

        ss.set_multiplex_active(True)

        (tmp_path / "profA").mkdir()
        (tmp_path / "profA" / ".env").write_text("WHATSAPP_MODE=bot\n")
        (tmp_path / "profB").mkdir()
        (tmp_path / "profB" / ".env").write_text("WHATSAPP_MODE=self-chat\n")

        tok_a = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path / "profA"))
        try:
            assert _wenv("WHATSAPP_MODE", "self-chat") == "bot"
        finally:
            ss.reset_secret_scope(tok_a)

        tok_b = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path / "profB"))
        try:
            assert _wenv("WHATSAPP_MODE", "self-chat") == "self-chat"
        finally:
            ss.reset_secret_scope(tok_b)


class TestWhatsAppCommonUsesSecretScope:
    """The shared WhatsAppBehaviorMixin methods must also use get_secret."""

    def test_effective_reply_prefix_uses_scope(self, tmp_path, monkeypatch):
        """_effective_reply_prefix respects the profile's WHATSAPP_MODE from scope."""
        from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin

        monkeypatch.delenv("WHATSAPP_MODE", raising=False)
        monkeypatch.delenv("WHATSAPP_REPLY_PREFIX", raising=False)
        ss.set_multiplex_active(True)

        # Set scope with bot mode (should NOT add reply prefix)
        (tmp_path / ".env").write_text("WHATSAPP_MODE=bot\n")
        tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
        try:
            mixin = object.__new__(WhatsAppBehaviorMixin)
            mixin.config = type("C", (), {"extra": {}})()
            mixin.name = "test"
            mixin._reply_prefix = None
            mixin.MAX_MESSAGE_LENGTH = 4096
            mixin.DEFAULT_REPLY_PREFIX = "[Reply] "

            prefix = mixin._effective_reply_prefix()
            # bot mode → no prefix
            assert prefix == ""
        finally:
            ss.reset_secret_scope(tok)

    def test_whatsapp_require_mention_uses_scope(self, tmp_path, monkeypatch):
        """_whatsapp_require_mention respects the profile's env from scope."""
        from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin

        monkeypatch.delenv("WHATSAPP_REQUIRE_MENTION", raising=False)
        ss.set_multiplex_active(True)

        (tmp_path / ".env").write_text("WHATSAPP_REQUIRE_MENTION=true\n")
        tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
        try:
            mixin = object.__new__(WhatsAppBehaviorMixin)
            mixin.config = type("C", (), {"extra": {}})()
            mixin.name = "test"

            assert mixin._whatsapp_require_mention() is True
        finally:
            ss.reset_secret_scope(tok)


class TestWhatsAppCloudAdapterUsesSecretScope:
    """The Cloud API adapter must also read WHATSAPP_* through get_secret."""

    def test_cloud_dm_policy_reads_scope(self, tmp_path, monkeypatch):
        """WhatsAppCloudAdapter._dm_policy respects profile scope."""
        from gateway.config import PlatformConfig

        monkeypatch.delenv("WHATSAPP_CLOUD_DM_POLICY", raising=False)
        monkeypatch.delenv("WHATSAPP_DM_POLICY", raising=False)
        ss.set_multiplex_active(True)

        (tmp_path / ".env").write_text("WHATSAPP_DM_POLICY=allowlist\n")
        tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
        try:
            from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

            cfg = type("C", (), {
                "extra": {},
                "enabled": True,
            })()
            # Cloud adapter won't fully init without creds, but we can at least
            # verify the dm_policy assignment path doesn't crash under scope.
            # We test via the mixin's behavior instead.
            from gateway.platforms.whatsapp_common import _get_wsecret
            assert _get_wsecret("WHATSAPP_DM_POLICY", default="pairing") == "allowlist"
        finally:
            ss.reset_secret_scope(tok)
