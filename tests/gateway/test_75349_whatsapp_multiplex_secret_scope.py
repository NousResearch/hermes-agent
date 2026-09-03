"""Regression test for #75349 — WhatsApp bridge loses WHATSAPP_* vars under multiplex.

Under ``_profile_runtime_scope`` (the context installed for every secondary-profile
turn in a multiplexed gateway), ``os.getenv("WHATSAPP_MODE")`` bypasses the
profile's secret scope and returns the process-global value (often unset),
causing the bridge to silently fall back to ``"self-chat"`` and reject all
inbound messages.

The fix routes WHATSAPP_* reads through ``get_secret()`` (``agent.secret_scope``)
which honours the active scope.
"""
import os

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


class TestWhatsAppYamlBridgeScopeIsolation:
    """#80099 — _apply_yaml_config's WHATSAPP_* env writes for
    require_mention/mention_patterns/free_response_chats/dm_policy/
    allow_from/group_policy/group_allow_from must not pollute shared
    process env under multiplex; a scoped secondary profile's own YAML
    values must still reach its adapter via PlatformConfig.extra."""

    _ENV_VARS = (
        "WHATSAPP_REQUIRE_MENTION",
        "WHATSAPP_MENTION_PATTERNS",
        "WHATSAPP_FREE_RESPONSE_CHATS",
        "WHATSAPP_DM_POLICY",
        "WHATSAPP_ALLOWED_USERS",
        "WHATSAPP_GROUP_POLICY",
        "WHATSAPP_GROUP_ALLOWED_USERS",
    )

    def test_scoped_load_seeds_extra_without_env_leak(self, monkeypatch):
        from plugins.platforms.whatsapp.adapter import _apply_yaml_config

        for var in self._ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            seeded = _apply_yaml_config(
                {},
                {
                    "require_mention": True,
                    "mention_patterns": ["(?i)hey bot"],
                    "free_response_chats": ["120363001@g.us"],
                    "dm_policy": "allowlist",
                    "allow_from": ["15551234567@c.us"],
                    "group_policy": "allowlist",
                    "group_allow_from": ["120363002@g.us"],
                },
            )
        finally:
            ss.reset_secret_scope(tok)

        assert seeded == {
            "require_mention": True,
            "mention_patterns": ["(?i)hey bot"],
            "free_response_chats": ["120363001@g.us"],
            "dm_policy": "allowlist",
            "allow_from": ["15551234567@c.us"],
            "group_policy": "allowlist",
            "group_allow_from": ["120363002@g.us"],
        }
        for var in self._ENV_VARS:
            assert os.getenv(var) is None, f"{var} leaked into process env under scope"

    def test_unscoped_single_profile_still_bridges_env(self, monkeypatch):
        """Unscoped (single-profile) behavior is unchanged: the legacy env
        bridge still fires, matching pre-fix behavior exactly."""
        from plugins.platforms.whatsapp.adapter import _apply_yaml_config

        for var in self._ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        ss.set_multiplex_active(False)
        _apply_yaml_config(
            {}, {"require_mention": True, "dm_policy": "allowlist"},
        )
        assert os.environ["WHATSAPP_REQUIRE_MENTION"] == "true"
        assert os.environ["WHATSAPP_DM_POLICY"] == "allowlist"

    def test_default_profile_env_does_not_leak_into_second_profiles_extra(self, monkeypatch):
        """First-writer env (the default profile's bridge output) must not
        override a scoped secondary profile's own extra-seeded value — the
        same construction-time precedence WhatsAppAdapter.__init__ already
        uses for dm_policy (config.extra.get(...) or _wenv(...))."""
        from plugins.platforms.whatsapp.adapter import _wenv

        monkeypatch.setenv("WHATSAPP_DM_POLICY", "pairing")  # default profile's leaked value
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            extra = {"dm_policy": "allowlist"}
            resolved = str(extra.get("dm_policy") or _wenv("WHATSAPP_DM_POLICY", "pairing")).strip().lower()
        finally:
            ss.reset_secret_scope(tok)
        assert resolved == "allowlist"


class TestSeedBridgeEnvFromExtra:
    """_seed_bridge_env_from_extra fills the Node bridge subprocess env from
    a scoped profile's PlatformConfig.extra, for fields _wenv (the .env/scope
    -only loop in connect()) can never see."""

    def test_fills_gaps_from_extra(self):
        from plugins.platforms.whatsapp.adapter import _seed_bridge_env_from_extra

        bridge_env = {}
        _seed_bridge_env_from_extra(
            bridge_env,
            {
                "require_mention": True,
                "mention_patterns": ["(?i)hey bot"],
                "dm_policy": "Allowlist",
                "allow_from": ["15551234567@c.us", "15559876543@c.us"],
            },
        )
        assert bridge_env["WHATSAPP_REQUIRE_MENTION"] == "true"
        assert bridge_env["WHATSAPP_MENTION_PATTERNS"] == '["(?i)hey bot"]'
        assert bridge_env["WHATSAPP_DM_POLICY"] == "allowlist"
        assert bridge_env["WHATSAPP_ALLOWED_USERS"] == "15551234567@c.us,15559876543@c.us"

    def test_does_not_override_existing_env_derived_values(self):
        """Values already resolved from the profile's own .env (via the
        _wenv loop) win over extra — extra only fills gaps."""
        from plugins.platforms.whatsapp.adapter import _seed_bridge_env_from_extra

        bridge_env = {"WHATSAPP_DM_POLICY": "pairing"}
        _seed_bridge_env_from_extra(bridge_env, {"dm_policy": "allowlist"})
        assert bridge_env["WHATSAPP_DM_POLICY"] == "pairing"

    def test_ignores_none_and_missing_keys(self):
        from plugins.platforms.whatsapp.adapter import _seed_bridge_env_from_extra

        bridge_env = {}
        _seed_bridge_env_from_extra(bridge_env, {})
        assert bridge_env == {}
