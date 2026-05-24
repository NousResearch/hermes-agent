"""Registry-driven plugin platform enablement gating (issue #31116).

The bug: ``gateway/config.py`` auto-enabled every plugin platform whose
``check_fn`` returned True. For Discord, ``check_fn`` lazy-installs the
``discord.py`` SDK and *always* returns True afterwards. So a fresh
gateway start would enable the Discord adapter even though the user
never configured ``DISCORD_BOT_TOKEN`` or picked Discord in the setup
wizard, producing repeating ``[Discord] No bot token configured``
errors and 60s reconnect spam.

The fix: prefer ``entry.is_connected(synthetic_cfg)`` (which checks
actual user-intent — env vars / config.yaml) over ``entry.check_fn``
(which only checks whether deps are installed). Plugins that haven't
defined ``is_connected`` keep the old behavior.
"""

from __future__ import annotations

import importlib
import os

import pytest


# ---------------------------------------------------------------------------
# Discord-specific: the original repro from the issue.
# ---------------------------------------------------------------------------


class TestDiscordNotEnabledWithoutToken:
    """User picked Matrix only — Discord must stay out of config.platforms."""

    def _clean_env(self, monkeypatch):
        for v in (
            "DISCORD_BOT_TOKEN",
            "DISCORD_HOME_CHANNEL",
            "DISCORD_REPLY_TO_MODE",
        ):
            monkeypatch.delenv(v, raising=False)

    def test_no_discord_token_means_no_discord_platform(
        self, monkeypatch, tmp_path
    ):
        """Pre-fix: this asserted Discord WAS in cfg.platforms (the bug).

        Post-fix: the registry-driven loop consults ``is_connected``, which
        for Discord requires ``DISCORD_BOT_TOKEN``. With no token set,
        Discord must not show up in the loaded config.
        """
        self._clean_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        from gateway.config import Platform, load_gateway_config

        cfg = load_gateway_config()
        assert Platform("discord") not in cfg.platforms, (
            "Discord platform was auto-enabled despite no DISCORD_BOT_TOKEN. "
            "The registry-driven enable loop is gating on check_fn (deps "
            "available) instead of is_connected (user actually configured). "
            "Issue #31116."
        )

    def test_discord_token_does_enable_discord_platform(
        self, monkeypatch, tmp_path
    ):
        """The fallback path: when the user *has* set the token, Discord
        must still be enabled. _apply_env_overrides handles this for
        built-in platforms; this test pins that the Discord plugin's own
        env-var handling continues to work alongside the new is_connected
        gate (the env-var code at gateway/config.py:_apply_env_overrides
        runs irrespective of the registry loop).
        """
        self._clean_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake-token-for-test")

        from gateway.config import Platform, load_gateway_config

        cfg = load_gateway_config()
        platform = Platform("discord")
        assert platform in cfg.platforms
        assert cfg.platforms[platform].enabled is True


# ---------------------------------------------------------------------------
# Behavior matrix: is_connected vs check_fn precedence.
# ---------------------------------------------------------------------------


class _FakeRegistry:
    """Stub platform_registry whose plugin_entries() returns a fixture set."""

    def __init__(self, entries):
        self._entries = entries

    def plugin_entries(self):
        return list(self._entries)

    def all_entries(self):  # pragma: no cover - exercised by other tests
        return list(self._entries)


class _FakeEntry:
    """Subset of PlatformEntry the gating loop reads from."""

    def __init__(
        self,
        name,
        *,
        check_fn,
        is_connected=None,
        env_enablement_fn=None,
        apply_yaml_config_fn=None,
    ):
        self.name = name
        self.check_fn = check_fn
        self.is_connected = is_connected
        self.env_enablement_fn = env_enablement_fn
        self.apply_yaml_config_fn = apply_yaml_config_fn


@pytest.fixture
def make_fake_platform_class(monkeypatch):
    """Patch ``gateway.config.Platform`` so unknown plugin names resolve."""

    from gateway.config import Platform as _RealPlatform

    def _make(extra_names):
        class _PatchedPlatform(_RealPlatform):
            @classmethod
            def _missing_(cls, value):
                if value in extra_names:
                    # Reuse the existing _missing_ on the real enum which
                    # creates dynamic members.
                    return _RealPlatform._missing_(value)
                return _RealPlatform._missing_(value)

        return _PatchedPlatform

    return _make


class TestRegistryDrivenEnableGating:
    """Black-box regression: plug a synthetic registry and watch which
    platforms end up enabled after _apply_env_overrides."""

    def _clean_discord_env(self, monkeypatch):
        for v in ("DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL"):
            monkeypatch.delenv(v, raising=False)

    def test_is_connected_false_skips_enable_even_if_check_fn_true(
        self, monkeypatch, tmp_path
    ):
        """The exact discord scenario: deps installed, no user config."""
        self._clean_discord_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        captured = {"check_fn_calls": 0, "is_connected_calls": 0}

        def _check_fn():
            captured["check_fn_calls"] += 1
            return True  # deps available

        def _is_connected(cfg):
            captured["is_connected_calls"] += 1
            return False  # no user config

        fake = _FakeRegistry([
            _FakeEntry(
                "discord",
                check_fn=_check_fn,
                is_connected=_is_connected,
            )
        ])
        import gateway.platform_registry as _pr_mod

        monkeypatch.setattr(_pr_mod, "platform_registry", fake)

        from gateway.config import Platform, load_gateway_config

        cfg = load_gateway_config()

        assert Platform("discord") not in cfg.platforms
        assert captured["is_connected_calls"] >= 1, (
            "is_connected must be consulted when defined"
        )

    def test_is_connected_true_enables_platform(
        self, monkeypatch, tmp_path
    ):
        self._clean_discord_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        fake = _FakeRegistry([
            _FakeEntry(
                "discord",
                check_fn=lambda: True,
                is_connected=lambda cfg: True,
            )
        ])
        import gateway.platform_registry as _pr_mod

        monkeypatch.setattr(_pr_mod, "platform_registry", fake)

        from gateway.config import Platform, load_gateway_config

        cfg = load_gateway_config()
        assert Platform("discord") in cfg.platforms
        assert cfg.platforms[Platform("discord")].enabled is True

    def test_no_is_connected_falls_back_to_check_fn(
        self, monkeypatch, tmp_path
    ):
        """Plugins predating the is_connected hook keep the old behavior:
        ``check_fn`` decides. This pins the back-compat path."""
        self._clean_discord_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # The "telegram" name happens to be a built-in Platform value; using
        # it lets ``Platform(name)`` succeed without any extra plumbing.
        fake = _FakeRegistry([
            _FakeEntry(
                "telegram",
                check_fn=lambda: True,
                is_connected=None,
            )
        ])
        import gateway.platform_registry as _pr_mod

        monkeypatch.setattr(_pr_mod, "platform_registry", fake)

        from gateway.config import Platform, load_gateway_config

        cfg = load_gateway_config()
        assert Platform("telegram") in cfg.platforms
        assert cfg.platforms[Platform("telegram")].enabled is True

    def test_is_connected_raising_does_not_enable(
        self, monkeypatch, tmp_path
    ):
        """Defensive: a raising is_connected must be treated as "not
        connected" rather than crashing the gateway boot."""
        self._clean_discord_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        def _boom(cfg):
            raise RuntimeError("synthetic plugin failure")

        fake = _FakeRegistry([
            _FakeEntry(
                "discord",
                check_fn=lambda: True,
                is_connected=_boom,
            )
        ])
        import gateway.platform_registry as _pr_mod

        monkeypatch.setattr(_pr_mod, "platform_registry", fake)

        from gateway.config import Platform, load_gateway_config

        cfg = load_gateway_config()  # must not raise
        assert Platform("discord") not in cfg.platforms
