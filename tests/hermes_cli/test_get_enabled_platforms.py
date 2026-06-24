"""Regression tests for `hermes_cli.tools_config._get_enabled_platforms`.

Covers #51771: the `hermes tools` curses UI does not list the `cron`
platform, so users cannot configure per-platform toolsets for scheduled
jobs from the documented setup wizard.

The function gates platforms by env-var presence for the messaging
adapters (telegram/discord/etc.). `cron` is a non-messaging platform
that needs no token, but the original implementation had no branch for
it at all — so even when no other platform is configured, the only
entry in the returned list is `cli`, and the cron option never appears.
The platform *is* registered in `hermes_cli.platforms.PLATFORMS` and the
docs (website/docs/user-guide/features/cron.md) explicitly tell users to
configure it via `hermes tools`, so the omission is a bug, not a design
choice.
"""

from __future__ import annotations

import pytest

from hermes_cli.tools_config import _get_enabled_platforms


# All env vars that the function consults. Any of these being set in the
# host environment would otherwise leak into the test (we cleared them in
# the previous test file at the module level, but be defensive per-test).
_PLATFORM_TOKEN_ENVS = (
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "SLACK_BOT_TOKEN",
    "WHATSAPP_ENABLED",
    "QQ_APP_ID",
)


@pytest.fixture(autouse=True)
def _clear_platform_env(monkeypatch):
    """Wipe platform-related env vars so tests are deterministic."""
    for env_var in _PLATFORM_TOKEN_ENVS:
        monkeypatch.delenv(env_var, raising=False)


class TestGetEnabledPlatforms:
    """The headline #51771 regression: `cron` must be in the list."""

    def test_cron_always_in_enabled_platforms(self):
        """The bug: with no platform tokens set, the returned list is
        just `['cli']` and `cron` is missing. The fix appends `cron`
        unconditionally because it is a local-only platform (no token
        gating) and the docs tell users to configure it via `hermes tools`."""
        enabled = _get_enabled_platforms()
        assert "cli" in enabled
        assert "cron" in enabled, (
            "cron platform must always be configurable — see "
            "https://hermes-agent.nousresearch.com/docs/user-guide/features/cron"
        )

    def test_cron_present_alongside_messaging_platforms(self, monkeypatch):
        """Adding a messaging platform token must NOT displace cron from
        the list — the user's cron toolset config is independent of
        their chat platforms."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token-for-test")
        enabled = _get_enabled_platforms()
        assert "cli" in enabled
        assert "telegram" in enabled
        assert "cron" in enabled

    def test_minimal_install_is_cli_plus_cron(self):
        """A fresh install with no platform tokens configured should
        show exactly the two always-available platforms: cli (the
        local REPL) and cron (scheduled jobs). This is the contract
        the docs promise for first-time setup."""
        enabled = _get_enabled_platforms()
        # Both are unconditionally available, both must be present
        assert {"cli", "cron"}.issubset(set(enabled))
        # No surprise messaging platforms on a clean install
        for p in ("telegram", "discord", "slack", "whatsapp", "qqbot"):
            assert p not in enabled, f"{p} should not be in enabled without a token"

    def test_messaging_platforms_gated_by_env(self, monkeypatch):
        """Each messaging platform only appears when its token is set."""
        cases = {
            "TELEGRAM_BOT_TOKEN": "telegram",
            "DISCORD_BOT_TOKEN": "discord",
            "SLACK_BOT_TOKEN": "slack",
            "WHATSAPP_ENABLED": "whatsapp",
            "QQ_APP_ID": "qqbot",
        }
        for env_var, platform in cases.items():
            # Clear all other platform tokens
            for other in cases:
                if other != env_var:
                    monkeypatch.delenv(other, raising=False)
            monkeypatch.delenv(env_var, raising=False)
            assert platform not in _get_enabled_platforms(), (
                f"{platform} should NOT appear without {env_var}"
            )
            monkeypatch.setenv(env_var, "fake-token-for-test")
            assert platform in _get_enabled_platforms(), (
                f"{platform} should appear when {env_var} is set"
            )
            monkeypatch.delenv(env_var, raising=False)


class TestCronListedAlongsideOtherPlatforms:
    """The user-visible contract from the docs and the issue report."""

    def test_cron_does_not_depend_on_any_token(self, monkeypatch):
        """Even with ALL messaging tokens set, cron must be in the list
        — it is a non-messaging platform and needs no credentials."""
        for env_var in _PLATFORM_TOKEN_ENVS:
            monkeypatch.setenv(env_var, "set")
        enabled = _get_enabled_platforms()
        assert "cron" in enabled

    def test_cron_in_enabled_platforms_even_when_first_in_alphabet(self):
        """The list is human-ordered (cli first, then messaging), not
        alphabetised. `cron` should appear where it makes sense in the
        ordering the UI uses — the test only requires it's present,
        which is the contract from the docs and the bug report."""
        enabled = _get_enabled_platforms()
        assert "cron" in enabled
