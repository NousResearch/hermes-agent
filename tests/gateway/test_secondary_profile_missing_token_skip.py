"""Secondary multiplexed profiles must skip token platforms with no credential.

Complement to #64674, which fixed the DEFAULT profile connecting a token
platform whose credential lived only in a secondary profile. The mirror case was
left open: a SECONDARY profile can have discord/telegram enabled — the plugin
registry auto-enables them for any profile listing those plugins — while the bot
credential lives only in the default profile's .env.

That is the normal shape for a profile used purely as a ``gateway.profile_routes``
target: it supplies the model/tools/memory/persona, and inbound arrives on the
default profile's single connection. ``_start_one_profile_adapters`` used to build
an adapter anyway, which logged a bare "No bot token configured" plus
"✗ <platform> failed to connect (profile: <name>)" on every gateway start and
could never succeed.

Unlike the sibling suite, these tests drive the real
``GatewayRunner._start_one_profile_adapters`` rather than re-implementing its
loop, so the guard cannot regress without failing here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig


def _make_runner(default_multiplex: bool = True, connect_succeeds: bool = True):
    """A runner stubbed richly enough to run the loop body to completion.

    Deliberately complete: if the skip guard regresses, the loop must reach
    _create_adapter and fail this suite's ``created == []`` assertion, rather
    than dying early on an incidental missing attribute.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=default_multiplex)
    runner._profile_adapters = {}
    runner.pairing_stores = {}
    runner.session_store = MagicMock()
    runner._busy_text_mode = "off"
    runner._adapter_credential_fingerprint = lambda a: None
    runner._configure_profile_adapter = MagicMock()

    async def _connect(adapter, platform):
        return connect_succeeds

    runner._connect_initial_adapter_with_timeout = _connect

    async def _disconnect(adapter, platform):
        return None

    runner._safe_adapter_disconnect = _disconnect
    return runner


def _patch_profile_cfg(monkeypatch, profile_cfg: GatewayConfig):
    """Make _start_one_profile_adapters see ``profile_cfg`` as the profile config."""
    import gateway.config as cfg_mod
    from gateway import run as run_mod

    monkeypatch.setattr(cfg_mod, "load_gateway_config", lambda: profile_cfg)

    class _NullScope:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(run_mod, "_profile_runtime_scope", lambda home: _NullScope())


class TestSecondaryProfileSkipsMissingCredential:
    @pytest.mark.asyncio
    async def test_skips_platform_with_no_token(self, monkeypatch, caplog):
        profile_cfg = GatewayConfig(multiplex_profiles=True)
        # token=None is the real shape: the key is absent from the profile .env,
        # not present-but-empty. Both must be skipped.
        profile_cfg.platforms[Platform.DISCORD] = PlatformConfig(enabled=True, token=None)
        profile_cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="")
        _patch_profile_cfg(monkeypatch, profile_cfg)

        # connect_succeeds=False reproduces the real adapter behaviour without a
        # token ("No bot token configured" → connect returns False), so a
        # regression surfaces as both a stray adapter and a "failed to connect".
        runner = _make_runner(connect_succeeds=False)
        created: list = []
        runner._create_adapter = lambda p, pc: created.append(p) or MagicMock()

        with caplog.at_level(logging.INFO, logger="gateway.run"):
            connected = await runner._start_one_profile_adapters(
                "mentor", Path("/nonexistent"), {}
            )

        assert connected == 0
        assert created == [], "no adapter may be built without a credential"
        assert runner._profile_adapters["mentor"] == {}
        text = caplog.text
        assert "Skipping discord on profile 'mentor'" in text
        assert "Skipping telegram on profile 'mentor'" in text
        # The failure this replaces must not appear.
        assert "failed to connect" not in text

    @pytest.mark.asyncio
    async def test_connects_platform_that_has_its_own_token(self, monkeypatch):
        profile_cfg = GatewayConfig(multiplex_profiles=True)
        profile_cfg.platforms[Platform.DISCORD] = PlatformConfig(
            enabled=True, token="secondary-own-bot-token"
        )
        _patch_profile_cfg(monkeypatch, profile_cfg)

        runner = _make_runner()
        adapter = MagicMock()
        runner._create_adapter = lambda p, pc: adapter
        # A distinct fingerprint: this profile owns its own bot, so the
        # same-token conflict detector must not claim it away.
        runner._adapter_credential_fingerprint = lambda a: "fp-secondary"

        connected = await runner._start_one_profile_adapters(
            "mentor", Path("/nonexistent"), {}
        )

        assert connected == 1
        assert runner._profile_adapters["mentor"][Platform.DISCORD] is adapter

    @pytest.mark.asyncio
    async def test_api_key_alone_still_counts_as_a_credential(self, monkeypatch):
        """Some adapters take api_key as the primary credential — don't skip those."""
        profile_cfg = GatewayConfig(multiplex_profiles=True)
        profile_cfg.platforms[Platform.DISCORD] = PlatformConfig(
            enabled=True, token=None, api_key="key-only-credential"
        )
        _patch_profile_cfg(monkeypatch, profile_cfg)

        runner = _make_runner()
        created: list = []
        runner._create_adapter = lambda p, pc: created.append(p) or MagicMock()

        connected = await runner._start_one_profile_adapters(
            "mentor", Path("/nonexistent"), {}
        )

        assert created == [Platform.DISCORD]
        assert connected == 1

    @pytest.mark.asyncio
    async def test_default_profile_token_in_process_env_is_not_borrowed(
        self, monkeypatch, caplog
    ):
        """The guard must not accept a credential that is only in ``os.environ``.

        This loop runs outside ``_profile_runtime_scope``, and that scope does
        not mutate ``os.environ`` — so a token sitting in the process env is the
        DEFAULT profile's, not this profile's. Borrowing it would start a second
        gateway session on the default profile's bot identity, with inbound
        landing on whichever connection won the race: a quieter, worse version
        of the bug this suite exists to prevent.

        Cross-PR guard: #68746 adds an ``os.environ`` fallback (and an in-place
        ``platform_config.token`` backfill) to the shared
        ``_platform_has_bot_credential``. That is correct for the paths acting
        on the default profile's behalf and wrong here, which is why this call
        site uses ``_profile_config_has_bot_credential``. If anyone repoints it
        at the shared helper, this test fails.
        """
        from gateway.config import PLATFORM_TOKEN_ENV_NAMES

        monkeypatch.setenv(PLATFORM_TOKEN_ENV_NAMES[Platform.DISCORD], "default-profile-bot-token")

        profile_cfg = GatewayConfig(multiplex_profiles=True)
        discord_cfg = PlatformConfig(enabled=True, token=None)
        profile_cfg.platforms[Platform.DISCORD] = discord_cfg
        _patch_profile_cfg(monkeypatch, profile_cfg)

        runner = _make_runner(connect_succeeds=False)
        created: list = []
        runner._create_adapter = lambda p, pc: created.append(p) or MagicMock()

        with caplog.at_level(logging.INFO, logger="gateway.run"):
            connected = await runner._start_one_profile_adapters(
                "mentor", Path("/nonexistent"), {}
            )

        assert created == [], "the default profile's env token is not this profile's"
        assert connected == 0
        assert "Skipping discord on profile 'mentor'" in caplog.text
        # A backfilling helper would leave the borrowed token on a config object
        # we go on to keep in _profile_adapters.
        assert discord_cfg.token is None, "guard must not write env tokens onto the config"

    @pytest.mark.asyncio
    async def test_non_token_platform_is_unaffected(self, monkeypatch):
        """Platforms outside PLATFORM_TOKEN_ENV_NAMES must never be skipped here."""
        from gateway.config import PLATFORM_TOKEN_ENV_NAMES

        assert Platform.SIGNAL not in PLATFORM_TOKEN_ENV_NAMES

        profile_cfg = GatewayConfig(multiplex_profiles=True)
        profile_cfg.platforms[Platform.SIGNAL] = PlatformConfig(enabled=True, token=None)
        _patch_profile_cfg(monkeypatch, profile_cfg)

        runner = _make_runner()
        created: list = []
        runner._create_adapter = lambda p, pc: created.append(p) or MagicMock()

        connected = await runner._start_one_profile_adapters(
            "mentor", Path("/nonexistent"), {}
        )

        assert created == [Platform.SIGNAL], "signal authenticates via its own session"
        assert connected == 1


class TestProfileConfigHasBotCredential:
    """Unit-level contract for the profile-scoped predicate.

    Kept separate from the shared ``_platform_has_bot_credential`` because the
    two answer different questions: that one may consult the process env on the
    default profile's behalf, this one must never do so.
    """

    def test_reads_token_from_config(self):
        from gateway.run import _profile_config_has_bot_credential

        assert _profile_config_has_bot_credential(
            Platform.DISCORD, PlatformConfig(enabled=True, token="own-token")
        ) is True

    def test_api_key_counts_as_a_credential(self):
        from gateway.run import _profile_config_has_bot_credential

        assert _profile_config_has_bot_credential(
            Platform.DISCORD, PlatformConfig(enabled=True, token=None, api_key="k")
        ) is True

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_missing_credential_is_false(self, empty):
        from gateway.run import _profile_config_has_bot_credential

        assert _profile_config_has_bot_credential(
            Platform.DISCORD, PlatformConfig(enabled=True, token=empty)
        ) is False

    def test_non_token_platform_always_true(self):
        from gateway.run import _profile_config_has_bot_credential

        assert _profile_config_has_bot_credential(
            Platform.SIGNAL, PlatformConfig(enabled=True, token=None)
        ) is True

    def test_ignores_process_env_and_does_not_mutate(self, monkeypatch):
        """The whole reason this predicate exists — see the call site in
        ``_start_one_profile_adapters``."""
        from gateway.config import PLATFORM_TOKEN_ENV_NAMES
        from gateway.run import _profile_config_has_bot_credential

        monkeypatch.setenv(PLATFORM_TOKEN_ENV_NAMES[Platform.DISCORD], "default-profile-bot-token")
        cfg = PlatformConfig(enabled=True, token=None)

        assert _profile_config_has_bot_credential(Platform.DISCORD, cfg) is False
        assert cfg.token is None
