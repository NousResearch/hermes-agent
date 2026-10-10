"""``channel_overrides`` must follow the transport-owning profile under multiplexing (#136198).

A Telegram DM's chat id is the user's id, shared by every profile's bot, so resolving
``channel_overrides`` from the launch profile's boot config let the launch profile's DM
override steer other bots' turns and silently ignore the routed profile's own ``model``.
"""

from dataclasses import replace
from pathlib import Path

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_identity import RoutingIdentity


def _runner(
    config: GatewayConfig, profile_configs: dict | None = None
) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner._profile_configs = profile_configs if profile_configs is not None else {}
    runner._primary_profile_name = "default"
    return runner


def _launch_config() -> GatewayConfig:
    return GatewayConfig(
        multiplex_profiles=True,
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "100": ChannelOverride(
                        model="launch/override-model", provider="launch"
                    ),
                },
            ),
        },
    )


def _source(profile: str | None = "work") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="100", user_id="100", profile=profile
    )


class TestChannelOverrideForSource:
    def test_launch_profile_override_never_reaches_a_secondary_profile_bot(self):
        # The reported bug: same user id DMs another bot, the launch profile's override fires.
        runner = _runner(_launch_config())
        assert (
            runner._channel_override_for_source(_source(), Platform.TELEGRAM, "100")
            is None
        )

    def test_secondary_profile_owns_its_chat_override(self):
        secondary = GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={"100": ChannelOverride(model="work/model")},
                ),
            },
        )
        runner = _runner(_launch_config(), profile_configs={"work": secondary})
        override = runner._channel_override_for_source(
            _source(), Platform.TELEGRAM, "100"
        )
        assert override is not None
        assert override.model == "work/model"

    def test_uncached_serving_profile_resolves_to_no_override(self):
        # Fail closed: an owner whose config never loaded gets nothing, never the launch profile's.
        runner = _runner(_launch_config(), profile_configs={})
        assert (
            runner._channel_override_for_source(_source(), Platform.TELEGRAM, "100")
            is None
        )

    def test_primary_profile_sources_still_read_the_bound_config(self):
        runner = _runner(_launch_config())
        override = runner._channel_override_for_source(
            _source(profile="default"), Platform.TELEGRAM, "100"
        )
        assert override is not None
        assert override.model == "launch/override-model"

    def test_pinned_identity_wins_over_the_source_profile_attribute(self):
        # resolve_identity pins the transport owner on the source; that provenance outranks the
        # stale routing result a reused source may still carry.
        secondary = GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={"100": ChannelOverride(model="work/model")},
                ),
            },
        )
        runner = _runner(_launch_config(), profile_configs={"work": secondary})
        source = _source(profile="default")  # stale routing result
        source._identity = RoutingIdentity(
            transport_profile="work",
            runtime_profile="work",
            authorization_home=Path("/tmp/h"),
            runtime_home=Path("/tmp/h"),
        )
        override = runner._channel_override_for_source(source, Platform.TELEGRAM, "100")
        assert override is not None
        assert override.model == "work/model"

    def test_standalone_gateway_reads_the_bound_config_even_with_a_profile_named(self):
        config = replace(_launch_config(), multiplex_profiles=False)
        runner = _runner(config)
        override = runner._channel_override_for_source(
            _source(), Platform.TELEGRAM, "100"
        )
        assert override is not None
        assert override.model == "launch/override-model"


class TestChannelOverrideSourcePlumbing:
    def test_resolve_model_for_channel_follows_the_source(self):
        from unittest.mock import patch

        runner = _runner(_launch_config())
        with patch(
            "gateway.run._resolve_gateway_model", return_value="work/global-model"
        ):
            model = runner._resolve_model_for_channel(
                Platform.TELEGRAM, "100", source=_source()
            )
        assert model == "work/global-model"

    def test_resolve_model_for_channel_without_source_keeps_legacy_lookup(self):
        runner = _runner(_launch_config())
        assert (
            runner._resolve_model_for_channel(Platform.TELEGRAM, "100")
            == "launch/override-model"
        )

    def test_get_system_prompt_for_channel_follows_the_source(self):
        from unittest.mock import patch
        from gateway.run_config_loaders import GatewayConfigLoadersMixin

        launch = GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "100": ChannelOverride(system_prompt="launch prompt")
                    },
                ),
            },
        )
        runner = _runner(launch)
        with patch.object(
            GatewayConfigLoadersMixin,
            "_load_ephemeral_system_prompt",
            return_value="global prompt",
        ):
            assert (
                runner._get_system_prompt_for_channel(
                    Platform.TELEGRAM, "100", source=_source()
                )
                == "global prompt"
            )
            assert (
                runner._get_system_prompt_for_channel(Platform.TELEGRAM, "100")
                == "launch prompt"
            )
