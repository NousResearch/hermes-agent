"""Bot Mode eligibility for exact routed Telegram topics."""

from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _runner(*, multiplex=True):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        multiplex_profiles=multiplex,
        profile_routes=parse_profile_routes(
            [
                {
                    "name": "builder-topic",
                    "platform": "telegram",
                    "chat_id": "42",
                    "thread_id": "200",
                    "profile": "builder",
                },
                {
                    "name": "disabled-topic",
                    "platform": "telegram",
                    "chat_id": "42",
                    "thread_id": "300",
                    "profile": "builder",
                    "enabled": False,
                },
            ]
        ),
    )
    return runner


def _source(*, thread_id="200", profile="builder", platform=Platform.TELEGRAM):
    return SessionSource(
        platform=platform,
        chat_id="42",
        chat_type="group",
        thread_id=thread_id,
        user_id="200",
        profile=profile,
    )


def test_exact_routed_telegram_topic_is_bot_mode_entry():
    assert _runner()._bot_mode_gateway_entry_state(_source()) is True


def test_unthreaded_group_user_suffix_is_not_mistaken_for_topic():
    assert _runner()._bot_mode_gateway_entry_state(_source(thread_id=None)) is False


def test_topic_requires_current_routed_profile_and_enabled_exact_route():
    runner = _runner()
    assert runner._bot_mode_gateway_entry_state(_source(profile="other")) is False
    assert runner._bot_mode_gateway_entry_state(_source(thread_id="300")) is False
    assert runner._bot_mode_gateway_entry_state(_source(thread_id="999")) is False


def test_topic_requires_multiplexing_and_telegram():
    assert _runner(multiplex=False)._bot_mode_gateway_entry_state(_source()) is False
    assert (
        _runner()._bot_mode_gateway_entry_state(
            _source(platform=Platform.DISCORD)
        )
        is False
    )


def test_route_match_failure_is_indeterminate():
    with patch(
        "gateway.profile_routing.match_profile_route", side_effect=RuntimeError("boom")
    ):
        assert _runner()._bot_mode_gateway_entry_state(_source()) is None


def test_exact_route_profile_comparison_uses_the_normalized_route_profile():
    """source.profile comes from the same normalized route, so casing of the
    raw config value can never demote a legitimate topic (reviewer #3)."""
    runner = _runner()
    assert runner._bot_mode_gateway_entry_state(_source(profile="Builder")) is False
    normalized = parse_profile_routes(
        [
            {
                "name": "builder-topic",
                "platform": "telegram",
                "chat_id": "42",
                "thread_id": "200",
                "profile": "BuIlDeR",
            }
        ]
    )
    stamped = _source(profile=normalized[0].profile)
    assert GatewayRunner._bot_mode_gateway_entry_state(
        _runner_with_routes(normalized), stamped
    ) is True


def _runner_with_routes(routes):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=True, profile_routes=routes)
    return runner
