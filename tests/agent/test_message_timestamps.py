"""Cross-channel timestamp rendering follows the messaging gateway contract."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest


def test_non_gateway_config_falls_back_to_legacy_without_overriding_gateway_route():
    from agent.message_timestamps import message_timestamps_enabled

    legacy_enabled = {"gateway": {"message_timestamps": {"enabled": True}}}
    conflicting_enabled = {
        "message_timestamps": {"enabled": False},
        "gateway": {"message_timestamps": {"enabled": True}},
    }
    conflicting_disabled = {
        "message_timestamps": {"enabled": True},
        "gateway": {"message_timestamps": {"enabled": False}},
    }

    assert message_timestamps_enabled(legacy_enabled) is True
    assert message_timestamps_enabled(conflicting_enabled) is False
    assert message_timestamps_enabled(conflicting_disabled) is True
    assert message_timestamps_enabled(conflicting_enabled, messaging_gateway=True) is True
    assert message_timestamps_enabled(conflicting_disabled, messaging_gateway=True) is False


@pytest.mark.parametrize(
    ("config", "non_gateway", "messaging_gateway"),
    [
        ({}, False, False),
        ({"gateway": {"message_timestamps": True}}, True, True),
        ({"gateway": {"message_timestamps": False}}, False, False),
        ({"message_timestamps": {"enabled": None}, "gateway": {"message_timestamps": True}}, True, True),
        ({"message_timestamps": {"enabled": False}, "gateway": {"message_timestamps": True}}, False, True),
        ({"message_timestamps": {"enabled": True}, "gateway": {"message_timestamps": False}}, True, False),
    ],
)
def test_timestamp_configuration_precedence_is_route_scoped(config, non_gateway, messaging_gateway):
    from agent.message_timestamps import message_timestamps_enabled

    assert message_timestamps_enabled(config) is non_gateway
    assert message_timestamps_enabled(config, messaging_gateway=True) is messaging_gateway


def test_timestamp_config_isolated_between_profile_homes(tmp_path, monkeypatch):
    """Each profile's resolved home controls the non-gateway toggle independently."""
    from agent.message_timestamps import message_timestamps_enabled
    from hermes_cli.config import load_config_readonly

    enabled_home = tmp_path / "profiles" / "enabled"
    disabled_home = tmp_path / "profiles" / "disabled"
    enabled_home.mkdir(parents=True)
    disabled_home.mkdir(parents=True)
    (enabled_home / "config.yaml").write_text(
        "message_timestamps:\n  enabled: true\n",
        encoding="utf-8",
    )
    (disabled_home / "config.yaml").write_text(
        "message_timestamps:\n  enabled: false\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(enabled_home))
    assert message_timestamps_enabled(load_config_readonly()) is True
    monkeypatch.setenv("HERMES_HOME", str(disabled_home))
    assert message_timestamps_enabled(load_config_readonly()) is False


def test_non_gateway_rendering_keeps_structured_content_and_clean_history_immutable():
    from agent.message_timestamps import render_message_timestamp_replay

    tz = ZoneInfo("Europe/Berlin")
    timestamp = datetime(2026, 4, 28, 13, 42, 10, tzinfo=tz).timestamp()
    history = [{"role": "user", "content": "earlier", "timestamp": timestamp}]
    structured = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]

    replay, current_idx = render_message_timestamp_replay(
        history + [{"role": "user", "content": structured}],
        enabled=True,
        current_turn_user_idx=1,
        tz=tz,
    )

    assert history == [{"role": "user", "content": "earlier", "timestamp": timestamp}]
    assert replay[0]["content"] == "[Tue 2026-04-28 13:42:10 CEST] earlier"
    assert replay[1]["content"] is structured
    assert current_idx == 1


def test_history_cleanup_invalidates_sidecars_and_keeps_compatible_rendered_sidecars_exact():
    from agent.message_timestamps import render_message_timestamp_replay

    tz = ZoneInfo("UTC")
    timestamp = datetime(2026, 8, 20, 12, 0, tzinfo=tz).timestamp()
    rendered = "[Thu 2026-08-20 12:00:00 UTC] retain this"
    history = [
        {
            "role": "user",
            "content": "retain this",
            "timestamp": timestamp,
            "api_content": rendered + "\n\nretrieval context",
        },
        {
            "role": "user",
            "content": "[2026-08-20T12:00:00+00:00] [System note: Your previous turn was interrupted. Continue.] actual question",
            "timestamp": timestamp + 60,
            "api_content": "stale sidecar",
        },
    ]

    replay, _ = render_message_timestamp_replay(
        history,
        enabled=True,
        current_turn_user_idx=len(history),
        tz=tz,
    )

    assert replay[0]["api_content"] == rendered + "\n\nretrieval context"
    assert replay[1]["content"] == "[Thu 2026-08-20 12:00:00 UTC] actual question"
    assert "api_content" not in replay[1]
    assert history[1]["content"].endswith("actual question")


def test_history_cleanup_runs_when_timestamp_rendering_is_disabled():
    from agent.message_timestamps import render_message_timestamp_replay

    history = [{
        "role": "user",
        "content": "[System note: Your previous turn was interrupted. Continue.] actual question",
        "api_content": "stale sidecar",
    }]

    replay, _ = render_message_timestamp_replay(
        history,
        enabled=False,
        current_turn_user_idx=len(history),
    )

    assert replay == [{"role": "user", "content": "actual question"}]


@pytest.mark.parametrize("sidecar", [None, "", {"invalid": True}, ["invalid"], 7])
def test_replay_drops_ineligible_sidecars_before_timestamp_compatibility(sidecar):
    """Pinned gateway replay admits only nonempty strings as sidecars."""
    from agent.message_timestamps import render_message_timestamp_replay

    timestamp = datetime(2026, 8, 20, 12, 0, tzinfo=ZoneInfo("UTC")).timestamp()
    replay, _ = render_message_timestamp_replay(
        [{"role": "user", "content": "question", "timestamp": timestamp, "api_content": sidecar}],
        enabled=True,
        current_turn_user_idx=1,
        tz=ZoneInfo("UTC"),
    )

    assert replay == [{"role": "user", "content": "[Thu 2026-08-20 12:00:00 UTC] question", "timestamp": timestamp}]


def test_merged_defaults_leave_global_setting_unset_for_legacy_fallback(tmp_path, monkeypatch):
    from agent.message_timestamps import message_timestamps_enabled
    from hermes_cli.config import load_config_readonly

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "gateway:\n  message_timestamps:\n    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_config_readonly()

    assert config["message_timestamps"]["enabled"] is None
    assert message_timestamps_enabled(config) is True
