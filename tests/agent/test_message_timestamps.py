from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.message_timestamps import (
    message_timestamps_enabled,
    render_turn_with_message_timestamps,
)
from agent.turn_context import substitute_api_content
from hermes_cli.config import DEFAULT_CONFIG, load_config_readonly


BERLIN = ZoneInfo("Europe/Berlin")


def _epoch(year, month, day, hour, minute, second):
    return datetime(year, month, day, hour, minute, second, tzinfo=BERLIN).timestamp()


def test_global_setting_is_canonical_with_gateway_compatibility():
    assert message_timestamps_enabled({"message_timestamps": {"enabled": True}}) is True
    assert message_timestamps_enabled(
        {"gateway": {"message_timestamps": {"enabled": True}}}
    ) is True
    assert message_timestamps_enabled(
        {
            "message_timestamps": {"enabled": False},
            "gateway": {"message_timestamps": {"enabled": True}},
        }
    ) is False


def test_default_config_declares_global_setting_without_masking_legacy_fallback():
    config = deepcopy(DEFAULT_CONFIG)

    assert "message_timestamps" in config
    config["gateway"]["message_timestamps"]["enabled"] = True
    assert message_timestamps_enabled(config) is True

    config["message_timestamps"]["enabled"] = False
    assert message_timestamps_enabled(config) is False


def test_loaded_legacy_gateway_config_survives_default_merge(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "gateway:\n  message_timestamps:\n    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_config_readonly()

    assert "message_timestamps" in config
    assert message_timestamps_enabled(config) is True


def test_render_turn_timestamps_history_and_current_user_without_mutating_input():
    prior_ts = _epoch(2026, 4, 28, 13, 40, 53)
    current_ts = _epoch(2026, 4, 28, 13, 42, 10)
    history = [
        {"role": "user", "content": "earlier", "timestamp": prior_ts},
        {"role": "assistant", "content": "reply"},
    ]

    rendered_history, rendered_user, persisted_ts = render_turn_with_message_timestamps(
        history,
        "now",
        config={"message_timestamps": {"enabled": True}},
        current_timestamp=current_ts,
        tz=BERLIN,
    )

    assert history[0]["content"] == "earlier"
    assert rendered_history[0]["content"] == "[Tue 2026-04-28 13:40:53 CEST] earlier"
    assert rendered_history[1]["content"] == "reply"
    assert rendered_user == "[Tue 2026-04-28 13:42:10 CEST] now"
    assert persisted_ts == current_ts


def test_rendered_history_keeps_timestamp_when_api_sidecar_is_substituted():
    prior_ts = _epoch(2026, 4, 28, 13, 40, 53)
    history = [
        {
            "role": "user",
            "content": "earlier",
            "api_content": "earlier\n\n<memory-context>prior context</memory-context>",
            "timestamp": prior_ts,
        }
    ]

    rendered_history, _, _ = render_turn_with_message_timestamps(
        history,
        "now",
        config={"message_timestamps": {"enabled": True}},
        current_timestamp=_epoch(2026, 4, 28, 13, 42, 10),
        tz=BERLIN,
    )
    api_message = dict(rendered_history[0])
    substitute_api_content(api_message)

    assert history[0]["api_content"].startswith("earlier")
    assert api_message["content"].startswith("[Tue 2026-04-28 13:40:53 CEST]")
    assert "<memory-context>prior context</memory-context>" in api_message["content"]


def test_render_turn_is_identity_preserving_when_disabled():
    history = [{"role": "user", "content": "clean", "timestamp": 1.0}]

    rendered_history, rendered_user, persisted_ts = render_turn_with_message_timestamps(
        history,
        "current",
        config={},
        current_timestamp=2.0,
        tz=BERLIN,
    )

    assert rendered_history == history
    assert rendered_history is not history
    assert rendered_user == "current"
    assert persisted_ts is None


def test_render_turn_timestamps_structured_user_content_without_touching_image_part():
    current_ts = _epoch(2026, 4, 28, 13, 42, 10)
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    user_message = [
        {"type": "text", "text": "describe this"},
        image_part,
    ]

    _, rendered_user, _ = render_turn_with_message_timestamps(
        [],
        user_message,
        config={"message_timestamps": {"enabled": True}},
        current_timestamp=current_ts,
        tz=BERLIN,
    )

    assert user_message[0]["text"] == "describe this"
    assert rendered_user[0]["text"] == "[Tue 2026-04-28 13:42:10 CEST] describe this"
    assert rendered_user[1] == image_part
