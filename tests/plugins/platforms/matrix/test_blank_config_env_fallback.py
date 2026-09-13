"""A present-but-blank ``matrix:`` key in config.yaml means "unset": the env-var fallback must fire
exactly as it does when the key is absent (0.21.2 started seeding blank YAML values into
``config.extra``, which flipped the precedence and silently disabled free-response rooms)."""

import pytest

from gateway.config import PlatformConfig


@pytest.mark.parametrize("blank", ["", "  \t "])
def test_blank_yaml_values_fall_through_to_env(monkeypatch, blank):
    from plugins.platforms.matrix.adapter import MatrixAdapter, _extra_csv_set, _resolve_max_message_length

    monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", "!home:example.org")
    monkeypatch.setenv("MATRIX_MAX_MESSAGE_LENGTH", "9000")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    config = PlatformConfig(enabled=True, extra={
        "free_response_rooms": blank, "max_message_length": blank, "auto_thread": blank})

    assert _extra_csv_set(config, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS") == {"!home:example.org"}
    assert _resolve_max_message_length(config) == 9000
    assert MatrixAdapter._extra_truthy(config, "auto_thread", "MATRIX_AUTO_THREAD", "true") is False


def test_explicit_yaml_values_still_beat_env(monkeypatch):
    from plugins.platforms.matrix.adapter import MatrixAdapter, _extra_csv_set, _resolve_max_message_length

    monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", "!env:example.org")
    monkeypatch.setenv("MATRIX_MAX_MESSAGE_LENGTH", "9000")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "true")
    config = PlatformConfig(enabled=True, extra={
        "free_response_rooms": ["!a:example.org", " !b:example.org "], "max_message_length": 4000,
        "auto_thread": False})

    assert _extra_csv_set(config, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS") == {"!a:example.org", "!b:example.org"}
    assert _resolve_max_message_length(config) == 4000
    assert MatrixAdapter._extra_truthy(config, "auto_thread", "MATRIX_AUTO_THREAD", "true") is False
    # An explicit empty list is a real "no rooms" value, not "unset".
    assert _extra_csv_set(PlatformConfig(enabled=True, extra={"free_response_rooms": []}),
                          "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS") == set()
