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


def test_explicit_env_beats_yaml_and_yaml_beats_default(monkeypatch):
    """Per-profile precedence: explicit scoped env → the profile's YAML → default. A blank env
    value is unset (it must not clobber YAML); an explicit empty list is a real "no rooms" value."""
    from plugins.platforms.matrix.adapter import MatrixAdapter, _extra_csv_set, _resolve_max_message_length

    yaml_config = PlatformConfig(enabled=True, extra={
        "free_response_rooms": ["!a:example.org", " !b:example.org "], "max_message_length": 4000,
        "auto_thread": False})

    monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", "!env:example.org")
    monkeypatch.setenv("MATRIX_MAX_MESSAGE_LENGTH", "9000")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "true")
    assert _extra_csv_set(yaml_config, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS") == {"!env:example.org"}
    assert _resolve_max_message_length(yaml_config) == 9000
    assert MatrixAdapter._extra_truthy(yaml_config, "auto_thread", "MATRIX_AUTO_THREAD", "true") is True

    for name in ("MATRIX_FREE_RESPONSE_ROOMS", "MATRIX_MAX_MESSAGE_LENGTH", "MATRIX_AUTO_THREAD"):
        monkeypatch.setenv(name, "  ")
    assert _extra_csv_set(yaml_config, "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS") == {"!a:example.org", "!b:example.org"}
    assert _resolve_max_message_length(yaml_config) == 4000
    assert MatrixAdapter._extra_truthy(yaml_config, "auto_thread", "MATRIX_AUTO_THREAD", "true") is False

    monkeypatch.delenv("MATRIX_AUTO_THREAD")
    assert MatrixAdapter._extra_truthy(PlatformConfig(enabled=True, extra={}), "auto_thread", "MATRIX_AUTO_THREAD", "true") is True
    assert _extra_csv_set(PlatformConfig(enabled=True, extra={"free_response_rooms": []}),
                          "free_response_rooms", "MATRIX_FREE_RESPONSE_ROOMS") == set()
