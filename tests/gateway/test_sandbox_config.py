"""Focused behavior tests for gateway terminal configuration overlays."""

import logging
import os


from gateway.sandbox_config import (
    apply_gateway_backend_to_env,
    get_gateway_sandbox_lifetime,
    get_gateway_terminal_backend,
    should_warn_insecure_gateway,
)


def test_gateway_backend_image_and_lifetime_override_terminal_settings(monkeypatch):
    config = {
        "terminal": {
            "backend": "local",
            "docker_image": "terminal:image",
            "lifetime_seconds": 120,
        },
        "gateway": {
            "terminal_backend": "docker",
            "sandbox_image": "gateway:image",
            "sandbox_lifetime": 42,
        },
    }
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "terminal:image")
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "120")

    apply_gateway_backend_to_env(config)

    assert os.environ["TERMINAL_ENV"] == "docker"
    assert os.environ["TERMINAL_DOCKER_IMAGE"] == "gateway:image"
    assert os.environ["TERMINAL_LIFETIME_SECONDS"] == "42"


def test_absent_gateway_settings_preserve_terminal_behavior(monkeypatch):
    config = {"terminal": {"backend": "ssh", "docker_image": "terminal:image", "lifetime_seconds": 90}}
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "terminal:image")
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "90")

    apply_gateway_backend_to_env(config)

    assert os.environ["TERMINAL_ENV"] == "ssh"
    assert os.environ["TERMINAL_DOCKER_IMAGE"] == "terminal:image"
    assert os.environ["TERMINAL_LIFETIME_SECONDS"] == "90"


def test_gateway_local_override_warns_based_on_effective_backend():
    config = {"terminal": {"backend": "docker"}, "gateway": {"terminal_backend": "local"}}

    assert should_warn_insecure_gateway(config) is True


def test_gateway_remote_override_suppresses_local_warning():
    config = {"terminal": {"backend": "local"}, "gateway": {"terminal_backend": "docker"}}

    assert should_warn_insecure_gateway(config) is False


def test_warning_uses_terminal_env_when_config_omits_backend(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    assert should_warn_insecure_gateway({}) is False


def test_invalid_gateway_backend_is_visible_and_preserves_terminal_env(monkeypatch, caplog):
    config = {"terminal": {"backend": "ssh"}, "gateway": {"terminal_backend": "invalid"}}
    monkeypatch.setenv("TERMINAL_ENV", "ssh")

    with caplog.at_level(logging.WARNING):
        apply_gateway_backend_to_env(config)

    assert os.environ["TERMINAL_ENV"] == "ssh"
    assert "gateway.terminal_backend" in caplog.text
    assert "invalid" in caplog.text


def test_invalid_gateway_lifetime_is_visible_and_preserves_terminal_env(monkeypatch, caplog):
    config = {"terminal": {"lifetime_seconds": 90}, "gateway": {"sandbox_lifetime": "forever"}}
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "90")

    with caplog.at_level(logging.WARNING):
        apply_gateway_backend_to_env(config)

    assert os.environ["TERMINAL_LIFETIME_SECONDS"] == "90"
    assert "gateway.sandbox_lifetime" in caplog.text
    assert "forever" in caplog.text


def test_lifetime_helper_returns_default_and_custom_values():
    assert get_gateway_sandbox_lifetime({}) == 3600
    assert get_gateway_sandbox_lifetime({"gateway": {"sandbox_lifetime": 15}}) == 15


def test_backend_helper_handles_missing_gateway_section():
    assert get_gateway_terminal_backend({}) is None
