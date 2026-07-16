"""Focused behavior tests for gateway terminal configuration overlays."""

import logging
import os


from gateway.sandbox_config import (
    apply_gateway_backend_to_env,
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


def test_gateway_image_override_is_stripped_before_env_write(monkeypatch):
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "terminal:image")

    apply_gateway_backend_to_env({"gateway": {"sandbox_image": " gateway:image "}})

    assert os.environ["TERMINAL_DOCKER_IMAGE"] == "gateway:image"


def test_absent_gateway_settings_preserve_terminal_behavior(monkeypatch):
    config = {"terminal": {"backend": "ssh", "docker_image": "terminal:image", "lifetime_seconds": 90}}
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "terminal:image")
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "90")

    apply_gateway_backend_to_env(config)

    assert os.environ["TERMINAL_ENV"] == "ssh"
    assert os.environ["TERMINAL_DOCKER_IMAGE"] == "terminal:image"
    assert os.environ["TERMINAL_LIFETIME_SECONDS"] == "90"


def test_gateway_local_override_warns_based_on_effective_backend(monkeypatch):
    config = {"terminal": {"backend": "docker"}, "gateway": {"terminal_backend": "local"}}
    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    apply_gateway_backend_to_env(config)
    assert should_warn_insecure_gateway(config) is True


def test_gateway_remote_override_suppresses_local_warning(monkeypatch):
    config = {"terminal": {"backend": "local"}, "gateway": {"terminal_backend": "docker"}}
    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    apply_gateway_backend_to_env(config)
    assert should_warn_insecure_gateway(config) is False


def test_warning_uses_terminal_env_when_config_omits_backend(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    assert should_warn_insecure_gateway({}) is False


def test_missing_gateway_config_warns_for_local_default(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    apply_gateway_backend_to_env({})
    assert should_warn_insecure_gateway({}) is True


def test_none_terminal_backend_warns_for_local_default(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    config = {"terminal": {"backend": None}}
    apply_gateway_backend_to_env(config)
    assert should_warn_insecure_gateway(config) is True


def test_non_mapping_gateway_config_warns_without_crashing(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    config = []
    apply_gateway_backend_to_env(config)
    assert should_warn_insecure_gateway(config) is True


def test_invalid_gateway_backend_is_visible_and_preserves_terminal_env(monkeypatch, caplog):
    config = {"terminal": {"backend": "ssh"}, "gateway": {"terminal_backend": "invalid"}}
    monkeypatch.setenv("TERMINAL_ENV", "ssh")

    with caplog.at_level(logging.WARNING):
        apply_gateway_backend_to_env(config)

    assert os.environ["TERMINAL_ENV"] == "ssh"
    assert "gateway.terminal_backend" in caplog.text
    assert "invalid" in caplog.text


def test_invalid_gateway_backend_keeps_local_warning(monkeypatch):
    config = {"terminal": {"backend": "local"}, "gateway": {"terminal_backend": "invalid"}}
    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    apply_gateway_backend_to_env(config)
    assert should_warn_insecure_gateway(config) is True


def test_invalid_gateway_lifetime_is_visible_and_preserves_terminal_env(monkeypatch, caplog):
    config = {"terminal": {"lifetime_seconds": 90}, "gateway": {"sandbox_lifetime": "forever"}}
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "90")

    with caplog.at_level(logging.WARNING):
        apply_gateway_backend_to_env(config)

    assert os.environ["TERMINAL_LIFETIME_SECONDS"] == "90"
    assert "gateway.sandbox_lifetime" in caplog.text
    assert "forever" in caplog.text


def test_fractional_gateway_lifetime_warns_and_preserves_terminal_env(monkeypatch, caplog):
    config = {"gateway": {"sandbox_lifetime": 1.5}}
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "90")

    with caplog.at_level(logging.WARNING):
        apply_gateway_backend_to_env(config)

    assert os.environ["TERMINAL_LIFETIME_SECONDS"] == "90"
    assert "gateway.sandbox_lifetime" in caplog.text




def test_non_finite_gateway_lifetime_warns_without_aborting(monkeypatch, caplog):
    config = {"gateway": {"sandbox_lifetime": float("inf")}}
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "90")

    with caplog.at_level(logging.WARNING):
        apply_gateway_backend_to_env(config)

    assert os.environ["TERMINAL_LIFETIME_SECONDS"] == "90"
    assert "gateway.sandbox_lifetime" in caplog.text
