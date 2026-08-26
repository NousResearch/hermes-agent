"""Tests for shared-metrics send configuration resolution."""

from __future__ import annotations

import logging

import pytest

from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.observability.shared_metrics_send_config import (
    DEFAULT_ENDPOINT,
    ENDPOINT_ENV_VAR,
    resolve_send_config,
    reset_warning_latch_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_latch():
    reset_warning_latch_for_tests()
    yield
    reset_warning_latch_for_tests()


def _config(**shared):
    return {"telemetry": {"shared_metrics": shared}}


class TestDefaults:
    def test_send_is_registered_disabled_by_default(self):
        shared = DEFAULT_CONFIG["telemetry"]["shared_metrics"]
        assert shared["enabled"] is False
        assert shared["send"] is False

    def test_default_endpoint_is_production(self):
        shared = DEFAULT_CONFIG["telemetry"]["shared_metrics"]
        assert shared["endpoint"] == DEFAULT_ENDPOINT
        assert DEFAULT_ENDPOINT.startswith("https://")

    def test_empty_config_sends_nothing(self):
        resolved = resolve_send_config({})
        assert resolved.enabled is False
        assert resolved.send is False

    def test_none_config_is_tolerated(self):
        assert resolve_send_config(None).send is False


class TestSendRequiresCollection:
    def test_collection_alone_does_not_send(self):
        resolved = resolve_send_config(_config(enabled=True))
        assert resolved.enabled is True
        assert resolved.send is False

    def test_send_with_collection_sends(self):
        resolved = resolve_send_config(_config(enabled=True, send=True))
        assert resolved.send is True

    def test_send_without_collection_is_refused(self):
        resolved = resolve_send_config(_config(enabled=False, send=True))
        assert resolved.send is False
        # send must never imply enabled
        assert resolved.enabled is False

    def test_send_without_collection_logs_an_error(self, caplog):
        with caplog.at_level(logging.ERROR):
            resolve_send_config(_config(enabled=False, send=True))
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        assert "enabled is false" in errors[0].getMessage()

    def test_the_error_is_logged_once_per_process(self, caplog):
        with caplog.at_level(logging.ERROR):
            for _ in range(5):
                resolve_send_config(_config(enabled=False, send=True))
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1, "misconfiguration must not spam every hook fire"


class TestEndpointPrecedence:
    def test_config_endpoint_overrides_default(self):
        resolved = resolve_send_config(
            _config(enabled=True, send=True, endpoint="https://example.test/v1")
        )
        assert resolved.endpoint == "https://example.test/v1"

    def test_env_var_overrides_config(self, monkeypatch):
        monkeypatch.setenv(ENDPOINT_ENV_VAR, "https://staging.test/v1")
        resolved = resolve_send_config(
            _config(enabled=True, send=True, endpoint="https://example.test/v1")
        )
        assert resolved.endpoint == "https://staging.test/v1"

    def test_blank_endpoint_falls_back_to_production(self):
        resolved = resolve_send_config(_config(enabled=True, send=True, endpoint="   "))
        assert resolved.endpoint == DEFAULT_ENDPOINT

    def test_endpoint_is_stripped(self, monkeypatch):
        monkeypatch.setenv(ENDPOINT_ENV_VAR, "  https://staging.test/v1  ")
        assert resolve_send_config(_config(enabled=True, send=True)).endpoint == (
            "https://staging.test/v1"
        )


class TestTransportSafety:
    def test_plaintext_endpoint_is_refused(self, caplog):
        with caplog.at_level(logging.ERROR):
            resolved = resolve_send_config(
                _config(enabled=True, send=True, endpoint="http://example.test/v1")
            )
        assert resolved.send is False, "telemetry must not go out in clear text"
        assert any("https" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://localhost:8099/v1/telemetry",
            "http://127.0.0.1:8099/v1/telemetry",
        ],
    )
    def test_loopback_http_is_allowed_for_testing(self, endpoint):
        resolved = resolve_send_config(
            _config(enabled=True, send=True, endpoint=endpoint)
        )
        assert resolved.send is True

    def test_nonsense_scheme_is_refused(self):
        resolved = resolve_send_config(
            _config(enabled=True, send=True, endpoint="ftp://example.test/v1")
        )
        assert resolved.send is False

    def test_unsafe_endpoint_does_not_block_collection(self):
        resolved = resolve_send_config(
            _config(enabled=True, send=True, endpoint="http://example.test/v1")
        )
        assert resolved.enabled is True
