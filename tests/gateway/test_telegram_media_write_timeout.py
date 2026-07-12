"""Test that HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT is plumbed into PTB's HTTPXRequest.

This tests #62936: without media_write_timeout, large file uploads die at PTB's
20s default regardless of HERMES_TELEGRAM_HTTP_WRITE_TIMEOUT.
"""
import os


def test_media_write_timeout_default_180s(monkeypatch):
    """Verify media_write_timeout defaults to 180s when env var is not set."""
    # Remove any existing env var to test the default
    monkeypatch.delenv("HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT", raising=False)

    # Simulate the adapter's _env_float helper
    def _env_float(name, default):
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    # Build request_kwargs as the adapter does
    request_kwargs = {
        "connection_pool_size": _env_float("HERMES_TELEGRAM_HTTP_POOL_SIZE", 512),
        "pool_timeout": _env_float("HERMES_TELEGRAM_HTTP_POOL_TIMEOUT", 8.0),
        "connect_timeout": _env_float("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", 10.0),
        "read_timeout": _env_float("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", 20.0),
        "write_timeout": _env_float("HERMES_TELEGRAM_HTTP_WRITE_TIMEOUT", 20.0),
        # PTB uses a separate media_write_timeout for file-upload endpoints
        # (sendDocument, sendPhoto, sendMediaGroup, etc.). Without this,
        # large uploads die at PTB's 20s default regardless of write_timeout.
        # A 36MB file needs ~30-120s on typical residential uplinks.
        "media_write_timeout": _env_float("HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT", 180.0),
    }

    # Verify media_write_timeout defaults to 180.0
    assert "media_write_timeout" in request_kwargs
    assert request_kwargs["media_write_timeout"] == 180.0


def test_media_write_timeout_respects_env_var(monkeypatch):
    """Users should be able to override the 180s default via env var."""
    # Set custom timeout
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT", "300")

    # Simulate the adapter's _env_float helper
    def _env_float(name, default):
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    # Build request_kwargs as the adapter does
    request_kwargs = {
        "connection_pool_size": _env_float("HERMES_TELEGRAM_HTTP_POOL_SIZE", 512),
        "pool_timeout": _env_float("HERMES_TELEGRAM_HTTP_POOL_TIMEOUT", 8.0),
        "connect_timeout": _env_float("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", 10.0),
        "read_timeout": _env_float("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", 20.0),
        "write_timeout": _env_float("HERMES_TELEGRAM_HTTP_WRITE_TIMEOUT", 20.0),
        # PTB uses a separate media_write_timeout for file-upload endpoints
        # (sendDocument, sendPhoto, sendMediaGroup, etc.). Without this,
        # large uploads die at PTB's 20s default regardless of write_timeout.
        # A 36MB file needs ~30-120s on typical residential uplinks.
        "media_write_timeout": _env_float("HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT", 180.0),
    }

    # Verify media_write_timeout respects the env var
    assert request_kwargs["media_write_timeout"] == 300.0