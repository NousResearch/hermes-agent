"""Tests for configurable platform timeouts."""
import os
import pytest


class TestMattermostTimeouts:
    def test_default_http_timeout(self):
        os.environ.pop("HERMES_MATTERMOST_HTTP_TIMEOUT", None)
        from gateway.platforms.mattermost import _MATTERMOST_HTTP_TIMEOUT
        assert _MATTERMOST_HTTP_TIMEOUT == 30.0

    def test_default_upload_timeout(self):
        os.environ.pop("HERMES_MATTERMOST_UPLOAD_TIMEOUT", None)
        from gateway.platforms.mattermost import _MATTERMOST_UPLOAD_TIMEOUT
        assert _MATTERMOST_UPLOAD_TIMEOUT == 60.0


class TestSignalTimeouts:
    def test_default_http_timeout(self):
        os.environ.pop("HERMES_SIGNAL_HTTP_TIMEOUT", None)
        from gateway.platforms.signal import _SIGNAL_HTTP_TIMEOUT
        assert _SIGNAL_HTTP_TIMEOUT == 30.0

    def test_default_poll_timeout(self):
        os.environ.pop("HERMES_SIGNAL_POLL_TIMEOUT", None)
        from gateway.platforms.signal import _SIGNAL_POLL_TIMEOUT
        assert _SIGNAL_POLL_TIMEOUT == 10.0

    def test_default_health_timeout(self):
        os.environ.pop("HERMES_SIGNAL_HEALTH_TIMEOUT", None)
        from gateway.platforms.signal import _SIGNAL_HEALTH_TIMEOUT
        assert _SIGNAL_HEALTH_TIMEOUT == 10.0


class TestRetryBackoff:
    def test_default_base_delay(self):
        os.environ.pop("HERMES_RETRY_BASE_DELAY", None)
        import importlib
        import agent.retry_utils as ru
        importlib.reload(ru)
        assert ru._RETRY_BASE_DELAY == 5.0

    def test_default_max_delay(self):
        os.environ.pop("HERMES_RETRY_MAX_DELAY", None)
        import importlib
        import agent.retry_utils as ru
        importlib.reload(ru)
        assert ru._RETRY_MAX_DELAY == 120.0
