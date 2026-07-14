"""Regression tests for HonchoMemoryProvider.shutdown join-timeout alignment.

Issue #33485: ``HonchoMemoryProvider.shutdown`` joined the dialectic /
prefetch threads with a fixed ``timeout=5.0``, but Honcho HTTP calls
default to a 30s client timeout.  When the join expired before the
HTTP call returned, the daemon thread was still alive when CPython
started ``Py_FinalizeEx`` — intermittently aborting the interpreter
with SIGABRT during clean shutdown.

These tests pin the new contract: shutdown must wait at least as long
as the configured Honcho HTTP timeout (with a 5s floor) before
declaring the worker abandoned, and ``resolve_http_timeout`` must
agree with the resolution chain inside ``get_honcho_client``.
"""

from unittest.mock import MagicMock, patch

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import (
    _DEFAULT_HTTP_TIMEOUT,
    HonchoClientConfig,
    resolve_http_timeout,
)


def _shutdown_with_threads(provider: HonchoMemoryProvider) -> tuple[float, float]:
    """Run shutdown with mocked alive threads and return the join timeouts."""
    prefetch = MagicMock(name="prefetch_thread")
    prefetch.is_alive.return_value = True
    sync = MagicMock(name="sync_thread")
    sync.is_alive.return_value = True

    provider._prefetch_thread = prefetch
    provider._sync_thread = sync
    provider._manager = None  # skip flush_all

    provider.shutdown()

    return (
        prefetch.join.call_args.kwargs["timeout"],
        sync.join.call_args.kwargs["timeout"],
    )


class TestShutdownJoinTimeout:
    """Provider shutdown must outlive a pending Honcho HTTP call."""

    def test_uses_explicit_config_timeout(self):
        """When honcho.json sets timeout=60, shutdown joins for 60s."""
        provider = HonchoMemoryProvider()
        provider._config = HonchoClientConfig(api_key="k", timeout=60.0)

        prefetch_to, sync_to = _shutdown_with_threads(provider)

        assert prefetch_to == 60.0
        assert sync_to == 60.0

    def test_falls_back_to_default_http_timeout(self):
        """With no explicit timeout, shutdown waits the SDK default (30s)."""
        provider = HonchoMemoryProvider()
        provider._config = HonchoClientConfig(api_key="k", timeout=None)

        # Patch out hermes_cli.config so the helper doesn't pick up a
        # real on-disk override and skew the assertion.
        with patch("hermes_cli.config.load_config", return_value={}):
            prefetch_to, sync_to = _shutdown_with_threads(provider)

        assert prefetch_to == _DEFAULT_HTTP_TIMEOUT
        assert sync_to == _DEFAULT_HTTP_TIMEOUT

    def test_keeps_five_second_floor_for_short_timeouts(self):
        """A user-configured 2s HTTP timeout must not shrink the join window
        below the original 5s grace period.
        """
        provider = HonchoMemoryProvider()
        provider._config = HonchoClientConfig(api_key="k", timeout=2.0)

        prefetch_to, sync_to = _shutdown_with_threads(provider)

        assert prefetch_to == 5.0
        assert sync_to == 5.0

    def test_handles_missing_config_gracefully(self):
        """Shutdown before initialize() must not crash; falls back to the
        SDK default timeout because ``resolve_http_timeout(None)`` returns
        ``_DEFAULT_HTTP_TIMEOUT``.
        """
        provider = HonchoMemoryProvider()
        # _config is None straight from __init__

        with patch("hermes_cli.config.load_config", return_value={}):
            prefetch_to, sync_to = _shutdown_with_threads(provider)

        assert prefetch_to == _DEFAULT_HTTP_TIMEOUT
        assert sync_to == _DEFAULT_HTTP_TIMEOUT


class TestResolveHttpTimeout:
    """``resolve_http_timeout`` must agree with ``get_honcho_client``'s chain."""

    def test_prefers_config_timeout_over_hermes_cli(self):
        cfg = HonchoClientConfig(timeout=11.0)
        with patch(
            "hermes_cli.config.load_config",
            return_value={"honcho": {"timeout": 99}},
        ):
            assert resolve_http_timeout(cfg) == 11.0

    def test_falls_back_to_hermes_cli_timeout(self):
        cfg = HonchoClientConfig(timeout=None)
        with patch(
            "hermes_cli.config.load_config",
            return_value={"honcho": {"timeout": 77}},
        ):
            assert resolve_http_timeout(cfg) == 77.0

    def test_request_timeout_alias_is_honored(self):
        cfg = HonchoClientConfig(timeout=None)
        with patch(
            "hermes_cli.config.load_config",
            return_value={"honcho": {"request_timeout": "55.5"}},
        ):
            assert resolve_http_timeout(cfg) == 55.5

    def test_defaults_to_thirty_seconds(self):
        cfg = HonchoClientConfig(timeout=None)
        with patch("hermes_cli.config.load_config", return_value={}):
            assert resolve_http_timeout(cfg) == _DEFAULT_HTTP_TIMEOUT

    def test_handles_none_config(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert resolve_http_timeout(None) == _DEFAULT_HTTP_TIMEOUT

    def test_swallows_load_config_failure(self):
        cfg = HonchoClientConfig(timeout=None)
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            assert resolve_http_timeout(cfg) == _DEFAULT_HTTP_TIMEOUT
