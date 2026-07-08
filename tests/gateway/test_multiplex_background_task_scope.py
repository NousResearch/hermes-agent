"""Regression: background tasks respect profile secret scope when multiplexing.

Issue #60726: /background command runs _run_background_task without a profile
scope, causing UnscopedSecretError when multiplexing is active and credentials
are profile-scoped.
"""
import pytest
from unittest import mock


class TestBackgroundTaskProfileScope:
    """_run_background_task installs _profile_runtime_scope when multiplexing is active."""

    def test_background_task_calls_inner_wrapped_in_scope_when_multiplex_active(self):
        """When multiplex_profiles is True, _run_background_task wraps call in _profile_runtime_scope."""
        from gateway.run import GatewayRunner

        config = {"multiplex_profiles": True}
        gw = GatewayRunner(config=config)
        gw._session_db = mock.MagicMock()
        gw._adapter_for_source = mock.MagicMock(return_value=mock.MagicMock())

        mock_inner = mock.AsyncMock(return_value=None)
        gw._run_background_task_inner = mock_inner

        import asyncio
        source = mock.MagicMock()
        source.profile_home = "/fake/profile"

        # Mock _profile_runtime_scope
        from gateway.run import _profile_runtime_scope
        with mock.patch("gateway.run._profile_runtime_scope") as mock_scope:
            mock_scope.return_value.__enter__ = mock.MagicMock()
            mock_scope.return_value.__exit__ = mock.MagicMock()

            asyncio.run(
                gw._run_background_task(
                    prompt="test",
                    source=source,
                    task_id="test_task",
                )
            )

            # _profile_runtime_scope should have been called with profile_home
            mock_scope.assert_called_once_with("/fake/profile")
            mock_inner.assert_called_once()

    def test_background_task_calls_inner_direct_when_multiplex_disabled(self):
        """When multiplex_profiles is False, _run_background_task calls inner directly."""
        from gateway.run import GatewayRunner

        config = {"multiplex_profiles": False}
        gw = GatewayRunner(config=config)
        gw._session_db = mock.MagicMock()
        gw._adapter_for_source = mock.MagicMock(return_value=mock.MagicMock())

        mock_inner = mock.AsyncMock(return_value=None)
        gw._run_background_task_inner = mock_inner

        import asyncio
        source = mock.MagicMock()

        with mock.patch("gateway.run._profile_runtime_scope") as mock_scope:
            asyncio.run(
                gw._run_background_task(
                    prompt="test",
                    source=source,
                    task_id="test_task",
                )
            )

            # _profile_runtime_scope should NOT have been called
            mock_scope.assert_not_called()
            mock_inner.assert_called_once()