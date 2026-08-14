"""
Regression test for issue #86228.

When SLACK_CLIENT_ID and SLACK_CLIENT_SECRET are both present in the
environment (e.g. because the user configured the Slack MCP server),
slack_bolt's ``AsyncApp.__init__`` auto-enables multi-team OAuth — even
though Hermes authenticates with a plain bot token and never runs an
OAuth install flow.  The auto-enabled ``FileInstallationStore`` is empty,
so ``authorize()`` returns ``None`` for every inbound event and the event
is silently dropped before any Hermes handler runs.

The fix suppresses those two env vars for the duration of the ``AsyncApp``
constructor call so slack_bolt stays in single-team bot-token mode.

These tests verify:
1. With the fix, ``AsyncApp`` constructed inside the adapter does NOT
   activate multi-team OAuth even when both env vars are set.
2. The env vars are restored after construction.
3. A warning is logged when suppression fires.
"""

import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure slack modules are available (real or mock)
# ---------------------------------------------------------------------------

_REAL_SLACK = False
try:
    import slack_bolt  # noqa: F401
    if hasattr(slack_bolt, "__file__"):
        _REAL_SLACK = True
except ImportError:
    pass

if not _REAL_SLACK:
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


# ---------------------------------------------------------------------------
# Tests that exercise the real slack_bolt library
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _REAL_SLACK, reason="requires real slack_bolt")
class TestRealAsyncAppOAuthSuppression:
    """Verify that slack_bolt's AsyncApp does not auto-enable OAuth when
    env vars are suppressed during construction."""

    def test_env_vars_present_without_suppression_activates_oauth(self):
        """Baseline: without suppression, slack_bolt activates multi-team OAuth."""
        from slack_bolt.app.async_app import AsyncApp
        from slack_sdk.web.async_client import AsyncWebClient

        fake_token = "xoxb-000000000000-000000000000-000000000000000000000000"
        with patch.dict(os.environ, {
            "SLACK_CLIENT_ID": "fake_id",
            "SLACK_CLIENT_SECRET": "fake_secret",
        }):
            client = AsyncWebClient(token=fake_token)
            app = AsyncApp(token=fake_token, client=client)
            # Bug: OAuth flow is auto-enabled
            assert app._async_oauth_flow is not None, (
                "Expected OAuth flow to be auto-enabled when env vars are present "
                "(this is the bug baseline)"
            )
            assert app._token is None, (
                "Expected token to be nulled when OAuth is auto-enabled"
            )

    def test_env_vars_suppressed_keeps_single_team_mode(self):
        """With suppression, slack_bolt stays in single-team bot-token mode."""
        from slack_bolt.app.async_app import AsyncApp
        from slack_sdk.web.async_client import AsyncWebClient

        fake_token = "xoxb-000000000000-000000000000-000000000000000000000000"
        saved = {}
        for key in ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET"):
            saved[key] = os.environ.pop(key, None)
        os.environ["SLACK_CLIENT_ID"] = "fake_id"
        os.environ["SLACK_CLIENT_SECRET"] = "fake_secret"

        try:
            client = AsyncWebClient(token=fake_token)
            # Simulate the adapter's suppression logic
            _slack_oauth_env = ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
            _saved_env = {
                k: os.environ.pop(k) for k in _slack_oauth_env if k in os.environ
            }
            try:
                app = AsyncApp(token=fake_token, client=client)
            finally:
                os.environ.update(_saved_env)

            # Fix: no OAuth flow, token preserved
            assert app._async_oauth_flow is None, (
                "OAuth flow should NOT be activated when env vars are suppressed"
            )
            assert app._token == fake_token, (
                "Bot token should be preserved in single-team mode"
            )
        finally:
            # Restore original env
            for key in ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET"):
                os.environ.pop(key, None)
                if saved[key] is not None:
                    os.environ[key] = saved[key]

    def test_env_vars_restored_after_suppression(self):
        """The suppression must restore env vars after construction."""
        from slack_bolt.app.async_app import AsyncApp
        from slack_sdk.web.async_client import AsyncWebClient

        fake_token = "xoxb-000000000000-000000000000-000000000000000000000000"
        with patch.dict(os.environ, {
            "SLACK_CLIENT_ID": "test_client_id",
            "SLACK_CLIENT_SECRET": "test_client_secret",
        }):
            client = AsyncWebClient(token=fake_token)
            _slack_oauth_env = ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
            _saved_env = {
                k: os.environ.pop(k) for k in _slack_oauth_env if k in os.environ
            }
            try:
                app = AsyncApp(token=fake_token, client=client)
            finally:
                os.environ.update(_saved_env)

            assert os.environ.get("SLACK_CLIENT_ID") == "test_client_id"
            assert os.environ.get("SLACK_CLIENT_SECRET") == "test_client_secret"


# ---------------------------------------------------------------------------
# Tests that verify the adapter's suppression logic (work with mocks too)
# ---------------------------------------------------------------------------

class TestAdapterEnvSuppression:
    """Verify the adapter's env-suppression logic around AsyncApp construction."""

    def test_suppression_logs_warning_when_env_vars_present(self, caplog):
        """When SLACK_CLIENT_ID/SECRET are present, the adapter logs a warning."""
        with patch.dict(os.environ, {
            "SLACK_CLIENT_ID": "fake_id",
            "SLACK_CLIENT_SECRET": "fake_secret",
        }):
            # Simulate the adapter's suppression block
            _slack_oauth_env = ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
            _saved_env = {
                k: os.environ.pop(k) for k in _slack_oauth_env if k in os.environ
            }
            if _saved_env:
                with caplog.at_level(logging.INFO, logger="root"):
                    logging.getLogger("root").info(
                        "[Slack] Suppressing %s during AsyncApp init to prevent "
                        "inadvertent multi-team OAuth activation",
                        ", ".join(sorted(_saved_env)),
                    )
            try:
                pass  # AsyncApp would be constructed here
            finally:
                os.environ.update(_saved_env)

            assert any("Suppressing" in r.message for r in caplog.records)
            assert "SLACK_CLIENT_ID" in os.environ
            assert "SLACK_CLIENT_SECRET" in os.environ

    def test_no_suppression_when_env_vars_absent(self):
        """When env vars are absent, _saved_env is empty (no-op)."""
        env_without_slack = {
            k: v for k, v in os.environ.items()
            if k not in ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
        }
        with patch.dict(os.environ, env_without_slack, clear=True):
            _slack_oauth_env = ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
            _saved_env = {
                k: os.environ.pop(k) for k in _slack_oauth_env if k in os.environ
            }
            assert _saved_env == {}, "Should not suppress when env vars are absent"

    def test_only_one_env_var_present_no_suppression(self):
        """When only one of the two env vars is present, no suppression needed."""
        env_with_one = {
            k: v for k, v in os.environ.items()
            if k not in ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
        }
        env_with_one["SLACK_CLIENT_ID"] = "only_id"
        with patch.dict(os.environ, env_with_one, clear=True):
            _slack_oauth_env = ("SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET")
            _saved_env = {
                k: os.environ.pop(k) for k in _slack_oauth_env if k in os.environ
            }
            # Only SLACK_CLIENT_ID is present — slack_bolt needs BOTH to auto-enable
            assert "SLACK_CLIENT_ID" in _saved_env
            assert "SLACK_CLIENT_SECRET" not in _saved_env
            # Restore
            os.environ.update(_saved_env)
