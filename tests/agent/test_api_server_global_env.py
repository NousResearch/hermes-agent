"""#69379: API_SERVER_* env vars must be treated as global (platform
enablement), not profile secrets. Docker compose environment: must remain
authoritative for platform enablement in multiplex mode."""

from __future__ import annotations

import os

import pytest

from agent.secret_scope import _is_global_env, get_secret, set_secret_scope, _MULTIPLEX_ACTIVE


class TestApiServerGlobalEnv:
    def test_api_server_enabled_is_global(self):
        assert _is_global_env("API_SERVER_ENABLED") is True

    def test_api_server_host_is_global(self):
        assert _is_global_env("API_SERVER_HOST") is True

    def test_api_server_port_is_global(self):
        assert _is_global_env("API_SERVER_PORT") is True

    def test_api_server_cors_origins_is_global(self):
        assert _is_global_env("API_SERVER_CORS_ORIGINS") is True

    def test_api_server_var_reads_from_os_environ_under_scope(self, monkeypatch):
        """When a secret scope is active, API_SERVER_* must still read
        from os.environ, not the scope (it's platform enablement, not
        a profile secret)."""
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        # Install a scope that does NOT contain API_SERVER_ENABLED
        scope = {"TELEGRAM_BOT_TOKEN": "some-token"}
        token = set_secret_scope(scope)
        try:
            val = get_secret("API_SERVER_ENABLED")
            assert val == "true", (
                "API_SERVER_ENABLED must read from os.environ even under "
                "secret scope — it's platform enablement, not a profile "
                "secret (#69379)"
            )
        finally:
            _reset = set_secret_scope(None)  # clear
            # Properly reset
            from agent.secret_scope import _SECRET_SCOPE
            _SECRET_SCOPE.reset(token)