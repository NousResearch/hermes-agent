"""Regression tests for redacting sensitive credentials from Gateway provider auth error responses."""

from unittest.mock import MagicMock, patch
import pytest

from gateway.run import GatewayRunner, SessionSource, Platform


class TestGatewayAuthErrorCredentialRedaction:
    """Verify raw exceptions in _run_agent mask secrets before sending responses."""

    def _make_runner(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._config = {}
        runner._session_store = MagicMock()
        runner._provider_routing = None
        runner.hooks = MagicMock()
        runner._running_agents = {}
        runner._active_session_leases = {}
        runner._get_proxy_url = MagicMock(return_value=None)
        runner._adapter_for_source = MagicMock(return_value=None)
        runner._get_system_prompt_for_channel = MagicMock(return_value="")
        return runner

    def _make_source(self):
        return SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")

    @pytest.mark.asyncio
    async def test_auth_error_redacts_api_keys(self):
        runner = self._make_runner()
        source = self._make_source()
        leak_msg = "Invalid API key provided: sk-ant-api03-abcdef1234567890ghijklmn-123456"

        with patch.object(runner, "_resolve_session_agent_runtime", side_effect=ValueError(leak_msg)):
            res = await runner._run_agent(
                "hello",
                "",
                [],
                source=source,
                session_key="user1",
                session_id="s1",
            )

        resp = res["final_response"]
        assert "abcdef1234567890ghijklmn" not in resp
        assert "sk-ant...3456" in resp
        assert "⚠️ Provider authentication failed:" in resp

    @pytest.mark.asyncio
    async def test_auth_error_preserves_safe_message(self):
        runner = self._make_runner()
        source = self._make_source()
        safe_msg = "Unknown provider: custom_foo"

        with patch.object(runner, "_resolve_session_agent_runtime", side_effect=ValueError(safe_msg)):
            res = await runner._run_agent(
                "hello",
                "",
                [],
                source=source,
                session_key="user1",
                session_id="s1",
            )

        resp = res["final_response"]
        assert "Unknown provider: custom_foo" in resp

    @pytest.mark.asyncio
    async def test_auth_error_truncates_long_message(self):
        runner = self._make_runner()
        source = self._make_source()
        long_msg = "A" * 500

        with patch.object(runner, "_resolve_session_agent_runtime", side_effect=RuntimeError(long_msg)):
            res = await runner._run_agent(
                "hello",
                "",
                [],
                source=source,
                session_key="user1",
                session_id="s1",
            )

        resp = res["final_response"]
        assert len(resp) <= 350
