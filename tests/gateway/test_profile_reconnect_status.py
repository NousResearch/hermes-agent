"""A multiplexed profile's adapter must read ``retrying`` while the runner reconnects it (#140451).

The primary ladder publishes ``retrying`` on a retryable fatal and again on every failed attempt. The
secondary path wrote nothing at all, so a profile-scoped adapter that the runner was actively retrying
kept the terminal ``fatal`` row its own ``_set_fatal_error`` had published, with that first failure's
error code and timestamp — an operator reading ``gateway_state.json`` / the dashboard sees a dead
channel, not a recovering one.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from tests.gateway.test_platform_reconnect import _make_runner, StubAdapter


def _retrying_writes(runner):
    return [
        call for call in runner._update_platform_runtime_status.call_args_list
        if call.kwargs.get("platform_state") == "retrying"
    ]


class TestProfileFatalPublishesRetrying:
    def _runner_with(self, adapter):
        runner = _make_runner()
        runner._profile_adapters = {"coder": {Platform.TELEGRAM: adapter}}
        runner._profile_failed_platforms = {}
        runner._update_platform_runtime_status = MagicMock()
        runner._safe_adapter_disconnect = AsyncMock()
        runner._schedule_secondary_profile_reconnect = MagicMock()
        return runner

    @pytest.mark.asyncio
    async def test_retryable_fatal_on_profile_adapter_reads_retrying(self):
        adapter = StubAdapter()
        runner = self._runner_with(adapter)
        adapter._set_fatal_error("telegram_network_error", "socket closed", retryable=True)
        runner._update_platform_runtime_status.reset_mock()

        with patch("gateway.run_adapters.record_platform_disconnect"), \
             patch("hermes_cli.profiles.get_profile_dir"):
            await runner._handle_profile_adapter_fatal_error("coder", Platform.TELEGRAM, adapter)

        writes = _retrying_writes(runner)
        assert writes, "a retrying profile adapter must not keep its terminal fatal row"
        assert writes[-1].args == ("coder:telegram",)
        assert writes[-1].kwargs["error_code"] == "telegram_network_error"
        assert writes[-1].kwargs["error_message"] == "socket closed"
        runner._schedule_secondary_profile_reconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_retryable_fatal_on_profile_adapter_stays_fatal(self):
        adapter = StubAdapter()
        runner = self._runner_with(adapter)
        adapter._set_fatal_error("email_auth_error", "bad app password", retryable=False)
        runner._update_platform_runtime_status.reset_mock()

        with patch("gateway.run_adapters.record_platform_disconnect"), \
             patch("hermes_cli.profiles.get_profile_dir"):
            await runner._handle_profile_adapter_fatal_error("coder", Platform.TELEGRAM, adapter)

        assert not _retrying_writes(runner)


class TestRetryLoopPublishesEachFailure:
    @pytest.mark.asyncio
    async def test_failed_attempt_republishes_retrying_with_this_failures_reason(self):
        runner = _make_runner()
        runner._profile_adapters, runner._profile_failed_platforms = {}, {}
        runner._sync_voice_mode_state_to_adapter = MagicMock()
        runner._update_platform_runtime_status = MagicMock()
        attempts = []

        async def attempt(profile, platform, predecessor=None):
            attempts.append(1)
            failed = StubAdapter(succeed=False, fatal_error="IMAP timeout")
            await failed.connect(is_reconnect=True)  # a real candidate's connect sets the retryable fatal
            if len(attempts) >= 2:
                runner._running = False  # stop the ladder after the second failed attempt
            return failed, False

        runner._secondary_reconnect_attempt = attempt
        with patch("gateway.run._reconnect_backoff", return_value=0), \
             patch("gateway.run._profile_runtime_scope", MagicMock()), \
             patch("gateway.run._platform_has_bot_credential", return_value=True), \
             patch("hermes_cli.profiles.get_profile_dir"), \
             patch("hermes_cli.env_loader.hydrate_profile_secret_sources"), \
             patch("gateway.config.load_gateway_config") as load:
            load.return_value.platforms = {}
            await asyncio.wait_for(
                runner._run_secondary_profile_reconnect("coder", Platform.TELEGRAM, None), timeout=10)

        assert len(attempts) >= 2, "the ladder must keep retrying a retryable failure"
        writes = _retrying_writes(runner)
        assert writes, "every failed profile reconnect attempt must republish retrying"
        reported = [w for w in writes if w.kwargs.get("error_code") == "test_error"]
        assert reported, f"the retry write must carry this failure's reason, got {writes}"
        assert reported[-1].args == ("coder:telegram",)
        assert reported[-1].kwargs["error_message"] == "IMAP timeout"
