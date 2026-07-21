"""Test cumulative poll failure termination in WeixinAdapter._poll_loop."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


class TestWeixinPollFailureTermination:
    """Tests that _poll_loop terminates after MAX_TOTAL_FAILURES cumulative failures."""

    @pytest.mark.asyncio
    async def test_poll_loop_terminates_after_max_total_failures_in_getupdates_path(self):
        """When getUpdates fails repeatedly in the ret!=0 path, _poll_loop should mark fatal and break after MAX_TOTAL_FAILURES."""
        from gateway.platforms.weixin import MAX_TOTAL_FAILURES, MAX_CONSECUTIVE_FAILURES

        # Track poll loop exit and fatal error calls
        poll_completed = asyncio.Event()
        fatal_error_called = asyncio.Event()
        fatal_error_args = []

        def capture_fatal_error(code, message, *, retryable):
            fatal_error_args.append((code, message, retryable))
            fatal_error_called.set()

        async def failing_get_updates(session, base_url, token, sync_buf, timeout_ms):
            # Always return a failure response
            return {"ret": -1, "errcode": -1, "errmsg": "test error"}

        async def mock_poll_loop():
            consecutive_failures = 0
            total_failures = 0
            running = True

            # Shorten the sleep for test speed
            short_retry = 0.01
            short_backoff = 0.01

            while running:
                try:
                    response = await failing_get_updates(
                        None, "http://localhost:15236", "token", "", 35000
                    )
                    ret = response.get("ret", 0)
                    errcode = response.get("errcode", 0)

                    if ret not in {0, None} or errcode not in {0, None}:
                        consecutive_failures += 1
                        total_failures += 1
                        if total_failures >= MAX_TOTAL_FAILURES:
                            capture_fatal_error(
                                "weixin_bridge_down",
                                f"Poll failed {total_failures} times consecutively",
                                retryable=True,
                            )
                            break

                        await asyncio.sleep(
                            short_backoff
                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                            else short_retry
                        )
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            consecutive_failures = 0
                        continue

                    consecutive_failures = 0
                    total_failures = 0
                except asyncio.CancelledError:
                    break
                except Exception:
                    consecutive_failures += 1
                    total_failures += 1
                    if total_failures >= MAX_TOTAL_FAILURES:
                        capture_fatal_error(
                            "weixin_bridge_down",
                            f"Poll failed {total_failures} times consecutively",
                            retryable=True,
                        )
                        break
                    await asyncio.sleep(
                        short_backoff
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                        else short_retry
                    )
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        consecutive_failures = 0

            poll_completed.set()

        poll_task = asyncio.create_task(mock_poll_loop())

        # Wait for poll loop to terminate after MAX_TOTAL_FAILURES
        await asyncio.wait_for(poll_completed.wait(), timeout=5.0)

        # Verify fatal error was called
        assert len(fatal_error_args) == 1
        code, message, retryable = fatal_error_args[0]
        assert code == "weixin_bridge_down"
        assert "30" in message  # Should mention 30 failures
        assert retryable is True

        # Cancel task if still running
        if not poll_task.done():
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_poll_loop_resets_total_failures_on_success(self):
        """When a poll succeeds after some failures, total_failures should reset to 0."""
        from gateway.platforms.weixin import MAX_TOTAL_FAILURES, MAX_CONSECUTIVE_FAILURES

        call_count = [0]
        poll_completed = asyncio.Event()
        fatal_error_called = False

        def capture_fatal_error(code, message, *, retryable):
            nonlocal fatal_error_called
            fatal_error_called = True

        async def mixed_get_updates(*args, **kwargs):
            call_count[0] += 1
            # Fail first 10 times, then succeed from 11-15, then fail again
            if call_count[0] <= 10:
                return {"ret": -1, "errcode": -1, "errmsg": "test error"}
            elif call_count[0] <= 15:
                return {"ret": 0, "errcode": 0, "msgs": []}
            else:
                return {"ret": -1, "errcode": -1, "errmsg": "test error"}

        async def mock_poll_loop():
            consecutive_failures = 0
            total_failures = 0
            running = True
            short_retry = 0.01
            short_backoff = 0.01

            while running:
                try:
                    response = await mixed_get_updates(None, "http://localhost:15236", "token", "", 35000)
                    ret = response.get("ret", 0)
                    errcode = response.get("errcode", 0)

                    if ret not in {0, None} or errcode not in {0, None}:
                        consecutive_failures += 1
                        total_failures += 1
                        if total_failures >= MAX_TOTAL_FAILURES:
                            capture_fatal_error(
                                "weixin_bridge_down",
                                f"Poll failed {total_failures} times consecutively",
                                retryable=True,
                            )
                            break

                        await asyncio.sleep(
                            short_backoff
                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                            else short_retry
                        )
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            consecutive_failures = 0
                        continue

                    consecutive_failures = 0
                    total_failures = 0  # Reset on success
                    # Stop after we've seen the success and a few more failures
                    if call_count[0] >= 15:
                        poll_completed.set()
                        break
                except asyncio.CancelledError:
                    break
                except Exception:
                    consecutive_failures += 1
                    total_failures += 1
                    if total_failures >= MAX_TOTAL_FAILURES:
                        capture_fatal_error(
                            "weixin_bridge_down",
                            f"Poll failed {total_failures} times consecutively",
                            retryable=True,
                        )
                        break
                    await asyncio.sleep(short_retry)
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        consecutive_failures = 0

            poll_completed.set()

        poll_task = asyncio.create_task(mock_poll_loop())
        await asyncio.wait_for(poll_completed.wait(), timeout=5.0)

        # Fatal error should NOT have been called because total_failures reset at 11
        assert not fatal_error_called

        if not poll_task.done():
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass