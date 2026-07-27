"""Progress-aware timeout around in-agent compress_context (#72016).

In-loop / preflight / manual ``/compress`` paths historically waited on
``compress_context`` with no host-level inactivity budget. Gateway session
hygiene already had a progress-aware wait; these tests pin the same contract
for the owned wrapper used when callers do not pass a ``commit_fence``.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from agent.conversation_compression import (
    CompressionCommitFence,
    resolve_context_compression_timeouts,
    run_compress_context_with_progress_timeout,
)


class TestResolveContextCompressionTimeouts:
    def test_defaults_when_empty_cfg(self):
        idle, ceiling = resolve_context_compression_timeouts({})
        assert idle == 120.0
        assert ceiling == 600.0

    def test_zero_idle_disables_wrapper(self):
        idle, ceiling = resolve_context_compression_timeouts(
            {"context_timeout_seconds": 0}
        )
        assert idle == 0.0
        assert ceiling == 600.0

    def test_ceiling_clamped_to_idle(self):
        idle, ceiling = resolve_context_compression_timeouts(
            {
                "context_timeout_seconds": 90,
                "context_total_ceiling_seconds": 30,
            }
        )
        assert idle == 90.0
        assert ceiling == 90.0


class TestRunCompressContextWithProgressTimeout:
    def test_silent_worker_times_out_and_preserves_messages(self):
        original = [{"role": "user", "content": "keep-me"}]
        started = threading.Event()
        release = threading.Event()
        commit_attempted = threading.Event()

        def worker(fence: CompressionCommitFence):
            started.set()
            assert release.wait(timeout=2)
            if not fence.begin_commit():
                return ([{"role": "assistant", "content": "should-not-land"}], "x")
            try:
                commit_attempted.set()
                return ([{"role": "assistant", "content": "too-late"}], "x")
            finally:
                fence.finish_commit()

        warnings = []

        result_msgs, result_prompt = run_compress_context_with_progress_timeout(
            worker=worker,
            messages=original,
            system_prompt_fallback="fallback-prompt",
            idle_timeout_seconds=0.05,
            total_ceiling_seconds=0.2,
            on_timeout=lambda idle, waited, since: warnings.append(
                (idle, waited, since)
            ),
        )

        assert started.wait(timeout=1)
        # Give the waiter time to cancel before releasing the worker.
        time.sleep(0.15)
        release.set()
        # Worker may still be winding down; fence cancel must have won.
        deadline = time.time() + 1.0
        while time.time() < deadline and not commit_attempted.is_set():
            time.sleep(0.01)

        assert result_msgs is original
        assert result_prompt == "fallback-prompt"
        assert warnings, "timeout callback should fire"
        assert not commit_attempted.is_set(), (
            "cancelled fence must block late session mutation"
        )

    def test_progress_extends_idle_budget_until_success(self):
        original = [{"role": "user", "content": "a"}]
        compressed = [{"role": "user", "content": "summarized"}]
        fence_holder: dict = {}

        def worker(fence: CompressionCommitFence):
            fence_holder["fence"] = fence
            # Keep ticking within each idle window so the waiter extends.
            for _ in range(6):
                time.sleep(0.04)
                fence.touch_progress()
            if not fence.begin_commit():
                return (original, "aborted")
            try:
                return (compressed, "ok-prompt")
            finally:
                fence.finish_commit()

        result_msgs, result_prompt = run_compress_context_with_progress_timeout(
            worker=worker,
            messages=original,
            system_prompt_fallback="fallback",
            idle_timeout_seconds=0.1,
            total_ceiling_seconds=1.0,
        )

        assert result_msgs == compressed
        assert result_prompt == "ok-prompt"
        assert "fence" in fence_holder

    def test_commit_started_before_timeout_returns_worker_result(self):
        original = [{"role": "user", "content": "a"}]
        compressed = [{"role": "assistant", "content": "done"}]
        entered = threading.Event()

        def worker(fence: CompressionCommitFence):
            assert fence.begin_commit()
            entered.set()
            try:
                time.sleep(0.2)
                return (compressed, "committed")
            finally:
                fence.finish_commit()

        result_msgs, result_prompt = run_compress_context_with_progress_timeout(
            worker=worker,
            messages=original,
            system_prompt_fallback="fallback",
            idle_timeout_seconds=0.05,
            total_ceiling_seconds=0.05,
        )

        assert entered.wait(timeout=1)
        assert result_msgs == compressed
        assert result_prompt == "committed"

    def test_rejects_non_positive_idle(self):
        with pytest.raises(ValueError):
            run_compress_context_with_progress_timeout(
                worker=lambda fence: ([], ""),
                messages=[],
                system_prompt_fallback="",
                idle_timeout_seconds=0,
                total_ceiling_seconds=1,
            )


    def test_propagates_conversation_context_into_worker(self):
        from agent.portal_tags import (
            get_conversation_context,
            reset_conversation_context,
            set_conversation_context,
        )

        seen = {}
        token = set_conversation_context("conv-timeout-ctx")
        try:
            def worker(fence: CompressionCommitFence):
                seen["ctx"] = get_conversation_context()
                if not fence.begin_commit():
                    return ([], "")
                try:
                    return ([{"role": "user", "content": "ok"}], "p")
                finally:
                    fence.finish_commit()

            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=[{"role": "user", "content": "x"}],
                system_prompt_fallback="fallback",
                idle_timeout_seconds=1.0,
                total_ceiling_seconds=2.0,
            )
        finally:
            reset_conversation_context(token)

        assert seen.get("ctx") == "conv-timeout-ctx"
        assert prompt == "p"
        assert msgs[0]["content"] == "ok"

    def test_runs_worker_off_caller_thread(self):
        """Mirror gateway run_in_executor: compress work must leave the caller thread."""
        caller = threading.current_thread().ident
        seen = {}

        def worker(fence: CompressionCommitFence):
            seen["worker"] = threading.current_thread().ident
            if not fence.begin_commit():
                return ([], "")
            try:
                return ([{"role": "user", "content": "ok"}], "p")
            finally:
                fence.finish_commit()

        msgs, prompt = run_compress_context_with_progress_timeout(
            worker=worker,
            messages=[],
            system_prompt_fallback="",
            idle_timeout_seconds=1.0,
            total_ceiling_seconds=1.0,
        )
        assert seen.get("worker") is not None
        assert seen["worker"] != caller
        assert prompt == "p"
        assert msgs[0]["content"] == "ok"

    def test_reuses_module_shared_executor(self):
        from tools.daemon_pool import DaemonThreadPoolExecutor
        from agent import conversation_compression as mod

        first = mod._get_compress_timeout_executor()
        second = mod._get_compress_timeout_executor()
        assert first is second
        assert isinstance(first, DaemonThreadPoolExecutor)


class TestCompressContextForwarderOwnsTimeout:
    """AIAgent._compress_context wraps when no caller fence is supplied."""

    def test_owned_timeout_skips_hung_compress(self, monkeypatch):
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.session_id = "s1"
        agent._cached_system_prompt = "sys"
        agent._emit_warning = MagicMock()
        agent._touch_activity = MagicMock()
        agent._build_system_prompt = MagicMock(return_value="sys")
        agent._conversation_root_id = MagicMock(return_value=None)
        agent.context_compressor = MagicMock()
        agent.context_compressor._consecutive_timeout_failures = 0
        agent.context_compressor._record_compression_failure_cooldown = MagicMock()

        hang = threading.Event()
        calls = {"n": 0}

        def fake_compress(agent_obj, messages, system_message, **kwargs):
            calls["n"] += 1
            fence = kwargs.get("commit_fence")
            assert fence is not None
            hang.wait(timeout=2)
            if not fence.begin_commit():
                return messages, "sys"
            try:
                return ([{"role": "assistant", "content": "nope"}], "sys")
            finally:
                fence.finish_commit()

        monkeypatch.setattr(
            "agent.conversation_compression.compress_context",
            fake_compress,
        )
        monkeypatch.setattr(
            "agent.conversation_compression.resolve_context_compression_timeouts",
            lambda compression_cfg=None: (0.05, 0.2),
        )
        monkeypatch.setattr(
            "agent.portal_tags.get_conversation_context",
            lambda: object(),
        )

        original = [{"role": "user", "content": "stay"}]
        out_msgs, out_prompt = AIAgent._compress_context(
            agent, original, "sys"
        )
        hang.set()

        assert out_msgs is original
        assert out_prompt == "sys"
        assert calls["n"] == 1
        agent._emit_warning.assert_called_once()
        assert agent.context_compressor._consecutive_timeout_failures == 1
        agent.context_compressor._record_compression_failure_cooldown.assert_called_once()
        cooldown_args = (
            agent.context_compressor._record_compression_failure_cooldown.call_args[0]
        )
        assert cooldown_args[0] == 60.0
        assert "host compress_context timeout" in cooldown_args[1]
        from agent.session_activity import ActivityProvenance

        agent._touch_activity.assert_called_with(
            "context compression timed out",
            provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        )

    def test_fallback_prompt_resolved_lazily_on_timeout(self, monkeypatch):
        """Eager prompt rebuild must not run before compression starts."""
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.session_id = "s1"
        agent._cached_system_prompt = None
        agent._emit_warning = MagicMock()
        agent._touch_activity = MagicMock()
        agent._conversation_root_id = MagicMock(return_value=None)
        agent.context_compressor = MagicMock()
        agent.context_compressor._consecutive_timeout_failures = 0
        agent.context_compressor._record_compression_failure_cooldown = MagicMock()
        builds = {"n": 0}

        def boom_build(*_a, **_kw):
            builds["n"] += 1
            raise RuntimeError("prompt rebuild boom")

        agent._build_system_prompt = boom_build

        hang = threading.Event()

        def fake_compress(agent_obj, messages, system_message, **kwargs):
            hang.wait(timeout=2)
            fence = kwargs.get("commit_fence")
            if fence is not None and not fence.begin_commit():
                return messages, "sys"
            return messages, "sys"

        monkeypatch.setattr(
            "agent.conversation_compression.compress_context",
            fake_compress,
        )
        monkeypatch.setattr(
            "agent.conversation_compression.resolve_context_compression_timeouts",
            lambda compression_cfg=None: (0.05, 0.2),
        )
        monkeypatch.setattr(
            "agent.portal_tags.get_conversation_context",
            lambda: object(),
        )

        original = [{"role": "user", "content": "stay"}]
        out_msgs, out_prompt = AIAgent._compress_context(
            agent, original, "sys"
        )
        hang.set()

        assert out_msgs is original
        assert out_prompt == "sys"
        # Fallback rebuild runs only on the timeout return path.
        assert builds["n"] == 1
        agent._emit_warning.assert_called_once()

    def test_caller_fence_bypasses_owned_wrapper(self, monkeypatch):
        from run_agent import AIAgent

        agent = object.__new__(AIAgent)
        agent.session_id = "s1"
        agent._cached_system_prompt = "sys"
        agent._conversation_root_id = MagicMock(return_value=None)

        seen = {}

        def fake_compress(agent_obj, messages, system_message, **kwargs):
            seen["fence"] = kwargs.get("commit_fence")
            return ([{"role": "assistant", "content": "ok"}], "sys")

        monkeypatch.setattr(
            "agent.conversation_compression.compress_context",
            fake_compress,
        )
        # If the owned wrapper ran, this would raise — prove we never call it.
        monkeypatch.setattr(
            "agent.conversation_compression.run_compress_context_with_progress_timeout",
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("owned wrapper must not run")
            ),
        )
        monkeypatch.setattr(
            "agent.portal_tags.get_conversation_context",
            lambda: object(),
        )

        fence = CompressionCommitFence()
        msgs, prompt = AIAgent._compress_context(
            agent,
            [{"role": "user", "content": "x"}],
            "sys",
            commit_fence=fence,
        )
        assert seen["fence"] is fence
        assert prompt == "sys"
        assert msgs[0]["content"] == "ok"
