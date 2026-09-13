"""Integration test for the codex_app_server runtime path through AIAgent.

Verifies that:
  - api_mode='codex_app_server' is accepted on AIAgent construction
  - run_conversation() takes the early-return path and never enters the
    chat completions loop
  - Projected messages from a fake Codex session land in the messages list
  - tool_iterations from the codex session tick the skill nudge counter
  - Memory nudge counter ticks once per turn
  - The returned dict has the same shape as the chat_completions path
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult


@pytest.fixture
def fake_session(monkeypatch):
    """Replace CodexAppServerSession with a stub that returns a fixed
    TurnResult, so we can drive AIAgent without spawning real codex."""

    def fake_run_turn(self, user_input: str, **kwargs):
        return TurnResult(
            final_text=f"echo: {user_input}",
            projected_messages=[
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "exec_1", "type": "function",
                                 "function": {"name": "exec_command",
                                              "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "exec_1", "content": "ok"},
                {"role": "assistant", "content": f"echo: {user_input}"},
            ],
            tool_iterations=1,
            interrupted=False,
            error=None,
            turn_id="turn-stub-1",
            thread_id="thread-stub-1",
        )

    monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
    monkeypatch.setattr(
        CodexAppServerSession, "ensure_started", lambda self: "thread-stub-1"
    )


def _make_codex_agent(**kwargs):
    """Construct an AIAgent in codex_app_server mode without contacting any
    real provider. We pass api_mode explicitly so the constructor takes the
    fast path for direct credentials."""
    return run_agent.AIAgent(
        api_key="stub",
        base_url="https://stub.invalid",
        provider="openai",
        api_mode="codex_app_server",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        **kwargs,
    )


class TestApiModeAccepted:
    def test_api_mode_is_codex_app_server(self):
        agent = _make_codex_agent()
        assert agent.api_mode == "codex_app_server"


class TestRunConversationCodexPath:
    def test_prestart_hard_interrupt_prevents_codex_session_creation(self, fake_session):
        agent = _make_codex_agent()
        agent.hard_interrupt("parent timed out")

        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("must not start")

        assert result["interrupted"] is True
        assert result["completed"] is False
        assert result["interrupt_message"] == "parent timed out"
        assert getattr(agent, "_codex_session", None) is None

    def test_turn_exception_obeys_cancel_fence_and_redacts_failure(
        self, monkeypatch, caplog
    ):
        from agent import codex_runtime

        sensitive = "private /Users/owner/codex.sock?token=child-secret"
        begin_called = threading.Event()

        def raising_run_turn(self, user_input: str, **kwargs):
            raise RuntimeError(sensitive)

        def cancel_wins(self, agent):
            begin_called.set()
            return False

        monkeypatch.setattr(CodexAppServerSession, "run_turn", raising_run_turn)
        monkeypatch.setattr(
            CodexAppServerSession,
            "ensure_started",
            lambda self: "thread-error-fence",
        )
        monkeypatch.setattr(
            codex_runtime._CodexTerminalCommitFence,
            "begin_commit",
            cancel_wins,
        )
        agent = _make_codex_agent()

        with caplog.at_level("ERROR"):
            result = agent.run_conversation("must not publish failure")

        assert begin_called.is_set()
        assert result["interrupted"] is True
        assert result["completed"] is False
        assert result["partial"] is True
        assert result["final_response"] == ""
        assert result["error"] is None
        assert sensitive not in caplog.text
        assert "child-secret" not in str(result)

    def test_cancel_winner_prevents_all_codex_terminal_mutations(self, monkeypatch):
        from agent import codex_runtime

        def fake_run_turn(self, user_input: str, **kwargs):
            return TurnResult(
                final_text="must not commit",
                projected_messages=[
                    {"role": "assistant", "content": "must not commit"}
                ],
                tool_iterations=3,
                turn_id="turn-cancel-wins",
                thread_id="thread-cancel-wins",
            )

        begin_entered = threading.Event()
        begin_release = threading.Event()
        cancel_admitted = threading.Event()
        original_begin = codex_runtime._CodexTerminalCommitFence.begin_commit
        original_cancel = (
            codex_runtime._CodexTerminalCommitFence.cancel_before_commit
        )

        def blocked_begin(self, agent):
            agent._flush_messages_to_session_db.reset_mock()
            begin_entered.set()
            assert begin_release.wait(timeout=2)
            return original_begin(self, agent)

        def recording_cancel(self, admit_cancel):
            result = original_cancel(self, admit_cancel)
            cancel_admitted.set()
            return result

        monkeypatch.setattr(
            codex_runtime._CodexTerminalCommitFence,
            "begin_commit",
            blocked_begin,
        )
        monkeypatch.setattr(
            codex_runtime._CodexTerminalCommitFence,
            "cancel_before_commit",
            recording_cancel,
        )
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(
            CodexAppServerSession,
            "ensure_started",
            lambda self: "thread-cancel-wins",
        )

        agent = _make_codex_agent()
        agent._iters_since_skill = 7
        agent._session_db = MagicMock()
        agent._flush_messages_to_session_db = MagicMock(return_value=True)
        result_holder = {}

        with patch.object(
            codex_runtime, "_record_codex_app_server_compaction"
        ) as record_compaction, patch.object(
            codex_runtime, "_record_codex_app_server_usage"
        ) as record_usage, patch.object(
            agent, "_sync_external_memory_for_turn"
        ) as memory_sync, patch.object(
            agent, "_spawn_background_review"
        ) as spawn_review:
            run_thread = threading.Thread(
                target=lambda: result_holder.setdefault(
                    "result", agent.run_conversation("hello")
                )
            )
            run_thread.start()
            assert begin_entered.wait(timeout=2)

            interrupt_thread = threading.Thread(
                target=lambda: agent.hard_interrupt("owner cancelled")
            )
            interrupt_thread.start()
            assert cancel_admitted.wait(timeout=2)
            begin_release.set()
            run_thread.join(timeout=2)
            interrupt_thread.join(timeout=2)

        assert not run_thread.is_alive()
        assert not interrupt_thread.is_alive()
        assert result_holder["result"]["interrupted"] is True
        assert agent._iters_since_skill == 7
        assert not any(
            message.get("content") == "must not commit"
            for message in result_holder["result"]["messages"]
        )
        agent._flush_messages_to_session_db.assert_not_called()
        record_compaction.assert_not_called()
        record_usage.assert_not_called()
        memory_sync.assert_not_called()
        spawn_review.assert_not_called()

    def test_parent_interrupt_surfaces_fixed_session_teardown_failure(self, caplog):
        class FailingInterruptSession:
            @staticmethod
            def request_interrupt():
                raise RuntimeError("/private/provider/path?token=secret-value")

        agent = _make_codex_agent()
        agent._codex_session = FailingInterruptSession()

        with caplog.at_level("DEBUG"), pytest.raises(RuntimeError) as exc_info:
            agent.hard_interrupt("owner cancelled")

        assert str(exc_info.value) == (
            "codex app-server cancellation could not be confirmed"
        )
        assert "secret-value" not in caplog.text
        assert "/private/provider/path" not in caplog.text

    @pytest.mark.parametrize("method_name", ["interrupt", "hard_interrupt"])
    def test_parent_interrupt_surfaces_fixed_child_failure_after_complete_fanout(
        self,
        caplog,
        method_name,
    ):
        class FailingChild:
            @staticmethod
            def interrupt(_message=None):
                raise RuntimeError("secret=/Users/private/child-token")

            @staticmethod
            def hard_interrupt(_message=None):
                raise RuntimeError("secret=/Users/private/child-token")

        class RecordingChild:
            interrupted = False

            def interrupt(self, _message=None):
                self.interrupted = True

            def hard_interrupt(self, _message=None):
                self.interrupted = True

        agent = _make_codex_agent()
        healthy_child = RecordingChild()
        getattr(agent, "_active_children").extend((FailingChild(), healthy_child))

        with caplog.at_level("DEBUG"), pytest.raises(RuntimeError) as exc_info:
            getattr(agent, method_name)("owner cancelled")

        assert str(exc_info.value) == (
            "child agent cancellation could not be confirmed"
        )
        assert healthy_child.interrupted is True
        assert "/Users/private" not in caplog.text
        assert "child-token" not in caplog.text

    def test_parent_hard_interrupt_treats_unsupported_child_as_failed_fanout(self):
        class UnsupportedChild:
            pass

        class RecordingChild:
            interrupted = False

            def hard_interrupt(self, _message=None):
                self.interrupted = True

        agent = _make_codex_agent()
        healthy_child = RecordingChild()
        agent._active_children.extend((UnsupportedChild(), healthy_child))

        with pytest.raises(RuntimeError) as exc_info:
            agent.hard_interrupt("owner cancelled")

        assert str(exc_info.value) == (
            "child agent cancellation could not be confirmed"
        )
        assert healthy_child.interrupted is True

    def test_parent_hard_interrupt_treats_false_child_result_as_failed_fanout(self):
        class UnconfirmedChild:
            @staticmethod
            def hard_interrupt(_message=None):
                return False

        class RecordingChild:
            interrupted = False

            def hard_interrupt(self, _message=None):
                self.interrupted = True
                return True

        agent = _make_codex_agent()
        healthy_child = RecordingChild()
        getattr(agent, "_active_children").extend((UnconfirmedChild(), healthy_child))

        with pytest.raises(RuntimeError) as exc_info:
            agent.hard_interrupt("owner cancelled")

        assert str(exc_info.value) == (
            "child agent cancellation could not be confirmed"
        )
        assert healthy_child.interrupted is True

    def test_parent_interrupt_treats_false_child_result_as_failed_fanout(self):
        class UnconfirmedChild:
            @staticmethod
            def interrupt(_message=None):
                return False

        class RecordingChild:
            interrupted = False

            def interrupt(self, _message=None):
                self.interrupted = True
                return True

        agent = _make_codex_agent()
        healthy_child = RecordingChild()
        getattr(agent, "_active_children").extend((UnconfirmedChild(), healthy_child))

        with pytest.raises(RuntimeError) as exc_info:
            agent.interrupt("owner cancelled")

        assert str(exc_info.value) == (
            "child agent cancellation could not be confirmed"
        )
        assert healthy_child.interrupted is True

    def test_run_conversation_returns_codex_shape(self, fake_session):
        agent = _make_codex_agent()
        # No background review fork during tests
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hello there")
        assert result["final_response"] == "echo: hello there"
        assert result["completed"] is True
        assert result["partial"] is False
        assert result["error"] is None
        assert result["api_calls"] == 1
        assert result["codex_thread_id"] == "thread-stub-1"
        assert result["codex_turn_id"] == "turn-stub-1"

    def test_terminal_commit_wins_while_parent_interrupt_waits_for_handoff(
        self, monkeypatch
    ):
        from agent import codex_runtime

        def fake_run_turn(self, user_input: str, **kwargs):
            return TurnResult(
                final_text="publish after commit wins",
                projected_messages=[
                    {"role": "assistant", "content": "publish after commit wins"}
                ],
                tool_iterations=1,
                turn_id="turn-handoff-1",
                thread_id="thread-handoff-1",
            )

        handoff_entered = threading.Event()
        handoff_release = threading.Event()
        interrupt_finished = threading.Event()
        cancel_entered = threading.Event()
        result_holder = {}

        original_cancel = codex_runtime._CodexTerminalCommitFence.cancel_before_commit

        def observe_cancel(self, admit_cancel):
            cancel_entered.set()
            return original_cancel(self, admit_cancel)

        def block_handoff(*_args, **_kwargs):
            handoff_entered.set()
            handoff_release.wait(timeout=2)

        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "thread-handoff-1"
        )
        monkeypatch.setattr(
            codex_runtime, "_record_codex_app_server_compaction", block_handoff
        )
        monkeypatch.setattr(
            codex_runtime._CodexTerminalCommitFence,
            "cancel_before_commit",
            observe_cancel,
        )

        agent = _make_codex_agent()
        agent._skill_nudge_interval = 1
        agent.valid_tool_names = set(agent.valid_tool_names)
        agent.valid_tool_names.add("skill_manage")

        with patch.object(
            agent, "_sync_external_memory_for_turn"
        ) as memory_sync, patch.object(
            agent, "_spawn_background_review", return_value=None
        ) as spawn_review:
            run_thread = threading.Thread(
                target=lambda: result_holder.update(
                    result=agent.run_conversation("finish concurrently")
                )
            )
            run_thread.start()
            assert handoff_entered.wait(timeout=1)

            interrupt_thread = threading.Thread(
                target=lambda: (
                    agent.hard_interrupt("owner cancelled"),
                    interrupt_finished.set(),
                )
            )
            interrupt_thread.start()
            assert cancel_entered.wait(timeout=1)
            assert not interrupt_finished.is_set()

            handoff_release.set()
            run_thread.join(timeout=2)
            interrupt_thread.join(timeout=2)

        assert not run_thread.is_alive()
        assert not interrupt_thread.is_alive()
        assert result_holder["result"]["interrupted"] is False
        assert result_holder["result"]["completed"] is True
        assert result_holder["result"]["final_response"] == "publish after commit wins"
        memory_sync.assert_called_once()
        spawn_review.assert_called_once()

    def test_terminal_commit_winner_does_not_poison_next_turn(
        self, fake_session, monkeypatch
    ):
        from agent import codex_runtime

        memory_entered = threading.Event()
        memory_release = threading.Event()
        interrupt_finished = threading.Event()
        cancel_entered = threading.Event()
        first_result = {}

        original_cancel = codex_runtime._CodexTerminalCommitFence.cancel_before_commit

        def observe_cancel(self, admit_cancel):
            cancel_entered.set()
            return original_cancel(self, admit_cancel)

        def block_memory_sync(**_kwargs):
            memory_entered.set()
            memory_release.wait(timeout=2)

        monkeypatch.setattr(
            codex_runtime._CodexTerminalCommitFence,
            "cancel_before_commit",
            observe_cancel,
        )
        agent = _make_codex_agent()
        with patch.object(
            agent, "_sync_external_memory_for_turn", side_effect=block_memory_sync
        ), patch.object(
            agent, "_spawn_background_review", return_value=None
        ):
            run_thread = threading.Thread(
                target=lambda: first_result.update(
                    result=agent.run_conversation("commit this turn")
                )
            )
            run_thread.start()
            assert memory_entered.wait(timeout=1)

            interrupt_thread = threading.Thread(
                target=lambda: (
                    agent.hard_interrupt("too late"),
                    interrupt_finished.set(),
                )
            )
            interrupt_thread.start()
            assert cancel_entered.wait(timeout=1)
            assert not interrupt_finished.is_set()

            memory_release.set()
            run_thread.join(timeout=2)
            interrupt_thread.join(timeout=2)

        assert not run_thread.is_alive()
        assert not interrupt_thread.is_alive()
        assert first_result["result"]["completed"] is True
        assert agent._interrupt_requested is False

        with patch.object(agent, "_spawn_background_review", return_value=None):
            next_result = agent.run_conversation("next turn")
        assert next_result["completed"] is True
        assert next_result["interrupted"] is False

    def test_codex_app_server_token_usage_updates_session_accounting(self, monkeypatch):
        def fake_run_turn(self, user_input: str, **kwargs):
            return TurnResult(
                final_text="done",
                projected_messages=[{"role": "assistant", "content": "done"}],
                turn_id="turn-usage-1",
                thread_id="thread-usage-1",
                token_usage_last={
                    "totalTokens": 130,
                    "inputTokens": 80,
                    "cachedInputTokens": 20,
                    "outputTokens": 25,
                    "reasoningOutputTokens": 5,
                },
                model_context_window=200000,
            )

        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "thread-usage-1"
        )
        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hello")

        assert result["api_calls"] == 1
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 25
        assert result["total_tokens"] == 130
        assert result["input_tokens"] == 80
        assert result["output_tokens"] == 25
        assert result["cache_read_tokens"] == 20
        assert result["cache_write_tokens"] == 0
        assert result["reasoning_tokens"] == 5
        assert result["last_prompt_tokens"] == 100

        assert agent.session_api_calls == 1
        assert agent.session_prompt_tokens == 100
        assert agent.session_completion_tokens == 25
        assert agent.session_total_tokens == 130
        assert agent.session_input_tokens == 80
        assert agent.session_output_tokens == 25
        assert agent.session_cache_read_tokens == 20
        assert agent.session_cache_write_tokens == 0
        assert agent.session_reasoning_tokens == 5
        assert agent.context_compressor.last_prompt_tokens == 100
        assert agent.context_compressor.last_completion_tokens == 25
        assert agent.context_compressor.last_total_tokens == 130
        assert agent.context_compressor.context_length == 200000

    def test_native_codex_compaction_updates_bookkeeping(self, monkeypatch):
        def fake_run_turn(self, user_input: str, **kwargs):
            return TurnResult(
                final_text="done",
                projected_messages=[{"role": "assistant", "content": "done"}],
                turn_id="turn-compact-1",
                thread_id="thread-compact-1",
                compacted=True,
                token_usage_last={
                    "totalTokens": 300_000,
                    "inputTokens": 300_000,
                    "cachedInputTokens": 0,
                    "outputTokens": 0,
                    "reasoningOutputTokens": 0,
                },
            )

        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "thread-compact-1"
        )
        events = []
        agent = _make_codex_agent(event_callback=lambda name, payload: events.append((name, payload)))

        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert agent.context_compressor.compression_count == 1
        # A compacted turn with real usage is judged against that same real
        # prompt count, exactly like a normal completed compression boundary.
        assert agent.context_compressor.last_prompt_tokens == 300_000
        assert agent.context_compressor.awaiting_real_usage_after_compression is False
        assert agent.context_compressor._ineffective_compression_count == 1
        assert events == [
            (
                "session:compress",
                {
                    "platform": "",
                    "session_id": agent.session_id,
                    "old_session_id": "",
                    "in_place": False,
                    "compression_count": 1,
                    "runtime": "codex_app_server",
                    "thread_id": "thread-compact-1",
                    "turn_id": "turn-compact-1",
                },
            )
        ]

    def test_native_compaction_retains_session_after_unconfirmed_teardown(
        self, monkeypatch
    ):
        from agent import conversation_compression

        class FailingSession:
            @staticmethod
            def compact_thread():
                return SimpleNamespace(
                    interrupted=True,
                    error="compact failed",
                    should_retire=True,
                )

            @staticmethod
            def close():
                raise RuntimeError("private teardown detail")

        heartbeat = MagicMock()
        heartbeat.start.return_value = heartbeat
        monkeypatch.setattr(
            conversation_compression,
            "_CompressionActivityHeartbeat",
            lambda _agent: heartbeat,
        )
        session = FailingSession()
        agent = SimpleNamespace(
            _codex_session=session,
            _codex_session_lock=threading.Lock(),
            _cached_system_prompt="system",
            _emit_status=MagicMock(),
            _emit_warning=MagicMock(),
            session_id="session",
        )
        messages = [{"role": "user", "content": "hello"}]

        returned_messages, returned_prompt = (
            conversation_compression._compress_context_via_codex_app_server(
                agent,
                messages,
                "system",
                force=True,
            )
        )

        assert returned_messages is messages
        assert returned_prompt == "system"
        assert agent._codex_session is session

    def test_native_compaction_retirement_cannot_clear_replacement_session(
        self, monkeypatch
    ):
        from agent import conversation_compression

        class RetiringSession:
            @staticmethod
            def compact_thread():
                return SimpleNamespace(
                    interrupted=True,
                    error="compact failed",
                    should_retire=True,
                )

            @staticmethod
            def close():
                return None

        class RacingAgent:
            def __init__(self, session):
                self._session = session
                self._codex_session_lock = threading.Lock()
                self._session_reads = 0
                self.identity_checked = threading.Event()
                self.replacement_published = threading.Event()
                self._cached_system_prompt = "system"
                self._emit_status = MagicMock()
                self._emit_warning = MagicMock()
                self.session_id = "session"

            @property
            def _codex_session(self):
                value = self._session
                self._session_reads += 1
                if self._session_reads == 2 and not self._codex_session_lock.locked():
                    self.identity_checked.set()
                    self.replacement_published.wait()
                return value

            @_codex_session.setter
            def _codex_session(self, value):
                self._session = value

        heartbeat = MagicMock()
        heartbeat.start.return_value = heartbeat
        monkeypatch.setattr(
            conversation_compression,
            "_CompressionActivityHeartbeat",
            lambda _agent: heartbeat,
        )
        session = RetiringSession()
        replacement = object()
        agent = RacingAgent(session)

        def replace_session():
            agent.identity_checked.wait()
            with agent._codex_session_lock:
                agent._codex_session = replacement
            agent.replacement_published.set()

        replacement_thread = threading.Thread(target=replace_session)
        replacement_thread.start()
        messages = [{"role": "user", "content": "hello"}]

        returned_messages, returned_prompt = (
            conversation_compression._compress_context_via_codex_app_server(
                agent,
                messages,
                "system",
                force=True,
            )
        )
        agent.identity_checked.set()
        replacement_thread.join()

        assert returned_messages is messages
        assert returned_prompt == "system"
        assert agent._codex_session is replacement

    def test_projected_messages_are_spliced(self, fake_session):
        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hello")
        msgs = result["messages"]
        # User message + 3 projected (assistant tool_call + tool + assistant text)
        assert len(msgs) >= 4
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"
        # Last assistant message has the final text
        final = [m for m in msgs if m.get("role") == "assistant"
                 and m.get("content") == "echo: hello"]
        assert final, f"expected final assistant message in {msgs}"

    def test_projected_messages_are_synced_to_external_memory(self, fake_session):
        agent = _make_codex_agent()
        agent._memory_manager = MagicMock()
        agent._memory_manager.build_system_prompt.return_value = ""

        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hello")

        agent._memory_manager.sync_all.assert_called_once()
        assert agent._memory_manager.sync_all.call_args.kwargs["messages"] == result["messages"]

    def test_nudge_counters_tick(self, fake_session):
        """The skill nudge counter must accumulate tool_iterations across
        turns. The memory nudge counter is gated on memory being configured
        (which we skip via skip_memory=True), so we don't assert on it here —
        a separate test below covers that path explicitly."""
        agent = _make_codex_agent()
        agent._iters_since_skill = 0
        agent._user_turn_count = 0
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("first")
        assert agent._iters_since_skill == 1  # one tool_iteration in fake turn
        # _user_turn_count is incremented by run_conversation pre-loop, not
        # by the codex helper — confirms we delegate that to the standard flow.
        assert agent._user_turn_count == 1
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("second")
        assert agent._iters_since_skill == 2
        assert agent._user_turn_count == 2

    def test_user_message_not_duplicated(self, fake_session):
        """Regression guard: the user message must appear exactly once in
        the messages list. The standard run_conversation pre-loop appends
        it, and the codex helper must NOT append again."""
        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("ping unique 12345")
        user_count = sum(
            1 for m in result["messages"]
            if m.get("role") == "user" and m.get("content") == "ping unique 12345"
        )
        assert user_count == 1, f"user message appeared {user_count}× in {result['messages']}"

    def test_background_review_NOT_invoked_below_threshold(self, fake_session):
        """A single turn shouldn't trigger background review — counters
        haven't reached the nudge interval (default 10)."""
        agent = _make_codex_agent()
        agent._memory_nudge_interval = 10
        agent._skill_nudge_interval = 10
        agent._iters_since_skill = 0
        with patch.object(agent, "_spawn_background_review",
                          return_value=None) as spawn:
            agent.run_conversation("ping")
        # Below threshold → review should NOT fire (was a real bug:
        # the helper was calling _spawn_background_review() with no
        # args after every turn, which would crash with TypeError).
        assert not spawn.called

    def test_background_review_skill_trigger_fires_above_threshold(
        self, monkeypatch
    ):
        """When tool iterations cross the skill nudge interval, the
        background review fires with review_skills=True and the right
        messages_snapshot signature."""
        from agent.transports.codex_app_server_session import (
            CodexAppServerSession, TurnResult,
        )
        # Make the fake session report 10 tool iterations in one turn
        # (matching the default skill threshold).
        def fake_run_turn(self, user_input: str, **kwargs):
            return TurnResult(
                final_text=f"echo: {user_input}",
                projected_messages=[
                    {"role": "assistant", "content": f"echo: {user_input}"},
                ],
                tool_iterations=10,
                turn_id="t1", thread_id="th1",
            )
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "th1"
        )

        agent = _make_codex_agent()
        agent._skill_nudge_interval = 10
        agent._iters_since_skill = 0
        # Make valid_tool_names include 'skill_manage' so the gate passes
        agent.valid_tool_names = set(getattr(agent, "valid_tool_names", set()))
        agent.valid_tool_names.add("skill_manage")

        with patch.object(agent, "_spawn_background_review",
                          return_value=None) as spawn:
            agent.run_conversation("do tool work")

        assert spawn.called, "skill threshold tripped but review didn't fire"
        # Verify the call signature matches what _spawn_background_review
        # actually expects — this is the regression guard for the original
        # bug where the codex path called it with no args at all.
        call = spawn.call_args
        assert "messages_snapshot" in call.kwargs
        assert isinstance(call.kwargs["messages_snapshot"], list)
        assert call.kwargs["review_skills"] is True
        # Counter should be reset after the review fires
        assert agent._iters_since_skill == 0

    def test_background_review_signature_never_breaks(self, fake_session):
        """Even when no trigger fires, the helper must never call
        _spawn_background_review with the wrong signature. Run a turn,
        then run another turn after manually tripping the skill counter
        and confirm the call shape is the kwargs-only form the function
        actually accepts."""
        agent = _make_codex_agent()
        agent._skill_nudge_interval = 1  # very low so any iter trips it
        agent._iters_since_skill = 0
        agent.valid_tool_names = set(getattr(agent, "valid_tool_names", set()))
        agent.valid_tool_names.add("skill_manage")

        with patch.object(agent, "_spawn_background_review",
                          return_value=None) as spawn:
            agent.run_conversation("first")
        # The fake session reports tool_iterations=1, which trips
        # _skill_nudge_interval=1. So review should fire.
        assert spawn.called
        # Critical invariant: positional args must be empty, all real
        # args must be kwargs (matching _spawn_background_review's
        # actual signature).
        call = spawn.call_args
        assert call.args == (), (
            f"expected no positional args, got {call.args!r} — "
            "would crash _spawn_background_review at runtime"
        )
        assert "messages_snapshot" in call.kwargs

    def test_chat_completions_loop_is_not_entered(self, fake_session):
        """The early-return must bypass the regular API call loop entirely.
        We confirm by patching the SDK call and asserting it's never invoked."""
        agent = _make_codex_agent()
        # The chat_completions loop calls self.client.chat.completions.create(...)
        # If our early-return works, that path is dead.
        with patch.object(agent, "client") as client_mock, patch.object(
            agent, "_spawn_background_review", return_value=None
        ):
            agent.run_conversation("hi")
        assert not client_mock.chat.completions.create.called

    def test_gateway_terminal_cwd_seeds_codex_thread_cwd(self, monkeypatch, tmp_path):
        """Gateway sessions set TERMINAL_CWD without pinning agent.session_cwd.
        Codex app-server must still start in that configured workspace instead
        of falling back to the Hermes daemon process cwd."""
        from agent.transports.codex_app_server_session import (
            CodexAppServerSession, TurnResult,
        )

        captured: dict[str, str] = {}

        def fake_init(self, **kwargs):
            captured["cwd"] = kwargs["cwd"]
            self._thread_id = "thread-stub-1"

        def fake_run_turn(self, user_input: str, **kwargs):
            return TurnResult(
                final_text="ok",
                projected_messages=[{"role": "assistant", "content": "ok"}],
                turn_id="turn-stub-1",
                thread_id="thread-stub-1",
            )

        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)

        agent = _make_codex_agent()
        assert agent.session_cwd is None
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("hi")

        assert captured["cwd"] == str(tmp_path)

    def _capture_routing_agent(self, monkeypatch):
        """Build a codex agent with a CodexAppServerSession stub that captures
        the request_routing passed at construction time, so we can assert how
        the gateway-context approval routing was resolved."""
        captured: dict = {}

        def fake_init(self, **kwargs):
            captured.update(kwargs)
            self._thread_id = "thread-stub-1"

        def fake_run_turn(self, user_input: str, **kwargs):
            return TurnResult(
                final_text="ok",
                projected_messages=[{"role": "assistant", "content": "ok"}],
                turn_id="turn-stub-1",
                thread_id="thread-stub-1",
            )

        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "thread-stub-1"
        )
        return captured

    def test_approvals_mode_off_auto_approves_codex_server_requests(
        self, monkeypatch
    ):
        """When the user disables Hermes approvals, codex app-server approval
        requests should not fail closed just because no interactive callback is
        wired (the typical gateway path). Codex's own sandbox permission
        profile remains the filesystem boundary."""
        captured = self._capture_routing_agent(monkeypatch)
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"approvals": {"mode": "off"}},
        ):
            agent = _make_codex_agent()
            with patch.object(
                agent, "_spawn_background_review", return_value=None
            ):
                agent.run_conversation("write something")
        routing = captured["request_routing"]
        assert routing.auto_approve_exec is True
        assert routing.auto_approve_apply_patch is True

    def test_yaml_boolean_false_approval_mode_also_auto_approves(
        self, monkeypatch
    ):
        """YAML 1.1 parses unquoted `off` as False; match the normal approval
        subsystem's compatibility behavior for codex app-server routing too."""
        captured = self._capture_routing_agent(monkeypatch)
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"approvals": {"mode": False}},
        ):
            agent = _make_codex_agent()
            with patch.object(
                agent, "_spawn_background_review", return_value=None
            ):
                agent.run_conversation("write something")
        routing = captured["request_routing"]
        assert routing.auto_approve_exec is True
        assert routing.auto_approve_apply_patch is True

    def test_manual_approvals_keep_codex_server_requests_fail_closed(
        self, monkeypatch
    ):
        """Default (manual) approvals must preserve the fail-closed behavior —
        this fix is a no-op for users who haven't opted out."""
        captured = self._capture_routing_agent(monkeypatch)
        with patch(
            "hermes_cli.config.load_config",
            return_value={"approvals": {"mode": "manual"}},
        ):
            agent = _make_codex_agent()
            with patch.object(
                agent, "_spawn_background_review", return_value=None
            ):
                agent.run_conversation("write something")
        routing = captured["request_routing"]
        assert routing.auto_approve_exec is False
        assert routing.auto_approve_apply_patch is False

    def test_frozen_yolo_env_auto_approves_codex_server_requests(
        self, monkeypatch
    ):
        """--yolo / HERMES_YOLO_MODE (frozen into _YOLO_MODE_FROZEN at import
        time — a prompt-injection-safe process-scoped bypass) should flow
        through to codex app-server routing so gateway/cron contexts do not
        fail closed when the user launched with yolo mode."""
        import tools.approval as _approval

        captured = self._capture_routing_agent(monkeypatch)
        monkeypatch.setattr(_approval, "_YOLO_MODE_FROZEN", True)
        with patch(
            "hermes_cli.config.load_config",
            return_value={"approvals": {"mode": "manual"}},
        ):
            agent = _make_codex_agent()
            with patch.object(
                agent, "_spawn_background_review", return_value=None
            ):
                agent.run_conversation("write something")
        routing = captured["request_routing"]
        assert routing.auto_approve_exec is True
        assert routing.auto_approve_apply_patch is True

    def test_session_yolo_auto_approves_codex_server_requests(
        self, monkeypatch
    ):
        """The /yolo session toggle should be honored at Codex session creation
        time, independent of the startup-time approvals config."""
        captured = self._capture_routing_agent(monkeypatch)
        with patch(
            "hermes_cli.config.load_config",
            return_value={"approvals": {"mode": "manual"}},
        ):
            agent = _make_codex_agent()
            with patch(
                "tools.approval.is_approval_bypass_active_for_session",
                return_value=True,
            ), patch.object(
                agent, "_spawn_background_review", return_value=None
            ):
                agent.run_conversation("write something")
        routing = captured["request_routing"]
        assert routing.auto_approve_exec is True
        assert routing.auto_approve_apply_patch is True


class TestReviewForkApiModeDowngrade:
    """When the parent agent runs on codex_app_server, the background
    review fork must downgrade to codex_responses — otherwise the fork
    can't dispatch agent-loop tools (memory, skill_manage) which is the
    whole point of the review."""

    def test_codex_app_server_parent_downgrades_review_fork(self):
        """Live test against the real _spawn_background_review code path:
        verify the review_agent gets api_mode=codex_responses when the
        parent is codex_app_server."""
        from unittest.mock import MagicMock, patch as _patch
        agent = _make_codex_agent()
        # Pretend memory + skills are configured so the review fork
        # reaches the AIAgent constructor.
        agent._memory_store = MagicMock()
        agent._memory_enabled = True
        agent._user_profile_enabled = True
        # Mock _current_main_runtime to return the parent's codex_app_server
        # state so we can confirm the helper detects + downgrades it.
        agent._current_main_runtime = lambda: {
            "api_mode": "codex_app_server",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "stub-token",
        }
        # Capture what AIAgent gets constructed with inside the helper.
        captured = {}

        def _capture_init(self, **kwargs):
            captured.update(kwargs)
            # Set bare attributes the rest of the spawn function reads
            # so it can finish without exploding.
            self.api_mode = kwargs.get("api_mode")
            self.provider = kwargs.get("provider")
            self.model = kwargs.get("model")
            self._memory_write_origin = None
            self._memory_write_context = None
            self._memory_store = None
            self._memory_enabled = False
            self._user_profile_enabled = False
            self._memory_nudge_interval = 0
            self._skill_nudge_interval = 0
            self.suppress_status_output = False
            self._session_messages = []

            def _no_op_run_conv(*a, **kw):
                return {"final_response": "", "messages": []}
            self.run_conversation = _no_op_run_conv

            def _no_op_close(*a, **kw):
                return None
            self.close = _no_op_close

        with _patch("run_agent.AIAgent.__init__", _capture_init):
            agent._spawn_background_review(
                messages_snapshot=[{"role": "user", "content": "x"}],
                review_memory=True,
                review_skills=False,
            )
            # Wait for the spawned thread to actually execute
            import time
            for _ in range(30):
                if "api_mode" in captured:
                    break
                time.sleep(0.1)

        assert captured.get("api_mode") == "codex_responses", (
            f"review fork should be downgraded to codex_responses when "
            f"parent is codex_app_server; got {captured.get('api_mode')!r}"
        )


class TestErrorHandling:
    def test_session_exception_returns_partial_with_error(self, monkeypatch):
        def boom_run_turn(self, user_input, **kwargs):
            raise RuntimeError("subprocess died")

        monkeypatch.setattr(CodexAppServerSession, "ensure_started",
                            lambda self: "t1")
        monkeypatch.setattr(CodexAppServerSession, "run_turn", boom_run_turn)

        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hi")
        assert result["completed"] is False
        assert result["partial"] is True
        assert result["error"] == "Codex app-server turn failed"
        assert "subprocess died" not in str(result)
        assert "codex-runtime auto" in result["final_response"]

    def test_interrupted_turn_marked_partial(self, monkeypatch):
        def interrupted_turn(self, user_input, **kwargs):
            return TurnResult(
                final_text="",
                projected_messages=[],
                tool_iterations=0,
                interrupted=True,
                error="user interrupted",
                turn_id="t",
                thread_id="th",
            )
        monkeypatch.setattr(CodexAppServerSession, "ensure_started",
                            lambda self: "th")
        monkeypatch.setattr(CodexAppServerSession, "run_turn", interrupted_turn)

        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hi")
        assert result["completed"] is False
        assert result["partial"] is True
        assert result["error"] == "user interrupted"


class TestSessionRetirementOnRunAgent:
    """run_agent.py side: when run_turn returns should_retire=True, the
    AIAgent must close + null _codex_session so the next turn respawns."""

    @pytest.mark.parametrize("method_name", ["release_clients", "close"])
    def test_agent_cleanup_retains_codex_session_until_close_is_confirmed(
        self, fake_session, monkeypatch, method_name
    ):
        attempts = []

        def flaky_close(session):
            attempts.append(session)
            assert getattr(agent, "_codex_session_lock").locked()
            if len(attempts) == 1:
                raise RuntimeError("private teardown detail")

        monkeypatch.setattr(CodexAppServerSession, "close", flaky_close)
        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("keep this normal session")
        session = getattr(agent, "_codex_session")

        getattr(agent, method_name)()

        assert getattr(agent, "_codex_session") is session

        getattr(agent, method_name)()

        assert getattr(agent, "_codex_session") is None

        getattr(agent, method_name)()

        assert attempts == [session, session]

    def test_should_retire_drops_session(self, monkeypatch):
        closes = {"count": 0}

        def fake_run_turn(self, user_input, **kwargs):
            return TurnResult(
                final_text="",
                projected_messages=[],
                tool_iterations=0,
                interrupted=True,
                error="turn timed out after 600.0s",
                turn_id="tu1",
                thread_id="th1",
                should_retire=True,
            )

        def fake_close(self):
            closes["count"] += 1

        monkeypatch.setattr(CodexAppServerSession, "ensure_started",
                            lambda self: "th1")
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(CodexAppServerSession, "close", fake_close)

        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hi")

        # The session was closed and cleared
        assert closes["count"] == 1
        assert getattr(agent, "_codex_session", "MISSING") is None
        # Partial result was still returned (caller still sees the error)
        assert result["partial"] is True
        assert result["error"] == "turn timed out after 600.0s"

    def test_should_retire_close_failure_is_fixed_and_session_retained(
        self, monkeypatch
    ):
        def fake_run_turn(self, user_input, **kwargs):
            return TurnResult(
                interrupted=True,
                error="turn timed out",
                turn_id="tu1",
                thread_id="th1",
                should_retire=True,
            )

        def failing_close(self):
            raise RuntimeError("/private/close/path?token=secret-value")

        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "th1"
        )
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(CodexAppServerSession, "close", failing_close)
        agent = _make_codex_agent()

        with pytest.raises(RuntimeError) as exc_info:
            agent.run_conversation("hi")

        assert str(exc_info.value) == "codex app-server session teardown unconfirmed"
        assert "secret-value" not in str(exc_info.value)
        assert getattr(agent, "_codex_session", None) is not None

    def test_normal_turn_keeps_session(self, fake_session):
        """fake_session fixture returns should_retire=False (default).
        The session must stay attached for the next turn to reuse."""
        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("hi")
        # Session was lazily created and still attached.
        assert getattr(agent, "_codex_session", None) is not None

    def test_exception_path_also_drops_session(self, monkeypatch):
        """Even if run_turn raises (not just sets should_retire), we must
        drop the session — a thrown exception is the strongest possible
        signal the process is dead."""
        closes = {"count": 0}

        def boom_run_turn(self, user_input, **kwargs):
            raise RuntimeError("codex segfaulted")

        def fake_close(self):
            closes["count"] += 1

        monkeypatch.setattr(CodexAppServerSession, "ensure_started",
                            lambda self: "th1")
        monkeypatch.setattr(CodexAppServerSession, "run_turn", boom_run_turn)
        monkeypatch.setattr(CodexAppServerSession, "close", fake_close)

        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hi")

        assert closes["count"] == 1
        assert agent._codex_session is None
        assert result["completed"] is False
        assert result["error"] == "Codex app-server turn failed"
        assert "codex segfaulted" not in str(result)

    def test_exception_path_close_failure_is_fixed_and_session_retained(
        self, monkeypatch
    ):
        def boom_run_turn(self, user_input, **kwargs):
            raise RuntimeError("codex turn failed")

        def failing_close(self):
            raise RuntimeError("/private/close/path?token=secret-value")

        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "th1"
        )
        monkeypatch.setattr(CodexAppServerSession, "run_turn", boom_run_turn)
        monkeypatch.setattr(CodexAppServerSession, "close", failing_close)
        agent = _make_codex_agent()

        with pytest.raises(RuntimeError) as exc_info:
            agent.run_conversation("hi")

        assert str(exc_info.value) == "codex app-server session teardown unconfirmed"
        assert "secret-value" not in str(exc_info.value)
        assert getattr(agent, "_codex_session", None) is not None


class TestCodexToolProgressBridge:
    """#38835 / #33200: Codex app-server item notifications must surface as
    Hermes tool-progress so gateways show verbose breadcrumbs on this route.
    The original item/started-only mapper was superseded by the full event
    bridge (make_codex_app_server_event_bridge); these tests pin the same
    mapping contract against the bridge helpers."""

    def test_mapper_command_execution(self):
        from agent.codex_runtime import (
            _codex_item_to_args,
            _codex_item_to_preview,
            _codex_item_to_tool_name,
        )
        item = {"type": "commandExecution", "command": "ls -la", "cwd": "/tmp"}
        assert _codex_item_to_tool_name(item) == "exec_command"
        assert _codex_item_to_preview(item) == "ls -la"
        assert _codex_item_to_args(item) == {"command": "ls -la", "cwd": "/tmp"}

    def test_mapper_file_change(self):
        from agent.codex_runtime import (
            _codex_item_to_preview,
            _codex_item_to_tool_name,
        )
        item = {
            "type": "fileChange",
            "changes": [{"path": "a.py"}, {"path": "b.py"}],
        }
        assert _codex_item_to_tool_name(item) == "apply_patch"
        assert _codex_item_to_preview(item) == "a.py, b.py"

    def test_mapper_mcp_and_dynamic_tool_calls(self):
        from agent.codex_runtime import (
            _codex_item_to_args,
            _codex_item_to_tool_name,
        )
        mcp = {"type": "mcpToolCall", "server": "fs", "tool": "read", "arguments": {"p": 1}}
        assert _codex_item_to_tool_name(mcp) == "mcp.fs.read"
        assert _codex_item_to_args(mcp) == {"p": 1}

        dyn = {"type": "dynamicToolCall", "tool": "web_search", "arguments": {"q": "x"}}
        assert _codex_item_to_tool_name(dyn) == "web_search"

    def test_bridge_ignores_non_tool_items_and_other_methods(self):
        from agent.codex_runtime import make_codex_app_server_event_bridge
        events = []
        agent = SimpleNamespace(
            tool_progress_callback=lambda *a, **kw: events.append(a),
            _fire_stream_delta=None,
            _fire_reasoning_delta=None,
            _emit_interim_assistant_message=None,
        )
        on_event = make_codex_app_server_event_bridge(agent)
        # agentMessage started items are not tool-shaped
        on_event({"method": "item/started", "params": {
            "item": {"type": "agentMessage", "text": "hi"}}})
        # malformed / empty notes
        on_event({"method": "item/completed", "params": {}})
        on_event({})
        assert events == []

    def test_session_wired_with_on_event_that_fires_tool_progress(self, monkeypatch):
        """The session is constructed with an on_event hook that, when fed an
        item/started note, calls the agent's tool_progress_callback."""
        captured_init = {}
        events = []

        def fake_init(self, **kwargs):
            captured_init.update(kwargs)
            # minimal attrs so the rest of run_turn stubs work
            self._client = None

        def fake_run_turn(self, user_input, **kwargs):
            # Exercise the wired on_event hook with a real item/started note.
            on_event = captured_init.get("on_event")
            if on_event:
                on_event({"method": "item/started", "params": {"item": {
                    "type": "commandExecution", "command": "pytest", "cwd": "/repo"}}})
            return TurnResult(final_text="done", projected_messages=[
                {"role": "assistant", "content": "done"}], turn_id="t1", thread_id="th1")

        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "th1")
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)

        agent = _make_codex_agent()
        agent.tool_progress_callback = lambda kind, name, preview, args: events.append(
            (kind, name, preview))
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("run the tests")

        assert "on_event" in captured_init and captured_init["on_event"] is not None
        assert ("tool.started", "exec_command", "pytest") in events
