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


def _capture_correlated_on_event(captured_init, session, kwargs):
    """Wrap fake-session callbacks with protocol-real item correlation."""
    wrapped = dict(kwargs)
    raw_on_event = wrapped.get("on_event")
    if raw_on_event is not None:
        def correlated_on_event(note):
            if isinstance(note, dict) and str(note.get("method", "")).startswith("item/"):
                params = note.get("params")
                if isinstance(params, dict):
                    params.setdefault("threadId", getattr(session, "_thread_id", None))
                    params.setdefault("turnId", getattr(session, "_active_turn_id", None))
            raw_on_event(note)

        wrapped["on_event"] = correlated_on_event
    captured_init.update(wrapped)


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

    def test_codex_user_projection_is_not_duplicated_or_persisted(self, monkeypatch):
        def fake_run_turn(self, user_input, **kwargs):
            return TurnResult(
                final_text="done",
                projected_messages=[
                    {"role": "user", "content": "request\nEPHEMERAL_RECALL"},
                    {"role": "assistant", "content": "done"},
                ],
                turn_id="turn-user-1",
                thread_id="thread-user-1",
            )

        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        agent = _make_codex_agent()

        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("request")

        user_messages = [m for m in result["messages"] if m.get("role") == "user"]
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == "request"
        assert all("EPHEMERAL_RECALL" not in str(m) for m in result["messages"])

    def test_context_memory_and_resumed_history_reach_codex(self, monkeypatch):
        captured = {}

        def fake_init(self, **kwargs):
            captured["init"] = kwargs

        def fake_run_turn(self, user_input, **kwargs):
            captured["turn_input"] = user_input
            return TurnResult(
                final_text="done",
                projected_messages=[{"role": "assistant", "content": "done"}],
                turn_id="turn-context-1",
                thread_id="thread-context-1",
            )

        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        agent = _make_codex_agent()
        agent._memory_manager = MagicMock()
        agent._memory_manager.build_system_prompt.return_value = ""
        agent._memory_manager.on_turn_start.return_value = None
        agent._memory_manager.prefetch_all.return_value = "HONCHO RECALL"

        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation(
                "current request",
                system_message="HERMES SOUL MEMORY PROFILE",
                conversation_history=[
                    {"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"},
                ],
            )

        assert result["final_response"] == "done"
        assert "HERMES SOUL MEMORY PROFILE" in (
            captured["init"]["developer_instructions"]
        )
        assert captured["init"]["initial_history_items"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "old question"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "old answer"}],
            },
        ]
        assert "current request" in captured["turn_input"]
        assert "HONCHO RECALL" in captured["turn_input"]

    def test_hermes_clarify_callback_is_wired_to_codex_session(self, monkeypatch):
        captured = {}

        def fake_init(self, **kwargs):
            captured.update(kwargs)

        def fake_run_turn(self, user_input, **kwargs):
            return TurnResult(
                final_text="done",
                projected_messages=[{"role": "assistant", "content": "done"}],
                turn_id="turn-input-1",
                thread_id="thread-input-1",
            )

        clarify_callback = lambda question, choices: "chosen"  # noqa: E731
        clarify_cancel_callback = lambda: None  # noqa: E731
        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        agent = _make_codex_agent(
            clarify_callback=clarify_callback,
            clarify_cancel_callback=clarify_cancel_callback,
        )

        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("ask if needed")

        assert captured["clarify_callback"] is clarify_callback
        assert captured["clarify_cancel_callback"] is clarify_cancel_callback

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

    def test_hermes_user_skills_are_forwarded_to_codex(self, monkeypatch, tmp_path):
        captured: dict = {}
        skills_root = tmp_path / "skills"
        skills_root.mkdir()

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

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)

        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("hi")

        assert captured["extra_skill_roots"] == [str(skills_root)]

    def test_gateway_terminal_cwd_seeds_codex_thread_cwd(self, monkeypatch, tmp_path):
        """Gateway sessions set TERMINAL_CWD without stamping agent.session_cwd.
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
        assert not hasattr(agent, "session_cwd")
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
            "hermes_cli.config.load_config",
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
            "hermes_cli.config.load_config",
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
                "tools.approval.is_current_session_yolo_enabled",
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
        assert "subprocess died" in result["error"]
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

    def test_normal_turn_keeps_session(self, fake_session):
        """fake_session fixture returns should_retire=False (default).
        The session must stay attached for the next turn to reuse."""
        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("hi")
        # Session was lazily created and still attached.
        assert getattr(agent, "_codex_session", None) is not None

    def test_dynamic_tool_catalog_change_restarts_session(self, fake_session, monkeypatch):
        closes = {"count": 0}
        original_close = CodexAppServerSession.close

        def counted_close(self):
            closes["count"] += 1
            return original_close(self)

        monkeypatch.setattr(CodexAppServerSession, "close", counted_close)
        agent = _make_codex_agent()
        agent.valid_tool_names.discard("todo")
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("first")
            first_session = agent._codex_session
            agent.valid_tool_names.add("todo")
            agent.run_conversation("second")

        assert closes["count"] == 1
        assert agent._codex_session is not first_session

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
        assert "codex segfaulted" in result["error"]


class TestCodexToolProgressBridge:
    """#38835: Codex app-server item/started notifications must surface as
    Hermes tool-progress so gateways show verbose breadcrumbs on this route."""

    def test_mapper_command_execution(self):
        from agent.codex_runtime import _codex_note_to_tool_progress
        note = {"method": "item/started", "params": {"item": {
            "type": "commandExecution", "id": "cmd-map", "command": "ls -la", "cwd": "/tmp"}}}
        name, preview, args = _codex_note_to_tool_progress(note)
        assert name == "exec_command"
        assert preview == "ls -la"
        assert args == {"command": "ls -la", "cwd": "/tmp"}

    def test_mapper_file_change(self):
        from agent.codex_runtime import _codex_note_to_tool_progress
        note = {"method": "item/started", "params": {"item": {
            "type": "fileChange",
            "id": "patch-map",
            "changes": [{"path": "a.py"}, {"path": "b.py"}]}}}
        name, preview, args = _codex_note_to_tool_progress(note)
        assert name == "apply_patch"
        assert preview == "a.py, b.py"

    def test_mapper_mcp_and_dynamic_tool_calls(self):
        from agent.codex_runtime import _codex_note_to_tool_progress
        mcp = {"method": "item/started", "params": {"item": {
            "type": "mcpToolCall", "id": "mcp-map", "server": "fs", "tool": "read", "arguments": {"p": 1}}}}
        name, preview, args = _codex_note_to_tool_progress(mcp)
        assert name == "mcp.fs.read"
        assert preview == "read"
        assert args == {"p": 1}

        dyn = {"method": "item/started", "params": {"item": {
            "type": "dynamicToolCall", "id": "dyn-map", "namespace": "hermes", "tool": "web_search", "arguments": {"q": "x"}}}}
        assert _codex_note_to_tool_progress(dyn)[0] == "web_search"

        assert _codex_note_to_tool_progress({
            "method": "item/started",
            "params": {"item": {"type": "mcpToolCall", "id": "mcp-missing-tool", "server": "fs"}},
        }) is None
        assert _codex_note_to_tool_progress({
            "method": "item/started",
            "params": {"item": {"type": "dynamicToolCall", "id": "dyn-missing-tool"}},
        }) is None

    def test_dynamic_tool_identity_requires_and_binds_namespace(self):
        from agent.codex_runtime import _codex_tool_identity

        base = {
            "type": "dynamicToolCall",
            "id": "dyn-namespace",
            "tool": "memory",
            "arguments": {},
        }
        assert _codex_tool_identity(base) is None
        assert _codex_tool_identity({**base, "namespace": ""}) is None
        assert _codex_tool_identity({**base, "namespace": "   "}) is None

        hermes = _codex_tool_identity({**base, "namespace": "hermes"})
        foreign = _codex_tool_identity({**base, "namespace": "foreign"})
        assert hermes is not None
        assert foreign is not None
        assert hermes[0] != foreign[0]

    def test_mcp_tool_identity_encoding_is_collision_free(self):
        from agent.codex_runtime import _codex_tool_identity

        left = _codex_tool_identity(
            {
                "type": "mcpToolCall",
                "id": "same-provider-id",
                "server": "a__b",
                "tool": "c",
            }
        )
        right = _codex_tool_identity(
            {
                "type": "mcpToolCall",
                "id": "same-provider-id",
                "server": "a",
                "tool": "b__c",
            }
        )
        assert left is not None
        assert right is not None
        assert left[0] != right[0]

    @pytest.mark.parametrize(
        "item",
        [
            {"type": "commandExecution", "id": "cmd-1", "command": "pwd"},
            {"type": "fileChange", "id": "patch-1", "changes": []},
            {
                "type": "mcpToolCall",
                "id": "mcp-1",
                "server": "fs",
                "tool": "read",
                "arguments": {},
            },
            {
                "type": "dynamicToolCall",
                "id": "dyn-1",
                "namespace": "hermes",
                "tool": "memory",
                "arguments": {},
            },
        ],
    )
    def test_live_tool_id_matches_persisted_projector(self, item):
        from agent.codex_runtime import _codex_tool_identity
        from agent.transports.codex_event_projector import CodexEventProjector

        live_identity = _codex_tool_identity(item)
        persisted = CodexEventProjector().project(
            {"method": "item/completed", "params": {"item": item}}
        )

        assert live_identity is not None
        assert live_identity[0] == persisted.messages[0]["tool_calls"][0]["id"]
        assert live_identity[1] == persisted.messages[0]["tool_calls"][0]["function"]["name"]

    def test_mapper_ignores_non_tool_items_and_other_methods(self):
        from agent.codex_runtime import _codex_note_to_tool_progress
        # agentMessage / reasoning items are not tool-shaped
        assert _codex_note_to_tool_progress({"method": "item/started", "params": {
            "item": {"type": "agentMessage", "text": "hi"}}}) is None
        # non-item/started methods
        assert _codex_note_to_tool_progress({"method": "item/completed", "params": {}}) is None
        assert _codex_note_to_tool_progress({}) is None

    def test_session_wired_with_on_event_that_fires_tool_progress(self, monkeypatch):
        """The session is constructed with an on_event hook that, when fed an
        item/started note, calls the agent's tool_progress_callback."""
        captured_init = {}
        events = []

        def fake_init(self, **kwargs):
            _capture_correlated_on_event(captured_init, self, kwargs)
            # minimal attrs so the rest of run_turn stubs work
            self._client = None
            self._thread_id = "th1"
            self._active_turn_id = "t1"

        def fake_run_turn(self, user_input, **kwargs):
            # Exercise the wired on_event hook with a real item/started note.
            on_event = captured_init.get("on_event")
            if on_event:
                on_event({
                    "method": "turn/started",
                    "params": {"threadId": "th1", "turn": {"id": "t1"}},
                })
                on_event({"method": "item/started", "params": {"item": {
                    "type": "commandExecution", "id": "cmd-progress", "command": "pytest", "cwd": "/repo"}}})
            return TurnResult(final_text="done", projected_messages=[
                {"role": "assistant", "content": "done"}], turn_id="t1", thread_id="th1")

        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "th1")
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)

        agent = _make_codex_agent()
        setattr(
            agent,
            "tool_progress_callback",
            lambda kind, name, preview, args, **kwargs: events.append((kind, name, preview)),
        )
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("run the tests")

        assert "on_event" in captured_init and captured_init["on_event"] is not None
        assert ("tool.started", "exec_command", "pytest") in events

    def test_session_wired_on_event_streams_text_reasoning_and_tool_lifecycle(
        self, monkeypatch
    ):
        """Codex App Server's live notifications must reach the same callbacks
        used by every other provider, not wait for the terminal TurnResult."""
        captured_init = {}
        streamed = []
        reasoning = []
        starts = []
        completes = []
        progress_events = []

        def fake_init(self, **kwargs):
            _capture_correlated_on_event(captured_init, self, kwargs)
            self._client = None
            self._thread_id = "th-live"
            self._active_turn_id = "t-live"

        def fake_run_turn(self, user_input, **kwargs):
            on_event = captured_init["on_event"]
            on_event({
                "method": "turn/started",
                "params": {"threadId": "th-live", "turn": {"id": "t-live"}},
            })
            on_event({
                "method": "item/agentMessage/delta",
                "params": {"itemId": "msg-1", "delta": "I’ll inspect it now."},
            })
            on_event({
                "method": "item/reasoning/summaryTextDelta",
                "params": {"itemId": "reason-1", "delta": "Checking the source."},
            })
            item = {
                "type": "commandExecution",
                "id": "cmd-1",
                "command": "pytest -q",
                "cwd": "/repo",
            }
            on_event({"method": "item/started", "params": {"item": item}})
            # Replayed notifications must not duplicate live cards.
            on_event({"method": "item/started", "params": {"item": item}})
            on_event({
                "method": "item/commandExecution/outputDelta",
                "params": {"itemId": "cmd-1", "delta": "collecting tests...\n"},
            })
            on_event({
                "method": "item/commandExecution/outputDelta",
                "params": {"itemId": "cmd-1", "delta": " done\n"},
            })
            on_event({
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "commandExecution",
                        "id": "cmd-1",
                        "status": "completed",
                        "aggregatedOutput": "1 passed\n",
                        "exitCode": 0,
                    }
                },
            })
            # Completion payloads may omit start-time args, and replays must be
            # ignored while retaining the authoritative start correlation.
            on_event({
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "commandExecution",
                        "id": "cmd-1",
                        "status": "completed",
                        "aggregatedOutput": "1 passed\n",
                        "exitCode": 0,
                    }
                },
            })
            dynamic_item = {
                "type": "dynamicToolCall",
                "id": "dyn-1",
                "namespace": "hermes",
                "tool": "memory",
                "arguments": {"action": "add", "content": "smoke"},
            }
            on_event({"method": "item/started", "params": {"item": dynamic_item}})
            on_event({
                "method": "item/completed",
                "params": {
                    "item": {
                        **dynamic_item,
                        "namespace": "foreign",
                        "success": True,
                        "contentItems": [
                            {"type": "inputText", "text": "foreign result"}
                        ],
                    }
                },
            })
            on_event({
                "method": "item/completed",
                "params": {
                    "item": {
                        **dynamic_item,
                        "success": True,
                        "contentItems": [
                            {"type": "inputText", "text": "memory stored"}
                        ],
                    }
                },
            })
            return TurnResult(
                final_text="done",
                projected_messages=[{"role": "assistant", "content": "done"}],
                turn_id="t-live",
                thread_id="th-live",
            )

        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "th-live")
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)

        agent = _make_codex_agent()
        setattr(agent, "reasoning_callback", reasoning.append)
        agent.tool_progress_callback = (
            lambda event_type, name, preview, args, **kwargs: progress_events.append(
                (event_type, name, preview, args, kwargs)
            )
        )
        agent.tool_start_callback = lambda call_id, name, args: starts.append(
            (call_id, name, args)
        )
        agent.tool_complete_callback = lambda call_id, name, args, result: completes.append(
            (call_id, name, args, result)
        )
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("inspect", stream_callback=streamed.append)

        assert streamed == ["I’ll inspect it now."]
        assert reasoning == ["Checking the source."]
        assert progress_events[1] == (
            "tool.progress",
            "exec_command",
            "collecting tests...\n",
            {"command": "pytest -q", "cwd": "/repo"},
            {"tool_call_id": "codex_4_exec_cmd-1"},
        )
        assert progress_events[2][2] == "collecting tests...\n done\n"
        assert starts == [
            (
                "codex_4_exec_cmd-1",
                "exec_command",
                {"command": "pytest -q", "cwd": "/repo"},
            ),
            (
                "codex_21_dyn_6_hermes_6_memory_dyn-1",
                "memory",
                {"action": "add", "content": "smoke"},
            ),
        ]
        assert completes == [
            (
                "codex_4_exec_cmd-1",
                "exec_command",
                {"command": "pytest -q", "cwd": "/repo"},
                "1 passed\n",
            ),
            (
                "codex_21_dyn_6_hermes_6_memory_dyn-1",
                "memory",
                {"action": "add", "content": "smoke"},
                "memory stored",
            ),
        ]

    def test_live_lifecycle_rejects_conflicting_same_id_item_starts(
        self, monkeypatch
    ):
        captured_init = {}
        starts = []
        completes = []

        def fake_init(self, **kwargs):
            _capture_correlated_on_event(captured_init, self, kwargs)
            self._client = None
            self._thread_id = "th-live-conflict"
            self._active_turn_id = "turn-live-conflict"

        def fake_run_turn(self, user_input, **kwargs):
            on_event = captured_init["on_event"]
            on_event({
                "method": "turn/started",
                "params": {
                    "threadId": "th-live-conflict",
                    "turn": {"id": "turn-live-conflict"},
                },
            })
            on_event({
                "method": "item/started",
                "params": {"item": {"type": "plan", "id": "shared-plan"}},
            })
            on_event({
                "method": "item/started",
                "params": {
                    "item": {
                        "type": "dynamicToolCall",
                        "id": "shared-plan",
                        "namespace": "hermes",
                        "tool": "memory",
                        "arguments": {},
                    }
                },
            })
            on_event({
                "method": "item/completed",
                "params": {"item": {"type": "plan", "id": "shared-plan"}},
            })

            command = {
                "type": "commandExecution",
                "id": "shared-command",
                "command": "pwd",
                "cwd": "/repo",
            }
            on_event({"method": "item/started", "params": {"item": command}})
            on_event({
                "method": "item/started",
                "params": {"item": {"type": "plan", "id": "shared-command"}},
            })
            on_event({
                "method": "item/completed",
                "params": {
                    "item": {
                        **command,
                        "status": "completed",
                        "aggregatedOutput": "/repo\n",
                        "exitCode": 0,
                    }
                },
            })
            return TurnResult(
                final_text="done",
                projected_messages=[{"role": "assistant", "content": "done"}],
                turn_id="turn-live-conflict",
                thread_id="th-live-conflict",
            )

        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "th-live-conflict"
        )
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)

        agent = _make_codex_agent()
        setattr(
            agent,
            "tool_start_callback",
            lambda call_id, name, args: starts.append((call_id, name, args)),
        )
        setattr(
            agent,
            "tool_complete_callback",
            lambda call_id, name, args, result: completes.append(
                (call_id, name, args, result)
            ),
        )
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("inspect")

        assert starts == [
            (
                "codex_4_exec_shared-command",
                "exec_command",
                {"command": "pwd", "cwd": "/repo"},
            )
        ]
        assert completes == [
            (
                "codex_4_exec_shared-command",
                "exec_command",
                {"command": "pwd", "cwd": "/repo"},
                "/repo\n",
            )
        ]

    def test_live_lifecycle_ignores_stale_turn_reset_and_orphan_completion(
        self, monkeypatch
    ):
        """Only the active turn may reset lifecycle state, and completions
        without a displayed start must not create orphan tool cards."""
        captured_init = {}
        streamed = []
        reasoning = []
        starts = []
        completes = []

        def fake_init(self, **kwargs):
            _capture_correlated_on_event(captured_init, self, kwargs)
            self._client = None
            self._thread_id = "th-active"
            self._active_turn_id = "turn-active"

        def fake_run_turn(self, user_input, **kwargs):
            on_event = captured_init["on_event"]
            item = {
                "type": "commandExecution",
                "id": "cmd-correlated",
                "command": "pytest -q",
                "cwd": "/repo",
            }
            # Item events outside a bound turn must not create live cards or
            # stream assistant text.
            on_event({
                "method": "item/agentMessage/delta",
                "params": {"itemId": "msg-before", "delta": "before"},
            })
            on_event({"method": "item/started", "params": {"item": {
                **item, "id": "cmd-before-turn"}}})
            on_event(
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": "th-active",
                        "turn": {"id": "turn-active", "status": "inProgress"},
                    },
                }
            )
            on_event({
                "method": "item/agentMessage/delta",
                "params": {
                    "turnId": "turn-stale",
                    "itemId": "msg-stale",
                    "delta": "stale",
                },
            })
            on_event({
                "method": "item/agentMessage/delta",
                "params": {"itemId": "msg-valid", "delta": "valid"},
            })
            on_event({
                "method": "item/reasoning/summaryTextDelta",
                "params": {"itemId": "reason-valid", "delta": "valid reason"},
            })
            on_event({"method": "item/started", "params": {"item": item}})
            # A replay of the active turn/start must not clear a tool already
            # displayed during that same turn.
            on_event(
                {
                    "method": "turn/started",
                    "params": {
                        "threadId": "th-active",
                        "turn": {"id": "turn-active", "status": "inProgress"},
                    },
                }
            )
            on_event(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "th-active",
                        "turn": {"id": "turn-stale", "status": "completed"},
                    },
                }
            )
            # A stale turn completion must not clear deduplication or the
            # authoritative arguments captured at start.
            on_event({"method": "item/started", "params": {"item": item}})
            # Explicitly conflicting item metadata must not mutate the active
            # card, even when the item id itself matches.
            on_event({
                "method": "item/completed",
                "params": {
                    "threadId": "th-active",
                    "turnId": "turn-stale",
                    "item": {
                        "type": "commandExecution",
                        "id": "cmd-correlated",
                        "aggregatedOutput": "stale result\n",
                    },
                },
            })
            # The same provider item id cannot switch tool identity between
            # start and completion. This must leave the command card open for
            # its matching completion below.
            on_event({
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "dynamicToolCall",
                        "id": "cmd-correlated",
                        "namespace": "hermes",
                        "tool": "memory",
                        "success": True,
                    }
                },
            })
            on_event(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "type": "commandExecution",
                            "id": "cmd-correlated",
                            "aggregatedOutput": "1 passed\n",
                        }
                    },
                }
            )
            # Valid ID and tool shape are insufficient without a live start.
            on_event(
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "type": "commandExecution",
                            "id": "cmd-orphan",
                            "command": "whoami",
                            "aggregatedOutput": "gabriel\n",
                        }
                    },
                }
            )
            on_event({
                "method": "turn/completed",
                "params": {
                    "threadId": "th-active",
                    "turn": {"id": "turn-active", "status": "completed"},
                },
            })
            on_event({
                "method": "item/agentMessage/delta",
                "params": {"itemId": "msg-after", "delta": "after"},
            })
            on_event({"method": "item/started", "params": {"item": {
                **item, "id": "cmd-after-turn"}}})
            return TurnResult(
                final_text="done",
                projected_messages=[{"role": "assistant", "content": "done"}],
                turn_id="turn-active",
                thread_id="th-active",
            )

        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "th-active"
        )
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)

        agent = _make_codex_agent()
        setattr(agent, "reasoning_callback", reasoning.append)
        setattr(agent, "tool_start_callback", lambda *args: starts.append(args))
        setattr(agent, "tool_complete_callback", lambda *args: completes.append(args))
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("inspect", stream_callback=streamed.append)

        assert streamed == ["valid"]
        assert reasoning == ["valid reason"]
        assert starts == [
            (
                "codex_4_exec_cmd-correlated",
                "exec_command",
                {"command": "pytest -q", "cwd": "/repo"},
            )
        ]
        assert completes == [
            (
                "codex_4_exec_cmd-correlated",
                "exec_command",
                {"command": "pytest -q", "cwd": "/repo"},
                "1 passed\n",
            )
        ]

    def test_aborted_turn_clears_live_state_before_reused_session(self, monkeypatch):
        captured_init = {}
        streamed = []
        calls = 0

        def fake_init(self, **kwargs):
            _capture_correlated_on_event(captured_init, self, kwargs)
            self._client = None
            self._thread_id = "th-reused"
            self._active_turn_id = None

        def fake_run_turn(self, user_input, **kwargs):
            nonlocal calls
            calls += 1
            on_event = captured_init["on_event"]
            turn_id = f"turn-{calls}"
            self._active_turn_id = turn_id
            if calls == 2:
                # This metadata-less item arrives before the next turn starts.
                # Cached correlation from the aborted first turn must not let it
                # leak into the second turn's live stream.
                on_event({
                    "method": "item/agentMessage/delta",
                    "params": {"itemId": "stale-buffered", "delta": "LEAK"},
                })
            on_event({
                "method": "turn/started",
                "params": {"threadId": "th-reused", "turn": {"id": turn_id}},
            })
            on_event({
                "method": "item/agentMessage/delta",
                "params": {"itemId": f"message-{calls}", "delta": f"valid-{calls}"},
            })
            return TurnResult(
                final_text=f"done-{calls}",
                projected_messages=[{"role": "assistant", "content": f"done-{calls}"}],
                interrupted=calls == 1,
                error="codex reported turn_aborted" if calls == 1 else None,
                turn_id=turn_id,
                thread_id="th-reused",
                should_retire=False,
            )

        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(
            CodexAppServerSession, "ensure_started", lambda self: "th-reused"
        )
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)

        agent = _make_codex_agent()
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("first", stream_callback=streamed.append)
            agent.run_conversation("second", stream_callback=streamed.append)

        assert streamed == ["valid-1", "valid-2"]

    def test_live_tool_display_force_redacts_sensitive_args_and_results(
        self, monkeypatch
    ):
        """Live-only Codex callback payloads are a security boundary even when
        normal secret-redaction settings are disabled."""
        captured_init = {}
        starts = []
        completes = []
        progress_events = []

        def fake_init(self, **kwargs):
            _capture_correlated_on_event(captured_init, self, kwargs)
            self._client = None
            self._thread_id = "th-redact"
            self._active_turn_id = "t-redact"

        def fake_run_turn(self, user_input, **kwargs):
            on_event = captured_init["on_event"]
            on_event({
                "method": "turn/started",
                "params": {"threadId": "th-redact", "turn": {"id": "t-redact"}},
            })
            item = {
                "type": "commandExecution",
                "id": "cmd-secret",
                "command": (
                    "OPENAI_API_KEY=live-command-secret "
                    "HTTPS://user:live-userinfo-arg-secret@example.test/run?"
                    "api%5Fkey=live-query-arg-secret&mode=test"
                ),
                "cwd": "/repo",
            }
            on_event({"method": "item/started", "params": {"item": item}})
            on_event({
                "method": "item/commandExecution/outputDelta",
                "params": {
                    "itemId": "cmd-secret",
                    "delta": (
                        "OPENAI_API_KEY=live-output-secret "
                        "HtTpS://user:live-userinfo-output-secret@example.test/progress?"
                        "token=live-query-output-secret\n"
                    ),
                },
            })
            on_event({
                "method": "item/commandExecution/outputDelta",
                "params": {
                    "itemId": "cmd-secret",
                    "delta": "OPENAI_API_KEY=" + ("long-secret-marker-" * 180) + "\n",
                },
            })
            on_event({
                "method": "item/commandExecution/outputDelta",
                "params": {"itemId": "cmd-secret", "delta": "OPENAI_API_"},
            })
            on_event({
                "method": "item/commandExecution/outputDelta",
                "params": {
                    "itemId": "cmd-secret",
                    "delta": "KEY=split-boundary-secret\n",
                },
            })
            on_event({
                "method": "item/completed",
                "params": {
                    "item": {
                        **item,
                        "aggregatedOutput": (
                            "OPENAI_API_KEY=live-result-secret "
                            "HTTPS://example.test/result?"
                            "client%5Fsecret=live-query-result-secret"
                        ),
                    }
                },
            })
            mcp_item = {
                "type": "mcpToolCall",
                "id": "mcp-secret",
                "server": "example",
                "tool": "lookup",
                "arguments": {"query": "safe"},
            }
            on_event({"method": "item/started", "params": {"item": mcp_item}})
            on_event({
                "method": "item/completed",
                "params": {
                    "item": {
                        **mcp_item,
                        "result": {
                            "padding": "x" * 3924,
                            "api_key": (
                                "UNIQUESECRET0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            ),
                        },
                    }
                },
            })
            return TurnResult(
                final_text="done",
                projected_messages=[{"role": "assistant", "content": "done"}],
                turn_id="t-redact",
                thread_id="th-redact",
            )

        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "th-redact")
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)

        agent = _make_codex_agent()
        agent.tool_progress_callback = (
            lambda event_type, name, preview, args, **kwargs: progress_events.append(
                (event_type, name, preview, args, kwargs)
            )
        )
        agent.tool_start_callback = lambda call_id, name, args: starts.append(
            (call_id, name, args)
        )
        agent.tool_complete_callback = lambda call_id, name, args, result: completes.append(
            (call_id, name, args, result)
        )

        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("inspect")

        displayed = repr((progress_events, starts, completes))
        assert "live-command-secret" not in displayed
        assert "live-output-secret" not in displayed
        assert "live-result-secret" not in displayed
        assert "live-query-arg-secret" not in displayed
        assert "live-query-output-secret" not in displayed
        assert "live-query-result-secret" not in displayed
        assert "live-userinfo-arg-secret" not in displayed
        assert "live-userinfo-output-secret" not in displayed
        assert "long-secret-marker" not in displayed
        assert "split-boundary-secret" not in displayed
        assert "UNIQUESECRET" not in displayed
        assert "api%5Fkey=***" in displayed
        assert "token=***" in displayed
        assert "client%5Fsecret=***" in displayed
        assert starts[0][2]["command"] != "OPENAI_API_KEY=live-command-secret"
        assert completes[0][3] != "OPENAI_API_KEY=live-result-secret"

    def test_live_tool_display_redaction_is_forced(self, monkeypatch):
        import agent.redact
        from agent.codex_runtime import (
            _redact_codex_live_display_args,
            _redact_codex_live_display_text,
        )

        real_redact = agent.redact.redact_sensitive_text
        force_flags = []

        def record_redaction(value, *args, **kwargs):
            force_flags.append(kwargs.get("force"))
            return real_redact(value, *args, **kwargs)

        monkeypatch.setattr(agent.redact, "redact_sensitive_text", record_redaction)

        assert _redact_codex_live_display_text("ordinary output") == "ordinary output"
        assert _redact_codex_live_display_args({"command": "pwd"}) == {"command": "pwd"}
        assert force_flags and all(force_flags)

    def test_live_tool_display_redacts_all_uri_userinfo(self):
        from agent.codex_runtime import _redact_codex_live_display_text

        displayed = _redact_codex_live_display_text(
            "https://abc123@example.test/path "
            "ssh://user:abc123@host/x custom://:p@host.test/p"
        )

        assert "abc123" not in displayed
        assert displayed == (
            "https://***@example.test/path "
            "ssh://user:***@host/x custom://:***@host.test/p"
        )

    def test_live_tool_display_redacts_sensitive_query_for_arbitrary_uri_scheme(self):
        from agent.codex_runtime import _redact_codex_live_display_text

        displayed = _redact_codex_live_display_text(
            "custom://host/path?token=SUPERSECRET&safe=value"
        )

        assert "SUPERSECRET" not in displayed
        assert displayed == "custom://host/path?token=***&safe=value"

    def test_live_tool_display_fails_closed_when_redaction_fails(self, monkeypatch):
        import agent.redact
        from agent.codex_runtime import (
            _redact_codex_live_display_args,
            _redact_codex_live_display_text,
        )

        def fail_redaction(*_args, **_kwargs):
            raise RuntimeError("redaction unavailable")

        monkeypatch.setattr(agent.redact, "redact_sensitive_text", fail_redaction)

        assert _redact_codex_live_display_text("api_key=live-secret") == ""
        assert _redact_codex_live_display_args({"api_key": "live-secret"}) == {}

    def test_live_tool_display_fails_closed_on_falsy_redaction(self, monkeypatch):
        import agent.redact
        from agent.codex_runtime import (
            _redact_codex_live_display_args,
            _redact_codex_live_display_text,
        )

        for falsy in (None, False, ""):
            monkeypatch.setattr(
                agent.redact,
                "redact_sensitive_text",
                lambda *_args, _value=falsy, **_kwargs: _value,
            )
            assert _redact_codex_live_display_text("api_key=live-secret") == ""
            assert _redact_codex_live_display_args({"api_key": "live-secret"}) == {}

    def test_live_lifecycle_ignores_malformed_or_idless_tool_items(self, monkeypatch):
        captured_init = {}
        starts = []
        completes = []
        progress_events = []

        def fake_init(self, **kwargs):
            _capture_correlated_on_event(captured_init, self, kwargs)
            self._client = None
            self._thread_id = "th-idless"
            self._active_turn_id = "t-idless"

        def fake_run_turn(self, user_input, **kwargs):
            on_event = captured_init["on_event"]
            on_event({
                "method": "turn/started",
                "params": {"threadId": "th-idless", "turn": {"id": "t-idless"}},
            })
            item = {"type": "commandExecution", "command": "pwd", "cwd": "/repo"}
            on_event({"method": "item/started", "params": {"item": item}})
            on_event({
                "method": "item/completed",
                "params": {"item": {**item, "aggregatedOutput": "/repo"}},
            })
            malformed_item = {**item, "id": 0}
            on_event({"method": "item/started", "params": {"item": malformed_item}})
            on_event({
                "method": "item/completed",
                "params": {"item": {**malformed_item, "aggregatedOutput": "/repo"}},
            })
            return TurnResult(
                final_text="done",
                projected_messages=[{"role": "assistant", "content": "done"}],
                turn_id="t-idless",
                thread_id="th-idless",
            )

        monkeypatch.setattr(CodexAppServerSession, "__init__", fake_init)
        monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "th-idless")
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)

        agent = _make_codex_agent()
        agent.tool_progress_callback = (
            lambda event_type, name, preview, args, **kwargs: progress_events.append(
                (event_type, name, preview, args, kwargs)
            )
        )
        agent.tool_start_callback = lambda *args: starts.append(args)
        agent.tool_complete_callback = lambda *args: completes.append(args)
        with patch.object(agent, "_spawn_background_review", return_value=None):
            agent.run_conversation("inspect")

        assert progress_events == []
        assert starts == []
        assert completes == []
