import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextCompressionCandidate
from run_agent import AIAgent


class _AsyncEngine:
    def __init__(self, replacement=None):
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.threshold_tokens = 1_000_000
        self.protect_first_n = 3
        self.protect_last_n = 6
        self.compression_count = 0
        self.replacement = replacement
        self.applied = []
        self.discarded = []
        self.reset_count = 0
        self.session_start_calls = []

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def should_prepare_async_compression(self, prompt_tokens=None, messages=None):
        return True

    def prepare_async_compression(self, messages, current_tokens=None, focus_topic=None):
        return self.replacement or [{"role": "user", "content": "compressed"}]

    def on_async_compression_applied(self, candidate, **kwargs):
        self.applied.append((candidate, kwargs))

    def on_async_compression_discarded(self, candidate, reason, **kwargs):
        self.discarded.append((candidate, reason, kwargs))

    def on_session_reset(self):
        self.reset_count += 1

    def on_session_start(self, session_id, **kwargs):
        self.session_start_calls.append((session_id, kwargs))


class _BlockingAsyncEngine(_AsyncEngine):
    def __init__(self, replacement=None):
        super().__init__(replacement=replacement)
        self.entered = threading.Event()
        self.release = threading.Event()

    def prepare_async_compression(self, messages, current_tokens=None, focus_topic=None):
        self.entered.set()
        assert self.release.wait(timeout=2), "test did not release async prepare"
        return super().prepare_async_compression(
            messages,
            current_tokens=current_tokens,
            focus_topic=focus_topic,
        )


class _ThresholdAsyncEngine(_AsyncEngine):
    def __init__(self, *, threshold=10_000, replacement=None):
        super().__init__(replacement=replacement)
        self.threshold = threshold
        self.should_tokens = []
        self.prepare_tokens = []

    def should_prepare_async_compression(self, prompt_tokens=None, messages=None):
        self.should_tokens.append(prompt_tokens)
        return bool(prompt_tokens and prompt_tokens >= self.threshold)

    def prepare_async_compression(self, messages, current_tokens=None, focus_topic=None):
        self.prepare_tokens.append(current_tokens)
        return super().prepare_async_compression(
            messages,
            current_tokens=current_tokens,
            focus_topic=focus_topic,
        )


def _minimal_agent(engine):
    agent = AIAgent.__new__(AIAgent)
    agent.compression_enabled = True
    agent.context_compressor = engine
    agent._async_context_lock = threading.Lock()
    agent._async_context_thread = None
    agent._async_context_inflight_digest = None
    agent._pending_async_context_candidate = None
    agent._async_context_generation = 0
    agent._cached_system_prompt = ""
    agent.tools = []
    agent.session_id = "sess"
    return agent


def _mock_response(content, usage=None):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="test/model",
        usage=SimpleNamespace(**usage) if usage else None,
    )


def _make_conversation_agent(engine, *, session_db=None, session_id=None):
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "web search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.context_compressor = engine
    agent.compression_enabled = True
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    return agent


def _wait_for_pending(agent, timeout=2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if agent._pending_async_context_candidate is not None:
            return agent._pending_async_context_candidate
        time.sleep(0.01)
    return None


def test_async_context_prepare_stores_candidate():
    engine = _AsyncEngine()
    agent = _minimal_agent(engine)
    messages = [{"role": "user", "content": "hello"}]

    assert agent._maybe_start_async_context_compression(messages, approx_tokens=25_000)

    candidate = _wait_for_pending(agent)
    assert isinstance(candidate, ContextCompressionCandidate)
    assert candidate.messages == [{"role": "user", "content": "compressed"}]
    assert candidate.base_message_count == len(messages)
    assert candidate.base_digest == agent._context_messages_digest(messages)


def test_async_context_apply_validates_digest_and_appends_new_suffix(monkeypatch):
    engine = _AsyncEngine()
    agent = _minimal_agent(engine)
    base_messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "done"},
    ]
    suffix = [{"role": "user", "content": "new ask"}]
    candidate_messages = [{"role": "user", "content": "summary"}]
    agent._pending_async_context_candidate = ContextCompressionCandidate(
        messages=candidate_messages,
        base_message_count=len(base_messages),
        base_digest=agent._context_messages_digest(base_messages),
    )

    def _fake_finalize(**kwargs):
        assert kwargs["original_messages"] == base_messages + suffix
        assert kwargs["rewritten_messages"] == candidate_messages + suffix
        assert kwargs["source_label"] == "async context compression"
        assert kwargs["notify_memory_pre_compress"] is True
        return kwargs["rewritten_messages"], "new system"

    monkeypatch.setattr(agent, "_finalize_context_rewrite", _fake_finalize)

    new_messages, new_prompt, applied = agent._maybe_apply_async_context_candidate(
        base_messages + suffix,
        "system",
        task_id="task",
    )

    assert applied is True
    assert new_prompt == "new system"
    assert new_messages == candidate_messages + suffix
    assert engine.applied
    assert not engine.discarded


def test_async_context_prepare_fills_missing_digest_for_partial_candidate(monkeypatch):
    replacement = [
        {"role": "user", "content": "compressed prefix"},
        {"role": "assistant", "content": "prefix summarized"},
    ]
    engine = _AsyncEngine(
        replacement=ContextCompressionCandidate(
            messages=replacement,
            base_message_count=1,
            base_digest="",
        )
    )
    agent = _minimal_agent(engine)
    messages = [
        {"role": "user", "content": "old prefix"},
        {"role": "assistant", "content": "suffix answer"},
    ]

    assert agent._maybe_start_async_context_compression(messages, approx_tokens=25_000)
    candidate = _wait_for_pending(agent)

    assert isinstance(candidate, ContextCompressionCandidate)
    assert candidate.base_message_count == 1
    assert candidate.base_digest == agent._context_messages_digest(messages[:1])

    def _fake_finalize(**kwargs):
        assert kwargs["rewritten_messages"] == replacement + messages[1:]
        return kwargs["rewritten_messages"], "new system"

    monkeypatch.setattr(agent, "_finalize_context_rewrite", _fake_finalize)

    new_messages, _new_prompt, applied = agent._maybe_apply_async_context_candidate(
        messages,
        "system",
        task_id="task",
    )

    assert applied is True
    assert new_messages == replacement + messages[1:]
    assert engine.applied
    assert not engine.discarded


def test_async_context_apply_discards_stale_candidate(monkeypatch):
    engine = _AsyncEngine()
    agent = _minimal_agent(engine)
    base_messages = [{"role": "user", "content": "old"}]
    live_messages = [{"role": "user", "content": "changed"}]
    agent._pending_async_context_candidate = ContextCompressionCandidate(
        messages=[{"role": "user", "content": "summary"}],
        base_message_count=len(base_messages),
        base_digest=agent._context_messages_digest(base_messages),
    )

    def _should_not_finalize(**_kwargs):
        raise AssertionError("stale candidates must not be applied")

    monkeypatch.setattr(agent, "_finalize_context_rewrite", _should_not_finalize)

    new_messages, new_prompt, applied = agent._maybe_apply_async_context_candidate(
        live_messages,
        "system",
        task_id="task",
    )

    assert applied is False
    assert new_prompt is None
    assert new_messages == live_messages
    assert engine.discarded
    assert engine.discarded[0][1] == "stale_digest"


def test_run_conversation_prepares_then_applies_candidate_on_next_turn():
    engine = _AsyncEngine(
        replacement=[
            {"role": "user", "content": "compressed prior turn"},
            {"role": "assistant", "content": "prior turn summarized"},
        ]
    )
    agent = _make_conversation_agent(engine)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            "first answer",
            usage={"prompt_tokens": 25_000, "completion_tokens": 10},
        ),
        _mock_response(
            "second answer",
            usage={"prompt_tokens": 2_000, "completion_tokens": 10},
        ),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("first ask")
        assert isinstance(_wait_for_pending(agent), ContextCompressionCandidate)

        second = agent.run_conversation(
            "second ask",
            conversation_history=first["messages"],
        )

    assert first["completed"] is True
    assert first["final_response"] == "first answer"
    assert second["completed"] is True
    assert second["final_response"] == "second answer"
    assert engine.applied
    assert second["messages"][0]["content"] == "compressed prior turn"
    assert second["messages"][1]["content"] == "prior turn summarized"
    assert second["messages"][-2]["content"] == "second ask"
    assert second["messages"][-1]["content"] == "second answer"
    assert not any(msg.get("content") == "first ask" for msg in second["messages"])


def test_async_apply_user_tail_keeps_current_turn_bookkeeping(monkeypatch):
    engine = _AsyncEngine()
    agent = _make_conversation_agent(engine)
    base_history = [
        {"role": "user", "content": "old ask"},
        {"role": "assistant", "content": "old answer"},
    ]
    candidate_messages = [{"role": "user", "content": "compressed summary"}]
    agent._pending_async_context_candidate = ContextCompressionCandidate(
        messages=candidate_messages,
        base_message_count=len(base_history),
        base_digest=agent._context_messages_digest(base_history),
    )
    agent.client.chat.completions.create.return_value = _mock_response(
        "final answer",
        usage={"prompt_tokens": 2_000, "completion_tokens": 10},
    )

    def _invoke_hook(name, **_kwargs):
        if name == "pre_llm_call":
            return [{"context": "PLUGIN CTX"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke_hook)

    with (
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "API ONLY current ask",
            conversation_history=base_history,
            persist_user_message="clean current ask",
        )

    api_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    api_roles = [msg.get("role") for msg in api_messages]
    assert ["user", "user"] not in [
        api_roles[idx: idx + 2] for idx in range(len(api_roles))
    ]
    user_api_messages = [msg for msg in api_messages if msg.get("role") == "user"]
    assert len(user_api_messages) == 2
    assert user_api_messages[0]["content"] == "compressed summary"
    current_api_content = user_api_messages[1]["content"]
    assert "API ONLY current ask" in current_api_content
    assert "PLUGIN CTX" in current_api_content

    assert result["messages"][0] == {"role": "user", "content": "compressed summary"}
    assert result["messages"][1] == {
        "role": "assistant",
        "content": AIAgent._ASYNC_CONTEXT_PRIOR_USER_TAIL_SEPARATOR_MESSAGE,
    }
    assert result["messages"][2] == {"role": "user", "content": "clean current ask"}
    assert result["messages"][-1]["content"] == "final answer"


def test_async_apply_user_tail_separates_multimodal_current_turn_with_boundary():
    engine = _AsyncEngine()
    agent = _make_conversation_agent(engine)
    base_history = [
        {"role": "user", "content": "old ask"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent._pending_async_context_candidate = ContextCompressionCandidate(
        messages=[{"role": "user", "content": "compressed summary"}],
        base_message_count=len(base_history),
        base_digest=agent._context_messages_digest(base_history),
    )
    agent.client.chat.completions.create.return_value = _mock_response(
        "vision answer",
        usage={"prompt_tokens": 2_000, "completion_tokens": 10},
    )
    multimodal_user = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
    ]

    with (
        patch.object(agent, "_model_supports_vision", return_value=True),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            multimodal_user,
            conversation_history=base_history,
        )

    api_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    api_roles = [msg.get("role") for msg in api_messages]
    assert ["user", "user"] not in [
        api_roles[idx: idx + 2] for idx in range(len(api_roles))
    ]

    user_api_messages = [msg for msg in api_messages if msg.get("role") == "user"]
    assert len(user_api_messages) == 2
    assert user_api_messages[0]["content"] == "compressed summary"
    assert user_api_messages[1]["content"] == multimodal_user
    assert result["messages"][0] == {"role": "user", "content": "compressed summary"}
    assert result["messages"][1] == {
        "role": "assistant",
        "content": AIAgent._ASYNC_CONTEXT_PRIOR_USER_TAIL_SEPARATOR_MESSAGE,
    }
    assert result["messages"][2]["content"] == multimodal_user
    assert result["messages"][-1]["content"] == "vision answer"


def test_supplied_history_user_tail_preserves_persistence_offset():
    engine = _AsyncEngine()
    agent = _make_conversation_agent(engine)
    history_tail = {"role": "user", "content": "interrupted old ask"}
    history = [history_tail]
    agent.client.chat.completions.create.return_value = _mock_response(
        "resumed answer",
        usage={"prompt_tokens": 2_000, "completion_tokens": 10},
    )
    persist_calls = []

    def _capture_persist(messages, conversation_history=None):
        persist_calls.append((list(messages), conversation_history))

    with (
        patch.object(agent, "_persist_session", side_effect=_capture_persist),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "current ask",
            conversation_history=history,
        )

    api_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    api_roles = [msg.get("role") for msg in api_messages]
    assert ["user", "user"] not in [
        api_roles[idx: idx + 2] for idx in range(len(api_roles))
    ]

    assert history_tail["content"] == "interrupted old ask"
    assert persist_calls
    persisted_messages, persisted_history = persist_calls[-1]
    assert persisted_history is history
    assert persisted_messages[len(history)] == {
        "role": "assistant",
        "content": AIAgent._ASYNC_CONTEXT_PRIOR_USER_TAIL_SEPARATOR_MESSAGE,
    }
    assert persisted_messages[len(history) + 1] == {
        "role": "user",
        "content": "current ask",
    }
    assert result["messages"][len(history)] == persisted_messages[len(history)]
    assert result["messages"][len(history) + 1] == persisted_messages[len(history) + 1]


def test_async_apply_preserves_pre_llm_continuation_state(monkeypatch):
    engine = _AsyncEngine()
    agent = _make_conversation_agent(engine)
    base_history = [
        {"role": "user", "content": "old ask"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent._pending_async_context_candidate = ContextCompressionCandidate(
        messages=[{"role": "user", "content": "compressed summary"}],
        base_message_count=len(base_history),
        base_digest=agent._context_messages_digest(base_history),
    )
    agent.client.chat.completions.create.return_value = _mock_response(
        "final answer",
        usage={"prompt_tokens": 2_000, "completion_tokens": 10},
    )
    hook_calls = []
    persist_calls = []

    def _invoke_hook(name, **kwargs):
        if name == "pre_llm_call":
            hook_calls.append(kwargs)
        return []

    def _capture_persist(messages, conversation_history=None):
        persist_calls.append((list(messages), conversation_history))

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke_hook)

    with (
        patch.object(agent, "_persist_session", side_effect=_capture_persist),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation(
            "current ask",
            conversation_history=base_history,
        )

    assert hook_calls
    assert hook_calls[-1]["is_first_turn"] is False
    assert hook_calls[-1]["conversation_history"]
    assert persist_calls
    assert persist_calls[-1][1] is None


def test_async_apply_runs_session_and_memory_side_effects(tmp_path):
    from hermes_state import SessionDB

    engine = _AsyncEngine()
    db = SessionDB(db_path=Path(tmp_path) / "session.db")
    agent = _make_conversation_agent(
        engine,
        session_db=db,
        session_id="original-session",
    )
    agent._memory_manager = MagicMock()
    agent.commit_memory_session = MagicMock()
    agent._build_system_prompt = MagicMock(return_value="rebuilt system")

    original_sid = agent.session_id
    original_messages = [
        {"role": "user", "content": "old context"},
        {"role": "assistant", "content": "old answer"},
    ]
    candidate_messages = [
        {"role": "user", "content": "compressed context"},
        {"role": "assistant", "content": "compressed answer"},
    ]
    agent._pending_async_context_candidate = ContextCompressionCandidate(
        messages=candidate_messages,
        base_message_count=len(original_messages),
        base_digest=agent._context_messages_digest(original_messages),
    )

    new_messages, new_prompt, applied = agent._maybe_apply_async_context_candidate(
        original_messages,
        "system message",
        approx_tokens=25_000,
        task_id="task",
    )

    assert applied is True
    assert new_messages == candidate_messages
    assert new_prompt == "rebuilt system"
    assert agent.session_id != original_sid
    agent._build_system_prompt.assert_called_once_with("system message")
    agent.commit_memory_session.assert_called_once_with(original_messages)
    agent._memory_manager.on_pre_compress.assert_called_once_with(original_messages)
    agent._memory_manager.on_session_switch.assert_called_once()
    switch_call = agent._memory_manager.on_session_switch.call_args
    assert switch_call.args == (agent.session_id,)
    assert switch_call.kwargs == {
        "parent_session_id": original_sid,
        "reset": False,
        "reason": "compression",
    }
    assert engine.session_start_calls
    new_sid, kwargs = engine.session_start_calls[-1]
    assert new_sid == agent.session_id
    assert kwargs["boundary_reason"] == "compression"
    assert kwargs["old_session_id"] == original_sid
    assert engine.applied


def test_builtin_compressor_does_not_opt_into_async_preparation():
    compressor = ContextCompressor.__new__(ContextCompressor)
    assert compressor.should_prepare_async_compression(10_000, []) is False
    assert compressor.prepare_async_compression([], current_tokens=10_000) is None


def test_default_noop_does_not_estimate_or_snapshot(monkeypatch):
    compressor = ContextCompressor.__new__(ContextCompressor)
    agent = _minimal_agent(compressor)
    messages = [{"role": "user", "content": "hello"}]

    def _fail_estimate(*_args, **_kwargs):
        raise AssertionError("default no-op path must not estimate tokens")

    def _fail_deepcopy(_value):
        raise AssertionError("default no-op path must not snapshot messages")

    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        _fail_estimate,
    )
    monkeypatch.setattr("agent.conversation_compression.copy.deepcopy", _fail_deepcopy)

    assert agent._maybe_start_async_context_compression(messages) is False
    assert agent._pending_async_context_candidate is None
    assert agent._async_context_thread is None


def test_async_context_estimates_tokens_before_threshold_gate_when_usage_missing(monkeypatch):
    engine = _ThresholdAsyncEngine(threshold=10_000)
    agent = _minimal_agent(engine)
    agent._cached_system_prompt = "system prompt"
    agent.tools = [{"type": "function", "function": {"name": "example"}}]
    messages = [{"role": "user", "content": "large transcript"}]
    estimate_calls = []

    def _estimate(messages_arg, *, system_prompt="", tools=None):
        estimate_calls.append((messages_arg, system_prompt, tools))
        return 25_000

    monkeypatch.setattr(
        "agent.conversation_compression.estimate_request_tokens_rough",
        _estimate,
    )

    assert agent._maybe_start_async_context_compression(messages) is True

    assert _wait_for_pending(agent) is not None
    assert estimate_calls == [(messages, "system prompt", agent.tools)]
    assert engine.should_tokens == [25_000]
    assert engine.prepare_tokens == [25_000]


def test_async_context_does_not_start_while_prepare_inflight():
    engine = _BlockingAsyncEngine()
    agent = _minimal_agent(engine)
    messages = [{"role": "user", "content": "hello"}]

    assert agent._maybe_start_async_context_compression(messages, approx_tokens=25_000)
    assert engine.entered.wait(timeout=2)

    try:
        assert agent._maybe_start_async_context_compression(messages, approx_tokens=25_000) is False
    finally:
        engine.release.set()

    thread = agent._async_context_thread
    if thread is not None:
        thread.join(timeout=2)
    assert isinstance(_wait_for_pending(agent), ContextCompressionCandidate)


def test_reset_discards_inflight_async_candidate():
    engine = _BlockingAsyncEngine()
    agent = _minimal_agent(engine)
    messages = [{"role": "user", "content": "hello"}]

    assert agent._maybe_start_async_context_compression(messages, approx_tokens=25_000)
    assert engine.entered.wait(timeout=2)

    agent.reset_session_state()
    engine.release.set()

    thread = agent._async_context_thread
    if thread is not None:
        thread.join(timeout=2)

    assert engine.reset_count == 1
    assert agent._pending_async_context_candidate is None
    assert agent._async_context_inflight_digest is None
