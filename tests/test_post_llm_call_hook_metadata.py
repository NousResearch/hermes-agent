from agent.turn_finalizer import finalize_turn


class _Budget:
    remaining = 5
    used = 1
    max_total = 5


class _Compressor:
    name = "lcm"
    last_prompt_tokens = 123
    api_key = "provider-secret-must-not-reach-hooks"


class _Agent:
    max_iterations = 5
    iteration_budget = _Budget()
    quiet_mode = True
    model = "test-model"
    provider = "test-provider"
    base_url = ""
    session_id = "session-1"
    platform = "discord"
    _chat_id = "channel-123"
    _chat_name = "Hermes LCM"
    _chat_type = "thread"
    _thread_id = "thread-456"
    _gateway_session_key = "agent:main:discord:thread:thread-456:thread-456"
    _user_id = "user-789"
    context_compressor = _Compressor()
    session_input_tokens = 1
    session_output_tokens = 2
    session_cache_read_tokens = 3
    session_cache_write_tokens = 4
    session_reasoning_tokens = 5
    session_prompt_tokens = 6
    session_completion_tokens = 7
    session_total_tokens = 8
    session_estimated_cost_usd = 0.0
    session_cost_status = "ok"
    session_cost_source = "test"
    _tool_guardrail_halt_decision = None
    _response_was_previewed = False
    _interrupt_message = ""
    _stream_callback = None
    _skill_nudge_interval = 0
    _iters_since_skill = 0
    valid_tool_names = set()

    def _emit_status(self, _msg):
        pass

    def _safe_print(self, _msg):
        pass

    def _handle_max_iterations(self, _messages, _api_call_count):
        return "max iteration summary"

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, _messages):
        pass

    def _persist_session(self, *_args, **_kwargs):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _format_turn_completion_explanation(self, _reason):
        return ""

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass

    def _spawn_background_review(self, **_kwargs):
        pass


def test_post_llm_call_hook_receives_active_context_engine_and_gateway_lane(monkeypatch):
    calls = []

    def fake_invoke_hook(name, **kwargs):
        calls.append((name, kwargs))
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "reply"},
    ]
    result = finalize_turn(
        _Agent(),
        final_response="reply",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hello",
        original_user_message="hello",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    assert result["final_response"] == "reply"
    post_call = next(kwargs for name, kwargs in calls if name == "post_llm_call")
    assert post_call["context_engine"] == "lcm"
    assert "context_compressor" not in post_call
    assert _Compressor.api_key not in repr(post_call)
    assert post_call["conversation_id"] == "agent:main:discord:thread:thread-456:thread-456"
    assert post_call["gateway_session_key"] == post_call["conversation_id"]
    assert post_call["sender_id"] == "user-789"
    assert post_call["chat_id"] == "channel-123"
    assert post_call["chat_type"] == "thread"
    assert post_call["thread_id"] == "thread-456"


def test_post_llm_call_hook_metadata_is_backward_compatible_for_telegram(monkeypatch):
    calls = []

    def fake_invoke_hook(name, **kwargs):
        calls.append((name, kwargs))
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)

    class _TelegramAgent(_Agent):
        platform = "telegram"
        _chat_id = "1782862480"
        _chat_name = "Home"
        _chat_type = "private"
        _thread_id = ""
        _gateway_session_key = "agent:main:telegram:private:1782862480"
        _user_id = "1782862480"

    messages = [
        {"role": "user", "content": "telegram hello"},
        {"role": "assistant", "content": "telegram reply"},
    ]
    result = finalize_turn(
        _TelegramAgent(),
        final_response="telegram reply",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task-2",
        turn_id="turn-2",
        user_message="telegram hello",
        original_user_message="telegram hello",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    assert result["final_response"] == "telegram reply"
    post_call = next(kwargs for name, kwargs in calls if name == "post_llm_call")
    assert post_call["platform"] == "telegram"
    assert post_call["context_engine"] == "lcm"
    assert "context_compressor" not in post_call
    assert post_call["conversation_id"] == "agent:main:telegram:private:1782862480"
    assert post_call["gateway_session_key"] == post_call["conversation_id"]
    assert post_call["sender_id"] == "1782862480"
    assert post_call["chat_id"] == "1782862480"
    assert post_call["chat_type"] == "private"
    assert post_call["thread_id"] == ""


def test_post_llm_call_hook_metadata_falls_back_to_public_lane_attrs(monkeypatch):
    calls = []

    def fake_invoke_hook(name, **kwargs):
        calls.append((name, kwargs))
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)

    class _PublicLaneAgent(_Agent):
        _chat_id = ""
        _chat_name = ""
        _chat_type = ""
        _thread_id = ""
        chat_id = "public-channel"
        chat_name = "Public Channel"
        chat_type = "channel"
        thread_id = "public-thread"
        _gateway_session_key = "agent:main:discord:channel:public-channel:public-thread"

    messages = [
        {"role": "user", "content": "public hello"},
        {"role": "assistant", "content": "public reply"},
    ]
    result = finalize_turn(
        _PublicLaneAgent(),
        final_response="public reply",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task-3",
        turn_id="turn-3",
        user_message="public hello",
        original_user_message="public hello",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    assert result["final_response"] == "public reply"
    post_call = next(kwargs for name, kwargs in calls if name == "post_llm_call")
    assert post_call["chat_id"] == "public-channel"
    assert post_call["chat_name"] == "Public Channel"
    assert post_call["chat_type"] == "channel"
    assert post_call["thread_id"] == "public-thread"
