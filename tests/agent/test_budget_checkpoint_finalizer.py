from types import SimpleNamespace


def test_handle_budget_checkpoint_requests_structured_packet_without_tools():
    from agent.chat_completion_helpers import handle_budget_checkpoint

    captured_kwargs = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="CONTINUATION PACKET"))])

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    class FakeTransport:
        def normalize_response(self, response, **kwargs):
            return SimpleNamespace(content=response.choices[0].message.content)

    class FakeAgent:
        model = "test-model"
        max_iterations = 30
        api_mode = "chat_completions"
        _cached_system_prompt = "You are helpful."
        ephemeral_system_prompt = ""
        prefill_messages = []
        provider = "openrouter"
        base_url = "https://openrouter.ai/api/v1"
        _base_url_lower = "https://openrouter.ai/api/v1"
        max_tokens = None
        reasoning_config = None
        providers_allowed = None
        providers_ignored = None
        providers_order = None
        provider_sort = None
        openrouter_min_coding_score = None
        _is_anthropic_oauth = False

        def _should_sanitize_tool_calls(self):
            return False

        def _copy_reasoning_content_for_api(self, msg, api_msg):
            return None

        def _sanitize_api_messages(self, messages):
            return messages

        def _drop_thinking_only_and_merge_users(self, messages):
            return messages

        def _supports_reasoning_extra_body(self):
            return False

        def _is_openrouter_url(self):
            return True

        def _max_tokens_param(self, max_tokens):
            return {"max_tokens": max_tokens}

        def _ensure_primary_openai_client(self, reason):
            return FakeClient()

        def _get_transport(self):
            return FakeTransport()

    messages = [{"role": "user", "content": "ship the issue"}]

    result = handle_budget_checkpoint(
        FakeAgent(),
        messages,
        api_call_count=27,
        budget_state="checkpoint 27/30",
        task_metadata={
            "issue": "Hermes: Budget-aware full-flow delivery",
            "lane": "FGDGHO / Wiring / Hermes",
            "owner": "Jimmy",
            "task_class": "full delivery",
            "current_phase": "validation",
        },
    )

    assert result == "CONTINUATION PACKET"
    assert "tools" not in captured_kwargs

    api_messages = captured_kwargs["messages"]
    checkpoint_prompt = api_messages[-1]["content"]
    assert "CONTINUATION PACKET" in checkpoint_prompt
    assert "Do not call tools" in checkpoint_prompt
    assert "issue" in checkpoint_prompt
    assert "lane" in checkpoint_prompt
    assert "owner" in checkpoint_prompt
    assert "task class" in checkpoint_prompt
    assert "budget state" in checkpoint_prompt
    assert "last completed phase" in checkpoint_prompt
    assert "current phase" in checkpoint_prompt
    assert "repo/worktree/branch" in checkpoint_prompt
    assert "commit SHA" in checkpoint_prompt
    assert "PR URL/state" in checkpoint_prompt
    assert "merge state" in checkpoint_prompt
    assert "package path" in checkpoint_prompt
    assert "Drive target" in checkpoint_prompt
    assert "Obsidian target" in checkpoint_prompt
    assert "Linear state" in checkpoint_prompt
    assert "validation" in checkpoint_prompt
    assert "remaining actions" in checkpoint_prompt
    assert "forbidden actions" in checkpoint_prompt
    assert "do-not-repeat list" in checkpoint_prompt
    assert "next safe resume packet" in checkpoint_prompt
    assert "FGD-21 footer" in checkpoint_prompt
    assert "summarizing what you've found" not in checkpoint_prompt


def test_handle_budget_checkpoint_strips_tools_from_codex_kwargs():
    from agent.chat_completion_helpers import handle_budget_checkpoint

    captured_kwargs = {}

    class FakeTransport:
        def normalize_response(self, response, **kwargs):
            return SimpleNamespace(content="CODEX CONTINUATION")

    class FakeAgent:
        model = "test-model"
        max_iterations = 30
        api_mode = "codex_responses"
        _cached_system_prompt = ""
        ephemeral_system_prompt = ""
        prefill_messages = []
        provider = "openai-codex"
        base_url = ""
        _base_url_lower = ""
        max_tokens = None
        reasoning_config = None
        providers_allowed = None
        providers_ignored = None
        providers_order = None
        provider_sort = None
        openrouter_min_coding_score = None
        _is_anthropic_oauth = False

        def _should_sanitize_tool_calls(self):
            return False

        def _copy_reasoning_content_for_api(self, msg, api_msg):
            return None

        def _sanitize_api_messages(self, messages):
            return messages

        def _drop_thinking_only_and_merge_users(self, messages):
            return messages

        def _build_api_kwargs(self, messages):
            return {"input": messages, "tools": [{"name": "should_not_survive"}]}

        def _run_codex_stream(self, kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace()

        def _get_transport(self):
            return FakeTransport()

    result = handle_budget_checkpoint(FakeAgent(), [{"role": "user", "content": "work"}], 27)

    assert result == "CODEX CONTINUATION"
    assert "tools" not in captured_kwargs
