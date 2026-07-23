from types import SimpleNamespace

from agent.chat_completion_helpers import handle_max_iterations
from agent.transports.chat_completions import ChatCompletionsTransport


def test_zai_chat_completion_rewrites_branded_system_prompt_without_mutating_input():
    transport = ChatCompletionsTransport()
    messages = [
        {
            "role": "system",
            "content": (
                "You are Hermes Agent, an intelligent AI assistant created by Nous Research.\n"
                "You run on Hermes Agent (by Nous Research).\n"
                "Use the provided tools."
            ),
        },
        {"role": "user", "content": "hello"},
    ]

    kwargs = transport.build_kwargs(
        model="glm-5.2",
        messages=messages,
        provider_name="custom",
        base_url="https://api.z.ai/api/coding/paas/v4",
    )

    assert kwargs["messages"] is not messages
    assert messages[0]["content"].startswith("You are Hermes Agent")
    system = kwargs["messages"][0]["content"]
    assert system.startswith("You are a precise local AI coding and operations assistant.")
    assert "Hermes" not in system
    assert "Nous Research" not in system
    assert "Use the provided tools." in system


def test_zai_chat_completion_inserts_neutral_system_prompt_when_missing():
    kwargs = ChatCompletionsTransport().build_kwargs(
        model="glm-5.2",
        messages=[{"role": "user", "content": "hello"}],
        provider_name="custom",
        base_url="https://api.z.ai/api/coding/paas/v4",
    )

    assert kwargs["messages"][0]["role"] == "system"
    assert "precise local AI" in kwargs["messages"][0]["content"]
    assert kwargs["messages"][1] == {"role": "user", "content": "hello"}


def test_zai_provider_alias_triggers_policy_without_base_url():
    messages = [{"role": "system", "content": "Hermes Agent by Nous Research"}]

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="glm-5.2",
        messages=messages,
        provider_name="zai",
        base_url="",
    )

    assert kwargs["messages"] is not messages
    assert "Hermes" not in kwargs["messages"][0]["content"]
    assert "Nous Research" not in kwargs["messages"][0]["content"]


def test_zhipu_bigmodel_endpoint_triggers_policy_for_custom_provider():
    messages = [{"role": "system", "content": "Hermes Agent by Nous Research"}]

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="glm-4.5",
        messages=messages,
        provider_name="custom",
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )

    assert kwargs["messages"] is not messages
    assert "Hermes Agent" not in kwargs["messages"][0]["content"]
    assert "Nous Research" not in kwargs["messages"][0]["content"]


def test_non_zai_request_preserves_message_identity_and_content():
    messages = [{"role": "system", "content": "You are Hermes Agent."}]

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="gpt-4.1",
        messages=messages,
        provider_name="openai",
        base_url="https://api.openai.com/v1",
    )

    assert kwargs["messages"] is messages
    assert kwargs["messages"][0]["content"] == "You are Hermes Agent."


def test_locally_hosted_glm_model_does_not_trigger_endpoint_policy():
    messages = [{"role": "system", "content": "You are Hermes Agent."}]

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="glm-5.2",
        messages=messages,
        provider_name="custom",
        base_url="http://127.0.0.1:8000/v1",
    )

    assert kwargs["messages"] is messages
    assert kwargs["messages"][0]["content"] == "You are Hermes Agent."


def test_zai_policy_preserves_operational_cli_paths_and_environment_names():
    operational = (
        "Run `hermes tools` and `hermes status`. "
        "Read ~/.hermes/config.yaml and .hermes.md. "
        "Keep HERMES_HOME unchanged. Active Hermes profile: default."
    )

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="glm-5.2",
        messages=[{"role": "system", "content": operational}],
        provider_name="custom",
        base_url="https://api.z.ai/api/coding/paas/v4",
    )

    system = kwargs["messages"][0]["content"]
    for fragment in (
        "hermes tools",
        "hermes status",
        "~/.hermes/config.yaml",
        ".hermes.md",
        "HERMES_HOME",
        "Active Hermes profile",
    ):
        assert fragment in system


def test_zai_policy_rewrites_multipart_system_text_without_mutating_input():
    content = [
        {
            "type": "text",
            "text": "You are Hermes Agent, an intelligent AI assistant created by Nous Research. Use tools.",
        }
    ]
    messages = [{"role": "system", "content": content}]

    kwargs = ChatCompletionsTransport().build_kwargs(
        model="glm-5.2",
        messages=messages,
        provider_name="custom",
        base_url="https://api.z.ai/api/coding/paas/v4",
    )

    assert messages[0]["content"] is content
    assert messages[0]["content"][0]["text"].startswith("You are Hermes Agent")
    rewritten = kwargs["messages"][0]["content"][0]["text"]
    assert rewritten.startswith("You are a precise local AI coding and operations assistant.")
    assert "Hermes Agent" not in rewritten
    assert "Nous Research" not in rewritten
    assert "Use tools." in rewritten


def test_zai_iteration_limit_summary_uses_prompt_policy():
    captured = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return object()

    class Transport:
        @staticmethod
        def normalize_response(_response):
            return SimpleNamespace(content="done")

    agent = SimpleNamespace(
        max_iterations=1,
        _should_sanitize_tool_calls=lambda: False,
        _copy_reasoning_content_for_api=lambda _source, _target: None,
        _cached_system_prompt=(
            "You are Hermes Agent, an intelligent AI assistant created by Nous Research.\n"
            "You run on Hermes Agent (by Nous Research)."
        ),
        ephemeral_system_prompt="",
        prefill_messages=[],
        _sanitize_api_messages=lambda messages: messages,
        _drop_thinking_only_and_merge_users=lambda messages: messages,
        model="glm-5.2",
        provider="custom",
        base_url="https://api.z.ai/api/coding/paas/v4",
        _base_url_lower="https://api.z.ai/api/coding/paas/v4",
        api_mode="chat_completions",
        reasoning_config=None,
        _supports_reasoning_extra_body=lambda: False,
        max_tokens=64,
        _max_tokens_param=lambda value: {"max_tokens": value},
        openrouter_min_coding_score=None,
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection=None,
        _is_openrouter_url=lambda: False,
        _ensure_primary_openai_client=lambda **_kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        ),
        _get_transport=lambda: Transport(),
    )

    result = handle_max_iterations(
        agent,
        [{"role": "user", "content": "original task"}],
        api_call_count=1,
    )

    assert result == "done"
    system = captured["messages"][0]["content"]
    assert system.startswith("You are a precise local AI coding and operations assistant.")
    assert "Hermes" not in system
    assert "Nous Research" not in system
