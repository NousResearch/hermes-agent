import importlib
import importlib.util
import inspect


def test_without_prefill_messages_leaves_request_untouched():
    prefill_messages = importlib.import_module("agent.prefill_messages")
    api_messages = [
        {"role": "system", "content": "base system"},
        {"role": "user", "content": "hello"},
    ]

    merged = prefill_messages.fold_system_prefill_messages(api_messages, [])

    assert merged is api_messages


def test_system_prefill_messages_fold_into_single_leading_system():
    spec = importlib.util.find_spec("agent.prefill_messages")
    assert spec is not None
    prefill_messages = importlib.import_module("agent.prefill_messages")

    api_messages = [
        {"role": "system", "content": "base system"},
        {"role": "user", "content": "hello"},
    ]
    prefill = [
        {"role": "system", "content": "persona rules"},
        {"role": "assistant", "content": "example answer"},
    ]

    merged = prefill_messages.fold_system_prefill_messages(api_messages, prefill)

    assert [msg["role"] for msg in merged] == ["system", "assistant", "user"]
    assert merged[0]["content"] == "base system\n\npersona rules"
    assert prefill[0]["content"] == "persona rules"


def test_system_prefill_creates_leading_system_when_request_has_none():
    spec = importlib.util.find_spec("agent.prefill_messages")
    assert spec is not None
    prefill_messages = importlib.import_module("agent.prefill_messages")

    merged = prefill_messages.fold_system_prefill_messages(
        [{"role": "user", "content": "hello"}],
        [
            {"role": "system", "content": "persona rules"},
            {"role": "user", "content": "few-shot question"},
        ],
    )

    assert [msg["role"] for msg in merged] == ["system", "user", "user"]
    assert merged[0]["content"] == "persona rules"


def test_non_system_prefill_messages_keep_their_existing_order():
    prefill_messages = importlib.import_module("agent.prefill_messages")

    merged = prefill_messages.fold_system_prefill_messages(
        [
            {"role": "system", "content": "base system"},
            {"role": "user", "content": "real question"},
        ],
        [
            {"role": "user", "content": "few-shot question"},
            {"role": "assistant", "content": "few-shot answer"},
        ],
    )

    assert merged == [
        {"role": "system", "content": "base system"},
        {"role": "user", "content": "few-shot question"},
        {"role": "assistant", "content": "few-shot answer"},
        {"role": "user", "content": "real question"},
    ]


def test_api_call_paths_use_system_prefill_folding():
    from agent import chat_completion_helpers, conversation_loop

    for func in (conversation_loop.run_conversation, chat_completion_helpers.handle_max_iterations):
        source = inspect.getsource(func)
        assert "fold_system_prefill_messages(" in source
