from types import SimpleNamespace

from agent.qwen_session_state import (
    build_qwen_session_metadata,
    is_qwen_session_provider,
    maybe_update_qwen_session_from_response,
)
from agent.transports.chat_completions import ChatCompletionsTransport


def _agent(**overrides):
    values = {
        "provider": "qwen-local",
        "base_url": "http://127.0.0.1:3264/api",
        "model": "qwen3.7-max",
        "session_id": "session-1",
        "platform": "telegram",
        "_chat_id": "330137562",
        "_thread_id": "201820",
        "_gateway_session_key": "agent:main:telegram:dm:330137562:201820",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen_local_metadata_uses_stable_conversation_id(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent()

    metadata = build_qwen_session_metadata(agent)

    assert metadata is not None
    assert metadata["conversation_id"].startswith("hermes_")
    assert metadata["sessionId"] == metadata["conversation_id"]
    assert "chatId" not in metadata
    assert build_qwen_session_metadata(agent)["conversation_id"] == metadata["conversation_id"]


def test_qwen_local_response_updates_persistent_chat_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent()
    response = SimpleNamespace(chatId="qwen-chat-1", parentId="qwen-parent-1")

    maybe_update_qwen_session_from_response(agent, response)
    metadata = build_qwen_session_metadata(_agent())

    assert metadata is not None
    assert metadata["chatId"] == "qwen-chat-1"
    assert metadata["parentId"] == "qwen-parent-1"


def test_qwen_portal_provider_keeps_native_metadata_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent(provider="qwen", base_url="https://portal.qwen.ai/v1")

    assert not is_qwen_session_provider(agent)
    assert build_qwen_session_metadata(agent) is None


def test_non_qwen_provider_does_not_emit_qwen_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent(provider="openrouter", base_url="https://openrouter.ai/api/v1")

    assert not is_qwen_session_provider(agent)
    assert build_qwen_session_metadata(agent) is None


def test_resolved_custom_qwen_provider_uses_original_provider_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    direct = _agent(provider="qwen-local")
    resolved_custom = _agent(
        provider="custom",
        _custom_providers=[
            {
                "provider_key": "qwen-local",
                "name": "Qwen Local",
                "base_url": "http://127.0.0.1:3264/api",
                "model": "qwen3.7-max",
            }
        ],
    )

    direct_metadata = build_qwen_session_metadata(direct)
    custom_metadata = build_qwen_session_metadata(resolved_custom)

    assert direct_metadata is not None
    assert is_qwen_session_provider(resolved_custom)
    assert custom_metadata is not None
    assert custom_metadata["conversation_id"] == direct_metadata["conversation_id"]


def test_plain_custom_provider_with_qwen_model_is_not_assumed_qwen(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = _agent(
        provider="custom",
        base_url="http://127.0.0.1:1234/v1",
        _custom_providers=[
            {
                "provider_key": "local-openai",
                "name": "Local OpenAI",
                "base_url": "http://127.0.0.1:1234/v1",
                "model": "qwen3.7-max",
            }
        ],
    )

    assert not is_qwen_session_provider(agent)
    assert build_qwen_session_metadata(agent) is None


def test_legacy_chat_completions_puts_qwen_metadata_top_level():
    transport = ChatCompletionsTransport()
    kwargs = transport.build_kwargs(
        model="qwen3.7-max",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        qwen_session_metadata={"conversation_id": "hermes_test"},
    )

    assert kwargs["metadata"] == {"conversation_id": "hermes_test"}
    assert "metadata" not in kwargs.get("extra_body", {})
