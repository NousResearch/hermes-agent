"""System-role prefill messages fold into the single leading system prompt.

Strict chat templates (Qwen-family Jinja templates served by llama.cpp / LM Studio)
reject a second system message with HTTP 500 "System message must be at the
beginning", and the native Anthropic converter keeps only the last system message.
Both outbound paths (the regular turn request and the max-iterations summary
request) must therefore send exactly one leading system message, preserve
structured (list-valued) system content, and keep non-system prefills in order.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

from agent.prefill_messages import fold_system_prefill_messages

_CACHE = {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# fold_system_prefill_messages
# ---------------------------------------------------------------------------


def test_without_prefill_messages_leaves_request_untouched():
    api_messages = [{"role": "system", "content": "base system"}, {"role": "user", "content": "hello"}]

    assert fold_system_prefill_messages(api_messages, []) is api_messages
    assert fold_system_prefill_messages(api_messages, None) is api_messages


def test_string_system_prefill_folds_into_leading_system():
    api_messages = [{"role": "system", "content": "base system"}, {"role": "user", "content": "hello"}]
    prefill = [
        {"role": "system", "content": "persona rules"},
        {"role": "assistant", "content": "example answer"},
    ]
    snapshot = copy.deepcopy((api_messages, prefill))

    merged = fold_system_prefill_messages(api_messages, prefill)

    assert merged == [
        {"role": "system", "content": "base system\n\npersona rules"},
        {"role": "assistant", "content": "example answer"},
        {"role": "user", "content": "hello"},
    ]
    assert (api_messages, prefill) == snapshot


def test_system_prefill_creates_leading_system_when_request_has_none():
    merged = fold_system_prefill_messages(
        [{"role": "user", "content": "hello"}],
        [
            {"role": "system", "content": "persona rules"},
            {"role": "user", "content": "few-shot question"},
        ],
    )

    assert merged == [
        {"role": "system", "content": "persona rules"},
        {"role": "user", "content": "few-shot question"},
        {"role": "user", "content": "hello"},
    ]


def test_non_system_prefill_messages_keep_their_order_after_the_system_prompt():
    merged = fold_system_prefill_messages(
        [{"role": "system", "content": "base system"}, {"role": "user", "content": "real question"}],
        [
            {"role": "user", "content": "few-shot question"},
            {"role": "system", "content": "persona rules"},
            {"role": "assistant", "content": "few-shot answer"},
        ],
    )

    assert merged == [
        {"role": "system", "content": "base system\n\npersona rules"},
        {"role": "user", "content": "few-shot question"},
        {"role": "assistant", "content": "few-shot answer"},
        {"role": "user", "content": "real question"},
    ]


def test_list_valued_system_prefill_keeps_its_content_blocks():
    persona_block = {"type": "text", "text": "persona rules", "cache_control": _CACHE}
    prefill = [{"role": "system", "content": [persona_block]}]
    snapshot = copy.deepcopy(prefill)

    merged = fold_system_prefill_messages(
        [{"role": "system", "content": "base system"}, {"role": "user", "content": "hello"}], prefill
    )

    assert merged[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "base system\n\n"}, persona_block],
    }
    assert [m["role"] for m in merged] == ["system", "user"]
    # The request owns its blocks: send-path rewrites never reach agent.prefill_messages.
    merged[0]["content"][1]["text"] = "rewritten"
    assert prefill == snapshot


def test_list_valued_leading_system_keeps_its_blocks_when_string_prefill_folds_in():
    base_blocks = [
        {"type": "text", "text": "static prefix", "cache_control": _CACHE},
        {"type": "text", "text": "dynamic suffix"},
    ]
    api_messages = [{"role": "system", "content": base_blocks}, {"role": "user", "content": "hello"}]
    snapshot = copy.deepcopy(api_messages)

    merged = fold_system_prefill_messages(api_messages, [{"role": "system", "content": "  persona rules  "}])

    assert merged[0]["content"] == [*base_blocks, {"type": "text", "text": "\n\npersona rules"}]
    assert api_messages == snapshot


def test_multiple_list_and_string_system_prefills_concatenate_in_order():
    merged = fold_system_prefill_messages(
        [{"role": "user", "content": "hello"}],
        [
            {"role": "system", "content": [{"type": "text", "text": "one"}]},
            {"role": "system", "content": "two"},
            {"role": "system", "content": [{"type": "text", "text": "three"}, {"type": "text", "text": "four"}]},
        ],
    )

    assert merged[0] == {
        "role": "system",
        "content": [{"type": "text", "text": t} for t in ("one", "\n\ntwo\n\n", "three", "four")],
    }
    assert [m["role"] for m in merged] == ["system", "user"]


@pytest.mark.parametrize("empty", [None, "", "   ", []])
def test_empty_system_prefill_leaves_the_system_prompt_alone(empty):
    merged = fold_system_prefill_messages(
        [{"role": "system", "content": "base system"}, {"role": "user", "content": "hello"}],
        [{"role": "system", "content": empty}],
    )

    assert merged == [{"role": "system", "content": "base system"}, {"role": "user", "content": "hello"}]


def test_unsupported_system_content_raises_instead_of_sending_a_repr():
    with pytest.raises(TypeError, match="system prefill content"):
        fold_system_prefill_messages(
            [{"role": "user", "content": "hello"}], [{"role": "system", "content": 42}]
        )


def test_native_anthropic_wire_keeps_base_prompt_and_prefill():
    """The Anthropic converter keeps only the last system message, so an unfolded system
    prefill used to replace the real system prompt on that route."""
    from agent.anthropic_message_convert import convert_messages_to_anthropic

    merged = fold_system_prefill_messages(
        [{"role": "system", "content": "base system"}, {"role": "user", "content": "hello"}],
        [{"role": "system", "content": "persona rules"}],
    )
    system, _ = convert_messages_to_anthropic(merged)

    assert system == "base system\n\npersona rules"


# ---------------------------------------------------------------------------
# Outbound requests: regular turn and max-iterations summary
# ---------------------------------------------------------------------------

_PREFILL = [
    {"role": "system", "content": "persona rules"},
    {"role": "user", "content": "few-shot question"},
    {"role": "assistant", "content": "few-shot answer"},
]


@pytest.fixture()
def loop_agent():
    """AIAgent on a chat-completions route with a mocked OpenAI client, so the
    outbound ``messages`` of every call can be captured."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _sent_messages(agent) -> list:
    call = agent.client.chat.completions.create.call_args
    return call.kwargs.get("messages") or call.args[0]["messages"]


def _run_turn(agent, prefill, text="real question") -> list:
    from tests.agent.test_run_agent import _mock_response

    agent.prefill_messages = prefill
    agent.client.chat.completions.create.side_effect = [_mock_response(content="answer", finish_reason="stop")]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation(text)
    assert agent.client.chat.completions.create.call_count == 1
    return _sent_messages(agent)


def _run_max_iterations_summary(agent, prefill) -> list:
    from agent.chat_completion_helpers import handle_max_iterations
    from tests.agent.test_run_agent import _mock_response

    agent.prefill_messages = prefill
    agent.client.chat.completions.create.side_effect = [_mock_response(content="summary", finish_reason="stop")]
    history = [{"role": "user", "content": "real question"}, {"role": "assistant", "content": "working on it"}]
    assert handle_max_iterations(agent, history, api_call_count=agent.max_iterations) == "summary"
    assert agent.client.chat.completions.create.call_count == 1
    return _sent_messages(agent)


def _roles_and_texts(sent: list) -> list:
    return [(m["role"], m["content"]) for m in sent[1:]]


@pytest.mark.parametrize("send", [_run_turn, _run_max_iterations_summary], ids=["regular", "max_iterations"])
def test_outbound_request_has_one_leading_system_and_ordered_prefill(loop_agent, send):
    prefill = copy.deepcopy(_PREFILL)

    sent = send(loop_agent, prefill)

    assert [m["role"] for m in sent].count("system") == 1
    assert sent[0]["role"] == "system"
    assert sent[0]["content"].startswith("You are helpful.")
    assert sent[0]["content"].endswith("\n\npersona rules")
    assert _roles_and_texts(sent)[:3] == [
        ("user", "few-shot question"),
        ("assistant", "few-shot answer"),
        ("user", "real question"),
    ]
    assert prefill == _PREFILL


@pytest.mark.parametrize("send", [_run_turn, _run_max_iterations_summary], ids=["regular", "max_iterations"])
def test_outbound_request_preserves_list_valued_system_prefill(loop_agent, send):
    persona_blocks = [{"type": "text", "text": "persona rules"}, {"type": "text", "text": "tone rules"}]
    prefill = [{"role": "system", "content": copy.deepcopy(persona_blocks)}, *copy.deepcopy(_PREFILL[1:])]

    sent = send(loop_agent, prefill)

    assert [m["role"] for m in sent].count("system") == 1
    system_content = sent[0]["content"]
    assert isinstance(system_content, list)
    assert system_content[0]["type"] == "text"
    assert system_content[0]["text"].startswith("You are helpful.")
    assert system_content[0]["text"].endswith("\n\n")
    assert system_content[1:] == persona_blocks
    assert _roles_and_texts(sent)[:2] == [("user", "few-shot question"), ("assistant", "few-shot answer")]
    assert prefill[0]["content"] == persona_blocks


@pytest.mark.parametrize("send", [_run_turn, _run_max_iterations_summary], ids=["regular", "max_iterations"])
def test_single_block_system_prefill_reads_as_separated_text(loop_agent, send):
    """The regular send path flattens one/two-part plain-text system lists with a ``""``-join
    (prompt-cache decoration undo); the fold's separator keeps the two prompts apart."""
    prefill = [{"role": "system", "content": [{"type": "text", "text": "persona rules"}]}]

    sent = send(loop_agent, prefill)

    assert [m["role"] for m in sent].count("system") == 1
    content = sent[0]["content"]
    text = content if isinstance(content, str) else "".join(p["text"] for p in content)
    assert text.startswith("You are helpful.")
    assert text.endswith("\n\npersona rules")


def test_responses_instructions_read_list_valued_system_as_text_not_repr():
    from agent.transports.codex import ResponsesApiTransport

    api_messages = fold_system_prefill_messages(
        [{"role": "system", "content": "base system"}, {"role": "user", "content": "hello"}],
        [{"role": "system", "content": [{"type": "text", "text": "persona rules", "cache_control": _CACHE}]}],
    )

    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-5.4", messages=api_messages, tools=None, base_url="https://api.openai.com/v1"
    )

    assert kwargs["instructions"] == "base system\n\npersona rules"
