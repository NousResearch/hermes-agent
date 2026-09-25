"""Tests for the AWS Bedrock Converse API adapter.

Covers:
  - AWS credential detection and region resolution
  - Message format conversion (OpenAI → Converse and back)
  - Tool definition conversion
  - Response normalization (non-streaming and streaming)
  - Model discovery with caching
  - Edge cases: empty messages, consecutive roles, image content
"""

import json
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# botocore import hygiene (anti-flake, Aug 2026)
#
# Several tests in this file plant fake ``botocore`` modules in sys.modules
# (to run without the real package / without touching the AWS credential
# chain). The real package's ``botocore.exceptions`` lazily executes
# ``from botocore.vendored import requests`` on FIRST import — if that first
# import happens while a fake parent is (or was) installed, the chain
# resolves against a module with no real __path__ and the whole file's
# exception tests die with ``No module named 'botocore.vendored'`` — but
# only in interpreter states where nothing imported it earlier (the exact
# CI-vs-local flake on PR #92617).
#
# Two defenses, both required:
#   1. Import the real exception types HERE, at module scope, before any
#      test can stub sys.modules. Once cached, later ``from
#      botocore.exceptions import X`` is a dict hit and can never
#      re-execute the vendored import under a poisoned parent.
#   2. An autouse fixture snapshots every boto* sys.modules entry before
#      each test and restores it after, so no stub window can leak state
#      into a later test regardless of ordering.
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised implicitly by every exception test
    from botocore.exceptions import (  # noqa: F401
        ClientError as _RealClientError,
        ConnectionClosedError as _RealConnectionClosedError,
    )
except Exception:  # botocore genuinely not installed / torn — tests skip
    _RealClientError = _RealConnectionClosedError = None

_BOTO_PREFIXES = ("botocore", "boto3")


@pytest.fixture(autouse=True)
def _boto_sys_modules_hygiene():
    """Restore every boto* sys.modules entry after each test (see above)."""
    import sys as _sys

    saved = {
        name: mod
        for name, mod in _sys.modules.items()
        if name.split(".", 1)[0] in _BOTO_PREFIXES
    }
    yield
    for name in [
        n for n in _sys.modules if n.split(".", 1)[0] in _BOTO_PREFIXES
    ]:
        _sys.modules.pop(name, None)
    _sys.modules.update(saved)


@contextmanager
def _mock_botocore_session(*, return_value=None, side_effect=None):
    """Patch botocore.session even when botocore is not installed."""
    botocore_mod = ModuleType("botocore")
    session_mod = ModuleType("botocore.session")
    session_mod.get_session = MagicMock(return_value=return_value, side_effect=side_effect)
    botocore_mod.session = session_mod
    with patch.dict("sys.modules", {"botocore": botocore_mod, "botocore.session": session_mod}):
        yield session_mod.get_session


# ---------------------------------------------------------------------------
# AWS credential detection
# ---------------------------------------------------------------------------

class TestResolveAwsAuthEnvVar:
    """Test AWS credential environment variable detection.

    Mirrors OpenClaw's resolveAwsSdkEnvVarName() priority order.
    """


    def test_requires_both_access_key_and_secret(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        # Only access key, no secret → should not match
        env = {"AWS_ACCESS_KEY_ID": "AKIA..."}
        assert resolve_aws_auth_env_var(env) != "AWS_ACCESS_KEY_ID"


    def test_returns_none_when_no_aws_auth(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        # Mock botocore to return no credentials (covers EC2 IMDS fallback)
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None
        with patch.dict("sys.modules", {"botocore": MagicMock(), "botocore.session": MagicMock()}):
            import botocore.session as _bs
            _bs.get_session = MagicMock(return_value=mock_session)
            assert resolve_aws_auth_env_var({}) is None


class TestHasAwsCredentials:
    def test_true_with_profile(self):
        from agent.bedrock_adapter import has_aws_credentials
        assert has_aws_credentials({"AWS_PROFILE": "default"}) is True

    def test_false_with_empty_env(self):
        from agent.bedrock_adapter import has_aws_credentials
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None
        with patch.dict("sys.modules", {"botocore": MagicMock(), "botocore.session": MagicMock()}):
            import botocore.session as _bs
            _bs.get_session = MagicMock(return_value=mock_session)
            assert has_aws_credentials({}) is False


class TestScopedAwsSessionKwargs:
    """A served multiplex profile never signs with the launch profile's ambient AWS chain (#116313)."""

    def test_two_homes_multiplex_refuses_ambient_chain_and_keeps_standalone(self, tmp_path, monkeypatch):
        """A -> B -> A over two real homes: A (own AWS_* in .env) gets its key pair, cred-less B is
        refused at the production client seam BEFORE boto3 is touched (``{}`` would let
        ``boto3.Session()`` sign as A) and its bearer read is scoped, A again is unaffected.
        Control: a standalone run keeps the ambient chain (``{}`` + process-env bearer)."""
        from agent import bedrock_adapter, secret_scope
        from agent.bedrock_adapter import _cached_client, resolve_bedrock_bearer_token, scoped_aws_session_kwargs
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        home_a, home_b = tmp_path / "home-A", tmp_path / "home-B"
        for home in (home_a, home_b):
            home.mkdir()
        (home_a / ".env").write_text(
            "AWS_ACCESS_KEY_ID=AKIA-A\nAWS_SECRET_ACCESS_KEY=secret-A\nAWS_BEARER_TOKEN_BEDROCK=bearer-A\n"
        )
        (home_b / ".env").write_text("")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-A")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-A")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bearer-A")

        def boom():
            raise AssertionError("boto3 must not be imported for a refused profile")
        monkeypatch.setattr(bedrock_adapter, "_require_boto3", boom)

        def in_scope(home, fn):
            h_tok = set_hermes_home_override(str(home))
            s_tok = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
            try:
                return fn()
            finally:
                secret_scope.reset_secret_scope(s_tok)
                reset_hermes_home_override(h_tok)

        # Control: standalone keeps the ambient chain.
        assert scoped_aws_session_kwargs() == {}
        assert resolve_bedrock_bearer_token() == "bearer-A"

        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        a_kwargs = {"aws_access_key_id": "AKIA-A", "aws_secret_access_key": "secret-A"}
        assert in_scope(home_a, scoped_aws_session_kwargs) == a_kwargs
        with pytest.raises(RuntimeError, match="refused for this profile"):
            in_scope(home_b, lambda: _cached_client({}, "bedrock-runtime", "us-east-1"))
        assert in_scope(home_b, resolve_bedrock_bearer_token) == ""
        assert in_scope(home_a, scoped_aws_session_kwargs) == a_kwargs
        assert in_scope(home_a, resolve_bedrock_bearer_token) == "bearer-A"


class TestResolveBedrocRegion:
    def test_prefers_aws_region(self):
        from agent.bedrock_adapter import resolve_bedrock_region
        env = {"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "us-west-2"}
        assert resolve_bedrock_region(env) == "eu-west-1"


    def test_defaults_to_us_east_1(self):
        from agent.bedrock_adapter import resolve_bedrock_region
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        mock_session.get_config_variable.return_value = None
        with _mock_botocore_session(return_value=mock_session):
            assert resolve_bedrock_region({}) == "us-east-1"


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------

class TestConvertToolsToConverse:
    """Test OpenAI → Bedrock Converse tool definition conversion."""

    def test_converts_single_tool(self):
        from agent.bedrock_adapter import convert_tools_to_converse
        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from disk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            },
        }]
        result = convert_tools_to_converse(tools)
        assert len(result) == 1
        spec = result[0]["toolSpec"]
        assert spec["name"] == "read_file"
        assert spec["description"] == "Read a file from disk"
        assert spec["inputSchema"]["json"]["type"] == "object"
        assert "path" in spec["inputSchema"]["json"]["properties"]


    def test_empty_tools(self):
        from agent.bedrock_adapter import convert_tools_to_converse
        assert convert_tools_to_converse([]) == []
        assert convert_tools_to_converse(None) == []


# ---------------------------------------------------------------------------
# Message conversion: OpenAI → Converse
# ---------------------------------------------------------------------------

class TestConvertMessagesToConverse:
    """Test OpenAI message format → Bedrock Converse format conversion."""

    def test_extracts_system_prompt(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        system, msgs = convert_messages_to_converse(messages)
        assert system is not None
        assert len(system) == 1
        assert system[0]["text"] == "You are a helpful assistant."
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"


    def test_assistant_with_tool_calls(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "I'll read that file.",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/test.txt"}',
                    },
                }],
            },
        ]
        system, msgs = convert_messages_to_converse(messages)
        # 3 messages: user, assistant, trailing user (Converse requires last=user)
        assert len(msgs) == 3
        assistant_content = msgs[1]["content"]
        # Should have text block + toolUse block
        assert any("text" in b for b in assistant_content)
        tool_use_blocks = [b for b in assistant_content if "toolUse" in b]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["toolUse"]["name"] == "read_file"
        assert tool_use_blocks[0]["toolUse"]["toolUseId"] == "call_123"
        assert tool_use_blocks[0]["toolUse"]["input"] == {"path": "/tmp/test.txt"}

    def test_tool_result_becomes_user_message(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "user", "content": "Read it"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents here"},
        ]
        system, msgs = convert_messages_to_converse(messages)
        # Tool result should be in a user-role message
        tool_result_msg = [m for m in msgs if m["role"] == "user" and any(
            "toolResult" in b for b in m["content"]
        )]
        assert len(tool_result_msg) == 1
        tr = [b for b in tool_result_msg[0]["content"] if "toolResult" in b][0]
        assert tr["toolResult"]["toolUseId"] == "call_1"
        assert tr["toolResult"]["content"][0]["text"] == "file contents here"




# ---------------------------------------------------------------------------
# Response normalization: Converse → OpenAI
# ---------------------------------------------------------------------------

class TestNormalizeConverseResponse:
    """Test Bedrock Converse response → OpenAI format conversion."""

    def test_text_response(self):
        from agent.bedrock_adapter import normalize_converse_response
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello, world!"}],
                },
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        result = normalize_converse_response(response)
        assert result.choices[0].message.content == "Hello, world!"
        assert result.choices[0].message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_cache_tokens_folded_into_prompt_tokens(self):
        """Converse's inputTokens excludes cache read/write tokens (unlike
        OpenAI's prompt_tokens). normalize_converse_response must add them
        back into prompt_tokens/total_tokens and surface the Anthropic-named
        fields so normalize_usage() picks them up via its existing fallback."""
        from agent.bedrock_adapter import normalize_converse_response
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 50,
                "outputTokens": 20,
                "cacheReadInputTokens": 900,
                "cacheWriteInputTokens": 300,
            },
        }
        result = normalize_converse_response(response)
        assert result.usage.prompt_tokens == 50 + 900 + 300
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 50 + 900 + 300 + 20
        assert result.usage.cache_read_input_tokens == 900
        assert result.usage.cache_creation_input_tokens == 300

    def test_tool_use_response(self):
        from agent.bedrock_adapter import normalize_converse_response
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "I'll read that file."},
                        {
                            "toolUse": {
                                "toolUseId": "call_abc",
                                "name": "read_file",
                                "input": {"path": "/tmp/test.txt"},
                            },
                        },
                    ],
                },
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 20, "outputTokens": 15},
        }
        result = normalize_converse_response(response)
        assert result.choices[0].message.content == "I'll read that file."
        assert result.choices[0].finish_reason == "tool_calls"
        tool_calls = result.choices[0].message.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_abc"
        assert tool_calls[0].function.name == "read_file"
        assert json.loads(tool_calls[0].function.arguments) == {"path": "/tmp/test.txt"}

    def test_redacted_reasoning_is_preserved_for_replay(self):
        from agent.bedrock_adapter import normalize_converse_response

        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"reasoningContent": {"redactedContent": b"opaque-bedrock-bytes"}},
                        {"toolUse": {"toolUseId": "call_1", "name": "read_file", "input": {}}},
                    ],
                },
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1, "outputTokens": 2},
        }

        result = normalize_converse_response(response)
        details = result.choices[0].message.reasoning_details
        assert details == [{
            "type": "redacted_thinking",
            "data": "b3BhcXVlLWJlZHJvY2stYnl0ZXM=",
        }]


    def test_redacted_reasoning_replays_as_bedrock_content_block(self):
        from agent.bedrock_adapter import convert_messages_to_converse

        _system, messages = convert_messages_to_converse([
            {
                "role": "assistant",
                "content": None,
                "reasoning_details": [{
                    "type": "redacted_thinking",
                    "data": "b3BhcXVlLWJlZHJvY2stYnl0ZXM=",
                }],
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ])
        assistant = next(m for m in messages if m["role"] == "assistant")
        assert assistant["content"][0] == {
            "reasoningContent": {"redactedContent": b"opaque-bedrock-bytes"}
        }

    def test_interleaved_reasoning_and_tools_keep_exact_order(self):
        from agent.bedrock_adapter import convert_messages_to_converse, normalize_converse_response

        normalized = normalize_converse_response({
            "output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"redactedContent": b"r1"}},
                {"toolUse": {"toolUseId": "t1", "name": "one", "input": {"n": 1}}},
                {"reasoningContent": {"redactedContent": b"r2"}},
                {"toolUse": {"toolUseId": "t2", "name": "two", "input": {"n": 2}}},
            ]}},
            "stopReason": "tool_use",
        })
        msg = normalized.choices[0].message
        _system, replay = convert_messages_to_converse([{
            "role": "user", "content": "go",
        }, {
            "role": "assistant", "content": msg.content,
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls],
            "reasoning_details": msg.reasoning_details,
            "bedrock_content_blocks": msg.bedrock_content_blocks,
        }])
        blocks = replay[1]["content"]
        assert [next(iter(block)) for block in blocks] == [
            "reasoningContent", "toolUse", "reasoningContent", "toolUse"
        ]
        assert blocks[0]["reasoningContent"]["redactedContent"] == b"r1"
        assert blocks[2]["reasoningContent"]["redactedContent"] == b"r2"


# ---------------------------------------------------------------------------
# Streaming response normalization
# ---------------------------------------------------------------------------

class TestNormalizeConverseStreamEvents:
    """Test Bedrock ConverseStream event → OpenAI format conversion."""

    def test_text_stream(self):
        from agent.bedrock_adapter import normalize_converse_stream_events
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": ", world!"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3}}},
        ]}
        result = normalize_converse_stream_events(events)
        assert result.choices[0].message.content == "Hello, world!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 3

    def test_redacted_reasoning_stream_is_preserved(self):
        from agent.bedrock_adapter import normalize_converse_stream_events

        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
            {"contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"redactedContent": b"stream-secret"}},
            }},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 2, "outputTokens": 3}}},
        ]}

        result = normalize_converse_stream_events(events)
        assert result.choices[0].message.reasoning_details == [{
            "type": "redacted_thinking",
            "data": "c3RyZWFtLXNlY3JldA==",
        }]

    def test_tool_use_stream(self):
        from agent.bedrock_adapter import normalize_converse_stream_events
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {
                "toolUse": {"toolUseId": "call_1", "name": "read_file"},
            }}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "toolUse": {"input": '{"path":'},
            }}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "toolUse": {"input": '"/tmp/f"}'},
            }}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 8}}},
        ]}
        result = normalize_converse_stream_events(events)
        assert result.choices[0].finish_reason == "tool_calls"
        tc = result.choices[0].message.tool_calls
        assert len(tc) == 1
        assert tc[0].id == "call_1"
        assert tc[0].function.name == "read_file"
        assert json.loads(tc[0].function.arguments) == {"path": "/tmp/f"}

    # Real ConverseStream wire shape (captured from global.anthropic.claude-opus-5): a text block gets NO
    # contentBlockStart, only deltas stamped contentBlockIndex=0; the toolUse block then starts at index 1.
    _LIVE_TEXT_THEN_TOOL_EVENTS = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "I"}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "'ll echo "}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "banana"}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": " now."}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "echo"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": ""}}}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"s": "banana"}'}}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 8}}},
    ]

    def test_text_deltas_without_content_block_start_stay_one_block_ahead_of_tool_use(self):
        """Regression: keying text deltas by a running counter instead of their contentBlockIndex shredded
        the text into one block per delta and let the toolUse start (index 1) overwrite the second fragment
        and sort into the middle — a ``[text, toolUse, text...]`` sidecar Claude 5 on Bedrock rejects on
        replay as "does not support assistant message prefill"."""
        from agent.bedrock_adapter import normalize_converse_stream_events
        result = normalize_converse_stream_events({"stream": list(self._LIVE_TEXT_THEN_TOOL_EVENTS)})
        msg = result.choices[0].message
        assert msg.content == "I'll echo banana now."
        assert msg.bedrock_content_blocks == [
            {"text": "I'll echo banana now."},
            {"toolUse": {"toolUseId": "tooluse_1", "name": "echo", "input": {"s": "banana"}}},
        ]
        assert [tc.function.name for tc in msg.tool_calls] == ["echo"]

    def test_events_without_content_block_index_fall_back_to_arrival_order(self):
        """Proxies/test doubles may omit contentBlockIndex: deltas continue the current block, a start opens a
        new one, so text still lands before the toolUse instead of being overwritten by it."""
        from agent.bedrock_adapter import normalize_converse_stream_events
        events = [{k: {kk: vv for kk, vv in v.items() if kk != "contentBlockIndex"} for k, v in e.items()}
                  for e in self._LIVE_TEXT_THEN_TOOL_EVENTS[:-2]]
        # Text after the tool's stop must open its own slot, not land on the closed tool block.
        events += [{"contentBlockDelta": {"delta": {"text": "done"}}}, {"contentBlockStop": {}},
                   {"messageStop": {"stopReason": "tool_use"}}]
        msg = normalize_converse_stream_events({"stream": events}).choices[0].message
        assert msg.content == "I'll echo banana now.\ndone"
        assert [list(b) for b in msg.bedrock_content_blocks] == [["text"], ["toolUse"], ["text"]]
        assert msg.bedrock_content_blocks[1]["toolUse"]["input"] == {"s": "banana"}


# ---------------------------------------------------------------------------
# build_converse_kwargs
# ---------------------------------------------------------------------------

class TestBuildConverseKwargs:
    """Test the high-level kwargs builder for Converse API calls."""

    def test_basic_kwargs(self):
        from agent.bedrock_adapter import build_converse_kwargs
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        kwargs = build_converse_kwargs(
            model="anthropic.claude-sonnet-4-6-20250514-v1:0",
            messages=messages,
            max_tokens=1024,
        )
        assert kwargs["modelId"] == "anthropic.claude-sonnet-4-6-20250514-v1:0"
        assert kwargs["inferenceConfig"]["maxTokens"] == 1024
        assert kwargs["system"] is not None
        assert len(kwargs["messages"]) >= 1

    def test_includes_tools(self):
        from agent.bedrock_adapter import build_converse_kwargs
        tools = [{"type": "function", "function": {
            "name": "test", "description": "Test", "parameters": {},
        }}]
        kwargs = build_converse_kwargs(
            model="test-model", messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        assert "toolConfig" in kwargs
        assert len(kwargs["toolConfig"]["tools"]) == 1


    def test_max_tokens_none_omits_cap(self):
        """max_tokens=None omits inferenceConfig.maxTokens so Bedrock uses the
        model's maximum allowed output (the Converse field is optional)."""
        from agent.bedrock_adapter import build_converse_kwargs
        kwargs = build_converse_kwargs(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=None,
            temperature=0.1,
        )
        assert "maxTokens" not in kwargs["inferenceConfig"]
        # Other inference params still flow through.
        assert kwargs["inferenceConfig"]["temperature"] == 0.1

    def test_max_tokens_none_and_no_sampling_drops_empty_inference_config(self):
        """When every inference param is absent, don't send an empty
        inferenceConfig object on the wire."""
        from agent.bedrock_adapter import build_converse_kwargs
        kwargs = build_converse_kwargs(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=None,
        )
        assert "inferenceConfig" not in kwargs

    def test_bedrock_xai_grok_models_never_receive_sampling_params(self):
        """Bedrock-hosted xAI Grok rejects temperature/topP in Converse with a hard 400
        (ValidationException: "This model doesn't support the temperature field"); the
        _forbids_sampling_params guard is Claude-only, so Grok needs its own denylist.
        Sibling Bedrock models keep receiving sampling params."""
        from agent.bedrock_adapter import build_converse_kwargs
        msgs = [{"role": "user", "content": "Hi"}]
        for model in ("us.xai.grok-4.6", "global.xai.grok-4.6"):
            cfg = build_converse_kwargs(
                model=model, messages=msgs, temperature=0.3, top_p=0.9
            )["inferenceConfig"]
            assert "temperature" not in cfg and "topP" not in cfg, model
        for model in ("test-model", "qwen.qwen3-vl-235b-a22b"):
            cfg = build_converse_kwargs(
                model=model, messages=msgs, temperature=0.3, top_p=0.9
            )["inferenceConfig"]
            assert cfg["temperature"] == 0.3 and cfg["topP"] == 0.9, model

    def test_cache_point_added_for_supported_model(self):
        """Claude and Nova on the Converse path get cachePoint markers on
        system, tools, and the message before the newest turn."""
        from agent.bedrock_adapter import build_converse_kwargs
        tools = [{"type": "function", "function": {
            "name": "test", "description": "Test", "parameters": {},
        }}]
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        kwargs = build_converse_kwargs(
            model="anthropic.claude-sonnet-4-6-20250514-v1:0",
            messages=messages,
            tools=tools,
        )
        assert kwargs["system"][-1] == {"cachePoint": {"type": "default"}}
        assert kwargs["toolConfig"]["tools"][-1] == {"cachePoint": {"type": "default"}}
        # Second-to-last converse message (the assistant "Reply" turn) carries
        # the checkpoint; the newest "Second" turn does not.
        marked = kwargs["messages"][-2]["content"]
        assert marked[-1] == {"cachePoint": {"type": "default"}}
        assert kwargs["messages"][-1]["content"][-1] != {"cachePoint": {"type": "default"}}

    def test_no_cache_point_for_unsupported_model(self):
        from agent.bedrock_adapter import build_converse_kwargs
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        kwargs = build_converse_kwargs(model="meta.llama3-70b-instruct-v1:0", messages=messages)
        assert {"cachePoint": {"type": "default"}} not in kwargs["system"]
        for m in kwargs["messages"]:
            assert {"cachePoint": {"type": "default"}} not in m["content"]


class TestConverseReasoningEffort:
    """Bedrock-hosted OpenAI GPT on Converse takes the effort as
    ``additionalModelRequestFields={"reasoning": {"effort": ...}}`` (live-verified: the nested shape is
    validated, the flat ``reasoning_effort`` key is ``unknown_parameter``). Every other family on
    Converse keeps its request unchanged."""

    MSGS = [{"role": "user", "content": "Hi"}]
    # Live-verified accepted levels per model generation (bedrock-runtime us-east-1).
    GPT6_LEVELS = {"none", "low", "medium", "high", "xhigh", "max"}
    GPT55_LEVELS = {"none", "low", "medium", "high", "xhigh"}

    @staticmethod
    def _fields(model, reasoning_config, messages=None):
        from agent.bedrock_adapter import build_converse_kwargs
        kwargs = build_converse_kwargs(
            model=model, messages=messages or TestConverseReasoningEffort.MSGS, reasoning_config=reasoning_config,
        )
        return kwargs.get("additionalModelRequestFields")

    @pytest.mark.parametrize("model", ["us.openai.gpt-6-sol", "global.openai.gpt-6-sol", "openai.gpt-6-sol",
                                       "us.openai.gpt-6-luna", "us.openai.gpt-5.6-sol"])
    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_supported_effort_reaches_the_wire_under_any_profile_prefix(self, model, effort):
        fields = self._fields(model, {"enabled": True, "effort": effort})
        assert fields == {"reasoning": {"effort": effort}}

    @pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"])
    def test_every_hermes_level_maps_into_the_accepted_vocabulary(self, effort):
        """No Hermes level may produce a value Converse rejects (``minimal`` is unsupported, ``ultra``
        is Hermes-internal; both are live 400s when sent verbatim)."""
        for model, accepted in (("us.openai.gpt-6-sol", self.GPT6_LEVELS), ("us.openai.gpt-5.5", self.GPT55_LEVELS)):
            cfg = {"enabled": False} if effort == "none" else {"enabled": True, "effort": effort}
            fields = self._fields(model, cfg)
            assert fields is not None, (model, effort)
            assert fields["reasoning"]["effort"] in accepted, (model, effort, fields)

    def test_unsupported_levels_clamp_to_the_nearest_accepted_level_without_escalating(self):
        gpt6 = lambda e: self._fields("us.openai.gpt-6-sol", {"enabled": True, "effort": e})["reasoning"]["effort"]
        gpt55 = lambda e: self._fields("us.openai.gpt-5.5", {"enabled": True, "effort": e})["reasoning"]["effort"]
        assert gpt6("ultra") == "max"
        assert gpt6("minimal") == "low"  # nothing weaker except "none", which would switch reasoning off
        assert gpt55("max") == gpt55("ultra") == "xhigh"

    def test_explicit_disable_sends_none_where_the_model_has_it(self):
        """An omitted field leaves the model's default effort on, so a disable must be sent explicitly."""
        assert self._fields("us.openai.gpt-6-sol", {"enabled": False}) == {"reasoning": {"effort": "none"}}

    def test_disable_is_omitted_on_a_model_without_none(self):
        """gpt-6-astra rejects ``none`` (live 400): the disable degrades to the route default, not a 400."""
        assert self._fields("us.openai.gpt-6-astra", {"enabled": False}) is None
        assert self._fields("us.openai.gpt-6-astra", {"enabled": True, "effort": "high"}) == {
            "reasoning": {"effort": "high"}}

    @pytest.mark.parametrize("cfg", [None, {}, {"enabled": True}, {"enabled": True, "effort": ""},
                                     {"enabled": True, "effort": "turbo"}])
    def test_unset_or_unmappable_effort_sends_nothing(self, cfg):
        """Unset stays unset (the model's own default), and a bespoke name never reaches a validating wire."""
        assert self._fields("us.openai.gpt-6-sol", cfg) is None

    @pytest.mark.parametrize("model", [
        "us.anthropic.claude-opus-4-6-v1", "anthropic.claude-sonnet-4-6-20250514-v1:0",
        "openai.gpt-oss-120b-1:0", "openai.gpt-oss-20b-1:0", "openai.gpt-oss-safeguard-120b",
        "us.amazon.nova-pro-v1:0", "deepseek.v3-v1:0", "us.xai.grok-4.6", "us.meta.llama4-maverick-17b-instruct-v1:0",
    ])
    def test_other_families_never_get_the_openai_reasoning_field(self, model):
        for cfg in ({"enabled": True, "effort": "high"}, {"enabled": False}):
            assert self._fields(model, cfg) is None, model

    def test_reasoning_field_is_byte_stable_as_the_conversation_grows(self):
        """Prompt caching: the field depends only on model + configured effort, never on history."""
        cfg = {"enabled": True, "effort": "high"}
        short = self._fields("us.openai.gpt-6-sol", cfg)
        grown = self._fields("us.openai.gpt-6-sol", cfg, messages=[
            {"role": "system", "content": "Be helpful."}, {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"}, {"role": "user", "content": "Second"},
        ])
        assert json.dumps(short, sort_keys=True) == json.dumps(grown, sort_keys=True)

    def test_call_converse_sends_the_field_to_boto3(self):
        from agent.bedrock_adapter import call_converse
        client = MagicMock()
        client.converse.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client):
            call_converse(region="us-east-1", model="us.openai.gpt-6-sol", messages=self.MSGS,
                          reasoning_config={"enabled": True, "effort": "low"})
        assert client.converse.call_args.kwargs["additionalModelRequestFields"] == {"reasoning": {"effort": "low"}}

    def test_transport_forwards_reasoning_config(self):
        import agent.transports.bedrock  # noqa: F401
        from agent.transports import get_transport
        kw = get_transport("bedrock_converse").build_kwargs(
            model="us.openai.gpt-6-sol", messages=self.MSGS, reasoning_config={"enabled": True, "effort": "xhigh"})
        assert kw["additionalModelRequestFields"] == {"reasoning": {"effort": "xhigh"}}


# ---------------------------------------------------------------------------
# cachePoint rejection self-heal (#97281)
# ---------------------------------------------------------------------------

CACHE_POINT = {"cachePoint": {"type": "default"}}

NOVA_TOOLS_REJECTION = (
    "An error occurred (ValidationException) when calling the ConverseStream "
    "operation: The model returned the following errors: Malformed input "
    "request: #/toolConfig/tools/18: extraneous key [cachePoint] is not "
    "permitted, please reformat your input and try again."
)


@pytest.fixture(autouse=True)
def _clean_cache_point_rejections():
    """Rejections are process-wide; keep them from leaking between tests."""
    from agent.bedrock_adapter import reset_cache_point_rejections
    reset_cache_point_rejections()
    yield
    reset_cache_point_rejections()


class TestCachePointRejectionRecovery:
    """Bedrock placement rules are per-family and per-field: Nova accepts
    cachePoint in system/messages but rejects it inside toolConfig.tools,
    failing 100% of tool-enabled turns (#97281). The server verdict is the
    authority - record it, drop that one marker, and retry."""

    def _nova_kwargs(self):
        from agent.bedrock_adapter import build_converse_kwargs
        return build_converse_kwargs(
            model="us.amazon.nova-pro-v1:0",
            messages=[
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Reply"},
                {"role": "user", "content": "Second"},
            ],
            tools=[{"type": "function", "function": {
                "name": "test", "description": "Test", "parameters": {},
            }}],
        )

    def test_classifies_tools_rejection(self):
        from agent.bedrock_adapter import cache_point_rejection_placement
        assert cache_point_rejection_placement(
            Exception(NOVA_TOOLS_REJECTION)
        ) == "tools"

    def test_classifies_system_and_messages_rejections(self):
        from agent.bedrock_adapter import cache_point_rejection_placement
        assert cache_point_rejection_placement(Exception(
            "Malformed input request: #/system/1: extraneous key [cachePoint] "
            "is not permitted"
        )) == "system"
        assert cache_point_rejection_placement(Exception(
            "Malformed input request: #/messages/2/content/3: extraneous key "
            "[cachePoint] is not permitted"
        )) == "messages"

    def test_ignores_unrelated_errors(self):
        from agent.bedrock_adapter import cache_point_rejection_placement
        assert cache_point_rejection_placement(
            Exception("ThrottlingException: Too many requests")
        ) is None
        assert cache_point_rejection_placement(Exception(
            "Malformed input request: #/toolConfig/tools/0: extraneous key "
            "[toolChoice] is not permitted"
        )) is None

    def test_strip_removes_only_the_rejected_placement(self):
        from agent.bedrock_adapter import strip_cache_points
        kwargs = self._nova_kwargs()
        assert CACHE_POINT in kwargs["toolConfig"]["tools"]
        stripped = strip_cache_points(kwargs, "tools")
        assert CACHE_POINT not in stripped["toolConfig"]["tools"]
        # system and messages markers survive - Nova accepts those.
        assert stripped["system"][-1] == CACHE_POINT
        assert stripped["messages"][-2]["content"][-1] == CACHE_POINT
        # Original kwargs are untouched (no in-place mutation).
        assert CACHE_POINT in kwargs["toolConfig"]["tools"]

    def test_strip_is_identity_when_marker_absent(self):
        """No marker to remove -> same object, so callers know a retry is futile."""
        from agent.bedrock_adapter import strip_cache_points
        kwargs = {"modelId": "x", "toolConfig": {"tools": [{"toolSpec": {}}]}}
        assert strip_cache_points(kwargs, "tools") is kwargs

    def test_recovery_records_verdict_so_later_turns_omit_the_marker(self):
        from agent.bedrock_adapter import recover_from_cache_point_rejection
        kwargs = self._nova_kwargs()
        retry = recover_from_cache_point_rejection(
            Exception(NOVA_TOOLS_REJECTION), kwargs
        )
        assert retry is not None
        assert CACHE_POINT not in retry["toolConfig"]["tools"]
        # Next turn is built clean without another round-trip failure.
        rebuilt = self._nova_kwargs()
        assert CACHE_POINT not in rebuilt["toolConfig"]["tools"]
        assert rebuilt["system"][-1] == CACHE_POINT
        assert rebuilt["messages"][-2]["content"][-1] == CACHE_POINT

    def test_verdict_is_scoped_to_the_rejecting_model(self):
        from agent.bedrock_adapter import (
            build_converse_kwargs,
            recover_from_cache_point_rejection,
        )
        recover_from_cache_point_rejection(
            Exception(NOVA_TOOLS_REJECTION), self._nova_kwargs()
        )
        claude = build_converse_kwargs(
            model="anthropic.claude-sonnet-4-6-20250514-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {
                "name": "test", "description": "Test", "parameters": {},
            }}],
        )
        assert claude["toolConfig"]["tools"][-1] == CACHE_POINT

    def test_recovery_declines_when_nothing_can_be_stripped(self):
        """A cachePoint rejection with no marker present must re-raise, not loop."""
        from agent.bedrock_adapter import recover_from_cache_point_rejection
        kwargs = {"modelId": "us.amazon.nova-pro-v1:0",
                  "toolConfig": {"tools": [{"toolSpec": {}}]}}
        assert recover_from_cache_point_rejection(
            Exception(NOVA_TOOLS_REJECTION), kwargs
        ) is None

    def test_call_converse_retries_without_the_marker(self):
        from agent.bedrock_adapter import call_converse
        client = MagicMock()
        client.converse.side_effect = [
            Exception(NOVA_TOOLS_REJECTION),
            {"output": {"message": {"role": "assistant",
                                    "content": [{"text": "ok"}]}},
             "stopReason": "end_turn",
             "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}},
        ]
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client",
                   return_value=client):
            response = call_converse(
                region="us-east-1",
                model="us.amazon.nova-pro-v1:0",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {
                    "name": "test", "description": "Test", "parameters": {},
                }}],
            )
        assert response.choices[0].message.content == "ok"
        assert client.converse.call_count == 2
        first, second = client.converse.call_args_list
        assert CACHE_POINT in first.kwargs["toolConfig"]["tools"]
        assert CACHE_POINT not in second.kwargs["toolConfig"]["tools"]

    def test_call_converse_reraises_unrelated_errors(self):
        from agent.bedrock_adapter import call_converse
        client = MagicMock()
        client.converse.side_effect = Exception("ThrottlingException")
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client",
                   return_value=client):
            with pytest.raises(Exception, match="ThrottlingException"):
                call_converse(
                    region="us-east-1",
                    model="us.amazon.nova-pro-v1:0",
                    messages=[{"role": "user", "content": "hi"}],
                )
        assert client.converse.call_count == 1


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

class TestDiscoverBedrockModels:
    """Test Bedrock model discovery with mocked AWS API calls."""


    def test_provider_filter(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-v2",
                    "modelName": "Claude v2",
                    "providerName": "Anthropic",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "modelLifecycle": {"status": "ACTIVE"},
                },
                {
                    "modelId": "amazon.titan-text",
                    "modelName": "Titan",
                    "providerName": "Amazon",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "modelLifecycle": {"status": "ACTIVE"},
                },
            ],
        }
        mock_client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}

        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=mock_client):
            models = discover_bedrock_models("us-east-1", provider_filter=["anthropic"])

        assert len(models) == 1
        assert models[0]["id"] == "anthropic.claude-v2"

    def test_caches_results(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [{
                "modelId": "test-model",
                "modelName": "Test",
                "providerName": "Test",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
                "modelLifecycle": {"status": "ACTIVE"},
            }],
        }
        mock_client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}

        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=mock_client):
            first = discover_bedrock_models("us-east-1")
            second = discover_bedrock_models("us-east-1")

        # Should only call the API once (second call uses cache)
        assert mock_client.list_foundation_models.call_count == 1
        assert first == second


    def test_handles_api_error_gracefully(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        with patch("agent.bedrock_adapter._get_bedrock_control_client", side_effect=Exception("No creds")):
            models = discover_bedrock_models("us-east-1")

        assert models == []


class TestExtractProviderFromArn:
    def test_extracts_anthropic(self):
        from agent.bedrock_adapter import _extract_provider_from_arn
        arn = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6"
        assert _extract_provider_from_arn(arn) == "anthropic"


    def test_returns_empty_for_invalid_arn(self):
        from agent.bedrock_adapter import _extract_provider_from_arn
        assert _extract_provider_from_arn("not-an-arn") == ""
        assert _extract_provider_from_arn("") == ""


# ---------------------------------------------------------------------------
# Client cache management
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Streaming with callbacks
# ---------------------------------------------------------------------------

class TestStreamConverseWithCallbacks:
    """Test real-time streaming with delta callbacks."""

    def test_cache_tokens_folded_into_prompt_tokens(self):
        """The streaming path must fold cacheRead/WriteInputTokens into
        prompt_tokens the same way the non-streaming path does (see
        TestNormalizeConverseResponse.test_cache_tokens_folded_into_prompt_tokens)."""
        from agent.bedrock_adapter import stream_converse_with_callbacks
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {
                "inputTokens": 50,
                "outputTokens": 20,
                "cacheReadInputTokens": 900,
                "cacheWriteInputTokens": 300,
            }}},
        ]}
        result = stream_converse_with_callbacks(events)
        assert result.usage.prompt_tokens == 50 + 900 + 300
        assert result.usage.total_tokens == 50 + 900 + 300 + 20
        assert result.usage.cache_read_input_tokens == 900
        assert result.usage.cache_creation_input_tokens == 300


    def test_text_deltas_suppressed_when_tool_use_present(self):
        """Text deltas should NOT fire when tool_use blocks are present."""
        from agent.bedrock_adapter import stream_converse_with_callbacks
        deltas = []
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Let me check."}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockStart": {"contentBlockIndex": 1, "start": {
                "toolUse": {"toolUseId": "c1", "name": "search"},
            }}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {
                "toolUse": {"input": '{"q":"test"}'},
            }}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0}}},
        ]}
        result = stream_converse_with_callbacks(
            events, on_text_delta=lambda t: deltas.append(t),
        )
        # Text delta for "Let me check." should fire (before tool_use was seen)
        assert "Let me check." in deltas
        # But the result should still have both text and tool calls
        assert result.choices[0].message.content == "Let me check."
        assert len(result.choices[0].message.tool_calls) == 1


# ---------------------------------------------------------------------------
# Guardrail config in build_converse_kwargs
# ---------------------------------------------------------------------------

class TestGuardrailConfig:
    """Test that guardrail configuration is correctly passed through."""

    def test_guardrail_included_in_kwargs(self):
        from agent.bedrock_adapter import build_converse_kwargs
        guardrail = {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1",
            "streamProcessingMode": "async",
            "trace": "enabled",
        }
        kwargs = build_converse_kwargs(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            guardrail_config=guardrail,
        )
        assert kwargs["guardrailConfig"] == guardrail


    def test_no_guardrail_when_empty_dict(self):
        from agent.bedrock_adapter import build_converse_kwargs
        kwargs = build_converse_kwargs(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            guardrail_config={},
        )
        # Empty dict is falsy, should not be included
        assert "guardrailConfig" not in kwargs


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestBedrockContextLength:
    """Test Bedrock model context length lookup."""


    def test_unknown_model_gets_default(self):
        from agent.bedrock_adapter import get_bedrock_context_length, BEDROCK_DEFAULT_CONTEXT_LENGTH
        assert get_bedrock_context_length("unknown.model-v1:0") == BEDROCK_DEFAULT_CONTEXT_LENGTH


    def test_no_region_skips_probe_uses_table(self):
        # Default call (no region) must NOT hit the network — returns the
        # static table value.  Guards backward compatibility for callers that
        # still invoke get_bedrock_context_length(model_id) with one arg.
        from agent.bedrock_adapter import get_bedrock_context_length
        with patch("agent.bedrock_adapter.probe_bedrock_context_length") as mock_probe:
            assert get_bedrock_context_length("anthropic.claude-opus-4-6") == 1_000_000
            mock_probe.assert_not_called()


class TestInferenceProfileContextLength:
    """Application-inference-profile ARNs name no model, so the window must come from the model the
    profile wraps via GetInferenceProfile — on the production call shape (no region, probe=False)."""

    ARN = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abcdef123456"

    def setup_method(self):
        from agent import bedrock_adapter
        bedrock_adapter._inference_profile_model_cache.clear()

    def test_arn_resolves_wrapped_model_window_in_the_arn_region(self):
        # No region passed (agent/model_metadata.py::_resolve_bedrock_context_length passes none) and
        # AWS_REGION elsewhere: the lookup must still run, in the ARN's own region, with the
        # parameter name botocore actually validates (inferenceProfileIdentifier).
        from agent.bedrock_adapter import get_bedrock_context_length
        client = MagicMock()
        client.get_inference_profile.return_value = {"models": [
            {"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-6"}]}
        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=client) as factory, \
                patch.dict("os.environ", {"AWS_REGION": "us-east-1"}):
            assert get_bedrock_context_length(self.ARN, probe=False) == 1_000_000
            assert get_bedrock_context_length(self.ARN, probe=False) == 1_000_000  # cached per process
        factory.assert_called_once_with("us-west-2")
        client.get_inference_profile.assert_called_once_with(inferenceProfileIdentifier=self.ARN)

    def test_resolution_denied_falls_back_to_default_with_warning(self, caplog):
        # Without bedrock:GetInferenceProfile the default window applies and the silence is broken
        # with a WARNING naming the profile and the explicit-config escape hatch.
        from agent.bedrock_adapter import get_bedrock_context_length, BEDROCK_DEFAULT_CONTEXT_LENGTH
        client = MagicMock()
        client.get_inference_profile.side_effect = Exception("AccessDeniedException")
        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=client), \
                caplog.at_level("WARNING", logger="agent.bedrock_adapter"):
            assert get_bedrock_context_length(self.ARN, probe=False) == BEDROCK_DEFAULT_CONTEXT_LENGTH
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1 and self.ARN in warnings[0].getMessage()
        assert "GetInferenceProfile" in warnings[0].getMessage()

    def test_profile_wrapping_claude_keeps_prompt_cache_markers(self):
        # build_converse_kwargs gates cachePoint on the model id; the opaque profile ARN must be
        # resolved to the wrapped Claude (cached lookup) or the profile silently loses prompt caching.
        from agent.bedrock_adapter import build_converse_kwargs
        client = MagicMock()
        client.get_inference_profile.return_value = {"models": [
            {"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-6"}]}
        messages = [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Hi"}]
        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=client):
            kwargs = build_converse_kwargs(model=self.ARN, messages=messages)
        assert kwargs["modelId"] == self.ARN  # the request still targets the profile
        assert kwargs["system"][-1] == {"cachePoint": {"type": "default"}}


class TestBedrockContextProbe:
    """Test the live context-window probe that reads the real window from
    Bedrock's 'prompt is too long' validation error."""

    def _client_raising(self, message):
        client = MagicMock()
        client.converse.side_effect = Exception(message)
        return client


    def test_probe_returns_none_when_client_unavailable(self):
        from agent.bedrock_adapter import probe_bedrock_context_length
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client",
                   side_effect=RuntimeError("boto3 missing")):
            assert probe_bedrock_context_length("any.model", "eu-central-1") is None

    def test_probe_result_beats_static_table(self):
        # A successful probe (1M) must override the stale table value (200K
        # via the 'anthropic.claude-opus-4' substring match).
        from agent.bedrock_adapter import get_bedrock_context_length
        err = "prompt is too long: 5000032 tokens > 1000000 maximum"
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client",
                   return_value=self._client_raising(err)):
            assert get_bedrock_context_length(
                "eu.anthropic.claude-opus-4-8",
                region="eu-central-1") == 1_000_000


# ---------------------------------------------------------------------------
# Tool-calling capability detection
# ---------------------------------------------------------------------------



class TestBuildConverseKwargsToolStripping:
    """Test that tools are stripped for non-tool-calling models."""


    def test_tools_stripped_for_deepseek_r1(self):
        from agent.bedrock_adapter import build_converse_kwargs
        tools = [{"type": "function", "function": {"name": "test", "description": "t", "parameters": {}}}]
        kwargs = build_converse_kwargs(
            model="us.deepseek.r1-v1:0",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        assert "toolConfig" not in kwargs


# ---------------------------------------------------------------------------
# Dual-path model routing
# ---------------------------------------------------------------------------

class TestIsAnthropicBedrockModel:
    """Test Claude model detection for dual-path routing."""

    def test_us_claude_sonnet(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("us.anthropic.claude-sonnet-4-6") is True


    def test_nova_is_not_anthropic(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("us.amazon.nova-pro-v1:0") is False


    def test_au_inference_profile(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("au.anthropic.claude-haiku-4-5-20251001-v1:0") is True
        assert is_anthropic_bedrock_model("au.anthropic.claude-sonnet-4-6") is True


class TestEmptyTextBlockFix:
    """Test that empty/whitespace-only text blocks are replaced with a
    non-whitespace placeholder (not a literal space, which is itself
    whitespace and gets rejected by the same Bedrock validation rule)."""

    def test_none_content_gets_placeholder(self):
        from agent.bedrock_adapter import _convert_content_to_converse, _EMPTY_TEXT_PLACEHOLDER
        blocks = _convert_content_to_converse(None)
        assert blocks[0]["text"] == _EMPTY_TEXT_PLACEHOLDER
        assert blocks[0]["text"].strip()




# ---------------------------------------------------------------------------
# Stale-connection detection and per-region client invalidation
# ---------------------------------------------------------------------------

class TestInvalidateRuntimeClient:
    """Per-region eviction used to discard dead/stale bedrock-runtime clients."""

    def test_evicts_only_the_target_region(self):
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            invalidate_runtime_client,
            reset_client_cache,
        )
        reset_client_cache()
        _bedrock_runtime_client_cache["us-east-1"] = "dead-client"
        _bedrock_runtime_client_cache["us-west-2"] = "live-client"

        evicted = invalidate_runtime_client("us-east-1")

        assert evicted is True
        assert "us-east-1" not in _bedrock_runtime_client_cache
        assert _bedrock_runtime_client_cache["us-west-2"] == "live-client"

    def test_returns_false_when_region_not_cached(self):
        from agent.bedrock_adapter import invalidate_runtime_client, reset_client_cache
        reset_client_cache()
        assert invalidate_runtime_client("eu-west-1") is False


class TestIsStaleConnectionError:
    """Classifier that decides whether an exception warrants client eviction."""


    def test_detects_botocore_read_timeout(self):
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")
        from agent.bedrock_adapter import is_stale_connection_error
        from botocore.exceptions import ReadTimeoutError
        exc = ReadTimeoutError(endpoint_url="https://bedrock.example")
        assert is_stale_connection_error(exc) is True


    def test_detects_library_internal_assertion_error(self):
        """A bare AssertionError raised from inside urllib3/botocore signals
        a corrupted connection-pool invariant and should trigger eviction."""
        from agent.bedrock_adapter import is_stale_connection_error

        # Fabricate an AssertionError whose traceback's last frame belongs
        # to a module named "urllib3.connectionpool". We do this by exec'ing
        # a tiny `assert False` under a fake globals dict — the resulting
        # frame's ``f_globals["__name__"]`` is what the classifier inspects.
        fake_globals = {"__name__": "urllib3.connectionpool"}
        try:
            exec("def _boom():\n    assert False\n_boom()", fake_globals)
        except AssertionError as exc:
            assert is_stale_connection_error(exc) is True
        else:
            pytest.fail("AssertionError not raised")


    def test_ignores_unrelated_exceptions(self):
        from agent.bedrock_adapter import is_stale_connection_error
        assert is_stale_connection_error(ValueError("bad input")) is False
        assert is_stale_connection_error(KeyError("missing")) is False


class TestCallConverseInvalidatesOnStaleError:
    """call_converse evicts the cached client only on a stale-connection error — so the
    next invocation reconnects instead of reusing the dead socket (the agent's streaming
    path is pinned in ``TestAgentBedrockStreamRecovery``)."""

    def test_converse_does_not_evict_on_non_stale_error(self):
        """Non-stale errors (e.g. ValidationException) leave the client cache alone."""
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            call_converse,
            reset_client_cache,
        )
        from botocore.exceptions import ClientError

        reset_client_cache()
        live_client = MagicMock()
        live_client.converse.side_effect = ClientError(
            error_response={"Error": {"Code": "ValidationException", "Message": "bad"}},
            operation_name="Converse",
        )
        _bedrock_runtime_client_cache["us-east-1"] = live_client

        with pytest.raises(ClientError):
            call_converse(
                region="us-east-1",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert _bedrock_runtime_client_cache.get("us-east-1") is live_client, (
            "validation errors do not indicate a dead connection — keep the client"
        )


class TestStreamingAccessDeniedDetection:
    """is_streaming_access_denied_error() recognizes IAM denials of
    bedrock:InvokeModelWithResponseStream (InvokeModel-only policies)."""

    def _denied_client_error(self):
        from botocore.exceptions import ClientError
        return ClientError(
            error_response={
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": (
                        "User: arn:aws:iam::123456789012:user/x is not "
                        "authorized to perform: "
                        "bedrock:InvokeModelWithResponseStream on resource: "
                        "arn:aws:bedrock:us-east-1::foundation-model/"
                        "anthropic.claude-3-sonnet-20240229-v1:0"
                    ),
                }
            },
            operation_name="ConverseStream",
        )

    def test_matches_access_denied_client_error(self):
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")
        from agent.bedrock_adapter import is_streaming_access_denied_error
        assert is_streaming_access_denied_error(self._denied_client_error()) is True


    def test_ignores_unrelated_errors(self):
        from agent.bedrock_adapter import is_streaming_access_denied_error
        assert is_streaming_access_denied_error(ValueError("boom")) is False
        assert is_streaming_access_denied_error(
            RuntimeError("stream not supported")
        ) is False


class TestAgentBedrockStreamRecovery:
    """The agent loop streams through ``chat_completion_helpers._bedrock_converse_call``;
    pin the recovery ladder on that live path:
    IAM streaming denial → ``_BedrockStream._fall_back_to_converse`` (non-streaming
    converse, streaming disabled for the session, client kept), stale connection →
    cached client evicted so the outer retry reconnects."""

    _KW = {"__bedrock_region__": "us-east-1", "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
           "messages": [{"role": "user", "content": [{"text": "hi"}]}]}

    def test_streaming_denial_falls_back_to_converse_via_bedrock_stream(self):
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")
        from types import SimpleNamespace
        from agent.bedrock_adapter import _bedrock_runtime_client_cache, reset_client_cache
        from agent.chat_completion_helpers import _BedrockStream
        from botocore.exceptions import ClientError

        reset_client_cache()
        client = MagicMock()
        client.converse_stream.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDeniedException", "Message": (
                "User is not authorized to perform: bedrock:InvokeModelWithResponseStream")}},
            operation_name="ConverseStream",
        )
        client.converse.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        _bedrock_runtime_client_cache["us-east-1"] = client
        agent = SimpleNamespace(_disable_streaming=False, _safe_print=MagicMock(), model="m", provider="bedrock")
        stream = _BedrockStream(agent, dict(self._KW), on_first_delta=None)

        result = stream._open_stream(dict(self._KW))

        client.converse.assert_called_once()
        assert "__bedrock_region__" not in client.converse.call_args.kwargs
        assert result.choices[0].message.content == "hi"
        assert agent._disable_streaming is True
        assert "InvokeModelWithResponseStream" in agent._safe_print.call_args.args[0]
        # Not a stale connection — client stays cached.
        assert _bedrock_runtime_client_cache.get("us-east-1") is client

    def test_stale_connection_evicts_client_on_agent_stream_path(self):
        pytest.importorskip("botocore.exceptions", reason="botocore (with working exceptions module) required")
        from agent.bedrock_adapter import _bedrock_runtime_client_cache, reset_client_cache
        from agent.chat_completion_helpers import _bedrock_converse_call
        from botocore.exceptions import ConnectionClosedError

        reset_client_cache()
        dead_client = MagicMock()
        dead_client.converse_stream.side_effect = ConnectionClosedError(endpoint_url="https://bedrock.example")
        _bedrock_runtime_client_cache["us-east-1"] = dead_client
        denied = MagicMock()

        with pytest.raises(ConnectionClosedError):
            _bedrock_converse_call(dict(self._KW), stream=True, on_stream_denied=denied)

        denied.assert_not_called()
        assert "us-east-1" not in _bedrock_runtime_client_cache


# ---------------------------------------------------------------------------
# boto3 version check
# ---------------------------------------------------------------------------


class TestRequireBoto3VersionCheck:
    """Test that _require_boto3() rejects boto3 versions older than 1.34.59."""

    def test_raises_runtime_error_when_boto3_too_old(self):
        """boto3 < 1.34.59 should raise RuntimeError with upgrade instructions."""
        from agent.bedrock_adapter import _require_boto3

        fake_boto3 = MagicMock()
        fake_boto3.__version__ = "1.34.46"
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            with pytest.raises(RuntimeError, match="does not support converse_stream"):
                _require_boto3()



class TestImageBase64Decoding:
    """Image data URLs must be decoded to raw bytes before passing to Converse API.

    boto3 re-encodes at the wire layer, so passing the base64 string directly
    results in double-encoding. Bedrock rejects with 'Failed to sanitize image'.
    Ref: #33317.
    """

    def test_data_url_decoded_to_bytes(self):
        from agent.bedrock_adapter import _convert_content_to_converse
        import base64

        # A tiny 1x1 red PNG
        raw_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

        content = [{"type": "image_url", "image_url": {"url": data_url}}]
        blocks = _convert_content_to_converse(content)

        assert len(blocks) == 1
        img_block = blocks[0]["image"]
        assert img_block["format"] == "png"
        # Must be raw bytes, not a base64 string
        assert isinstance(img_block["source"]["bytes"], bytes)
        assert img_block["source"]["bytes"] == raw_png

    def test_invalid_base64_falls_back_to_encode(self):
        from agent.bedrock_adapter import _convert_content_to_converse

        data_url = "data:image/jpeg;base64,NOT_VALID_BASE64!!!"
        content = [{"type": "image_url", "image_url": {"url": data_url}}]
        blocks = _convert_content_to_converse(content)

        # Should not crash — falls back to encoding the string as bytes
        assert len(blocks) == 1
        assert isinstance(blocks[0]["image"]["source"]["bytes"], bytes)


class TestBearerTokenRoutesToConverse:
    """Bearer Token users must go through Converse API, not AnthropicBedrock SDK.

    The AnthropicBedrock SDK only supports SigV4 signing — it cannot use
    AWS_BEARER_TOKEN_BEDROCK. Ref: #28156.
    """

    def _resolve(self, monkeypatch, *, bearer: bool):
        import os

        from hermes_cli import runtime_provider as rp

        if bearer:
            monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bearer-token-123")
        else:
            monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        assert "AWS_BEARER_TOKEN_BEDROCK" in os.environ or not bearer

        monkeypatch.setattr(
            rp,
            "_get_model_config",
            lambda: {
                "default": "us.anthropic.claude-sonnet-4-6",
                "provider": "bedrock",
            },
        )
        monkeypatch.setattr(rp, "load_config", lambda: {"bedrock": {}})
        return rp.resolve_runtime_provider(requested="bedrock")

    def test_bearer_token_forces_converse_for_claude(self, monkeypatch):
        """Claude model + Bearer Token → bedrock_converse, not anthropic_messages."""
        runtime = self._resolve(monkeypatch, bearer=True)
        assert runtime["api_mode"] == "bedrock_converse"
        assert "bedrock_anthropic" not in runtime

    def test_sigv4_claude_still_uses_anthropic_bedrock_sdk(self, monkeypatch):
        """Without a bearer token, Claude keeps the AnthropicBedrock SDK path."""
        runtime = self._resolve(monkeypatch, bearer=False)
        assert runtime["api_mode"] == "anthropic_messages"
        assert runtime.get("bedrock_anthropic") is True


# ---------------------------------------------------------------------------
# Reasoning replay through the Converse tagged union + sealed-blob resend-once (#115865)
# ---------------------------------------------------------------------------

CROSS_REGION_REJECTION = (
    "An error occurred (ValidationException) when calling the ConverseStream operation: The model returned "
    'the following errors: {"error":{"code":"validation_error","message":"Encrypted content cannot be used in a '
    'different region from the one that created it.","param":null,"type":"invalid_request_error"}}'
)


def _kimi_turn_history():
    """Turn 1 as the ConverseStream path captures it: signed thinking, a sealed blob, then a tool call."""
    from agent.bedrock_adapter import normalize_converse_stream_events
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "let me think"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig-1"}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"reasoningContent": {"redactedContent": b"sealed"}}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"contentBlockStart": {"contentBlockIndex": 2, "start": {"toolUse": {"toolUseId": "t1", "name": "read_file"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 2, "delta": {"toolUse": {"input": "{}"}}}},
        {"contentBlockStop": {"contentBlockIndex": 2}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
    ]
    msg = normalize_converse_stream_events({"stream": events}).choices[0].message
    return [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "reasoning_content": msg.reasoning_content,
         "reasoning_details": msg.reasoning_details, "bedrock_content_blocks": msg.bedrock_content_blocks,
         "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
    ]


def _ok_converse_response():
    return {"output": {"message": {"role": "assistant", "content": [{"text": "done"}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1}}


class TestReasoningReplaySchema:
    """Converse ``reasoningContent`` is a tagged union (``reasoningText{text,signature}`` | ``redactedContent``);
    replaying captured thinking as a bare ``text`` key dies client-side with ParamValidationError (#115865)."""

    def test_call_converse_replays_thinking_botocore_accepts(self):
        pytest.importorskip("botocore.session", reason="botocore (bedrock extra) required")
        import botocore.session
        from botocore.validate import validate_parameters
        from agent.bedrock_adapter import call_converse
        shape = botocore.session.get_session().get_service_model("bedrock-runtime").operation_model("Converse").input_shape
        client = MagicMock()

        def converse(**kwargs):
            validate_parameters(kwargs, shape)  # the real client's client-side validation
            return _ok_converse_response()
        client.converse.side_effect = converse
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client):
            response = call_converse(region="us-east-1", model="global.moonshotai.kimi-k3", messages=_kimi_turn_history())
        assert response.choices[0].message.content == "done"
        replayed = client.converse.call_args.kwargs["messages"][1]["content"]
        assert replayed[0] == {"reasoningContent": {"reasoningText": {"text": "let me think", "signature": "sig-1"}}}
        assert replayed[1] == {"reasoningContent": {"redactedContent": b"sealed"}}
        assert "toolUse" in replayed[2]

    def test_sync_response_reads_nested_reasoning_text(self):
        from agent.bedrock_adapter import normalize_converse_response
        msg = normalize_converse_response({
            "output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "hmm", "signature": "s"}}}, {"text": "hi"}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1},
        }).choices[0].message
        assert msg.reasoning_content == "hmm"
        assert msg.bedrock_content_blocks[0] == {"reasoningContent": {"text": "hmm", "signature": "s"}}


class TestSealedReasoningResendOnce:
    """A redacted blob is sealed to the region/model that minted it; a ``global.*`` profile routed elsewhere
    rejects it with ValidationException. Drop the sealed blocks, keep everything else, resend once (#115865)."""

    def test_call_converse_strips_sealed_blocks_and_keeps_other_turns_intact(self):
        from agent.bedrock_adapter import call_converse
        client = MagicMock()
        client.converse.side_effect = [Exception(CROSS_REGION_REJECTION), _ok_converse_response()]
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client):
            response = call_converse(region="us-east-1", model="global.moonshotai.kimi-k3", messages=_kimi_turn_history())
        assert response.choices[0].message.content == "done"
        assert client.converse.call_count == 2
        first, resent = (c.kwargs["messages"] for c in client.converse.call_args_list)
        assert resent[1]["content"] == [b for b in first[1]["content"] if "redactedContent" not in b.get("reasoningContent", {})]
        assert resent[0] == first[0] and resent[2] == first[2]  # untouched turns are replayed verbatim

    def test_call_converse_reraises_when_nothing_sealed_remains(self):
        from agent.bedrock_adapter import call_converse
        client = MagicMock()
        client.converse.side_effect = Exception(CROSS_REGION_REJECTION)
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client):
            with pytest.raises(Exception, match="ValidationException"):
                call_converse(region="us-east-1", model="m", messages=[{"role": "user", "content": "hi"}])
        assert client.converse.call_count == 1
