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
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# AWS credential detection
# ---------------------------------------------------------------------------

class TestResolveAwsAuthEnvVar:
    """Test AWS credential environment variable detection.

    Mirrors OpenClaw's resolveAwsSdkEnvVarName() priority order.
    """

    def test_prefers_bearer_token_over_access_keys_and_profile(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        env = {
            "AWS_BEARER_TOKEN_BEDROCK": "bearer-token",
            "AWS_ACCESS_KEY_ID": "AKIA...",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_PROFILE": "default",
        }
        assert resolve_aws_auth_env_var(env) == "AWS_BEARER_TOKEN_BEDROCK"

    def test_uses_access_keys_when_bearer_token_missing(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        env = {
            "AWS_ACCESS_KEY_ID": "AKIA...",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_PROFILE": "default",
        }
        assert resolve_aws_auth_env_var(env) == "AWS_ACCESS_KEY_ID"

    def test_requires_both_access_key_and_secret(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        # Only access key, no secret → should not match
        env = {"AWS_ACCESS_KEY_ID": "AKIA..."}
        assert resolve_aws_auth_env_var(env) != "AWS_ACCESS_KEY_ID"

    def test_uses_profile_when_no_keys(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        env = {"AWS_PROFILE": "production"}
        assert resolve_aws_auth_env_var(env) == "AWS_PROFILE"

    def test_uses_container_credentials(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        env = {"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials/..."}
        assert resolve_aws_auth_env_var(env) == "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"

    def test_uses_web_identity(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        env = {"AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/secrets/token"}
        assert resolve_aws_auth_env_var(env) == "AWS_WEB_IDENTITY_TOKEN_FILE"

    def test_returns_none_when_no_aws_auth(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        # Mock botocore to return no credentials (covers EC2 IMDS fallback)
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None
        with patch.dict("sys.modules", {"botocore": MagicMock(), "botocore.session": MagicMock()}):
            import botocore.session as _bs
            _bs.get_session = MagicMock(return_value=mock_session)
            assert resolve_aws_auth_env_var({}) is None

    def test_ignores_whitespace_only_values(self):
        from agent.bedrock_adapter import resolve_aws_auth_env_var
        env = {"AWS_PROFILE": "  ", "AWS_ACCESS_KEY_ID": " "}
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None
        with patch.dict("sys.modules", {"botocore": MagicMock(), "botocore.session": MagicMock()}):
            import botocore.session as _bs
            _bs.get_session = MagicMock(return_value=mock_session)
            assert resolve_aws_auth_env_var(env) is None


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


class TestResolveBedrocRegion:
    def test_prefers_aws_region(self):
        from agent.bedrock_adapter import resolve_bedrock_region
        env = {"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "us-west-2"}
        assert resolve_bedrock_region(env) == "eu-west-1"

    def test_falls_back_to_default_region(self):
        from agent.bedrock_adapter import resolve_bedrock_region
        env = {"AWS_DEFAULT_REGION": "ap-northeast-1"}
        assert resolve_bedrock_region(env) == "ap-northeast-1"

    def test_defaults_to_us_east_1(self):
        from agent.bedrock_adapter import resolve_bedrock_region
        from unittest.mock import patch, MagicMock
        mock_session = MagicMock()
        mock_session.get_config_variable.return_value = None
        with patch("botocore.session.get_session", return_value=mock_session):
            assert resolve_bedrock_region({}) == "us-east-1"

    def test_falls_back_to_botocore_profile_region(self):
        from agent.bedrock_adapter import resolve_bedrock_region
        from unittest.mock import patch, MagicMock
        mock_session = MagicMock()
        mock_session.get_config_variable.return_value = "eu-central-1"
        with patch("botocore.session.get_session", return_value=mock_session):
            assert resolve_bedrock_region({}) == "eu-central-1"

    def test_botocore_failure_falls_back_to_us_east_1(self):
        from agent.bedrock_adapter import resolve_bedrock_region
        from unittest.mock import patch
        with patch("botocore.session.get_session", side_effect=Exception("no botocore")):
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

    def test_converts_multiple_tools(self):
        from agent.bedrock_adapter import convert_tools_to_converse
        tools = [
            {"type": "function", "function": {"name": "tool_a", "description": "A", "parameters": {}}},
            {"type": "function", "function": {"name": "tool_b", "description": "B", "parameters": {}}},
        ]
        result = convert_tools_to_converse(tools)
        assert len(result) == 2
        assert result[0]["toolSpec"]["name"] == "tool_a"
        assert result[1]["toolSpec"]["name"] == "tool_b"

    def test_empty_tools(self):
        from agent.bedrock_adapter import convert_tools_to_converse
        assert convert_tools_to_converse([]) == []
        assert convert_tools_to_converse(None) == []

    def test_missing_parameters_gets_default(self):
        from agent.bedrock_adapter import convert_tools_to_converse
        tools = [{"type": "function", "function": {"name": "noop", "description": "No-op"}}]
        result = convert_tools_to_converse(tools)
        schema = result[0]["toolSpec"]["inputSchema"]["json"]
        assert schema == {"type": "object", "properties": {}}


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

    def test_user_message_text(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [{"role": "user", "content": "What is 2+2?"}]
        system, msgs = convert_messages_to_converse(messages)
        assert system is None
        assert len(msgs) == 1
        assert msgs[0]["content"][0]["text"] == "What is 2+2?"

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

    def test_merges_consecutive_user_messages(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        system, msgs = convert_messages_to_converse(messages)
        # Should be merged into one user message (Converse requires alternation)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        texts = [b["text"] for b in msgs[0]["content"] if "text" in b]
        assert "First" in texts
        assert "Second" in texts

    def test_merges_consecutive_assistant_messages(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Part 1"},
            {"role": "assistant", "content": "Part 2"},
        ]
        system, msgs = convert_messages_to_converse(messages)
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1

    def test_first_message_must_be_user(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "assistant", "content": "I'm ready"},
            {"role": "user", "content": "Go"},
        ]
        system, msgs = convert_messages_to_converse(messages)
        assert msgs[0]["role"] == "user"

    def test_last_message_must_be_user(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        system, msgs = convert_messages_to_converse(messages)
        assert msgs[-1]["role"] == "user"

    def test_empty_content_gets_placeholder(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [{"role": "user", "content": ""}]
        system, msgs = convert_messages_to_converse(messages)
        # Empty string should get a space placeholder
        assert msgs[0]["content"][0]["text"].strip() != "" or msgs[0]["content"][0]["text"] == " "

    def test_image_data_url_converted(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgo=",
                }},
            ],
        }]
        system, msgs = convert_messages_to_converse(messages)
        content = msgs[0]["content"]
        assert any("text" in b for b in content)
        image_blocks = [b for b in content if "image" in b]
        assert len(image_blocks) == 1
        assert image_blocks[0]["image"]["format"] == "png"

    def test_multiple_system_messages_merged(self):
        from agent.bedrock_adapter import convert_messages_to_converse
        messages = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Go"},
        ]
        system, msgs = convert_messages_to_converse(messages)
        assert system is not None
        assert len(system) == 2
        assert system[0]["text"] == "Rule 1"
        assert system[1]["text"] == "Rule 2"


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

    def test_multiple_tool_calls(self):
        from agent.bedrock_adapter import normalize_converse_response
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "c1", "name": "tool_a", "input": {}}},
                        {"toolUse": {"toolUseId": "c2", "name": "tool_b", "input": {"x": 1}}},
                    ],
                },
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 0, "outputTokens": 0},
        }
        result = normalize_converse_response(response)
        assert len(result.choices[0].message.tool_calls) == 2
        assert result.choices[0].finish_reason == "tool_calls"

    def test_stop_reason_mapping(self):
        from agent.bedrock_adapter import _converse_stop_reason_to_openai
        assert _converse_stop_reason_to_openai("end_turn") == "stop"
        assert _converse_stop_reason_to_openai("stop_sequence") == "stop"
        assert _converse_stop_reason_to_openai("tool_use") == "tool_calls"
        assert _converse_stop_reason_to_openai("max_tokens") == "length"
        assert _converse_stop_reason_to_openai("content_filtered") == "content_filter"
        assert _converse_stop_reason_to_openai("guardrail_intervened") == "content_filter"
        assert _converse_stop_reason_to_openai("unknown_reason") == "stop"

    def test_empty_content(self):
        from agent.bedrock_adapter import normalize_converse_response
        response = {
            "output": {"message": {"role": "assistant", "content": []}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 0, "outputTokens": 0},
        }
        result = normalize_converse_response(response)
        assert result.choices[0].message.content is None
        assert result.choices[0].message.tool_calls is None

    def test_tool_calls_override_stop_finish_reason(self):
        """When tool_calls are present but stopReason is end_turn, finish_reason should be tool_calls."""
        from agent.bedrock_adapter import normalize_converse_response
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "c1", "name": "t", "input": {}}},
                    ],
                },
            },
            "stopReason": "end_turn",  # Bedrock sometimes sends this with tool_use
            "usage": {"inputTokens": 0, "outputTokens": 0},
        }
        result = normalize_converse_response(response)
        assert result.choices[0].finish_reason == "tool_calls"


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

    def test_mixed_text_and_tool_stream(self):
        from agent.bedrock_adapter import normalize_converse_stream_events
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            # Text block
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Let me check."}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            # Tool block
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
        result = normalize_converse_stream_events(events)
        assert result.choices[0].message.content == "Let me check."
        assert len(result.choices[0].message.tool_calls) == 1

    def test_empty_stream(self):
        from agent.bedrock_adapter import normalize_converse_stream_events
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0}}},
        ]}
        result = normalize_converse_stream_events(events)
        assert result.choices[0].message.content is None
        assert result.choices[0].message.tool_calls is None


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

    def test_includes_temperature_and_top_p(self):
        from agent.bedrock_adapter import build_converse_kwargs
        kwargs = build_converse_kwargs(
            model="test-model", messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7, top_p=0.9,
        )
        assert kwargs["inferenceConfig"]["temperature"] == 0.7
        assert kwargs["inferenceConfig"]["topP"] == 0.9

    def test_includes_guardrail_config(self):
        from agent.bedrock_adapter import build_converse_kwargs
        guardrail = {
            "guardrailIdentifier": "gr-123",
            "guardrailVersion": "1",
        }
        kwargs = build_converse_kwargs(
            model="test-model", messages=[{"role": "user", "content": "Hi"}],
            guardrail_config=guardrail,
        )
        assert kwargs["guardrailConfig"] == guardrail

    def test_no_system_when_absent(self):
        from agent.bedrock_adapter import build_converse_kwargs
        kwargs = build_converse_kwargs(
            model="test-model", messages=[{"role": "user", "content": "Hi"}],
        )
        assert "system" not in kwargs

    def test_no_tool_config_when_empty(self):
        from agent.bedrock_adapter import build_converse_kwargs
        kwargs = build_converse_kwargs(
            model="test-model", messages=[{"role": "user", "content": "Hi"}],
            tools=[],
        )
        assert "toolConfig" not in kwargs


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

class TestDiscoverBedrockModels:
    """Test Bedrock model discovery with mocked AWS API calls."""

    def test_discovers_foundation_models(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-sonnet-4-6-20250514-v1:0",
                    "modelName": "Claude Sonnet 4.6",
                    "providerName": "Anthropic",
                    "inputModalities": ["TEXT", "IMAGE"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "modelLifecycle": {"status": "ACTIVE"},
                },
                {
                    "modelId": "amazon.nova-pro-v1:0",
                    "modelName": "Nova Pro",
                    "providerName": "Amazon",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "modelLifecycle": {"status": "ACTIVE"},
                },
            ],
        }
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [],
        }

        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=mock_client):
            models = discover_bedrock_models("us-east-1")

        assert len(models) == 2
        ids = [m["id"] for m in models]
        assert "anthropic.claude-sonnet-4-6-20250514-v1:0" in ids
        assert "amazon.nova-pro-v1:0" in ids

    def test_filters_inactive_models(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "old-model",
                    "modelName": "Old",
                    "providerName": "Test",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "modelLifecycle": {"status": "LEGACY"},
                },
            ],
        }
        mock_client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}

        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=mock_client):
            models = discover_bedrock_models("us-east-1")

        assert len(models) == 0

    def test_filters_non_streaming_models(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "embed-model",
                    "modelName": "Embeddings",
                    "providerName": "Test",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["EMBEDDING"],
                    "responseStreamingSupported": False,
                    "modelLifecycle": {"status": "ACTIVE"},
                },
            ],
        }
        mock_client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}

        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=mock_client):
            models = discover_bedrock_models("us-east-1")

        assert len(models) == 0

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

    def test_discovers_inference_profiles(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {"modelSummaries": []}
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "us.anthropic.claude-sonnet-4-6",
                    "inferenceProfileName": "US Claude Sonnet 4.6",
                    "status": "ACTIVE",
                    "models": [{"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6"}],
                },
            ],
        }

        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=mock_client):
            models = discover_bedrock_models("us-east-1")

        assert len(models) == 1
        assert models[0]["id"] == "us.anthropic.claude-sonnet-4-6"

    def test_global_profiles_sorted_first(self):
        from agent.bedrock_adapter import discover_bedrock_models, reset_discovery_cache
        reset_discovery_cache()

        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [{
                "modelId": "anthropic.claude-v2",
                "modelName": "Claude v2",
                "providerName": "Anthropic",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
                "modelLifecycle": {"status": "ACTIVE"},
            }],
        }
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [{
                "inferenceProfileId": "global.anthropic.claude-v2",
                "inferenceProfileName": "Global Claude v2",
                "status": "ACTIVE",
                "models": [],
            }],
        }

        with patch("agent.bedrock_adapter._get_bedrock_control_client", return_value=mock_client):
            models = discover_bedrock_models("us-east-1")

        assert models[0]["id"] == "global.anthropic.claude-v2"

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

    def test_extracts_amazon(self):
        from agent.bedrock_adapter import _extract_provider_from_arn
        arn = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"
        assert _extract_provider_from_arn(arn) == "amazon"

    def test_returns_empty_for_invalid_arn(self):
        from agent.bedrock_adapter import _extract_provider_from_arn
        assert _extract_provider_from_arn("not-an-arn") == ""
        assert _extract_provider_from_arn("") == ""


# ---------------------------------------------------------------------------
# Client cache management
# ---------------------------------------------------------------------------

class TestClientCache:
    def test_reset_clears_caches(self):
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            _bedrock_control_client_cache,
            reset_client_cache,
        )
        _bedrock_runtime_client_cache["test"] = "dummy"
        _bedrock_control_client_cache["test"] = "dummy"
        reset_client_cache()
        assert len(_bedrock_runtime_client_cache) == 0
        assert len(_bedrock_control_client_cache) == 0


# ---------------------------------------------------------------------------
# Streaming with callbacks
# ---------------------------------------------------------------------------

class TestStreamConverseWithCallbacks:
    """Test real-time streaming with delta callbacks."""

    def test_text_deltas_fire_callback(self):
        from agent.bedrock_adapter import stream_converse_with_callbacks
        deltas = []
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": " world"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3}}},
        ]}
        result = stream_converse_with_callbacks(
            events, on_text_delta=lambda t: deltas.append(t),
        )
        assert deltas == ["Hello", " world"]
        assert result.choices[0].message.content == "Hello world"

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

    def test_tool_start_callback_fires(self):
        from agent.bedrock_adapter import stream_converse_with_callbacks
        tools_started = []
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {
                "toolUse": {"toolUseId": "c1", "name": "read_file"},
            }}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "toolUse": {"input": '{"path":"/tmp/f"}'},
            }}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0}}},
        ]}
        result = stream_converse_with_callbacks(
            events, on_tool_start=lambda name: tools_started.append(name),
        )
        assert tools_started == ["read_file"]

    def test_interrupt_stops_processing(self):
        from agent.bedrock_adapter import stream_converse_with_callbacks
        deltas = []
        call_count = {"n": 0}
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "A"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "B"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "C"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0}}},
        ]}

        def check_interrupt():
            call_count["n"] += 1
            return call_count["n"] >= 3  # Interrupt after 2 events

        result = stream_converse_with_callbacks(
            events,
            on_text_delta=lambda t: deltas.append(t),
            on_interrupt_check=check_interrupt,
        )
        # Should have processed fewer than all deltas
        assert len(deltas) < 3

    def test_reasoning_delta_callback(self):
        from agent.bedrock_adapter import stream_converse_with_callbacks
        reasoning = []
        events = {"stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "reasoningContent": {"text": "Let me think..."},
            }}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "Answer."}}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0}}},
        ]}
        result = stream_converse_with_callbacks(
            events, on_reasoning_delta=lambda t: reasoning.append(t),
        )
        assert reasoning == ["Let me think..."]
        assert result.choices[0].message.reasoning_content == "Let me think..."


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

    def test_no_guardrail_when_none(self):
        from agent.bedrock_adapter import build_converse_kwargs
        kwargs = build_converse_kwargs(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            guardrail_config=None,
        )
        assert "guardrailConfig" not in kwargs

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

class TestBedrockErrorClassification:
    """Test Bedrock-specific error classification."""

    def test_context_overflow_validation_exception(self):
        from agent.bedrock_adapter import classify_bedrock_error
        assert classify_bedrock_error(
            "ValidationException: input is too long for model"
        ) == "context_overflow"

    def test_context_overflow_max_tokens(self):
        from agent.bedrock_adapter import classify_bedrock_error
        assert classify_bedrock_error(
            "ValidationException: exceeds the maximum number of input tokens"
        ) == "context_overflow"

    def test_context_overflow_stream_error(self):
        from agent.bedrock_adapter import classify_bedrock_error
        assert classify_bedrock_error(
            "ModelStreamErrorException: Input is too long"
        ) == "context_overflow"

    def test_rate_limit_throttling(self):
        from agent.bedrock_adapter import classify_bedrock_error
        assert classify_bedrock_error("ThrottlingException: Rate exceeded") == "rate_limit"

    def test_rate_limit_concurrent(self):
        from agent.bedrock_adapter import classify_bedrock_error
        assert classify_bedrock_error("Too many concurrent requests") == "rate_limit"

    def test_overloaded_not_ready(self):
        from agent.bedrock_adapter import classify_bedrock_error
        assert classify_bedrock_error("ModelNotReadyException") == "overloaded"

    def test_overloaded_timeout(self):
        from agent.bedrock_adapter import classify_bedrock_error
        assert classify_bedrock_error("ModelTimeoutException") == "overloaded"

    def test_unknown_error(self):
        from agent.bedrock_adapter import classify_bedrock_error
        assert classify_bedrock_error("SomeRandomError: something went wrong") == "unknown"


class TestBedrockContextLength:
    """Test Bedrock model context length lookup."""

    def test_claude_opus_4_6(self):
        from agent.bedrock_adapter import get_bedrock_context_length
        assert get_bedrock_context_length("anthropic.claude-opus-4-6-20250514-v1:0") == 200_000

    def test_claude_sonnet_versioned(self):
        from agent.bedrock_adapter import get_bedrock_context_length
        assert get_bedrock_context_length("anthropic.claude-sonnet-4-6-20250514-v1:0") == 200_000

    def test_nova_pro(self):
        from agent.bedrock_adapter import get_bedrock_context_length
        assert get_bedrock_context_length("amazon.nova-pro-v1:0") == 300_000

    def test_nova_micro(self):
        from agent.bedrock_adapter import get_bedrock_context_length
        assert get_bedrock_context_length("amazon.nova-micro-v1:0") == 128_000

    def test_unknown_model_gets_default(self):
        from agent.bedrock_adapter import get_bedrock_context_length, BEDROCK_DEFAULT_CONTEXT_LENGTH
        assert get_bedrock_context_length("unknown.model-v1:0") == BEDROCK_DEFAULT_CONTEXT_LENGTH

    def test_inference_profile_resolves(self):
        from agent.bedrock_adapter import get_bedrock_context_length
        # Cross-region inference profiles contain the base model ID
        assert get_bedrock_context_length("us.anthropic.claude-sonnet-4-6") == 200_000

    def test_longest_prefix_wins(self):
        from agent.bedrock_adapter import get_bedrock_context_length
        # "anthropic.claude-3-5-sonnet" should match before "anthropic.claude-3"
        assert get_bedrock_context_length("anthropic.claude-3-5-sonnet-20240620-v1:0") == 200_000


# ---------------------------------------------------------------------------
# Tool-calling capability detection
# ---------------------------------------------------------------------------

class TestModelSupportsToolUse:
    """Test non-tool-calling model detection."""

    def test_claude_supports_tools(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("us.anthropic.claude-sonnet-4-6") is True

    def test_nova_supports_tools(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("us.amazon.nova-pro-v1:0") is True

    def test_deepseek_v3_supports_tools(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("deepseek.v3.2") is True

    def test_llama_supports_tools(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("us.meta.llama4-scout-17b-instruct-v1:0") is True

    def test_deepseek_r1_no_tools(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("us.deepseek.r1-v1:0") is False

    def test_deepseek_r1_alt_format_no_tools(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("deepseek-r1") is False

    def test_stability_no_tools(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("stability.stable-diffusion-xl") is False

    def test_embedding_no_tools(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("cohere.embed-v4") is False

    def test_unknown_model_defaults_to_true(self):
        from agent.bedrock_adapter import _model_supports_tool_use
        assert _model_supports_tool_use("some-future-model-v1") is True


class TestBuildConverseKwargsToolStripping:
    """Test that tools are stripped for non-tool-calling models."""

    def test_tools_included_for_claude(self):
        from agent.bedrock_adapter import build_converse_kwargs
        tools = [{"type": "function", "function": {"name": "test", "description": "t", "parameters": {}}}]
        kwargs = build_converse_kwargs(
            model="us.anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )
        assert "toolConfig" in kwargs

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

    def test_global_claude_opus(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("global.anthropic.claude-opus-4-6-v1") is True

    def test_bare_claude(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("anthropic.claude-haiku-4-5-20251001-v1:0") is True

    def test_nova_is_not_anthropic(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("us.amazon.nova-pro-v1:0") is False

    def test_deepseek_is_not_anthropic(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("deepseek.v3.2") is False

    def test_llama_is_not_anthropic(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("us.meta.llama4-scout-17b-instruct-v1:0") is False

    def test_mistral_is_not_anthropic(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("mistral.mistral-large-3-675b-instruct") is False

    def test_eu_claude(self):
        from agent.bedrock_adapter import is_anthropic_bedrock_model
        assert is_anthropic_bedrock_model("eu.anthropic.claude-sonnet-4-6") is True


class TestEmptyTextBlockFix:
    """Test that empty text blocks are replaced with space placeholders."""

    def test_none_content_gets_space(self):
        from agent.bedrock_adapter import _convert_content_to_converse
        blocks = _convert_content_to_converse(None)
        assert blocks[0]["text"] == " "

    def test_empty_string_gets_space(self):
        from agent.bedrock_adapter import _convert_content_to_converse
        blocks = _convert_content_to_converse("")
        assert blocks[0]["text"] == " "

    def test_whitespace_only_gets_space(self):
        from agent.bedrock_adapter import _convert_content_to_converse
        blocks = _convert_content_to_converse("   ")
        assert blocks[0]["text"] == " "

    def test_real_text_preserved(self):
        from agent.bedrock_adapter import _convert_content_to_converse
        blocks = _convert_content_to_converse("Hello")
        assert blocks[0]["text"] == "Hello"


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

    def test_detects_botocore_connection_closed_error(self):
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
        from agent.bedrock_adapter import is_stale_connection_error
        from botocore.exceptions import ConnectionClosedError
        exc = ConnectionClosedError(endpoint_url="https://bedrock.example")
        assert is_stale_connection_error(exc) is True

    def test_detects_botocore_endpoint_connection_error(self):
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
        from agent.bedrock_adapter import is_stale_connection_error
        from botocore.exceptions import EndpointConnectionError
        exc = EndpointConnectionError(endpoint_url="https://bedrock.example")
        assert is_stale_connection_error(exc) is True

    def test_detects_botocore_read_timeout(self):
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
        from agent.bedrock_adapter import is_stale_connection_error
        from botocore.exceptions import ReadTimeoutError
        exc = ReadTimeoutError(endpoint_url="https://bedrock.example")
        assert is_stale_connection_error(exc) is True

    def test_detects_urllib3_protocol_error(self):
        from agent.bedrock_adapter import is_stale_connection_error
        from urllib3.exceptions import ProtocolError
        exc = ProtocolError("Connection broken")
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

    def test_detects_botocore_internal_assertion_error(self):
        """Same as above but for a frame inside the botocore namespace."""
        from agent.bedrock_adapter import is_stale_connection_error
        fake_globals = {"__name__": "botocore.httpsession"}
        try:
            exec("def _boom():\n    assert False\n_boom()", fake_globals)
        except AssertionError as exc:
            assert is_stale_connection_error(exc) is True
        else:
            pytest.fail("AssertionError not raised")

    def test_ignores_application_assertion_error(self):
        """AssertionError from application code (not urllib3/botocore) should
        NOT be classified as stale — those are real test/code bugs."""
        from agent.bedrock_adapter import is_stale_connection_error
        try:
            assert False, "test-only"  # noqa: B011
        except AssertionError as exc:
            assert is_stale_connection_error(exc) is False

    def test_ignores_unrelated_exceptions(self):
        from agent.bedrock_adapter import is_stale_connection_error
        assert is_stale_connection_error(ValueError("bad input")) is False
        assert is_stale_connection_error(KeyError("missing")) is False


class TestCallConverseInvalidatesOnStaleError:
    """call_converse / call_converse_stream evict the cached client when the
    boto3 call raises a stale-connection error — so the next invocation
    reconnects instead of reusing the dead socket."""

    def test_converse_evicts_client_on_stale_error(self):
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            call_converse,
            reset_client_cache,
        )
        from botocore.exceptions import ConnectionClosedError

        reset_client_cache()
        dead_client = MagicMock()
        dead_client.converse.side_effect = ConnectionClosedError(
            endpoint_url="https://bedrock.example",
        )
        _bedrock_runtime_client_cache["us-east-1"] = dead_client

        with pytest.raises(ConnectionClosedError):
            call_converse(
                region="us-east-1",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert "us-east-1" not in _bedrock_runtime_client_cache, (
            "stale client should have been evicted so the retry reconnects"
        )

    def test_converse_stream_evicts_client_on_stale_error(self):
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            call_converse_stream,
            reset_client_cache,
        )
        from botocore.exceptions import ConnectionClosedError

        reset_client_cache()
        dead_client = MagicMock()
        dead_client.converse_stream.side_effect = ConnectionClosedError(
            endpoint_url="https://bedrock.example",
        )
        _bedrock_runtime_client_cache["us-east-1"] = dead_client

        with pytest.raises(ConnectionClosedError):
            call_converse_stream(
                region="us-east-1",
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert "us-east-1" not in _bedrock_runtime_client_cache

    def test_converse_does_not_evict_on_non_stale_error(self):
        """Non-stale errors (e.g. ValidationException) leave the client cache alone."""
        pytest.importorskip("botocore", reason="botocore required for Bedrock exception tests")
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

    def test_converse_leaves_successful_client_in_cache(self):
        from agent.bedrock_adapter import (
            _bedrock_runtime_client_cache,
            call_converse,
            reset_client_cache,
        )

        reset_client_cache()
        live_client = MagicMock()
        live_client.converse.return_value = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        _bedrock_runtime_client_cache["us-east-1"] = live_client

        call_converse(
            region="us-east-1",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert _bedrock_runtime_client_cache.get("us-east-1") is live_client
