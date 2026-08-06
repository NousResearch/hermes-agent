"""Contract for the one-shot Claude Agent SDK auxiliary adapter.

Five things are pinned down here:

1. An auxiliary call in subscription mode goes through the SDK and **never**
   through the pre-SDK direct-OAuth HTTP adapter.
2. The adapter exposes no tools and cannot touch the conversation's session.
3. An explicit ``anthropic`` API-key auxiliary config is completely unaffected.
4. The result is the same OpenAI-shaped object every other auxiliary transport
   returns, so no call site in ``auxiliary_client.py`` has to special-case it.
5. Image input either goes out on the SDK's streaming-input path or fails with
   a message naming ``auxiliary.vision`` — never silently dropped.

``claude-agent-sdk`` is an optional extra. Every test that makes a call installs
a fake ``claude_agent_sdk`` module, so the suite behaves identically with and
without the real package (and never touches the network either way). The two
tests that inspect the *real* package's signature skip when it is absent.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from agent import claude_auxiliary, claude_sdk_input
from agent.claude_auxiliary import (
    ClaudeAuxiliaryError,
    build_claude_auxiliary_client,
    build_claude_auxiliary_options,
    is_claude_subscription_provider,
    split_messages,
)


# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------


@dataclass
class FakeClaudeAgentOptions:
    system_prompt: Any = None
    tools: Any = None
    allowed_tools: list = field(default_factory=list)
    mcp_servers: dict = field(default_factory=dict)
    strict_mcp_config: bool = False
    setting_sources: Any = None
    max_turns: Any = None
    model: Any = None
    permission_mode: Any = None
    include_partial_messages: bool = False
    continue_conversation: bool = False
    resume: Any = None
    fork_session: bool = False
    session_store: Any = None


@dataclass
class TextBlock:
    text: str


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-sonnet-5"


@dataclass
class ResultMessage:
    result: str = ""
    usage: dict | None = None
    session_id: str = "sdk-throwaway-session"
    is_error: bool = False


class _Recorder:
    """Captures what the adapter handed the SDK."""

    def __init__(self) -> None:
        self.prompts: list = []
        self.options: list = []
        self.calls = 0


def _install_fake_sdk(monkeypatch, *, messages=None, recorder=None, delay=0.0):
    """Install a fake ``claude_agent_sdk`` and return its recorder."""
    recorder = recorder or _Recorder()
    payload = list(
        messages
        if messages is not None
        else [
            AssistantMessage(content=[TextBlock("a title")]),
            ResultMessage(result="a title", usage={"input_tokens": 11, "output_tokens": 3}),
        ]
    )

    async def _query(*, prompt, options=None, transport=None):
        recorder.calls += 1
        # A streaming prompt is an async iterable; materialise it exactly as the
        # real SDK does so the test sees the frames that would hit the CLI.
        if isinstance(prompt, str):
            recorder.prompts.append(prompt)
        else:
            frames = [frame async for frame in prompt]
            recorder.prompts.append(frames)
        recorder.options.append(options)
        if delay:
            import asyncio

            await asyncio.sleep(delay)
        for message in payload:
            yield message

    class _FakeSDKClient:
        async def query(self, prompt, session_id="default"):  # pragma: no cover
            raise AssertionError("the auxiliary path must not use ClaudeSDKClient")

    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = FakeClaudeAgentOptions
    module.query = _query
    module.ClaudeSDKClient = _FakeSDKClient
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return recorder


@pytest.fixture
def fake_sdk(monkeypatch):
    return _install_fake_sdk(monkeypatch)


@pytest.fixture
def sdk_absent(monkeypatch):
    """Make ``import claude_agent_sdk`` fail, as it does without the extra."""
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)


@pytest.fixture
def streaming_input_supported(monkeypatch):
    monkeypatch.setattr(claude_sdk_input, "sdk_supports_streaming_input", lambda: True)
    monkeypatch.setattr(claude_auxiliary, "sdk_supports_streaming_input", lambda: True)


@pytest.fixture
def streaming_input_unsupported(monkeypatch):
    monkeypatch.setattr(claude_sdk_input, "sdk_supports_streaming_input", lambda: False)
    monkeypatch.setattr(claude_auxiliary, "sdk_supports_streaming_input", lambda: False)


# ---------------------------------------------------------------------------
# Options — no tools, no session state
# ---------------------------------------------------------------------------


class TestOptions:
    def test_no_tools_of_any_kind(self, fake_sdk):
        options = build_claude_auxiliary_options(
            system_prompt="be terse", model="claude-sonnet-5"
        )
        assert options.tools == []
        assert options.allowed_tools == []
        assert options.mcp_servers == {}
        # Without strict mode a project .mcp.json could reintroduce a toolset.
        assert options.strict_mcp_config is True
        # Without this the CLI loads settings, hooks, skills and plugins.
        assert options.setting_sources == []

    def test_bounded(self, fake_sdk):
        options = build_claude_auxiliary_options(system_prompt=None, model=None)
        assert options.max_turns == claude_auxiliary.CLAUDE_AUX_MAX_TURNS
        assert options.max_turns >= 1

    def test_carries_no_conversation_session_state(self, fake_sdk):
        options = build_claude_auxiliary_options(system_prompt=None, model=None)
        assert options.resume is None
        assert options.continue_conversation is False
        assert options.fork_session is False
        assert options.session_store is None
        assert getattr(options, "session_id", None) is None

    def test_real_sdk_accepts_the_option_set(self):
        """The locked-down option set must be constructible on the real SDK."""
        pytest.importorskip("claude_agent_sdk")
        options = build_claude_auxiliary_options(
            system_prompt="hi", model="claude-sonnet-5"
        )
        assert options.tools == []
        assert options.strict_mcp_config is True
        assert options.setting_sources == []
        assert options.resume is None


class TestMissingExtra:
    def test_actionable_import_error(self, sdk_absent):
        with pytest.raises(ImportError) as excinfo:
            build_claude_auxiliary_client("claude-sonnet-5")
        message = str(excinfo.value)
        assert "claude-code" in message
        assert "pip install" in message

    def test_options_also_report_the_missing_extra(self, sdk_absent):
        with pytest.raises(ImportError):
            build_claude_auxiliary_options(system_prompt=None, model=None)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


class TestOneShotCall:
    def test_returns_openai_shaped_response(self, fake_sdk):
        client = build_claude_auxiliary_client("claude-sonnet-5")
        response = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "title this"},
            ],
        )
        assert response.choices[0].message.content == "a title"
        assert response.choices[0].message.tool_calls is None
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 11
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 14

    def test_system_message_becomes_an_option_not_a_turn(self, fake_sdk):
        client = build_claude_auxiliary_client("claude-sonnet-5")
        client.chat.completions.create(
            messages=[
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "title this"},
            ]
        )
        assert fake_sdk.options[0].system_prompt == "be terse"
        assert fake_sdk.prompts == ["title this"]

    def test_uses_one_shot_query_not_a_client(self, fake_sdk):
        """``ClaudeSDKClient`` would mean a second long-lived CLI subprocess."""
        client = build_claude_auxiliary_client("claude-sonnet-5")
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
        assert fake_sdk.calls == 1

    def test_tools_are_refused_loudly(self, fake_sdk):
        client = build_claude_auxiliary_client("claude-sonnet-5")
        with pytest.raises(ClaudeAuxiliaryError) as excinfo:
            client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "terminal"}}],
            )
        assert "no tools" in str(excinfo.value)
        assert fake_sdk.calls == 0

    def test_stream_request_degrades_to_a_complete_response(self, fake_sdk):
        client = build_claude_auxiliary_client("claude-sonnet-5")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}], stream=True
        )
        assert response.choices[0].message.content == "a title"

    def test_sdk_error_result_raises(self, monkeypatch):
        _install_fake_sdk(
            monkeypatch,
            messages=[ResultMessage(result="rate limited", is_error=True)],
        )
        client = build_claude_auxiliary_client("claude-sonnet-5")
        with pytest.raises(ClaudeAuxiliaryError) as excinfo:
            client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
        assert "rate limited" in str(excinfo.value)

    def test_timeout_is_bounded_and_reported(self, monkeypatch):
        _install_fake_sdk(monkeypatch, delay=5.0)
        client = build_claude_auxiliary_client("claude-sonnet-5", timeout=0.2)
        with pytest.raises(ClaudeAuxiliaryError) as excinfo:
            client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
        assert "timed out" in str(excinfo.value)

    def test_no_worker_thread_survives_a_call(self, fake_sdk):
        import threading

        client = build_claude_auxiliary_client("claude-sonnet-5")
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
        assert not [
            t for t in threading.enumerate() if t.name == "hermes-claude-aux" and t.is_alive()
        ]


class TestSessionIsolation:
    def test_does_not_mutate_the_conversation_session_id(self, fake_sdk):
        """The SDK mints a throwaway session; the conversation keeps its own."""
        agent = SimpleNamespace(_claude_sdk_session_id="conversation-session")
        client = build_claude_auxiliary_client("claude-sonnet-5")
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
        assert agent._claude_sdk_session_id == "conversation-session"
        assert fake_sdk.options[0].resume is None
        assert getattr(fake_sdk.options[0], "session_id", None) is None

    def test_streaming_frame_carries_no_session_id(
        self, monkeypatch, streaming_input_supported
    ):
        """A pinned session_id is exactly how an aux call would leak into chat."""
        recorder = _install_fake_sdk(monkeypatch)
        client = build_claude_auxiliary_client("claude-sonnet-5")
        client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,QUJD"},
                        },
                    ],
                }
            ]
        )
        frames = recorder.prompts[0]
        assert isinstance(frames, list)
        assert "session_id" not in frames[0]


# ---------------------------------------------------------------------------
# Message flattening
# ---------------------------------------------------------------------------


class TestSplitMessages:
    def test_single_user_turn_passes_through(self):
        system, content = split_messages([{"role": "user", "content": "hi"}])
        assert system is None
        assert content == "hi"

    def test_system_turns_are_hoisted(self):
        system, content = split_messages(
            [
                {"role": "system", "content": "rule one"},
                {"role": "system", "content": "rule two"},
                {"role": "user", "content": "go"},
            ]
        )
        assert system == "rule one\n\nrule two"
        assert content == "go"

    def test_multi_turn_examples_are_kept_not_dropped(self):
        system, content = split_messages(
            [
                {"role": "user", "content": "example in"},
                {"role": "assistant", "content": "example out"},
                {"role": "user", "content": "real one"},
            ]
        )
        assert system is None
        assert "example in" in content
        assert "example out" in content
        assert "real one" in content


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


class TestImages:
    def test_streaming_probe_matches_the_installed_sdk(self):
        pytest.importorskip("claude_agent_sdk")
        # 0.2.128 takes ``str | AsyncIterable[dict]`` on both entry points.
        assert claude_sdk_input.sdk_supports_streaming_input() is True

    def test_probe_is_false_without_the_extra(self, sdk_absent):
        assert claude_sdk_input.sdk_supports_streaming_input() is False

    def test_probe_is_false_for_a_string_only_build(self, monkeypatch):
        async def _query(*, prompt: str, options=None, transport=None):  # noqa: ARG001
            yield None

        class _Client:
            async def query(self, prompt: str, session_id="default"):
                pass

        module = types.ModuleType("claude_agent_sdk")
        module.query = _query
        module.ClaudeSDKClient = _Client
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
        assert claude_sdk_input.sdk_supports_streaming_input() is False

    def test_image_goes_out_as_an_anthropic_block(
        self, monkeypatch, streaming_input_supported
    ):
        recorder = _install_fake_sdk(monkeypatch)
        client = build_claude_auxiliary_client("claude-sonnet-5")
        client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,QUJD"},
                        },
                    ],
                }
            ]
        )
        frames = recorder.prompts[0]
        blocks = frames[0]["message"]["content"]
        assert blocks[0] == {"type": "text", "text": "describe"}
        assert blocks[1] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
        }

    def test_unsupported_build_names_the_vision_config(
        self, fake_sdk, streaming_input_unsupported
    ):
        client = build_claude_auxiliary_client("claude-sonnet-5")
        with pytest.raises(ClaudeAuxiliaryError) as excinfo:
            client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,QUJD"},
                            }
                        ],
                    }
                ]
            )
        message = str(excinfo.value)
        assert "auxiliary.vision" in message
        assert "NOT sent" in message
        assert fake_sdk.calls == 0

    def test_text_only_content_never_uses_the_streaming_path(self, fake_sdk):
        client = build_claude_auxiliary_client("claude-sonnet-5")
        client.chat.completions.create(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        )
        assert fake_sdk.prompts == ["hi"]

    def test_unencodable_image_becomes_a_visible_note(self):
        blocks = claude_sdk_input.content_to_sdk_blocks(
            [{"type": "image_url", "image_url": {"url": "file:///tmp/x.png"}}]
        )
        assert blocks == [
            {"type": "text", "text": "[an image was attached but could not be encoded]"}
        ]


# ---------------------------------------------------------------------------
# Routing inside auxiliary_client
# ---------------------------------------------------------------------------


@pytest.fixture
def subscription_gate_open(monkeypatch):
    """Open the Claude subscription gate for provider normalisation."""
    monkeypatch.setattr(
        "hermes_cli.claude_code.subscription_enabled", lambda config=None: True
    )


class TestAuxiliaryRouting:
    def test_provider_slug_is_never_anthropic(self):
        assert is_claude_subscription_provider("claude-code") is True
        assert is_claude_subscription_provider("anthropic") is False
        assert is_claude_subscription_provider("") is False

    def test_subscription_task_never_touches_the_http_adapter(
        self, fake_sdk, subscription_gate_open
    ):
        from agent import auxiliary_client

        with patch.object(auxiliary_client, "_try_anthropic") as http_path, patch.object(
            auxiliary_client, "_create_openai_client"
        ) as openai_path:
            client, model = auxiliary_client.resolve_provider_client(
                "claude-code", model="claude-sonnet-5"
            )

        assert isinstance(client, claude_auxiliary.ClaudeAuxiliaryClient)
        assert model == "claude-sonnet-5"
        http_path.assert_not_called()
        openai_path.assert_not_called()

    def test_resolved_client_holds_no_credential(self, fake_sdk, subscription_gate_open):
        from agent import auxiliary_client

        client, _ = auxiliary_client.resolve_provider_client(
            "claude-code", model="claude-sonnet-5"
        )
        assert client.api_key == ""
        assert not client.base_url.startswith("http")

    def test_async_mode_returns_the_async_facade(self, fake_sdk, subscription_gate_open):
        from agent import auxiliary_client

        client, _ = auxiliary_client.resolve_provider_client(
            "claude-code", model="claude-sonnet-5", async_mode=True
        )
        assert isinstance(client, claude_auxiliary.AsyncClaudeAuxiliaryClient)

    def test_missing_extra_degrades_instead_of_raising(
        self, sdk_absent, subscription_gate_open
    ):
        from agent import auxiliary_client

        client, model = auxiliary_client.resolve_provider_client(
            "claude-code", model="claude-sonnet-5"
        )
        assert client is None and model is None

    def test_missing_claude_cli_degrades_instead_of_raising(
        self, fake_sdk, subscription_gate_open
    ):
        from agent import auxiliary_client
        from hermes_cli.auth import AuthError

        with patch(
            "hermes_cli.auth.resolve_external_process_provider_credentials",
            side_effect=AuthError("Could not find the `claude` CLI."),
        ):
            client, model = auxiliary_client.resolve_provider_client(
                "claude-code", model="claude-sonnet-5"
            )
        assert client is None and model is None

    def test_credential_material_is_refused(self, fake_sdk, subscription_gate_open):
        """A non-empty key here would mean Hermes started holding Claude auth."""
        from agent import auxiliary_client

        with patch(
            "hermes_cli.auth.resolve_external_process_provider_credentials",
            return_value={"api_key": "sk-ant-oat01-leaked", "base_url": "x"},
        ):
            client, model = auxiliary_client.resolve_provider_client(
                "claude-code", model="claude-sonnet-5"
            )
        assert client is None and model is None

    def test_streaming_flag_is_not_sent_to_the_sdk_client(self, fake_sdk):
        from agent import auxiliary_client

        client = build_claude_auxiliary_client("claude-sonnet-5")
        assert auxiliary_client._client_streams_internally(client) is True


class TestAnthropicApiKeyConfigUnchanged:
    """An explicit ``anthropic`` API-key auxiliary config must not shift."""

    def test_api_key_provider_still_builds_the_http_adapter(self, monkeypatch):
        from agent import auxiliary_client

        monkeypatch.setattr(
            auxiliary_client, "_read_main_provider", lambda: "claude-code"
        )
        built = {}

        def _fake_build(token, base_url, **kwargs):
            built["token"] = token
            return SimpleNamespace(base_url=base_url)

        monkeypatch.setattr(
            "agent.anthropic_adapter.build_anthropic_client", _fake_build
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.resolve_anthropic_token", lambda: ""
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._is_oauth_token", lambda tok: False
        )

        client, model = auxiliary_client._try_anthropic(
            explicit_api_key="sk-ant-api03-real-key"
        )
        assert client is not None
        assert built["token"] == "sk-ant-api03-real-key"
        assert model

    def test_oauth_token_is_refused_when_the_subscription_runtime_is_active(
        self, monkeypatch
    ):
        """Otherwise an aux fallback quietly resumes extra-usage billing."""
        from agent import auxiliary_client

        monkeypatch.setattr(
            auxiliary_client, "_read_main_provider", lambda: "claude-code"
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.resolve_anthropic_token",
            lambda: "sk-ant-oat01-subscription",
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter._is_oauth_token", lambda tok: True
        )

        def _must_not_build(*args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("the legacy direct-OAuth path was reached")

        monkeypatch.setattr(
            "agent.anthropic_adapter.build_anthropic_client", _must_not_build
        )

        client, model = auxiliary_client._try_anthropic()
        assert client is None and model is None

    def test_oauth_token_still_works_for_a_non_subscription_user(self, monkeypatch):
        from agent import auxiliary_client

        monkeypatch.setattr(auxiliary_client, "_read_main_provider", lambda: "anthropic")
        monkeypatch.setattr(
            "agent.anthropic_adapter.resolve_anthropic_token",
            lambda: "sk-ant-oat01-legacy",
        )
        monkeypatch.setattr("agent.anthropic_adapter._is_oauth_token", lambda tok: True)
        monkeypatch.setattr(
            "agent.anthropic_adapter.build_anthropic_client",
            lambda token, base_url, **kw: SimpleNamespace(base_url=base_url),
        )

        client, _ = auxiliary_client._try_anthropic()
        assert client is not None
