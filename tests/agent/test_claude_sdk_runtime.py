"""Tests for the claude-agent-sdk runtime (#25267).

Covers the three new modules end-to-end without requiring the optional
``claude-agent-sdk`` extra: the projector and session duck-type on class
NAMES, so local stand-in classes named like the SDK's types are the fixture.

Plant-the-failure discipline: every guard here is exercised RED first —
the auth classifier has a negative control (an ordinary error must NOT
produce the re-auth hint), and the session's error path is asserted to
retire the client rather than silently continue.
"""

import asyncio
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from agent.conversation_loop import _handle_claude_sdk_turn_with_fallback
from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
    classify_auth_failure,
)
from agent.transports.claude_sdk_event_projector import (
    ClaudeSdkEventProjector,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Every `agent.claude_agent_sdk` flag now resolves from config.yaml only.

    Without this, `_provider_config()` reads the DEVELOPER'S REAL config.yaml:
    a machine with `allow_metered_key: true` set would silently invert the
    metered-billing refusal assertions, and a real `append_file` would leak into
    the system-prompt tests. Default to an empty block; tests that care patch
    `load_config_readonly` themselves (the last patch wins).
    """
    import hermes_cli.config as cfg
    from gateway.session_context import reset_session_vars
    from tools.terminal_tool import set_approval_callback

    # Tests in this module create gateway-shaped contextvars and CLI callbacks.
    # Reset both around every case so a later bare-CLI assertion cannot inherit
    # state from a prior test in the same process.
    reset_session_vars()
    set_approval_callback(None)
    monkeypatch.setattr(cfg, "load_config_readonly", lambda *a, **k: {}, raising=False)
    yield
    set_approval_callback(None)
    reset_session_vars()


# ---------- SDK stand-in types (duck-typed by class NAME) ----------


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str = ""


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: Optional[bool] = None


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-opus-4-8"
    parent_tool_use_id: Optional[str] = None


@dataclass
class UserMessage:
    content: Any = None


@dataclass
class SystemMessage:
    subtype: str = "init"
    data: dict = field(default_factory=dict)
    session_id: Optional[str] = None


@dataclass
class RateLimitInfo:
    status: str = "allowed"
    rate_limit_type: Optional[str] = "five_hour"
    overage_status: Optional[str] = None
    overage_disabled_reason: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass
class RateLimitEvent:
    rate_limit_info: RateLimitInfo
    uuid: str = "rate-1"
    session_id: str = "sdk-session-1"


@dataclass
class ServerToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class StreamEvent:
    uuid: str = "se-1"
    session_id: str = "sdk-session-1"
    event: dict = field(default_factory=dict)
    parent_tool_use_id: Optional[str] = None


def _text_delta_event(text, parent_tool_use_id=None):
    return StreamEvent(
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
        parent_tool_use_id=parent_tool_use_id,
    )


@dataclass
class ResultMessage:
    subtype: str = "success"
    duration_ms: int = 1
    duration_api_ms: int = 1
    is_error: bool = False
    num_turns: int = 1
    session_id: str = "sdk-session-1"
    result: Optional[str] = None
    usage: Optional[dict] = None
    uuid: Optional[str] = "uuid-1"
    errors: Optional[list] = None
    api_error_status: Optional[int] = None
    total_cost_usd: Optional[float] = None


class TestClaudeSdkFallbackBridge:
    def test_quota_error_activates_configured_fallback(self):
        class Agent:
            def __init__(self):
                self.reason, self.status = None, []
            def _run_claude_agent_sdk_turn(self, **kwargs):
                return {"error": "SDK result error (HTTP 429): You've hit your session limit"}
            def _try_activate_fallback(self, reason=None):
                self.reason = reason
                return True
            def _buffer_status(self, text):
                self.status.append(text)
        agent = Agent()
        assert _handle_claude_sdk_turn_with_fallback(
            agent, user_message="continue", original_user_message="continue",
            messages=[], effective_task_id="test", should_review_memory=False,
        ) is None
        assert agent.reason.value == "rate_limit"
        assert agent.status == ["⚠️ Claude session limit reached — switching to fallback provider..."]


# ---------- projector ----------


class TestProjector:
    def test_assistant_text(self):
        p = ClaudeSdkEventProjector()
        out = p.project(AssistantMessage(content=[TextBlock("hello")]))
        assert out.messages == [{"role": "assistant", "content": "hello"}]
        assert out.final_text == "hello"
        assert not out.is_tool_iteration

    def test_assistant_tool_use_and_thinking(self):
        p = ClaudeSdkEventProjector()
        # Thinking arrives first, stashes onto the next assistant entry.
        p.project(AssistantMessage(content=[ThinkingBlock("pondering")]))
        out = p.project(
            AssistantMessage(
                content=[ToolUseBlock(id="t1", name="Bash", input={"command": "ls"})]
            )
        )
        (msg,) = out.messages
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert msg["reasoning"] == "pondering"
        (call,) = msg["tool_calls"]
        assert call["id"] == "t1"
        assert call["function"]["name"] == "Bash"
        assert '"command": "ls"' in call["function"]["arguments"]

    def test_tool_result_projection(self):
        p = ClaudeSdkEventProjector()
        out = p.project(
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="ok")])
        )
        assert out.is_tool_iteration
        assert out.messages == [
            {"role": "tool", "tool_call_id": "t1", "content": "ok"}
        ]

    def test_tool_result_error_and_list_content(self):
        p = ClaudeSdkEventProjector()
        out = p.project(
            UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="t2",
                        content=[{"type": "text", "text": "boom"}],
                        is_error=True,
                    )
                ]
            )
        )
        assert out.messages[0]["content"] == "[error] boom"

    def test_tool_result_truncation(self):
        p = ClaudeSdkEventProjector()
        out = p.project(
            UserMessage(
                content=[ToolResultBlock(tool_use_id="t3", content="x" * 9000)]
            )
        )
        assert len(out.messages[0]["content"]) == 4000

    def test_result_message_sets_final_text(self):
        p = ClaudeSdkEventProjector()
        out = p.project(ResultMessage(result="the answer"))
        assert out.is_result
        assert out.final_text == "the answer"
        assert out.messages == []

    def test_server_tool_use_never_emits_dangling_tool_calls(self):
        # Validator C8: server tools (web_search, ...) execute API-side and
        # never produce a {role:'tool'} echo — emitting a tool_calls entry
        # for them leaves a dangling tool_call_id that can break replay
        # through a native provider after a /model switch.
        p = ClaudeSdkEventProjector()
        out = p.project(
            AssistantMessage(content=[
                ServerToolUseBlock(id="srv-1", name="web_search", input={"query": "x"}),
                TextBlock("found it"),
            ])
        )
        (msg,) = out.messages
        assert msg.get("tool_calls") in (None, [],) or "srv-1" not in str(msg.get("tool_calls"))
        assert msg["content"] == "found it"

    def test_lifecycle_messages_ignored(self):
        p = ClaudeSdkEventProjector()
        assert p.project(SystemMessage()).messages == []
        # A plain-text user echo must not duplicate the real user turn.
        assert p.project(UserMessage(content="hi")).messages == []


# ---------- auth classifier (with negative control) ----------


class TestAuthClassifier:
    def test_auth_failure_produces_hint(self):
        hint = classify_auth_failure("HTTP 401 unauthorized: oauth token expired")
        assert hint is not None
        assert "setup-token" in hint

    def test_hint_redacts_underlying_error(self):
        hint = classify_auth_failure(
            "HTTP 401 unauthorized: oauth token expired; Authorization: Bearer sk-secret-test-token"
        )
        assert hint is not None
        assert "401 unauthorized" in hint
        assert "sk-secret-test-token" not in hint

    def test_negative_control_ordinary_error_no_hint(self):
        # RED-first: an unrelated failure must surface verbatim, never as a
        # re-auth redirect.
        assert classify_auth_failure("connection reset by peer") is None
        assert classify_auth_failure("") is None

    def test_negative_control_overbroad_substrings(self):
        # RED-first against the original hint list: codex's
        # _OAUTH_REFRESH_FAILURE_HINTS has "401 unauthorized", never bare
        # "401", and no bare "credentials" — a tool id or an MCP server's
        # own file complaint must not retire the session as an auth failure.
        assert classify_auth_failure("tool_use toolu_401abc failed at 4012") is None
        assert (
            classify_auth_failure(
                "mcp server hermes-tools: could not read credentials file"
            )
            is None
        )
        assert (
            classify_auth_failure(
                "mcp server weather: Unauthorized — invalid region scope"
            )
            is None
        )

    @pytest.mark.parametrize(
        "message",
        [
            "HTTP 401: Unauthorized",
            "request failed: status code 401, Unauthorized",
            "request failed: status code 401 - Unauthorized",
            "request failed: status code 401 (Unauthorized)",
            "request failed: status code 401\nUnauthorized",
            "SDK result error: Unauthorized (HTTP 401)",
        ],
    )
    def test_delimited_401_is_auth_failure(self, message):
        assert classify_auth_failure(message) is not None

    @pytest.mark.parametrize(
        "message",
        [
            "mcp__calendar failed: HTTP 401 Unauthorized",
            'mcp__calendar failed: {"type":"authentication_error"}',
            "mcp server failed: invalid api key",
            "mcp__tool failed; Anthropic OAuth error: HTTP 401 Unauthorized",
        ],
    )
    def test_mcp_auth_payload_is_not_claude_auth_failure(self, message):
        assert classify_auth_failure(message) is None

    def test_structured_mcp_provenance_suppresses_unlabeled_auth_payload(self):
        assert (
            classify_auth_failure(
                "HTTP 401 Unauthorized: authentication_error",
                mcp_attributed=True,
            )
            is None
        )


# ---------- session (fake client) ----------


_EOS = object()  # the fake CLI process exited: the message stream ends here


class _FakeClient:
    """Stub ClaudeSDKClient: async surface over ONE continuous message stream.

    Mirrors the real SDK shape the session depends on:

    - ``receive_messages()`` is the single continuous stream for the client's
      lifetime (what the session's persistent reader owns);
    - ``receive_response()`` is the SDK's drain-until-ResultMessage wrapper
      over that same stream — kept so the PRE-fix session code also runs
      against this fake, which is what makes the desync regression tests
      below RED-provable on the buggy implementation;
    - ``query()`` makes the scripted turn output appear on the stream, the
      way the CLI answers a stdin message. A script with NO ResultMessage
      models a CLI that died mid-turn: the stream ends right after it.
    - ``feed(*messages)`` injects CLI-initiated output with no query — a
      finished background Agent task reporting in. That is the trigger shape
      of the 2026-07-25 stale-answer incident (dasbrow-hermes-coder#2).
    """

    def __init__(self, options=None, script=None, connect_exc=None):
        self.options = options
        self._script = list(script or [])
        self._connect_exc = connect_exc
        self.queried: list[Any] = []
        self.disconnected = False
        self.interrupted = False
        self._pending: deque = deque()

    def feed(self, *messages):
        """Thread-safe injection of unsolicited CLI output (deque append is
        GIL-atomic; the consumer polls on the session loop)."""
        self._pending.extend(messages)

    async def connect(self):
        if self._connect_exc is not None:
            raise self._connect_exc

    async def query(self, text):
        if isinstance(text, str):
            self.queried.append(text)
        else:
            self.queried.append([message async for message in text])
        self._pending.extend(self._script)
        if not any(type(m).__name__ == "ResultMessage" for m in self._script):
            self._pending.append(_EOS)

    async def receive_messages(self):
        while True:
            try:
                message = self._pending.popleft()
            except IndexError:
                await asyncio.sleep(0.005)
                continue
            if message is _EOS:
                return
            yield message

    async def receive_response(self):
        async for message in self.receive_messages():
            yield message
            if type(message).__name__ == "ResultMessage":
                return

    async def interrupt(self):
        self.interrupted = True

    async def disconnect(self):
        self.disconnected = True


def _make_session(script=None, connect_exc=None, **kwargs):
    holder = {}

    def factory(options=None):
        holder["client"] = _FakeClient(
            options=options, script=script, connect_exc=connect_exc
        )
        return holder["client"]

    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=factory, **kwargs
    )
    return session, holder


class TestSession:
    def test_reported_api_key_source_fails_closed(self):
        session, holder = _make_session(
            script=[
                SystemMessage(
                    data={"apiKeySource": "ANTHROPIC_API_KEY"},
                    session_id="sdk-keyed",
                ),
                ResultMessage(result="must not be delivered"),
            ]
        )
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert "API-key source" in turn.error
        assert turn.final_text == ""
        assert turn.should_retire is True
        assert turn.fatal_reason == "startup"
        assert holder["client"].interrupted is True

    def test_enabled_subscription_overage_fails_closed_before_fallback(self):
        session, holder = _make_session(
            script=[
                RateLimitEvent(
                    RateLimitInfo(
                        overage_status="allowed",
                        raw={"isUsingOverage": False},
                    )
                ),
                ResultMessage(result="must not be delivered"),
            ]
        )
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert "extra usage is enabled" in turn.error
        assert turn.final_text == ""
        assert turn.should_retire is True
        assert turn.fatal_reason == "startup"
        assert holder["client"].interrupted is True

    def test_disabled_overage_and_oauth_source_verify_subscription_lane(self):
        session, _holder = _make_session(
            script=[
                SystemMessage(
                    data={"apiKeySource": "none"},
                    session_id="sdk-subscription",
                ),
                RateLimitEvent(
                    RateLimitInfo(
                        overage_status="rejected",
                        raw={"isUsingOverage": False},
                    )
                ),
                ResultMessage(result="included"),
            ]
        )
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "included"
        assert turn.billing_mode == "subscription_included"
        assert turn.billing_evidence == {
            "api_key_source": "none",
            "is_using_overage": False,
            "overage_status": "rejected",
            "rate_limit_type": "five_hour",
        }

    def test_explicit_metered_opt_in_allows_and_labels_reported_lane(
        self, monkeypatch
    ):
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"allow_metered_key": True}}
            },
            raising=False,
        )
        session, _holder = _make_session(
            script=[
                SystemMessage(data={"apiKeySource": "ANTHROPIC_API_KEY"}),
                RateLimitEvent(
                    RateLimitInfo(
                        rate_limit_type="overage",
                        overage_status="allowed",
                        raw={"isUsingOverage": True},
                    )
                ),
                ResultMessage(result="metered", total_cost_usd=0.25),
            ]
        )
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "metered"
        assert turn.billing_mode == "sdk_reported_metered"
        assert turn.total_cost_usd == 0.25

    def test_empty_turn_rejects_before_start_and_following_turn_works(self):
        session, holder = _make_session(script=[ResultMessage(result="usable")])
        session.request_interrupt()
        try:
            for content in ([], "", "   "):
                turn = session.run_turn(content)
                assert turn.error == turn.final_text
                assert turn.api_call_made is False
                assert not turn.should_retire
            assert holder == {}
            usable = session.run_turn("normal turn")
        finally:
            session.close()
        assert usable.error is None
        assert holder["client"].queried == ["normal turn"]

    def test_successful_mcp_use_does_not_mask_terminal_401(self):
        session, _ = _make_session(script=[
            AssistantMessage(content=[
                ToolUseBlock(
                    id="mcp-1",
                    name="mcp__calendar__list_events",
                    input={},
                )
            ]),
            ResultMessage(
                result="",
                errors=["HTTP 401 Unauthorized"],
                is_error=True,
                subtype="error_during_execution",
                api_error_status=401,
            ),
        ])
        try:
            turn = session.run_turn("list events")
        finally:
            session.close()
        assert "HTTP 401" in (turn.error or "")
        assert turn.should_retire
        assert turn.fatal_reason == "auth"

    @pytest.mark.parametrize("mcp_tool", [True, False])
    def test_interrupted_auth_result_respects_mcp_provenance(self, mcp_tool):
        script = []
        if mcp_tool:
            script.append(AssistantMessage(content=[
                ToolUseBlock(
                    id="mcp-1",
                    name="mcp__calendar__list_events",
                    input={},
                )
            ]))
        script.append(ResultMessage(
            result="",
            errors=["HTTP 401 Unauthorized"],
            is_error=True,
            subtype="error_during_execution",
            api_error_status=401,
        ))
        session, _ = _make_session(script=script)
        original_factory = session._client_factory

        def interrupting_factory(*args, **kwargs):
            client = original_factory(*args, **kwargs)
            original_query = client.query

            async def query_then_interrupt(text):
                await original_query(text)
                session._interrupt_event.set()

            client.query = query_then_interrupt
            return client

        session._client_factory = interrupting_factory
        try:
            turn = session.run_turn("list events")
        finally:
            session.close()
        assert turn.interrupted
        # A successful MCP call is not provenance for a later terminal Claude
        # auth failure, including when the user interrupts the turn.
        assert "setup-token" in (turn.error or "")
        assert turn.should_retire
        assert turn.fatal_reason == "auth"

    def test_quota_429_contradictory_success_surfaces_result_for_fallback(self):
        session, _ = _make_session(script=[ResultMessage(
            result="You've hit your session limit", is_error=True,
            subtype="success", api_error_status=429,
        )])
        try:
            turn = session.run_turn("continue")
        finally:
            session.close()
        assert "HTTP 429" in (turn.error or "")
        assert "session limit" in (turn.error or "").lower()
        # Terminal SDK diagnostics must not become assistant history before the
        # provider-routing layer switches to its fallback.
        assert not turn.final_text

    def test_happy_turn(self):
        script = [
            AssistantMessage(
                content=[ToolUseBlock(id="t1", name="Read", input={"file_path": "/x"})]
            ),
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="data")]),
            AssistantMessage(content=[TextBlock("done reading")]),
            ResultMessage(
                result="done reading",
                usage={"input_tokens": 10, "output_tokens": 5},
            ),
        ]
        session, holder = _make_session(script=script)
        try:
            turn = session.run_turn("read /x please")
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "done reading"
        assert turn.tool_iterations == 1
        assert turn.token_usage_last == {"input_tokens": 10, "output_tokens": 5}
        assert turn.thread_id == "sdk-session-1"
        # assistant(tool_call) + tool + assistant(text)
        assert [m["role"] for m in turn.projected_messages] == [
            "assistant", "tool", "assistant",
        ]
        assert holder["client"].queried == ["read /x please"]
        assert not turn.should_retire

    def test_mixed_text_and_data_image_reaches_sdk_as_native_content(self):
        session, holder = _make_session(script=[ResultMessage(result="a diagram")])
        try:
            turn = session.run_turn([
                {"type": "text", "text": "Inspect this diagram."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aGVsbG8=",
                    },
                },
            ])
        finally:
            session.close()

        assert turn.error is None
        assert holder["client"].queried == [[{
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this diagram."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8=",
                        },
                    },
                ],
            },
            "parent_tool_use_id": None,
        }]]

    def test_image_only_url_reaches_sdk_without_fabricated_prompt(self):
        session, holder = _make_session(script=[ResultMessage(result="a photo")])
        try:
            turn = session.run_turn([{
                "type": "image_url",
                "image_url": {"url": "https://example.test/photo.png"},
            }])
        finally:
            session.close()

        assert turn.error is None
        (query,), = holder["client"].queried
        assert query["message"]["content"] == [{
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.test/photo.png",
            },
        }]

    def test_malformed_image_is_disclosed_instead_of_claimed_attached(
        self, caplog
    ):
        session, holder = _make_session(script=[ResultMessage(result="cannot inspect")])
        with caplog.at_level(logging.WARNING):
            try:
                turn = session.run_turn([{
                    "type": "image_url",
                    "image_url": {"url": "not-an-image-source"},
                }])
            finally:
                session.close()

        assert turn.error is None
        (query,) = holder["client"].queried
        assert "image attachment unavailable" in query.lower()
        assert "what do you see" not in query.lower()
        assert "image attachment" in caplog.text.lower()

    def test_sdk_error_result_surfaces(self):
        script = [
            ResultMessage(
                result="RAW terminal diagnostic",
                subtype="error_max_turns",
                is_error=False,
            )
        ]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert "error_max_turns" in turn.error
        assert not turn.final_text

    def test_contradictory_success_envelope_not_an_error(self):
        # 2026-08-11 incident: the CLI emitted is_error=True with
        # subtype="success" and an empty errors list; the fabricated
        # "SDK result error (subtype=success): success" killed the cron
        # job that had already produced its answer. The contradictory
        # envelope loses to its own subtype. A genuine failure keeps the
        # honest path: non-empty errors (see test_auth_error_marks_retire,
        # same subtype) or a non-success subtype.
        script = [
            AssistantMessage(content=[TextBlock("nightly summary saved")]),
            ResultMessage(
                result="nightly summary saved", is_error=True, subtype="success"
            ),
        ]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("consolidate")
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "nightly summary saved"
        assert not turn.should_retire

    def test_auth_error_marks_retire(self):
        script = [
            ResultMessage(
                subtype="success",
                is_error=True,
                errors=["401 unauthorized: invalid bearer token"],
            )
        ]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.should_retire
        assert "setup-token" in (turn.error or "")

    def test_connect_failure_fails_closed(self):
        session, _ = _make_session(connect_exc=RuntimeError("not logged in"))
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.should_retire
        assert turn.error is not None

    def test_option_fields_shape(self):
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        options = holder["client"].options
        assert options["model"] == "claude-opus-4-8"
        assert options["system_prompt"]["preset"] == "claude_code"
        assert "hermes-tools" in options["mcp_servers"]
        mcp = options["mcp_servers"]["hermes-tools"]
        assert mcp["args"] == [
            "-m",
            "agent.transports.hermes_tools_mcp_server",
            "--profile",
            "claude-agent-sdk",
        ]
        # Hard rule: a metered key never reaches any child of this runtime.
        assert "ANTHROPIC_API_KEY" not in (mcp.get("env") or {})
        # Agent SDK defaults fail closed to `default`: this is the only mode
        # that installs Hermes' approval bridge. The terminal `auto` posture
        # maps here too, so green-field deployments retain that bridge.
        assert options["permission_mode"] == "default"
        # Explicit SDK isolation: None would load ALL of ~/.claude and
        # .claude/settings*, letting ambient settings shadow the gateway's
        # approval posture. The empty list is the SDK's isolation mode.
        assert options["setting_sources"] == []

    def test_native_read_is_disallowed_in_favor_of_bounded_mcp_read(self):
        # Native Read must remain behind Hermes's protected-path-aware bounded
        # MCP read surface under every supported SDK permission mode.
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        fields = session.build_option_fields()
        assert fields["disallowed_tools"] == ["AskUserQuestion", "Read"]
        assert "Bash" not in fields["disallowed_tools"]
        assert "Edit" not in fields["disallowed_tools"]
        assert "Write" not in fields["disallowed_tools"]
        assert "mcp__hermes-tools__read_file" not in fields["disallowed_tools"]

    def test_config_permission_mode_overrides_env_mapping(self, monkeypatch):
        # agent.claude_agent_sdk.permission_mode (an SDK literal) wins over
        # the HERMES_TERMINAL_SECURITY_MODE mapping; explicit constructor
        # arg still wins over both.
        import hermes_cli.config as cfg

        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "unrestricted")
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"permission_mode": "plan"}}
            },
            raising=False,
        )
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["permission_mode"] == "plan"

        explicit = ClaudeAgentSdkSession(
            cwd="/tmp", permission_mode="default", client_factory=MagicMock()
        )
        assert explicit.build_option_fields()["permission_mode"] == "default"

    def test_invalid_config_permission_mode_falls_back(self, monkeypatch):
        # A typo must never silently loosen the posture — it falls back to the
        # SDK's safe default, which retains the Hermes approval bridge.
        import hermes_cli.config as cfg

        monkeypatch.delenv("HERMES_TERMINAL_SECURITY_MODE", raising=False)
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"permission_mode": "yolo"}}
            },
            raising=False,
        )
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["permission_mode"] == "default"

    def test_unknown_terminal_security_mode_falls_back_to_default(self, monkeypatch):
        """Unknown terminal modes must retain the approval-bridge posture."""
        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "unexpected-mode")
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["permission_mode"] == "default"

    def test_empty_config_permission_mode_keeps_env_mapping(self, monkeypatch):
        # "" (the canonical default) = current behavior: the
        # HERMES_TERMINAL_SECURITY_MODE mapping stands.
        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "approval-required")
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["permission_mode"] == "default"

    def test_max_buffer_size_set_above_sdk_default(self):
        # The SDK's 1 MiB default kills the turn outright when one CLI stdout
        # message clears it (2026-08-17 21:03 EDT production kill). The option
        # must always be present — falling back to the SDK default silently
        # reintroduces that failure.
        from agent.transports.claude_agent_sdk_session import (
            _DEFAULT_MAX_BUFFER_SIZE,
        )

        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["max_buffer_size"] == (
            _DEFAULT_MAX_BUFFER_SIZE
        )
        assert _DEFAULT_MAX_BUFFER_SIZE > 1024 * 1024

    def test_config_max_buffer_size_overrides_default(self, monkeypatch):
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"max_buffer_size": 4194304}}
            },
            raising=False,
        )
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["max_buffer_size"] == 4194304

    @pytest.mark.parametrize("bad", [True, "big", 0, -1, None, 1.5, float("inf")])
    def test_invalid_max_buffer_size_falls_back(self, monkeypatch, bad):
        # A typo must not disable the only backstop against an unterminated
        # line growing until the host OOMs, nor silently drop to 1 MiB.
        import hermes_cli.config as cfg
        from agent.transports.claude_agent_sdk_session import (
            _DEFAULT_MAX_BUFFER_SIZE,
        )

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"max_buffer_size": bad}}
            },
            raising=False,
        )
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["max_buffer_size"] == (
            _DEFAULT_MAX_BUFFER_SIZE
        )

    def test_metered_key_scrubbed_from_mcp_env(self, monkeypatch):
        # RED-first: with the ambient var set, the builder must scrub it.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        fields = session.build_option_fields()
        assert "ANTHROPIC_API_KEY" not in fields["mcp_servers"]["hermes-tools"]["env"]

    def test_metered_vectors_neutralized_in_cli_env(self, monkeypatch):
        # The SDK spawns the claude CLI with the FULL parent env and merges
        # options.env ON TOP ({**os.environ, **options.env}), so the scrub
        # must override each present metered vector with "" — a filtered
        # copy could never remove an inherited key. Simulate the SDK merge
        # to prove the neutralization end-to-end.
        import os as _os

        metered = {
            "ANTHROPIC_API_KEY": "sk-ant-api03-fake",
            "ANTHROPIC_AUTH_TOKEN": "fake-bearer",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_USE_VERTEX": "1",
            "AWS_ACCESS_KEY_ID": "AKIAFAKE",
            "AWS_SECRET_ACCESS_KEY": "fake-secret",
            "AWS_SESSION_TOKEN": "fake-session",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/fake-sa.json",
        }
        for key, value in metered.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-subscription")

        session, _ = _make_session(script=[ResultMessage(result="ok")])
        fields = session.build_option_fields()

        # Every present metered vector is overridden to "" (empty = unset
        # for the CLI and the AWS/GCP credential chains).
        for key in metered:
            assert fields["env"][key] == "", key

        # The SDK-side merge — the actual child env — sees them neutralized.
        merged = {**_os.environ, **fields["env"]}
        for key in metered:
            assert merged[key] == "", key

        # Benign keys and the subscription token flow are NOT overridden:
        # absent from options.env, so the inherited values survive the merge.
        for benign in ("HOME", "PATH", "CLAUDE_CODE_OAUTH_TOKEN"):
            assert benign not in fields["env"], benign
        assert merged["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-subscription"

    def test_absent_metered_vectors_are_not_invented(self, monkeypatch):
        # Only PRESENT vectors are overridden — writing "" for absent ones
        # would hand the child empty vars it never had (an empty
        # AWS_ACCESS_KEY_ID can itself break AWS credential chains).
        for key in (
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS",
        ):
            monkeypatch.delenv(key, raising=False)
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["env"] == {}

    def test_allow_metered_key_disables_the_scrub(self, monkeypatch):
        # allow_metered_key: true is the operator's explicit metered opt-in
        # (the startup guard honors it); the scrub must honor it too, or the
        # documented escape hatch would hand the CLI a blanked key.
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"allow_metered_key": True}}
            },
            raising=False,
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["env"] == {}

    def test_metered_key_refuses_startup_fail_closed(self, monkeypatch):
        # The hard rule enforced at the front door: a present metered key
        # must abort the REAL runtime startup path, never silently rebill.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        session = ClaudeAgentSdkSession(cwd="/tmp")  # no factory → real path
        turn = session.run_turn("hi")
        assert turn.should_retire
        assert "ANTHROPIC_API_KEY" in (turn.error or "")


# ---------- runtime glue ----------


def _make_turn(**overrides):
    base = dict(
        interrupted=False,
        error=None,
        thread_id="sdk-session-1",
        turn_id="uuid-1",
        projected_messages=[{"role": "assistant", "content": "SDK_ASSISTANT"}],
        tool_iterations=2,
        final_text="SDK_ASSISTANT",
        should_retire=False,
        billing_mode="subscription_included",
        token_usage_last={"input_tokens": 7, "output_tokens": 3},
        token_usage_total=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_agent():
    agent = MagicMock()
    agent._claude_sdk_session = MagicMock()
    agent._claude_sdk_session.run_turn.return_value = _make_turn()
    agent.tool_progress_callback = None
    agent._interrupt_requested = False
    agent._persist_disabled = False
    agent.skip_background_review = False
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent._session_db_created = True
    agent.session_id = "sess-1"
    agent.session_api_calls = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.context_compressor = None
    agent.model = "claude-opus-4-8"
    agent.provider = "claude-agent-sdk"
    agent.base_url = ""
    return agent


class TestRuntimeGlue:
    def test_turn_contract(self):
        agent = _make_agent()
        messages = [{"role": "user", "content": "hi"}]
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages,
            effective_task_id="task-1",
        )
        assert result["final_response"] == "SDK_ASSISTANT"
        assert result["completed"] is True
        assert result["agent_persisted"] is True
        assert result["cost_status"] == "included"
        assert result["cost_source"] == "claude-subscription"
        # Projected messages spliced after the (pre-appended) user turn.
        assert messages[-1]["content"] == "SDK_ASSISTANT"
        # Skill-nudge counter parity with the codex path.
        assert agent._iters_since_skill == 2

    def test_compact_boundary_completes_once_before_turn_end(self, monkeypatch):
        """The stream boundary is primary; terminal completion is fallback only."""
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        callbacks = {}

        class SpySession:
            def __init__(self, **kwargs):
                callbacks.update(kwargs)

            def context_usage(self):
                return None

            def set_turn_visibility_callbacks(self, **kwargs):
                pass

            def run_turn(self, user_input):
                callbacks["on_compaction"]("auto")
                callbacks["on_compact_boundary"]("auto")
                return _make_turn()

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        emitted = []
        agent._emit_status = emitted.append
        agent.status_callback = lambda kind, text: emitted.append((kind, text))

        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )

        assert agent._sdk_compaction_pending is False
        completions = [item for item in emitted if isinstance(item, tuple)]
        assert len(completions) == 1
        assert completions[0][0] == "compaction"
        assert "compaction complete" in completions[0][1].lower()

    def test_empty_rich_input_rejects_before_digest_or_session_creation(self):
        agent = _make_agent()
        agent._claude_sdk_session = None
        result = run_claude_agent_sdk_turn(
            agent,
            user_message=[],
            original_user_message=[],
            messages=[
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": []},
            ],
            effective_task_id="task-1",
        )
        assert result["api_calls"] == 0
        assert result["partial"] is True
        assert result["interrupted"] is False
        assert agent._claude_sdk_session is None

    def test_empty_rejection_consumes_interrupts_and_next_turn_runs(self):
        agent = _make_agent()
        session = agent._claude_sdk_session
        agent._interrupt_requested = True
        rejected = run_claude_agent_sdk_turn(
            agent,
            user_message="   ",
            original_user_message="   ",
            messages=[{"role": "user", "content": "   "}],
            effective_task_id="task-1",
        )
        assert rejected["api_calls"] == 0
        assert agent._interrupt_requested is False
        session.consume_interrupt.assert_called_once()
        session.run_turn.assert_not_called()

        accepted = run_claude_agent_sdk_turn(
            agent,
            user_message="continue",
            original_user_message="continue",
            messages=[{"role": "user", "content": "continue"}],
            effective_task_id="task-2",
        )
        assert accepted["api_calls"] == 1
        session.run_turn.assert_called_once_with(user_input="continue")

    def test_native_image_with_continuity_digest_preserves_blocks(
        self, monkeypatch
    ):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        seen = {}

        class SpySession:
            def __init__(self, **kwargs):
                pass

            def run_turn(self, user_input):
                seen["input"] = user_input
                return _make_turn()

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        image = {
            "type": "image_url",
            "image_url": {"url": "https://example.test/photo.png"},
        }
        run_claude_agent_sdk_turn(
            agent,
            user_message=[image],
            original_user_message=[image],
            messages=[
                {"role": "assistant", "content": "prior answer"},
                {"role": "user", "content": [image]},
            ],
            effective_task_id="task-1",
        )
        assert seen["input"][0]["type"] == "text"
        assert "continuity" in seen["input"][0]["text"].lower()
        assert seen["input"][1] == {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.test/photo.png",
            },
        }

    def test_transport_local_result_is_visible_without_usage_count(self):
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            api_call_made=False,
            error="local rejection",
            final_text="local rejection",
            projected_messages=[],
            token_usage_last=None,
        )
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="non-empty local input",
            original_user_message="non-empty local input",
            messages=[{"role": "user", "content": "non-empty local input"}],
            effective_task_id="task-1",
        )
        assert result["final_response"] == "local rejection"
        assert result["api_calls"] == 0
        assert result["partial"] is True
        assert agent.session_api_calls == 0

    def test_retire_closes_session(self):
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            should_retire=True, error="turn timed out after 600s",
            projected_messages=[], final_text="", token_usage_last=None,
        )
        stale = agent._claude_sdk_session
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        stale.close.assert_called_once()
        assert agent._claude_sdk_session is None
        assert result["partial"] is True


# ---------- background review spawns only when routed off this runtime ----------


class TestBackgroundReviewRouting:
    @staticmethod
    def _route(monkeypatch, routed):
        import agent.background_review as br

        monkeypatch.setattr(
            br, "_resolve_review_runtime", lambda _agent: {"routed": routed}
        )

    def _run(self, agent, *, want_memory=False):
        return run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
            should_review_memory=want_memory,
        )

    def test_unrouted_memory_nudge_does_not_spawn(self, monkeypatch):
        self._route(monkeypatch, False)
        agent = _make_agent()
        self._run(agent, want_memory=True)
        agent._spawn_background_review.assert_not_called()

    def test_unrouted_skill_nudge_does_not_spawn_but_counter_still_ticks(
        self, monkeypatch
    ):
        self._route(monkeypatch, False)
        agent = _make_agent()
        agent._skill_nudge_interval = 1
        agent.valid_tool_names = set()
        self._run(agent)
        agent._spawn_background_review.assert_not_called()
        assert agent._iters_since_skill == 0

    def test_routed_memory_nudge_spawns_the_review(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        self._run(agent, want_memory=True)
        agent._spawn_background_review.assert_called_once()
        assert agent._spawn_background_review.call_args.kwargs["review_memory"]

    def test_routed_skill_nudge_spawns_the_review(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        agent._skill_nudge_interval = 1
        agent.valid_tool_names = set()
        self._run(agent)
        agent._spawn_background_review.assert_called_once()
        assert agent._spawn_background_review.call_args.kwargs["review_skills"]

    def test_skill_review_does_not_need_foreground_skill_manage(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        agent._skill_nudge_interval = 1
        agent.valid_tool_names = set()
        self._run(agent)
        agent._spawn_background_review.assert_called_once()
        assert agent._spawn_background_review.call_args.kwargs["review_skills"]

    def test_skip_background_review_blocks_routed_skill_review(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        agent.skip_background_review = True
        agent._skill_nudge_interval = 1
        agent.valid_tool_names = set()
        self._run(agent)
        agent._spawn_background_review.assert_not_called()

    def test_dead_turn_never_spawns_even_when_routed(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            interrupted=True, final_text=""
        )
        self._run(agent, want_memory=True)
        agent._spawn_background_review.assert_not_called()

    def test_broken_resolver_falls_back_to_skipping(self, monkeypatch):
        import agent.background_review as br

        def _boom(_agent):
            raise RuntimeError("no config")

        monkeypatch.setattr(br, "_resolve_review_runtime", _boom)
        agent = _make_agent()
        result = self._run(agent, want_memory=True)
        agent._spawn_background_review.assert_not_called()
        assert result["final_response"] == "SDK_ASSISTANT"

    def test_raising_spawn_does_not_take_the_turn_down(self, monkeypatch):
        self._route(monkeypatch, True)
        agent = _make_agent()
        agent._spawn_background_review.side_effect = RuntimeError("thread fail")
        result = self._run(agent, want_memory=True)
        agent._spawn_background_review.assert_called_once()
        assert result["final_response"] == "SDK_ASSISTANT"


# ---------- direct HTTP MCP security -------------------------------------


class TestHttpMcpSecurity:
    @pytest.fixture(autouse=True)
    def _enable_direct_http_opt_in(self, monkeypatch):
        from agent.transports import claude_agent_sdk_session as mod

        monkeypatch.setattr(
            mod,
            "_provider_config",
            lambda: {"hybrid_mcp_bridge": True},
        )

    def test_helper_is_default_off_even_when_called_directly(
        self, monkeypatch, tmp_path
    ):
        from agent.transports import claude_agent_sdk_session as mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "mcp_servers:\n  public:\n    url: https://mcp.example.test\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "_provider_config", lambda: {})
        monkeypatch.setattr(
            mod,
            "_configured_hybrid_exclude",
            MagicMock(side_effect=AssertionError("must not read exclusions")),
        )
        assert mod._http_mcp_entries_from_config() == {}

    def test_default_off_never_discovers_direct_http_servers(self, monkeypatch):
        from agent.transports import claude_agent_sdk_session as mod

        discover = MagicMock(return_value={
            "remote": {"type": "http", "url": "https://mcp.example.test"}
        })
        monkeypatch.setattr(mod, "_http_mcp_entries_from_config", discover)
        session, _ = _make_session(script=[ResultMessage(result="ok")])

        fields = session.build_option_fields()

        discover.assert_not_called()
        assert "remote" not in fields["mcp_servers"]

    def test_header_bearing_server_is_refused_without_secret_in_logs(
        self, monkeypatch, tmp_path, caplog
    ):
        from agent.transports.claude_agent_sdk_session import (
            _http_mcp_entries_from_config,
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PROFILE", "test")
        monkeypatch.setenv("PRIVATE_MCP_TOKEN", "super-secret-token")
        (tmp_path / "config.yaml").write_text(
            """mcp_servers:
  private-search:
    url: https://mcp.example.test
    headers:
      Authorization: Bearer ${PRIVATE_MCP_TOKEN}
""",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            entries = _http_mcp_entries_from_config()

        assert entries == {}
        assert "private-search" in caplog.text
        assert "super-secret-token" not in caplog.text
        assert "Authorization" not in caplog.text

    def test_headerless_server_is_safe_to_register(self, monkeypatch, tmp_path):
        from agent.transports.claude_agent_sdk_session import (
            _http_mcp_entries_from_config,
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PROFILE", "test")
        (tmp_path / "config.yaml").write_text(
            """mcp_servers:
  public-search:
    url: https://mcp.example.test
""",
            encoding="utf-8",
        )

        assert _http_mcp_entries_from_config() == {
            "public-search": {
                "type": "http",
                "url": "https://mcp.example.test",
            }
        }

    def test_malformed_resolved_url_is_refused_without_url_in_logs(
        self, monkeypatch, tmp_path, caplog
    ):
        from agent.transports.claude_agent_sdk_session import (
            _http_mcp_entries_from_config,
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PROFILE", "test")
        monkeypatch.setenv("MISSING_MCP_HOST", "expanded-secret-host")
        (tmp_path / "config.yaml").write_text(
            """mcp_servers:
  broken-search:
    url: https://${MISSING_MCP_HOST}/mcp
""",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            entries = _http_mcp_entries_from_config()

        assert entries == {}
        assert "broken-search" in caplog.text
        assert "https:///mcp" not in caplog.text
        assert "expanded-secret-host" not in caplog.text

    @pytest.mark.parametrize(
        "url",
        [
            "https://mcp.example.test/${lowercase_secret}/mcp",
            "https://user:password@mcp.example.test/mcp",
            "https://mcp.example.test/mcp?api-key=value",
            "https://mcp.example.test/mcp?apikey=value",
            "https://mcp.example.test/mcp?token=value",
            "https://mcp.example.test/mcp?%74oken=value",
            "https://mcp.example.test/mcp?%2574oken=value",
            "https://mcp.example.test/mcp?signature=value",
            "https://mcp.example.test/mcp#authorization=value",
            "https://mcp.example.test/mcp#%61uth=value",
        ],
    )
    def test_templated_or_credential_bearing_url_is_refused(
        self, monkeypatch, tmp_path, url
    ):
        from agent.transports.claude_agent_sdk_session import (
            _http_mcp_entries_from_config,
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            f"mcp_servers:\n  private:\n    url: {url}\n",
            encoding="utf-8",
        )
        assert _http_mcp_entries_from_config() == {}

    def test_truthy_non_mapping_headers_are_refused(self, monkeypatch, tmp_path):
        from agent.transports.claude_agent_sdk_session import (
            _http_mcp_entries_from_config,
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "mcp_servers:\n  malformed:\n    url: https://mcp.example.test\n"
            "    headers: bearer-secret\n",
            encoding="utf-8",
        )
        assert _http_mcp_entries_from_config() == {}

    def test_http_server_name_honors_exclusion(self, monkeypatch, tmp_path):
        from agent.transports import claude_agent_sdk_session as mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            mod,
            "_provider_config",
            lambda: {
                "hybrid_mcp_bridge": True,
                "hybrid_mcp_bridge_exclude": ["public-search"],
            },
        )
        (tmp_path / "config.yaml").write_text(
            "mcp_servers:\n  public-search:\n"
            "    url: https://mcp.example.test\n",
            encoding="utf-8",
        )
        assert mod._http_mcp_entries_from_config() == {}

    def test_hermes_tools_collision_preserves_hybrid_entry(
        self, monkeypatch, tmp_path
    ):
        from agent.transports import hermes_hybrid_mcp

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            "mcp_servers:\n"
            "  hermes-tools:\n    url: https://mcp.example.test/collision\n"
            "  public:\n    url: https://mcp.example.test/public\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            hermes_hybrid_mcp,
            "build_hybrid_mcp_server",
            lambda *args, server_name, **kwargs: {
                "type": "sdk",
                "name": server_name,
            },
        )
        session = ClaudeAgentSdkSession(
            cwd="/tmp",
            agent=object(),
            tools={"read_file": object()},
        )
        fields = session.build_option_fields()
        assert fields["mcp_servers"]["hermes-tools"]["type"] == "sdk"
        assert fields["mcp_servers"]["public"] == {
            "type": "http",
            "url": "https://mcp.example.test/public",
        }


# ---------- hermes session id plumbing to the MCP shims (#26567) ----------


class TestMcpEnvMinimal:
    def test_mcp_env_carries_no_secrets(self, monkeypatch):
        # Validator C4 (HIGH): the SDK inlines the MCP config -- env
        # included -- into the claude CLI argv, world-readable via ps. The
        # env must be a minimal allowlist, never the credentialed environ.
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-fake")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
        # (ANTHROPIC_AUTH_TOKEN deliberately NOT set here — the C5 fail-closed
        # guard would refuse startup before the MCP config is even built,
        # which is its own test below. The allowlist excludes it regardless.)
        monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-test-home")
        session, holder = _make_session(
            script=[ResultMessage(result="ok")], hermes_session_id="sess-9"
        )
        try:
            session.run_turn("ping")
        finally:
            session.close()
        env = holder["client"].options["mcp_servers"]["hermes-tools"]["env"]
        for secret in ("CLAUDE_CODE_OAUTH_TOKEN", "OPENROUTER_API_KEY",
                       "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
            assert secret not in env, f"{secret} leaked into the MCP argv env"
        assert "PYTHONPATH" in env
        assert env["HERMES_SESSION_ID"] == "sess-9"
        assert env["HERMES_HOME"] == "/tmp/hermes-test-home"

    def test_state_db_override_rides_the_mcp_env(self, monkeypatch):
        # Validator N1 (round 3): the C4 allowlist dropped HERMES_MCP_STATE_DB,
        # silently killing the shims' documented state-DB override — the MCP
        # subprocess searched the DEFAULT DB with no error. A path, not a
        # secret, so it belongs on the allowlist.
        monkeypatch.setenv("HERMES_MCP_STATE_DB", "/tmp/custom-state.db")
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        env = holder["client"].options["mcp_servers"]["hermes-tools"]["env"]
        assert env["HERMES_MCP_STATE_DB"] == "/tmp/custom-state.db"

    def test_anthropic_auth_token_refuses_startup(self, monkeypatch):
        # Validator C5: the CLI also honors ANTHROPIC_AUTH_TOKEN (bearer,
        # typically metered/proxy) — same fail-closed class as the API key.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "fake-bearer")
        session = ClaudeAgentSdkSession(cwd="/tmp")  # no factory → real path
        turn = session.run_turn("hi")
        assert turn.should_retire
        assert "ANTHROPIC_AUTH_TOKEN" in (turn.error or "")

    def test_allow_metered_key_via_config_yaml(self, monkeypatch):
        # The explicit override is a config.yaml key (AGENTS.md: behavioral
        # settings live in config, not env); the guard steps aside and the
        # fake-backed session starts normally.
        import hermes_cli.config as cfg

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"allow_metered_key": True}}
            },
        )
        session, _holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            turn = session.run_turn("ping")
        finally:
            session.close()
        assert not turn.should_retire
        assert turn.error is None

    def test_half_connected_client_is_reaped_on_close(self):
        # Validator C6: on a connect failure the client was assigned only
        # AFTER connect() returned, so close() skipped disconnect and the
        # CLI subprocess was orphaned.
        session, holder = _make_session(connect_exc=RuntimeError("connect blew up"))
        turn = session.run_turn("hi")
        assert turn.should_retire
        session.close()
        assert holder["client"].disconnected is True

    def test_mid_stream_interrupt_breaks_and_discards_tail(self):
        # Validator HIGH test-gap: the /stop-arriving-DURING-streaming path
        # was never exercised at session level.
        holder = {}

        class MidStreamClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                self._pending.append(
                    AssistantMessage(content=[TextBlock("first chunk")])
                )
                holder["session"]._interrupt_event.set()
                self._pending.append(
                    AssistantMessage(content=[TextBlock("tail that must be discarded")])
                )
                self._pending.append(ResultMessage(result="tail that must be discarded"))

        def factory(options=None):
            client = MidStreamClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        holder["session"] = session
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.interrupted is True
        assert all("discarded" not in str(m.get("content")) for m in turn.projected_messages)


class TestStreamOwnership:
    """Regression tests for the 2026-07-25 stale-answer incident
    (dasbrow-hermes-coder#2): the Claude Code CLI runs FULL unsolicited turns
    when background Agent tasks complete, leaving unconsumed ResultMessages in
    the shared FIFO. ``receive_response()`` then serves the OLDEST buffered
    result to the next turn — a permanent, silent off-by-N. Every test here is
    RED on the pre-fix implementation."""

    def _wait_unsolicited(self, session, n, timeout=5.0):
        """Sync point for the fixed code (reader routes idle-time messages
        within ms); a bounded no-op on the pre-fix code, which has no reader."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(session, "_unsolicited_results", 0) >= n:
                return True
            time.sleep(0.01)
        return False

    def test_unsolicited_result_while_idle_is_not_served_as_next_answer(self):
        session, holder = _make_session(
            script=[
                AssistantMessage(content=[TextBlock("fresh answer")]),
                ResultMessage(result="fresh answer", uuid="fresh-1"),
            ]
        )
        try:
            session.ensure_started()
            # A background Agent task finished while nobody asked anything:
            # the CLI ran a full turn on its own initiative.
            holder["client"].feed(
                AssistantMessage(
                    content=[TextBlock("stale answer to an earlier question")]
                ),
                ResultMessage(
                    result="stale answer to an earlier question", uuid="stale-1"
                ),
            )
            self._wait_unsolicited(session, 1)
            turn = session.run_turn("new question", turn_timeout=15.0)
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "fresh answer"
        assert getattr(session, "_unsolicited_results", None) == 1

    def test_offset_does_not_accumulate_across_unsolicited_turns(self):
        # The live incident: 4 unsolicited turns -> every later reply answered
        # a question 4 back. N unsolicited results must be dropped, not queued.
        session, holder = _make_session(
            script=[
                AssistantMessage(content=[TextBlock("the real answer")]),
                ResultMessage(result="the real answer", uuid="real-1"),
            ]
        )
        try:
            session.ensure_started()
            for i in range(3):
                holder["client"].feed(
                    AssistantMessage(content=[TextBlock(f"unsolicited {i}")]),
                    ResultMessage(result=f"unsolicited {i}", uuid=f"u-{i}"),
                )
            self._wait_unsolicited(session, 3)
            turn = session.run_turn("a question", turn_timeout=15.0)
        finally:
            session.close()
        assert turn.error is None
        assert turn.final_text == "the real answer"
        assert getattr(session, "_unsolicited_results", None) == 3

    def test_interrupted_turn_consumes_its_own_result(self):
        # Second entry point to the same corruption: breaking out of the
        # message loop on interrupt used to orphan that turn's ResultMessage
        # in the stream, where it became the NEXT turn's answer.
        holder = {}

        class InterruptingClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                if len(self.queried) == 1:
                    self._pending.append(
                        AssistantMessage(content=[TextBlock("turn1 partial")])
                    )
                    holder["session"]._interrupt_event.set()
                    self._pending.append(
                        ResultMessage(result="turn1 stale result", uuid="r1")
                    )
                else:
                    self._pending.append(
                        AssistantMessage(content=[TextBlock("turn2 answer")])
                    )
                    self._pending.append(
                        ResultMessage(result="turn2 answer", uuid="r2")
                    )

        def factory(options=None):
            client = InterruptingClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        holder["session"] = session
        try:
            turn1 = session.run_turn("first", turn_timeout=15.0)
            assert turn1.interrupted is True
            turn2 = session.run_turn("second", turn_timeout=15.0)
        finally:
            session.close()
        assert turn2.interrupted is False
        assert turn2.final_text == "turn2 answer"  # NOT "turn1 stale result"

    def test_residue_after_result_is_not_carried_into_next_turn(self):
        # A CLI turn that completes WHILE ours is running parks its result
        # behind ours in the stream. It must be routed away as unsolicited,
        # not served as the next turn's answer. (Mid-flight overlap — the one
        # window the idle-time tests above don't cover. Ported from the
        # independent re-derivation of this fix, commit 09537f965.)
        #
        # Updated pin (2026-08-07): the original version routed ALL residue
        # to the background-delivery lane with no content discrimination —
        # which ships a turn's OWN answer as a fake background completion,
        # the 2026-08-06 incident class. New intent: residue never becomes
        # the next turn's answer (unchanged) AND the delivery lane splits by
        # content — genuinely different residue (this test) still delivers
        # as a background burst; own-answer residue is suppressed (see
        # test_own_answer_residue_never_delivered_as_background_result).
        holder = {}
        got = []

        class OverlappingClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                if len(self.queried) == 1:
                    self._pending.append(ResultMessage(result="FIRST", uuid="f-1"))
                    self._pending.append(
                        ResultMessage(
                            result="RESIDUE from an overlapping CLI turn",
                            uuid="res-1",
                        )
                    )
                else:
                    self._pending.append(ResultMessage(result="SECOND", uuid="s-1"))

        def factory(options=None):
            client = OverlappingClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(
            cwd="/tmp", client_factory=factory, on_unsolicited_result=got.append
        )
        try:
            first = session.run_turn("one", turn_timeout=15.0)
            assert first.final_text == "FIRST"
            assert self._wait_unsolicited(session, 1), (
                "residue was left in the stream to poison the next turn"
            )
            second = session.run_turn("two", turn_timeout=15.0)
        finally:
            session.close()
        assert second.final_text == "SECOND"
        # Delivery split: differing residue is a REAL background completion —
        # it must still reach the delivery lane, not be swallowed.
        assert got == [["RESIDUE from an overlapping CLI turn"]]

    def test_own_answer_residue_never_delivered_as_background_result(
        self, caplog
    ):
        # D2 rework (2026-08-06 incident class): a residue ResultMessage that
        # repeats the just-finished turn's OWN answer must be suppressed —
        # dedup-marked and WARN'd, never handed to the background-delivery
        # callback as a fake completion.
        holder = {}
        got = []

        class OwnEchoClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                if len(self.queried) == 1:
                    self._pending.append(ResultMessage(result="FIRST", uuid="f-1"))
                    self._pending.append(
                        ResultMessage(result="FIRST", uuid="own-dup")
                    )
                else:
                    self._pending.append(ResultMessage(result="SECOND", uuid="s-1"))

        def factory(options=None):
            client = OwnEchoClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(
            cwd="/tmp", client_factory=factory, on_unsolicited_result=got.append
        )
        with caplog.at_level(
            logging.WARNING, logger="agent.transports.claude_agent_sdk_session"
        ):
            try:
                first = session.run_turn("one", turn_timeout=15.0)
                assert first.final_text == "FIRST"
                # run_turn returns only after the residue drain — the
                # suppression already happened; a bounded wait just proves
                # nothing arrives late either.
                self._wait(lambda: got, timeout=0.5)
                second = session.run_turn("two", turn_timeout=15.0)
            finally:
                session.close()
        assert got == [], "own-answer residue was delivered as a fake background result"
        assert second.final_text == "SECOND"
        assert session._unsolicited_results == 1  # routed away, still counted
        assert "own-dup" in session._unsolicited_delivered
        assert any(
            "matches this turn's own answer" in r.getMessage()
            for r in caplog.records
        ), "suppression must WARN, never silently drop"

    def test_genuine_overlap_residue_still_delivers_as_burst(self):
        # The delivery split's other half, explicit: residue with DIFFERENT
        # content is deliver_background_results working — never suppressed.
        holder = {}
        got = []

        class OverlapClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                self._pending.append(ResultMessage(result="FIRST", uuid="f-1"))
                self._pending.append(
                    ResultMessage(result="DIFFERENT bg completion", uuid="bg-9")
                )

        def factory(options=None):
            client = OverlapClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(
            cwd="/tmp", client_factory=factory, on_unsolicited_result=got.append
        )
        try:
            first = session.run_turn("one", turn_timeout=15.0)
            assert first.final_text == "FIRST"
            assert self._wait(lambda: got, timeout=5.0)
        finally:
            session.close()
        assert got == [["DIFFERENT bg completion"]]

    def test_stale_unsolicited_text_never_attaches_to_later_result(
        self, caplog
    ):
        # Leak fix: text buffered idle-time whose terminal ResultMessage never
        # arrived is discarded (with WARN) at the next turn's start — a later
        # unrelated result must never pick it up as its own burst.
        got = []
        session, holder = _make_session(
            script=[
                AssistantMessage(content=[TextBlock("fresh answer")]),
                ResultMessage(result="fresh answer", uuid="fresh-1"),
            ],
            on_unsolicited_result=got.append,
        )
        with caplog.at_level(
            logging.WARNING, logger="agent.transports.claude_agent_sdk_session"
        ):
            try:
                session.ensure_started()
                # A background turn started streaming but its result never
                # came (CLI died / mid-burst) — text sits in the buffer.
                holder["client"].feed(
                    AssistantMessage(content=[TextBlock("orphaned partial text")])
                )
                assert self._wait(lambda: session._unsolicited_text)
                turn = session.run_turn("new question", turn_timeout=15.0)
                assert turn.final_text == "fresh answer"
                # A later, unrelated background completion arrives idle-time.
                holder["client"].feed(
                    ResultMessage(result="unrelated bg answer", uuid="bg-x")
                )
                assert self._wait(lambda: got)
            finally:
                session.close()
        assert got == [["unrelated bg answer"]], (
            "stale pre-turn text misattached to an unrelated later result"
        )
        assert any(
            "stale unsolicited text" in r.getMessage()
            for r in caplog.records
        ), "turn-start discard must WARN, never silently drop"

    @staticmethod
    def _wait(cond, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.01)
        return False

    def test_stream_death_mid_turn_fails_fast_instead_of_hanging(self):
        # A script with no ResultMessage models the CLI dying mid-turn. The
        # turn must surface an error promptly — pre-fix the loop ended and the
        # turn returned as an empty SUCCESS; an unguarded reader design would
        # instead hang until turn_timeout.
        session, _holder = _make_session(
            script=[AssistantMessage(content=[TextBlock("half an answer")])]
        )
        started = time.monotonic()
        try:
            turn = session.run_turn("hi", turn_timeout=15.0)
        finally:
            session.close()
        elapsed = time.monotonic() - started
        assert elapsed < 10.0, f"turn took {elapsed:.1f}s — hung on a dead stream"
        assert turn.error is not None and "stream ended" in turn.error


class TestHermesSessionIdPlumbing:
    def test_session_id_rides_mcp_env(self):
        session, holder = _make_session(
            script=[ResultMessage(result="ok")], hermes_session_id="sess-42"
        )
        try:
            session.run_turn("ping")
        finally:
            session.close()
        env = holder["client"].options["mcp_servers"]["hermes-tools"]["env"]
        assert env["HERMES_SESSION_ID"] == "sess-42"
        # The invented pre-fix name must never come back: the shim consumer
        # reads only the canonical HERMES_SESSION_ID.
        assert "HERMES_MCP_SESSION_ID" not in env

    def test_no_session_id_no_env_var(self):
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        env = holder["client"].options["mcp_servers"]["hermes-tools"]["env"]
        assert "HERMES_SESSION_ID" not in env
        assert "HERMES_MCP_SESSION_ID" not in env

    def test_runtime_passes_agent_session_id(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        assert captured.get("hermes_session_id") == "sess-1"

    def test_runtime_passes_context_to_append_builder(self, monkeypatch):
        # W2: the append builder receives the agent's platform/session/model
        # so the session line and platform hint reflect the live session. The
        # SDK process uses the validated runtime cwd, while prompt discovery
        # keeps None as the native fallback sentinel so the install-tree guard
        # can distinguish fallback from an operator-selected directory.
        import agent.claude_sdk_runtime as rt
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        captured = {}
        session_captured = {}

        def fake_append(**kwargs):
            captured.update(kwargs)
            return "APPEND-UNDER-TEST"

        class SpySession:
            def __init__(self, **kwargs):
                session_captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(rt, "build_system_prompt_append", fake_append)
        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        import agent.runtime_cwd as runtime_cwd

        monkeypatch.setattr(runtime_cwd, "resolve_agent_cwd", lambda: "/resolved-workspace")
        monkeypatch.setattr(runtime_cwd, "resolve_context_cwd", lambda: None)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent.session_cwd = "/unvalidated-stale-workspace"
        agent.skip_context_files = True
        agent.platform = "telegram"
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        # Per-key pins (not whole-dict equality): the contract is that the
        # builder receives the live session's platform/session/model — a new
        # kwarg added later must not break these unrelated assertions.
        assert captured["platform"] == "telegram"
        assert captured["session_id"] == "sess-1"
        assert captured["model"] == "claude-opus-4-8"
        assert captured["cwd"] is None
        assert captured["include_project_context"] is False
        assert session_captured["cwd"] == "/resolved-workspace"

        # A validated, explicitly configured context cwd is forwarded as an
        # explicit path and project files return to their default-on posture.
        captured.clear()
        session_captured.clear()
        agent._claude_sdk_session = None
        agent.skip_context_files = False
        monkeypatch.setattr(
            runtime_cwd,
            "resolve_context_cwd",
            lambda: "/configured-workspace",
        )
        run_claude_agent_sdk_turn(
            agent,
            user_message="again",
            original_user_message="again",
            messages=[{"role": "user", "content": "again"}],
            effective_task_id="task-2",
        )
        assert captured["cwd"] == "/configured-workspace"
        assert captured["include_project_context"] is True
        assert session_captured["cwd"] == "/resolved-workspace"

    @staticmethod
    def _run_with_spy_session(monkeypatch, config_block):
        """Drive one runtime turn with a kwargs-capturing session and the
        given agent.claude_agent_sdk config block; returns captured kwargs."""
        import agent.claude_sdk_runtime as rt
        import agent.transports.claude_agent_sdk_session as sdk_session_mod
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": config_block}},
            raising=False,
        )
        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(rt, "build_system_prompt_append", lambda **k: None)
        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        return captured

    def test_max_budget_usd_config_reaches_the_session(self, monkeypatch):
        captured = self._run_with_spy_session(
            monkeypatch, {"max_budget_usd": 2.5}
        )
        assert captured["max_budget_usd"] == 2.5

    def test_max_budget_usd_default_is_no_budget(self, monkeypatch):
        captured = self._run_with_spy_session(monkeypatch, {})
        assert captured["max_budget_usd"] is None

    def test_max_budget_usd_invalid_values_ignored(self, monkeypatch):
        # A typo or a nonsense cap (0 would fail every turn instantly) must
        # never become a silent behavior change — no budget is passed.
        for bad in ("not-a-number", 0, -3, True):
            captured = self._run_with_spy_session(
                monkeypatch, {"max_budget_usd": bad}
            )
            assert captured["max_budget_usd"] is None, bad


# ---------- interrupt routes to the SDK session (W4) ----------


class TestInterruptRoutesToSdkSession:
    """/stop and new-message preemption call AIAgent.interrupt(); the SDK
    session's request_interrupt (event + client.interrupt()) already works —
    this pins the one missing caller."""

    @staticmethod
    def _make_real_agent():
        from run_agent import AIAgent

        return AIAgent(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    def test_interrupt_reaches_live_sdk_session(self):
        agent = self._make_real_agent()
        agent._claude_sdk_session = MagicMock()
        agent.interrupt()
        agent._claude_sdk_session.request_interrupt.assert_called_once()

    def test_interrupt_without_sdk_session_stays_safe(self):
        agent = self._make_real_agent()
        agent._claude_sdk_session = None
        agent.interrupt()  # must not raise

    def test_release_clients_disconnects_sdk_session(self):
        # Adversarial-review HIGH: the gateway's ROUTINE evictions (LRU cap,
        # idle-TTL sweep, model switch) release via release_clients(), which
        # never touched the SDK session — leaking the loop thread + the
        # Claude CLI subprocess per eviction on a 24/7 gateway.
        agent = self._make_real_agent()
        sdk_session = MagicMock()
        agent._claude_sdk_session = sdk_session
        agent.release_clients()
        sdk_session.close.assert_called_once()
        assert agent._claude_sdk_session is None

    def test_pending_interrupt_flag_short_circuits_cold_turn(self, monkeypatch):
        # Adversarial-review MEDIUM: an interrupt landing before the SDK
        # session exists set only agent._interrupt_requested, which the SDK
        # path never read — the turn ran uninterruptible for up to 600s.
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        instances = []

        class SpySession:
            def __init__(self, **kwargs):
                instances.append(self)

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._interrupt_requested = True
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert instances == []  # no session created, no subscription burn
        assert result["completed"] is False and result["partial"] is True
        assert agent._interrupt_requested is False  # consumed, next turn runs

    def test_honored_interrupt_consumes_agent_flag(self, monkeypatch):
        # Live-gate catch: after an interrupt was honored mid-turn, the
        # agent-level flag stayed set and the cold-flag check short-circuited
        # the NEXT turn into an empty answer. Honoring must consume it.
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._session_db = None

        class SpySession:
            def __init__(self, **kwargs):
                pass

            def run_turn(self, user_input):
                agent._interrupt_requested = True  # user hit /stop mid-turn
                return _make_turn(interrupted=True, final_text="", projected_messages=[])

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert result["partial"] is True
        assert agent._interrupt_requested is False  # consumed — next turn runs

    def test_thread_id_captured_from_init_message(self):
        # A FIRST-turn interrupt used to lose the resume id (only the final
        # ResultMessage carried it). The SDK announces session_id in its init
        # SystemMessage — capture it from any message.
        session, _ = _make_session(script=[SystemMessage(session_id="sdk-early-7")])
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.thread_id == "sdk-early-7"

    def test_pre_set_interrupt_event_honored_then_next_turn_runs(self):
        # Adversarial-review MEDIUM: run_turn unconditionally CLEARED the
        # interrupt event after connect — an interrupt arriving during the
        # (up to 60s) connect window was silently erased. It must instead be
        # honored by THIS turn, and must not bleed into the next one.
        session, holder = _make_session(
            script=[ResultMessage(result="ok")]
        )
        try:
            session.ensure_started()
            session.request_interrupt()
            turn1 = session.run_turn("first")
            assert turn1.interrupted is True
            assert holder["client"].queried == []  # never reached the model
            turn2 = session.run_turn("second")
            assert turn2.interrupted is False
            assert holder["client"].queried == ["second"]
        finally:
            session.close()


# ---------- streaming deltas (W4, env-gated default OFF) ----------


class TestStreaming:
    def test_env_var_cannot_enable_streaming(self, monkeypatch):
        # AGENTS.md:102-107 keeps behavioural settings out of HERMES_* env
        # vars. The old HERMES_CLAUDE_SDK_STREAMING override is gone, so
        # setting it must have NO effect — config.yaml is the only interface.
        monkeypatch.setenv("HERMES_CLAUDE_SDK_STREAMING", "1")
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert "include_partial_messages" not in holder["client"].options

    def test_option_absent_by_default(self):
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert "include_partial_messages" not in holder["client"].options

    def test_config_yaml_is_the_operator_interface(self, monkeypatch):
        # AGENTS.md: behavioral settings live in config.yaml, not env.
        # agent.claude_agent_sdk.streaming turns the option on without any env.
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": {"streaming": True}}},
        )
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["include_partial_messages"] is True

    def test_env_var_cannot_disable_config_streaming(self, monkeypatch):
        # The mirror of the test above: an explicit env "0" must NOT be able to
        # veto config.yaml either. Together the pair pins the override as fully
        # inert in both directions, so it cannot creep back in unnoticed.
        import hermes_cli.config as cfg

        monkeypatch.setenv("HERMES_CLAUDE_SDK_STREAMING", "0")
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": {"streaming": True}}},
        )
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["include_partial_messages"] is True

    def test_setting_sources_isolated_by_default(self):
        # Absent config → full isolation: the SDK loads NO filesystem
        # settings, so ambient ~/.claude / project files cannot
        # re-permission tools underneath the configured posture.
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["setting_sources"] == []

    def test_setting_sources_config_opt_in(self, monkeypatch):
        # Deployments whose operating model stores tool grants in the
        # operator's own ~/.claude/settings.json (unattended cron turns that
        # must pre-approve WebSearch/MCP tools) opt back in explicitly.
        # Regression: the hardening initially shipped setting_sources
        # hardcoded [] and silently cut a production box's cron jobs off
        # from their allowlist (2026-07-26).
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"setting_sources": ["user"]}}
            },
        )
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["setting_sources"] == ["user"]

    def test_setting_sources_invalid_entries_dropped(self, monkeypatch):
        # A typo must never silently load an unintended source; valid
        # entries survive, invalid ones are dropped (with a warning).
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {
                    "claude_agent_sdk": {
                        "setting_sources": ["user", "bogus", "project"]
                    }
                }
            },
        )
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["setting_sources"] == ["user", "project"]

    def test_deltas_reach_callback_and_never_the_transcript(self):
        got = []
        script = [
            _text_delta_event("Hel"),
            _text_delta_event("lo"),
            AssistantMessage(content=[TextBlock("Hello")]),
            ResultMessage(result="Hello"),
        ]
        session, _ = _make_session(script=script, on_stream_delta=got.append)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert got == ["Hel", "lo"]
        # Display-only: deltas never become transcript rows.
        assert [m["role"] for m in turn.projected_messages] == ["assistant"]
        assert turn.final_text == "Hello"

    def test_tool_adjacent_assistant_text_reaches_interim_callback_once(self):
        commentary = []
        script = [
            AssistantMessage(content=[
                TextBlock("I will inspect the project first."),
                ToolUseBlock(id="t1", name="Bash", input={"command": "pwd"}),
            ]),
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="/tmp")]),
            ResultMessage(result="Final answer"),
        ]
        session, _ = _make_session(script=script, on_interim_assistant=commentary.append)
        try:
            turn = session.run_turn("inspect")
        finally:
            session.close()
        assert commentary == ["I will inspect the project first."]
        assert turn.final_text == "Final answer"

    def test_tool_iteration_callback_runs_before_turn_returns(self):
        iterations = []
        script = [
            AssistantMessage(content=[ToolUseBlock(id="t1", name="Bash", input={"command": "pwd"})]),
            UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="/tmp")]),
            ResultMessage(result="Final answer"),
        ]
        session, _ = _make_session(script=script, on_tool_iteration=lambda: iterations.append(True))
        try:
            turn = session.run_turn("inspect")
        finally:
            session.close()
        assert iterations == [True]
        assert turn.tool_iterations == 1

    def test_subagent_deltas_are_not_forwarded(self):
        got = []
        script = [
            _text_delta_event("sub", parent_tool_use_id="tool-1"),
            ResultMessage(result="done"),
        ]
        session, _ = _make_session(script=script, on_stream_delta=got.append)
        try:
            session.run_turn("hi")
        finally:
            session.close()
        assert got == []

    def test_runtime_wires_late_bound_stream_callback(self, monkeypatch):
        # The gateway assigns agent.stream_delta_callback per turn AFTER the
        # session exists — the wiring must read it at call time.
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        relay = captured.get("on_stream_delta")
        assert callable(relay)
        cli_seen = []
        gateway_seen = []
        agent.stream_delta_callback = cli_seen.append  # assigned AFTER creation
        agent._stream_callback = gateway_seen.append
        relay("delta-text")
        assert cli_seen == ["delta-text"]
        assert gateway_seen == ["delta-text"]

        # The CLI/TUI callback is cleared between turns, but the gateway sink
        # remains the desktop's message.delta lane and must keep receiving.
        agent.stream_delta_callback = None
        relay("gateway-only")
        assert cli_seen == ["delta-text"]
        assert gateway_seen == ["delta-text", "gateway-only"]

        agent._stream_callback = None  # both cleared → no crash
        relay("dropped")
        assert gateway_seen == ["delta-text", "gateway-only"]


# ---------- continuity: resume + digest fallback (W3) ----------


class TestContinuity:
    """Retire matrix under test:
      /new, expiry      → new Hermes session row → no persisted id → FRESH
      restart/eviction  → same row, id persisted → RESUME
      error retire      → persisted id CLEARED → next turn fresh + digest
      stale resume      → retire → clear → ONE fresh retry with digest
    """

    @staticmethod
    def _db_agent(persisted_sdk_id=None):
        agent = _make_agent()
        agent._claude_sdk_session = None
        db = MagicMock()
        db.get_session.return_value = {"claude_sdk_session_id": persisted_sdk_id}
        agent._session_db = db
        agent._session_db_created = True
        return agent, db

    @staticmethod
    def _spy_sessions(monkeypatch, behaviors):
        """Install a SpySession whose Nth instance behaves per behaviors[N]:
        a TurnResult-like object to return, or an Exception to raise."""
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        instances = []

        class SpySession:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.inputs = []
                instances.append(self)

            def run_turn(self, user_input):
                self.inputs.append(user_input)
                behavior = behaviors[len(instances) - 1]
                if isinstance(behavior, Exception):
                    raise behavior
                return behavior

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        return instances

    def test_creation_resumes_from_persisted_id(self, monkeypatch):
        agent, _db = self._db_agent(persisted_sdk_id="sdk-old-1")
        instances = self._spy_sessions(monkeypatch, [_make_turn()])
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert instances[0].kwargs.get("resume_session_id") == "sdk-old-1"
        # A resumed session already holds the context — no digest.
        assert instances[0].inputs == ["hi"]

    def test_successful_turn_persists_thread_id(self, monkeypatch):
        agent, db = self._db_agent()
        self._spy_sessions(monkeypatch, [_make_turn(thread_id="sdk-new-9")])
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        db.update_claude_sdk_session_id.assert_called_with("sess-1", "sdk-new-9")

    def test_error_retire_clears_persisted_id(self, monkeypatch):
        agent, db = self._db_agent()
        self._spy_sessions(monkeypatch, [_make_turn(
            should_retire=True, error="turn timed out", projected_messages=[],
            final_text="", token_usage_last=None,
        )])
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        db.update_claude_sdk_session_id.assert_called_with("sess-1", None)

    def test_digest_prepended_on_fresh_session_with_history(self, monkeypatch):
        agent, _db = self._db_agent(persisted_sdk_id=None)
        instances = self._spy_sessions(monkeypatch, [_make_turn()])
        messages = [
            {"role": "user", "content": "the linter flags shadowed imports"},
            {"role": "assistant", "content": "Fixed by renaming the local."},
            {"role": "user", "content": "and the tests?"},
        ]
        run_claude_agent_sdk_turn(
            agent, user_message="and the tests?", original_user_message="and the tests?",
            messages=messages, effective_task_id="t",
        )
        sent = instances[0].inputs[0]
        assert sent.startswith("[Continuity digest")
        assert "shadowed imports" in sent
        assert sent.endswith("and the tests?")

    def test_projected_bg_row_excluded_from_continuity_digest(self):
        # Binding amendment (sdk-echo-approval-fixes): rows projected by the
        # background-result lane are the agent's OWN delivered answers —
        # the digest re-presenting them is double-presentation, the exact
        # pathology the lane fixes. Marked rows never enter the digest.
        from agent.claude_sdk_runtime import _render_continuity_digest

        digest = _render_continuity_digest([
            {"role": "user", "content": "run the research"},
            {
                "role": "assistant",
                "content": "the full background report",
                "display_kind": "sdk_background_result",
            },
            {"role": "assistant", "content": "a normal reply"},
        ])
        assert "the full background report" not in digest
        assert "run the research" in digest
        assert "a normal reply" in digest

    def test_no_digest_on_brand_new_conversation(self, monkeypatch):
        agent, _db = self._db_agent(persisted_sdk_id=None)
        instances = self._spy_sessions(monkeypatch, [_make_turn()])
        run_claude_agent_sdk_turn(
            agent, user_message="hello", original_user_message="hello",
            messages=[{"role": "user", "content": "hello"}], effective_task_id="t",
        )
        assert instances[0].inputs == ["hello"]

    def test_stale_resume_retires_then_retries_fresh_with_digest(self, monkeypatch):
        # The Pi probe: a stale resume id fails the session. The runtime
        # must clear the id and retry ONCE fresh (digest included) — the
        # user gets an answer, not an error.
        agent, db = self._db_agent(persisted_sdk_id="sdk-stale-7")
        instances = self._spy_sessions(monkeypatch, [
            _make_turn(should_retire=True, error="resume failed",
                       projected_messages=[], final_text="", token_usage_last=None),
            _make_turn(final_text="fresh answer",
                       projected_messages=[{"role": "assistant", "content": "fresh answer"}]),
        ])
        messages = [
            {"role": "user", "content": "earlier context line"},
            {"role": "assistant", "content": "earlier reply"},
            {"role": "user", "content": "current question"},
        ]
        result = run_claude_agent_sdk_turn(
            agent, user_message="current question",
            original_user_message="current question",
            messages=messages, effective_task_id="t",
        )
        assert result["final_response"] == "fresh answer"
        assert len(instances) == 2
        assert instances[0].kwargs.get("resume_session_id") == "sdk-stale-7"
        assert instances[1].kwargs.get("resume_session_id") is None
        assert instances[1].inputs[0].startswith("[Continuity digest")
        db.update_claude_sdk_session_id.assert_any_call("sess-1", None)

    def test_cold_short_circuit_consumes_live_session_event_too(self, monkeypatch):
        # Validator C1: an interrupt racing turn completion sets BOTH the
        # agent flag and the live session's event. The short-circuit consumed
        # only the flag — the NEXT legit message then died on the stale
        # session event with no model call. Honoring must consume both.
        agent, _db = self._db_agent()
        live = MagicMock()
        agent._claude_sdk_session = live
        agent._interrupt_requested = True
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert result["partial"] is True
        live.consume_interrupt.assert_called_once()
        live.run_turn.assert_not_called()

    def test_resume_id_persisted_after_flush_and_gated_on_persist_disabled(self, monkeypatch):
        # Validator C9: the resume-id UPDATE ran BEFORE the flush that
        # (re)creates the session row after a transient turn-start lock —
        # silently discarding continuity. Order must be flush-then-store.
        agent, db = self._db_agent()
        order = []
        agent._flush_messages_to_session_db = MagicMock(
            side_effect=lambda *a, **k: order.append("flush"))
        db.update_claude_sdk_session_id.side_effect = (
            lambda *a, **k: order.append("store"))
        self._spy_sessions(monkeypatch, [_make_turn(thread_id="sdk-z-1")])
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert "store" in order and "flush" in order
        assert order.index("flush") < order.index("store")
        # And a fork with persistence disabled must never touch the parent row.
        agent2, db2 = self._db_agent(persisted_sdk_id="sdk-parent-1")
        agent2._persist_disabled = True
        self._spy_sessions(monkeypatch, [_make_turn(thread_id="sdk-fork-9")])
        run_claude_agent_sdk_turn(
            agent2, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        db2.update_claude_sdk_session_id.assert_not_called()

    def test_interrupted_turn_retires_client_but_persists_resume_id(self, monkeypatch):
        # Adversarial-review HIGH: breaking out of receive_response() on
        # interrupt leaves the interrupted turn's ResultMessage queued in the
        # client's stream — a REUSED client would serve it as the NEXT turn's
        # answer. The runtime must retire the client (clean stream) while
        # persisting the SDK id, so the next turn RESUMES the conversation.
        agent, db = self._db_agent()
        self._spy_sessions(monkeypatch, [_make_turn(
            interrupted=True, final_text="partial answer", thread_id="sdk-live-3",
        )])
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert agent._claude_sdk_session is None  # client retired
        db.update_claude_sdk_session_id.assert_called_with("sess-1", "sdk-live-3")
        assert result["partial"] is True

    def test_fresh_retire_does_not_retry(self, monkeypatch):
        # Only a RESUMED session earns the retry — a fresh session that
        # retires is a real error and must surface, never loop.
        agent, _db = self._db_agent(persisted_sdk_id=None)
        instances = self._spy_sessions(monkeypatch, [_make_turn(
            should_retire=True, error="boom", projected_messages=[],
            final_text="", token_usage_last=None,
        )])
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert len(instances) == 1
        assert result["partial"] is True

    def test_interrupted_retire_does_not_retry(self, monkeypatch):
        # A RESUMED session that retires on an INTERRUPTED turn (a /stop
        # that killed the CLI, a hard watchdog trip) must NOT re-run the
        # turn — the retry would evaporate the stop and deliver the answer
        # anyway. Only non-interrupted resume failures earn the retry.
        agent, _db = self._db_agent(persisted_sdk_id="sdk-live-1")
        instances = self._spy_sessions(monkeypatch, [_make_turn(
            should_retire=True, interrupted=True,
            error="SDK message stream ended before this turn's result",
            projected_messages=[], final_text="", token_usage_last=None,
        )])
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert len(instances) == 1  # no second full-budget run
        assert result["partial"] is True

    def test_pre_turn_interrupt_short_circuit_reports_interrupted(self):
        # The top-of-turn short-circuit consumes a pre-turn /stop without a
        # model call. Its result dict must carry interrupted=True — without
        # the key the gateway's empty-response normalizer has NO branch to
        # take (api_calls 0, partial True) and the user's message dies in
        # total silence.
        agent, _db = self._db_agent()
        agent._claude_sdk_session = None
        agent._interrupt_requested = True
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert result["interrupted"] is True
        assert result["api_calls"] == 0
        assert agent._interrupt_requested is False  # consumed

    def test_effective_prompt_snapshot_replaces_native_one(self, monkeypatch):
        # The prologue persists Hermes' native composed prompt — a prompt
        # this runtime never sends. The runtime overwrites the snapshot with
        # the EFFECTIVE prompt so the audit trail tells the truth.
        agent, db = self._db_agent()
        self._spy_sessions(monkeypatch, [_make_turn()])
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        args = db.update_system_prompt.call_args
        assert args is not None
        assert args.args[0] == "sess-1"
        assert args.args[1].startswith("[claude_code preset]")


class TestSessionResumeField:
    def test_resume_rides_options_when_set(self):
        session, holder = _make_session(
            script=[ResultMessage(result="ok")], resume_session_id="sdk-abc"
        )
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert holder["client"].options["resume"] == "sdk-abc"

    def test_no_resume_field_when_unset(self):
        session, holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            session.run_turn("ping")
        finally:
            session.close()
        assert "resume" not in holder["client"].options


# ---------- agent close() releases the SDK session ----------


class TestAgentCloseClosesSdkSession:
    """AIAgent.close() runs on /new, session expiry, and agent-cache
    eviction. Without an explicit disconnect the SDK client (and its CLI
    subprocess) is dropped to GC — a leak. (#25267)"""

    @staticmethod
    def _make_real_agent():
        from run_agent import AIAgent

        return AIAgent(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    def test_close_disconnects_claude_sdk_session(self):
        agent = self._make_real_agent()
        sdk_session = MagicMock()
        agent._claude_sdk_session = sdk_session
        agent.close()
        sdk_session.close.assert_called_once()
        assert agent._claude_sdk_session is None

    def test_close_without_sdk_session_stays_safe(self):
        # Negative control: an agent that never created an SDK session (or
        # already closed it) must close without raising — idempotency.
        agent = self._make_real_agent()
        agent.close()
        agent._claude_sdk_session = None
        agent.close()


# ---------- provider wiring ----------


class TestProviderWiring:
    def test_profile_registered_with_aliases(self):
        from providers import get_provider_profile

        profile = get_provider_profile("claude-agent-sdk")
        assert profile is not None
        assert profile.api_mode == "claude_agent_sdk"
        assert profile.auth_type == "oauth_external"
        assert get_provider_profile("claude-sdk") is profile
        # The anthropic profile keeps its own alias namespace untouched.
        anthropic = get_provider_profile("claude")
        assert anthropic is not None and anthropic.name == "anthropic"

    def test_runtime_resolution_short_circuit(self):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="claude-agent-sdk")
        assert runtime["provider"] == "claude-agent-sdk"
        assert runtime["api_mode"] == "claude_agent_sdk"
        # No credential-pool machinery, no metered key.
        assert runtime["api_key"] == "claude-subscription-oauth"

    def test_api_mode_accepted_by_agent_init(self):
        from hermes_cli.runtime_provider import _parse_api_mode

        assert _parse_api_mode("claude_agent_sdk") == "claude_agent_sdk"


class TestSystemPromptAppend:
    # W2 (composer parity): the append is composed from Hermes' NATIVE
    # builders — memory gauge via MemoryStore.format_for_system_prompt,
    # guidance constants from agent.prompt_builder, the skills index via
    # build_skills_system_prompt — never re-implemented formats. Guidance
    # appears ONLY for tools that are actually callable through the MCP
    # shims. Deliberate pin updates from W1 are annotated inline.

    @staticmethod
    def _home(tmp_path, monkeypatch, *, soul=None, memory=None, user=None, budget=None):
        hermes_home = tmp_path / "hermes"
        memories = hermes_home / "memories"
        memories.mkdir(parents=True)
        if memory is not None:
            (memories / "MEMORY.md").write_text(memory)
        if user is not None:
            (memories / "USER.md").write_text(user)
        import hermes_cli.config as cfg

        append_file = ""
        if soul is not None:
            soul_file = tmp_path / "SOUL.md"
            soul_file.write_text(soul)
            append_file = str(soul_file)
        # config.yaml is the only interface for the persona file
        # (agent.claude_agent_sdk.append_file); the old env var is gone.
        # Patching unconditionally also isolates the suite from a developer's
        # real config.yaml, which would otherwise leak a live append_file in.
        sdk_cfg = {"append_file": append_file}
        # Only inject the budget key when a test asks for one: absent must stay
        # distinguishable from present-but-invalid, which take different paths.
        if budget is not None:
            sdk_cfg["append_total_max_chars"] = budget
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": sdk_cfg}},
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        return hermes_home

    def test_soul_first_and_user_content_present(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(
            tmp_path, monkeypatch,
            soul="# I am the persona under test",
            user="The user prefers concise results",
        )
        out = build_system_prompt_append()
        assert out is not None
        assert out.startswith("# I am the persona under test")
        assert "The user prefers concise results" in out

    def test_native_soul_md_autoloads_when_append_file_unset(
        self, tmp_path, monkeypatch
    ):
        # R2 (#65982, romain-bury): the native composer treats
        # $HERMES_HOME/SOUL.md as identity slot #1; W2 composer parity means
        # this path must load it too when no explicit append_file overrides.
        from agent.claude_sdk_runtime import build_system_prompt_append

        home = self._home(tmp_path, monkeypatch)
        (home / "SOUL.md").write_text("# Native soul identity")
        out = build_system_prompt_append()
        assert out is not None
        assert out.startswith("# Native soul identity")

    def test_append_file_wins_over_native_soul_md(self, tmp_path, monkeypatch):
        # append_file stays the explicit operator override.
        from agent.claude_sdk_runtime import build_system_prompt_append

        home = self._home(
            tmp_path, monkeypatch, soul="# Override persona"
        )
        (home / "SOUL.md").write_text("# Native soul identity")
        out = build_system_prompt_append()
        assert out is not None
        assert out.startswith("# Override persona")
        assert "# Native soul identity" not in out

    def test_workspace_context_file_is_in_sdk_append(self, tmp_path, monkeypatch):
        """SDK turns must receive the same Hermes project instructions as native turns."""
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(tmp_path, monkeypatch)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".hermes.md").write_text(
            "# Workspace contract\nRun the project check before reporting success."
        )

        out = build_system_prompt_append(cwd=str(workspace)) or ""

        assert "# Project Context" in out
        assert "Run the project check before reporting success." in out
        # Context-file discovery uses a process-local warning queue. Drain it
        # so this direct builder test cannot leak state into native prompt tests.
        from agent.prompt_builder import drain_truncation_warnings

        drain_truncation_warnings()

    def test_coding_workspace_snapshot_is_in_sdk_append(self, tmp_path, monkeypatch):
        """SDK turns retain the workspace and operator tail, not false tool guidance."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.coding_context as coding_context

        self._home(tmp_path, monkeypatch)
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: (["Coding posture"], ["Workspace snapshot"], ["Coding tail"]),
        )

        out = build_system_prompt_append(cwd=str(tmp_path), platform="telegram") or ""

        # The native prefix names patch/write_file/terminal/todo, which are not
        # present on the default SDK surface. The claude_code preset plus the
        # SDK-specific inspection guidance own behavior; only the workspace
        # snapshot and configured operator tail are portable here.
        assert "Coding posture" not in out
        assert "Workspace snapshot" in out
        assert "Coding tail" in out
        from agent.prompt_builder import drain_truncation_warnings

        drain_truncation_warnings()

    def test_project_context_preserves_native_fallback_policy(self, tmp_path, monkeypatch):
        """A fallback cwd stays None so prompt_builder can guard install trees."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.coding_context as coding_context
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch)
        captured = {}

        def fake_project_context(**kwargs):
            captured.update(kwargs)
            return ""

        monkeypatch.setattr(prompt_builder, "build_context_files_prompt", fake_project_context)
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: ([], [], []),
        )

        build_system_prompt_append(cwd=None, platform="telegram")

        assert captured["cwd"] is None
        assert captured["skip_soul"] is True
        assert captured["allow_install_tree_fallback"] is False

    def test_skip_project_context_keeps_coding_snapshot(self, tmp_path, monkeypatch):
        """skip_context_files does not disable the independent coding snapshot."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.coding_context as coding_context
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch)

        def unexpected_project_context(**_kwargs):
            raise AssertionError("project context must stay disabled")

        monkeypatch.setattr(
            prompt_builder,
            "build_context_files_prompt",
            unexpected_project_context,
        )
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: ([], ["WORKSPACE-WITH-PROJECT-FILES-DISABLED"], []),
        )

        out = build_system_prompt_append(
            cwd=str(tmp_path),
            include_project_context=False,
        ) or ""

        assert "WORKSPACE-WITH-PROJECT-FILES-DISABLED" in out

    def test_workspace_snapshot_survives_large_project_context(self, tmp_path, monkeypatch):
        """A large project file must not silently evict the SDK workspace snapshot."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.coding_context as coding_context
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch)
        monkeypatch.setattr(
            prompt_builder,
            "build_context_files_prompt",
            lambda **_kwargs: "PROJECT-CONTEXT " + "p" * 16_000,
        )
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: ([], ["WORKSPACE-SNAPSHOT " + "w" * 3_500], []),
        )

        out = build_system_prompt_append(cwd=str(tmp_path)) or ""

        assert "PROJECT-CONTEXT" in out
        assert "WORKSPACE-SNAPSHOT" in out

    def test_oversized_workspace_snapshot_is_capped_not_dropped(self, tmp_path, monkeypatch):
        """An oversized coding snapshot remains represented within the workspace cap."""
        from agent.claude_sdk_runtime import (
            _APPEND_TOTAL_MAX_CHARS,
            build_system_prompt_append,
        )
        import agent.coding_context as coding_context
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch)
        monkeypatch.setattr(prompt_builder, "build_context_files_prompt", lambda **_kwargs: "")
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: ([], ["OVERSIZED-WORKSPACE " + "w" * 22_000], []),
        )

        out = build_system_prompt_append(cwd=str(tmp_path)) or ""

        assert "OVERSIZED-WORKSPACE" in out
        assert len(out) <= _APPEND_TOTAL_MAX_CHARS

    def test_workspace_snapshot_survives_capped_soul(self, tmp_path, monkeypatch):
        """The workspace block must fit after the maximum-size SDK soul block."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.coding_context as coding_context
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch, soul="CAPPED-SOUL " + "s" * 8_000)
        monkeypatch.setattr(prompt_builder, "build_context_files_prompt", lambda **_kwargs: "")
        monkeypatch.setattr(
            coding_context,
            "coding_system_prompt_parts",
            lambda **_kwargs: ([], ["SOUL-COMPATIBLE-WORKSPACE " + "w" * 12_000], []),
        )

        out = build_system_prompt_append(cwd=str(tmp_path)) or ""

        assert "CAPPED-SOUL" in out
        assert "SOUL-COMPATIBLE-WORKSPACE" in out

    def test_project_context_warning_queue_drains_after_builder_error(self, tmp_path, monkeypatch):
        """A failed SDK context build cannot leak warnings into later native prompts."""
        from agent.claude_sdk_runtime import build_system_prompt_append
        import agent.prompt_builder as prompt_builder

        self._home(tmp_path, monkeypatch)

        def record_then_fail(**_kwargs):
            prompt_builder._record_truncation_warning("SDK test warning")
            raise RuntimeError("transient context read failure")

        monkeypatch.setattr(prompt_builder, "build_context_files_prompt", record_then_fail)
        build_system_prompt_append(cwd=str(tmp_path))

        assert prompt_builder.drain_truncation_warnings() == []

    def test_gauge_blocks_are_the_native_render(self, tmp_path, monkeypatch):
        # Byte-pin: the memory/user blocks are EXACTLY what the native
        # composer injects (MemoryStore.format_for_system_prompt output,
        # gauge header included) — never a re-implementation.
        from agent.claude_sdk_runtime import build_system_prompt_append
        from tools.memory_tool import load_on_disk_store

        self._home(
            tmp_path, monkeypatch,
            memory="ci runs on the drone server",
            user="prefers squash merges",
        )
        store = load_on_disk_store()
        expected_memory = store.format_for_system_prompt("memory")
        expected_user = store.format_for_system_prompt("user")
        assert "MEMORY (your personal notes) [" in expected_memory  # sanity
        assert "USER PROFILE (who the user is) [" in expected_user

        out = build_system_prompt_append()
        assert expected_memory in out
        assert expected_user in out

    def test_mcp_inspection_preference_is_in_effective_sdk_prompt(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(tmp_path, monkeypatch)
        out = build_system_prompt_append() or ""
        assert "For multi-step tool work, provide a brief user-facing status" in out
        assert "never reveal private reasoning" in out
        assert "prefer the Hermes MCP `read_file` and `search_files` tools before Bash" in out
        assert "database client, process/service state, network operation" in out
        assert "Bash remains subject to normal approval" in out

    def test_memory_guidance_present_skill_sentence_stripped(self, tmp_path, monkeypatch):
        # MEMORY_GUIDANCE ships verbatim EXCEPT its one sentence instructing
        # the skill tool (skill_manage is not exposed — checklist #3:
        # guidance only for callable tools). The strip must be a pure
        # deletion of a sentence that actually exists in the native constant
        # — if upstream rewords it, this test goes red and we re-derive.
        from agent.claude_sdk_runtime import (
            _strip_uncallable_tool_guidance,
            build_system_prompt_append,
        )
        from agent.prompt_builder import MEMORY_GUIDANCE

        self._home(tmp_path, monkeypatch, memory="uses trunk-based development")
        stripped = _strip_uncallable_tool_guidance(MEMORY_GUIDANCE)
        assert stripped != MEMORY_GUIDANCE, "skill sentence not found — upstream reworded it"
        assert "save it as a skill with the skill tool" not in stripped

        out = build_system_prompt_append()
        assert "You have persistent memory across sessions" in out
        assert stripped in out
        assert "save it as a skill with the skill tool" not in out
        # Disambiguation addendum (caught live): the claude_code preset has
        # its own file-based memory convention; the append must pin the
        # hermes-tools memory tool as the ONLY durable store.
        assert "ONLY durable memory" in out
        assert "hermes-tools MCP server" in out
        # Reworded after the adversarial review PROVED the preset's memory
        # dir DOES persist per-cwd: the addendum must state true facts
        # (unmanaged/disposable), never the false "will not be injected".
        assert "disposable" in out
        assert "will not be injected" not in out

    def test_skills_guidance_never_injected(self, tmp_path, monkeypatch):
        # SKILLS_GUIDANCE instructs skill_manage — unexposed by design.
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(tmp_path, monkeypatch, memory="a fact")
        out = build_system_prompt_append()
        assert "skill_manage" not in out

    def test_session_search_guidance_always_present(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append
        from agent.prompt_builder import SESSION_SEARCH_GUIDANCE

        self._home(tmp_path, monkeypatch)  # no memory files at all
        out = build_system_prompt_append()
        assert out is not None
        assert SESSION_SEARCH_GUIDANCE in out
        # Query-style addendum (observed live: ANDy multi-term queries miss).
        assert "ALL terms must match" in out

    def test_memory_disabled_removes_blocks_and_guidance(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append
        import hermes_cli.config as cfg

        self._home(tmp_path, monkeypatch, memory="should not appear")
        monkeypatch.setattr(
            cfg, "load_config", lambda *a, **k: {"memory": {"memory_enabled": False}}
        )
        out = build_system_prompt_append()
        assert "should not appear" not in (out or "")
        assert "You have persistent memory" not in (out or "")
        # session_search still works when memory is off — its guidance stays.
        assert "session_search" in (out or "")

    def test_external_memory_provider_removes_tool_guidance(self, tmp_path, monkeypatch):
        # memory.provider: honcho (or ANY external backend) leaves the memory
        # shim UNREGISTERED (hermes_tools_mcp_server._stateless_shim_defs
        # requires enabled AND no external provider), so the append must not
        # instruct or advertise an absent tool. The on-disk store block stays:
        # external providers run alongside the builtin store, and its facts
        # remain readable. Proven red-first against the enabled-only gate.
        import agent.prompt_builder as pb
        import hermes_cli.config as cfg
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(tmp_path, monkeypatch, memory="a durable fact")
        monkeypatch.setattr(
            cfg,
            "load_config",
            lambda *a, **k: {
                "memory": {"memory_enabled": True, "provider": "honcho"}
            },
        )
        captured = {}

        def fake_index(**kwargs):
            captured.update(kwargs)
            return ""

        monkeypatch.setattr(pb, "build_skills_system_prompt", fake_index)
        out = build_system_prompt_append() or ""
        assert "You have persistent memory" not in out
        assert "ONLY durable memory" not in out
        # The store block itself survives — facts stay readable.
        assert "a durable fact" in out
        # session_search is unaffected.
        assert "session_search" in out
        # And the skills filter is not told the tool exists.
        tools = captured.get("available_tools") or set()
        assert "memory" not in tools
        assert "session_search" in tools

    def test_session_line_and_platform_hint(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append
        from agent.prompt_builder import PLATFORM_HINTS

        self._home(tmp_path, monkeypatch)
        out = build_system_prompt_append(
            platform="telegram", session_id="sess-77", model="claude-opus-4-8"
        )
        assert "Conversation started:" in out  # date-only, native format
        assert "Session ID: sess-77" in out
        assert "Model: claude-opus-4-8" in out
        assert "Provider: claude-agent-sdk" in out
        assert PLATFORM_HINTS["telegram"].strip() in out

    def test_unknown_platform_no_hint_and_none_safe(self, tmp_path, monkeypatch):
        from agent.claude_sdk_runtime import build_system_prompt_append

        self._home(tmp_path, monkeypatch)
        out = build_system_prompt_append(platform="faxmachine")
        assert out is not None  # None-safe, no crash, no bogus hint

    def test_budget_skips_oversized_block_keeps_later_blocks(self, tmp_path, monkeypatch):
        # Whole-block budget policy: a block that does not fit is SKIPPED
        # entirely (never truncated mid-block) and later, smaller blocks
        # still make it in. An oversized hand-edited MEMORY.md must not
        # evict the guidance. (Deliberate pin update from W1's 8000-char
        # raw-file cap: the store renders whole blocks; the budget governs.)
        from agent.claude_sdk_runtime import (
            _APPEND_TOTAL_MAX_CHARS,
            build_system_prompt_append,
        )

        self._home(tmp_path, monkeypatch, memory="y" * (_APPEND_TOTAL_MAX_CHARS + 5000))
        out = build_system_prompt_append()
        assert "yyyyyyyyyy" not in out  # oversized memory block skipped whole
        assert "session_search" in out  # later block survived
        assert len(out) <= _APPEND_TOTAL_MAX_CHARS

    def test_budget_eviction_warns_and_names_the_block(
        self, tmp_path, monkeypatch, caplog
    ):
        # An eviction deletes standing instructions the operator believes are
        # in force, so it must be audible. This was DEBUG — which is how the
        # identity fix could have silently traded SOUL.md in for the
        # MCP-inspection block with nothing in the log to say so.
        import logging

        from agent.claude_sdk_runtime import (
            _APPEND_TOTAL_MAX_CHARS,
            build_system_prompt_append,
        )

        oversized = "# Hand-edited memory\n" + "y" * (_APPEND_TOTAL_MAX_CHARS + 5000)
        self._home(tmp_path, monkeypatch, memory=oversized)
        with caplog.at_level(logging.WARNING):
            out = build_system_prompt_append()

        assert "yyyyyyyyyy" not in out
        evictions = [
            r for r in caplog.records
            if "append budget" in r.getMessage() and r.levelno >= logging.WARNING
        ]
        assert evictions, "an evicted block must be reported above DEBUG"
        message = evictions[0].getMessage()
        # Named by the store's own header, so the loss is actionable...
        assert "MEMORY" in message
        # ...but header ONLY: the body must never reach the log.
        assert "yyyyyyyyyy" not in message

    def test_budget_override_seats_a_block_the_default_evicts(
        self, tmp_path, monkeypatch
    ):
        # The ceiling is a per-box cost decision, so it must be reachable from
        # config without a code edit.
        from agent.claude_sdk_runtime import (
            _APPEND_TOTAL_MAX_CHARS,
            build_system_prompt_append,
        )

        # Separate homes: _home() is not re-entrant on one tmp_path, and the
        # two halves must differ ONLY in the configured ceiling.
        big = "# Big memory\n" + "y" * (_APPEND_TOTAL_MAX_CHARS + 5000)
        self._home(tmp_path / "default", monkeypatch, memory=big)
        assert "yyyyyyyyyy" not in build_system_prompt_append()  # evicted at default

        self._home(
            tmp_path / "raised",
            monkeypatch,
            memory=big,
            budget=_APPEND_TOTAL_MAX_CHARS * 3,
        )
        assert "yyyyyyyyyy" in build_system_prompt_append()  # seated when raised

    @pytest.mark.parametrize(
        "bad", ["not-a-number", 0, -5, True, 1.5, float("inf")]
    )
    def test_invalid_budget_override_falls_back_to_default_with_warning(
        self, tmp_path, monkeypatch, caplog, bad
    ):
        # A typo'd or zero ceiling must never become a silent behaviour
        # change: 0 would strip the entire append, identity included.
        import logging

        from agent.claude_sdk_runtime import (
            _APPEND_TOTAL_MAX_CHARS,
            _append_total_max_chars,
        )

        self._home(tmp_path, monkeypatch, budget=bad)
        with caplog.at_level(logging.WARNING):
            assert _append_total_max_chars() == _APPEND_TOTAL_MAX_CHARS
        assert any(
            "append_total_max_chars" in r.getMessage() for r in caplog.records
        )

    def test_restoring_identity_does_not_evict_a_previously_seated_block(
        self, tmp_path, monkeypatch
    ):
        # The regression the independent review caught, stated as the
        # relationship it actually is: seating SOUL.md must not be paid for by
        # silently dropping a block that fit without it. Sizes are calibrated
        # (memory 8400) so the identity block genuinely pushes the total past
        # the historical 20000 ceiling — at 20000 this test is RED, which is
        # the whole point of the raised default.
        from agent.claude_sdk_runtime import (
            _MCP_INSPECTION_PREFERENCE,
            build_system_prompt_append,
        )

        mcp = _MCP_INSPECTION_PREFERENCE.strip()
        common = dict(user="u" * 4921, memory="m" * 8400)

        # Baseline: no append_file, no hand-written identity file (native
        # load_soul_md() scaffolds a short default), historical ceiling —
        # MCP block fits.
        self._home(tmp_path / "no-soul", monkeypatch, budget=20000, **common)
        assert mcp in build_system_prompt_append()

        # Same load, at the shipped default, with a real identity file
        # written straight into $HERMES_HOME (append_file stays unset, so
        # this exercises the native load_soul_md() path): identity seated
        # AND the MCP block survives. (At budget=20000 the MCP block is
        # evicted — calibrated and verified 2026-08-18 on Main.)
        home = self._home(tmp_path / "with-soul", monkeypatch, **common)
        (home / "SOUL.md").write_text("I am the agent.\n" + "s" * 3200)
        out = build_system_prompt_append()

        assert "I am the agent." in out  # identity seated
        assert "m" * 100 in out and "u" * 100 in out  # memory + profile seated
        assert mcp in out  # ...and NOT paid for with this one

    def test_eviction_warning_never_derives_label_from_block_content(
        self, tmp_path, monkeypatch, caplog
    ):
        import logging

        import agent.prompt_builder as pb
        from agent.claude_sdk_runtime import build_system_prompt_append

        secret = "private skill detail that must never reach logs"
        self._home(tmp_path, monkeypatch, budget=1000)
        monkeypatch.setattr(
            pb,
            "build_skills_system_prompt",
            lambda **_kwargs: secret + "x" * 5000,
        )

        with caplog.at_level(logging.WARNING):
            build_system_prompt_append()

        messages = [record.getMessage() for record in caplog.records]
        assert any("skills index" in message for message in messages)
        assert all(secret not in message for message in messages)

    def test_skills_index_wiring(self, tmp_path, monkeypatch):
        # The index rides the NATIVE builder; we pin OUR wiring — called
        # with the honest MCP-exposed tool set (shims included).
        import agent.prompt_builder as pb
        from agent.claude_sdk_runtime import build_system_prompt_append
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

        self._home(tmp_path, monkeypatch)
        captured = {}

        def fake_index(**kwargs):
            captured.update(kwargs)
            # Includes the index's real unconditional boilerplate sentence —
            # caught LIVE on the deployed box: the native index instructs
            # skill_manage regardless of available_tools, and the strip must
            # remove it (a tmp home's empty index made the old pin vacuous).
            return (
                "## Skills (mandatory)\n"
                "If a skill has issues, fix it with skill_manage(action='patch').\n"
                "- fixture-skill: proves the wiring"
            )

        monkeypatch.setattr(pb, "build_skills_system_prompt", fake_index)
        out = build_system_prompt_append()
        assert "fixture-skill: proves the wiring" in out
        assert "skill_manage" not in out
        tools = captured.get("available_tools") or set()
        assert "memory" in tools and "session_search" in tools
        assert {"read_file", "search_files"} <= tools
        assert not tools & {"terminal", "shell", "write_file", "patch", "process"}
        assert set(EXPOSED_TOOLS) <= tools

    def test_root_files_are_not_read(self, tmp_path, monkeypatch):
        # Negative control (W1): ONE canonical location. Files left at the
        # HERMES_HOME root must NOT be injected.
        from agent.claude_sdk_runtime import build_system_prompt_append

        hermes_home = tmp_path / "hermes"
        (hermes_home / "memories").mkdir(parents=True)
        (hermes_home / "USER.md").write_text("stale root copy")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        assert "stale root copy" not in (build_system_prompt_append() or "")

    def test_memory_shim_write_is_visible_to_next_append(self, tmp_path, monkeypatch):
        # The loop closes: a fact saved through the stateless MCP shim must
        # appear in the next session's system-prompt append.
        from agent.claude_sdk_runtime import build_system_prompt_append
        from agent.transports.hermes_tools_mcp_server import dispatch_memory

        self._home(tmp_path, monkeypatch)
        dispatch_memory(
            {"action": "add", "target": "memory", "content": "the beta build ships friday"}
        )
        out = build_system_prompt_append()
        assert out is not None
        assert "the beta build ships friday" in out

    def test_empty_home_still_provides_guidance(self, tmp_path, monkeypatch):
        # Deliberate pin update (was: no sources → None). Since W2 the
        # append always carries the recall/memory behavior contract — a
        # brand-new box still gets guidance, so the brain knows its tools.
        from agent.claude_sdk_runtime import build_system_prompt_append

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # empty dir
        out = build_system_prompt_append()
        assert out is not None
        assert "session_search" in out


class TestAuxLaneSubscriptionRouting:
    def test_aux_auto_detect_uses_same_sdk_subscription_lane(self, monkeypatch):
        # Auto auxiliary work must not fall through to a metered provider, but
        # it can safely use the same subscription-owned Agent SDK one-shot path.
        from agent.auxiliary_client import _resolve_auto
        from agent.claude_sdk_aux_client import ClaudeSdkAuxClient

        monkeypatch.setenv("OPENROUTER_API_KEY", "«redacted:sk-…»")
        client, model = _resolve_auto(main_runtime={
            "provider": "claude-agent-sdk",
            "model": "claude-opus-4-8",
            "api_mode": "claude_agent_sdk",
            "base_url": "",
            "api_key": "claude-subscription-oauth",
        })
        assert isinstance(client, ClaudeSdkAuxClient)
        assert model == "claude-opus-4-8"


class TestSdkAvailabilityGate:
    def test_check_routes_through_lazy_install_lane(self, monkeypatch):
        # F1 (deps): the SDK is an opt-in extra excluded from [all], so the
        # availability gate must offer the lazy-install lane first — the
        # exact pattern anthropic_adapter._get_anthropic_sdk uses for
        # provider.anthropic. A lean install otherwise dead-ends on
        # ImportError with no self-serve path.
        import tools.lazy_deps as lazy_deps
        from agent.transports.claude_agent_sdk_session import (
            check_claude_sdk_available,
        )

        assert "provider.claude_agent_sdk" in lazy_deps.LAZY_DEPS
        called = {}

        def fake_ensure(feature, *, prompt=True):
            called["feature"] = feature
            called["prompt"] = prompt

        monkeypatch.setattr(lazy_deps, "ensure", fake_ensure)
        # Pin the LEAN install this lane exists for: a None entry in
        # sys.modules makes `import claude_agent_sdk` raise ImportError.
        import sys as _sys

        monkeypatch.setitem(_sys.modules, "claude_agent_sdk", None)
        check_claude_sdk_available()
        assert called == {"feature": "provider.claude_agent_sdk", "prompt": False}

    def test_check_skips_lazy_lane_when_sdk_already_imports(self, monkeypatch):
        # ensure() can shell out to `uv pip install` and calls
        # importlib.invalidate_caches(). Running it immediately before
        # `import claude_agent_sdk -> mcp -> anyio` rewrites site-packages and
        # drops import caches under a live interpreter, intermittently
        # corrupting that very import ("KeyError: 'anyio'" out of
        # importlib._bootstrap._find_and_load). When the extra is ALREADY
        # importable the installer must not run at all.
        import sys as _sys
        import types as _types

        import tools.lazy_deps as lazy_deps
        from agent.transports.claude_agent_sdk_session import (
            check_claude_sdk_available,
        )

        called = {}

        def fake_ensure(feature, *, prompt=True):
            called["feature"] = feature

        monkeypatch.setattr(lazy_deps, "ensure", fake_ensure)
        monkeypatch.setitem(
            _sys.modules, "claude_agent_sdk", _types.ModuleType("claude_agent_sdk")
        )
        assert check_claude_sdk_available() == (True, "ok")
        assert called == {}

    def test_lazy_lane_pin_matches_pyproject_extra(self):
        # The LAZY_DEPS lane must mirror the pyproject extra in lockstep
        # (same contract test_pyproject_and_lazy_deps_pins_agree enforces
        # globally; pinned here so the SDK lane keeps a single exact spec).
        from tools.lazy_deps import LAZY_DEPS

        specs = LAZY_DEPS["provider.claude_agent_sdk"]
        assert len(specs) == 1
        assert specs[0].startswith("claude-agent-sdk==")

    def test_check_reports_missing_sdk(self, monkeypatch):
        # RED-first negative control: with the import broken, the gate must
        # fail with the install hint — never silently pass. The lazy lane is
        # stubbed to FeatureUnavailable (lazy installs disabled / offline) so
        # the test never triggers a real multi-MB SDK download on CI.
        import builtins

        import tools.lazy_deps as lazy_deps

        def _unavailable(feature, *, prompt=True):
            raise lazy_deps.FeatureUnavailable(feature, (), "disabled in test")

        monkeypatch.setattr(lazy_deps, "ensure", _unavailable)

        real_import = builtins.__import__

        def _broken(name, *args, **kwargs):
            if name == "claude_agent_sdk":
                raise ImportError("No module named 'claude_agent_sdk'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _broken)
        from agent.transports.claude_agent_sdk_session import (
            check_claude_sdk_available,
        )

        ok, msg = check_claude_sdk_available()
        assert ok is False
        assert "hermes-agent[claude-agent-sdk]" in msg


# ---------- fatal-reason plumbing: refusals must be machine-readable ----------
# Clean-checkout E2E finding on #65982 (jefftropeano): a fatal metered-billing
# refusal exited 0 because the runtime never sets "failed"/"failure_reason" —
# the fields the chat_completions path sets (conversation_loop) and the -Q
# exit path keys on. TurnResult.fatal_reason carries the classification out
# of run_turn (the refusal exception never propagates past it).


class TestFatalReason:
    def test_metered_refusal_sets_fatal_reason_startup(self, monkeypatch):
        # "startup", deliberately NOT "billing": the kanban -Q exit contract
        # maps failure_reason "billing" to the transient EX_TEMPFAIL requeue
        # sentinel, and a present metered key is a config error retries can't
        # fix — it must count as a real failure everywhere.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
        session = ClaudeAgentSdkSession(cwd="/tmp")  # no factory → real path
        turn = session.run_turn("hi")
        assert turn.should_retire
        assert turn.fatal_reason == "startup"

    def test_auth_classified_startup_failure_sets_fatal_reason_auth(self):
        session, _ = _make_session(
            connect_exc=RuntimeError("401 unauthorized: invalid bearer token")
        )
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.should_retire
        assert turn.fatal_reason == "auth"

    def test_startup_traceback_keeps_frames_but_redacts_exception(self, caplog):
        secret = "sk-ant-api03-SUPERSECRET"
        session, _ = _make_session(
            connect_exc=RuntimeError(f"connect failed with {secret}")
        )
        with caplog.at_level(
            logging.WARNING,
            logger="agent.transports.claude_agent_sdk_session",
        ):
            try:
                turn = session.run_turn("hi")
            finally:
                session.close()

        assert turn.should_retire
        assert secret not in caplog.text
        assert "claude-agent-sdk startup failed" in caplog.text
        assert "Traceback (most recent call last)" in caplog.text
        assert "connect" in caplog.text

    def test_sdk_error_result_is_not_fatal(self):
        # An in-turn SDK error (e.g. error_max_turns) is turn-scoped, not a
        # startup/auth/billing refusal — it must stay non-fatal so one bad
        # turn can't flip an integration's exit code.
        script = [ResultMessage(subtype="error_max_turns", is_error=False)]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert turn.fatal_reason is None

    def test_runtime_glue_maps_fatal_reason_to_failed(self):
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            should_retire=True,
            error="claude-agent-sdk startup failed: refused",
            fatal_reason="startup",
            projected_messages=[],
            final_text="",
            token_usage_last=None,
        )
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        assert result["failed"] is True
        assert result["failure_reason"] == "startup"
        assert result["partial"] is True

    def test_runtime_glue_transient_error_stays_unfailed(self):
        # A retire without fatal_reason (timeout, transient turn error) keeps
        # today's contract: partial, no "failed" key — gateway/CLI treat it
        # as a recoverable turn, not a dead run.
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            should_retire=True,
            error="turn timed out after 600s",
            projected_messages=[],
            final_text="",
            token_usage_last=None,
        )
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        assert not result.get("failed")
        assert "failure_reason" not in result


# ---------- SDK permission-result stand-ins (planted as the module) ----------
# _make_can_use_tool lazy-imports PermissionResultAllow/Deny from
# claude_agent_sdk at CALL time — the only import of the real SDK package any
# test in this file can reach. Upstream CI installs no claude-agent-sdk
# extra, so tests that INVOKE the callback must plant a stand-in module
# first (the header's contract: stand-in classes named like the SDK's
# types). Planted unconditionally: the tests exercise identical code
# whether or not the real SDK is installed.


class PermissionResultAllow:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class PermissionResultDeny:
    def __init__(self, message: str = "", **kwargs: Any) -> None:
        self.message = message
        self.__dict__.update(kwargs)


def _plant_claude_agent_sdk_stand_in(monkeypatch) -> None:
    module = ModuleType("claude_agent_sdk")
    module.PermissionResultAllow = PermissionResultAllow
    module.PermissionResultDeny = PermissionResultDeny
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)


# ---------- gateway approval bridge: SDK permission prompts reach the chat ----------
# Production finding (dasbrow 24/7 box): under the gateway, _create_session's
# thread-local CLI callback is always None, so mode=default wired
# can_use_tool=None and the SDK denied every un-allowlisted tool silently —
# no Telegram prompt ever reached the operator, though the gateway registers
# a notify channel around every turn. The bridge routes SDK permission
# requests onto that same tools.approval queue.


class TestSdkBoundedMcpInspectionPermissions:
    """Only fixed Hermes MCP file inspection tools bypass SDK prompting."""

    @pytest.fixture(autouse=True)
    def _sdk_permission_results(self, monkeypatch):
        _plant_claude_agent_sdk_stand_in(monkeypatch)

    @pytest.mark.parametrize("tool_name", [
        "mcp__hermes-tools__read_file",
        "mcp__hermes-tools__search_files",
    ])
    def test_bounded_inspection_mcp_tools_are_auto_allowed(self, tool_name):
        calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(tool_name, {}, None))
        assert type(result).__name__ == "PermissionResultAllow"
        assert calls == []

    @pytest.mark.parametrize("tool_name", [
        "mcp__hermes-tools__write_file",
        "mcp__hermes-tools__read_file_evil",
        "mcp__other-server__read_file",
        "Bash",
    ])
    def test_non_bounded_tools_still_use_approval_bridge(self, tool_name):
        calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(tool_name, {"command": "ls"}, None))
        assert type(result).__name__ == "PermissionResultAllow"
        assert len(calls) == 1


class TestGatewayApprovalBridge:
    @pytest.fixture(autouse=True)
    def _sdk_permission_results(self, monkeypatch):
        _plant_claude_agent_sdk_stand_in(monkeypatch)

    def _gateway_ctx(self, monkeypatch, session_key):
        from tools import approval as approval_mod

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        token = approval_mod.set_current_session_key(session_key)
        return approval_mod, token

    def test_builder_returns_none_outside_gateway_context(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        from tools.approval import build_sdk_gateway_approval_callback

        assert build_sdk_gateway_approval_callback() is None

    def test_builder_returns_none_for_cron_sessions(self, monkeypatch):
        # UPDATED (W9): this test used to pin builder→None for cron contexts
        # — which FROZE a session first created during a cron turn into
        # callback=None forever (silent deny in every later interactive
        # turn, the sticky-session incident defect). The builder now wires a
        # callback for any gateway-shaped surface and resolves cron-ness per
        # CALL. Cron posture is preserved: settings allow-rules suppress
        # prompts before can_use_tool is consulted, and a would-be prompt
        # during a cron turn denies IMMEDIATELY with an honest reason —
        # never blocks, never pages, never enqueues.
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        from tools import approval as approval_mod

        cb = approval_mod.build_sdk_gateway_approval_callback()
        assert cb is not None  # gateway-shaped: no more cron-born freeze
        result = cb("Bash(ls)", "Claude requests tool Bash")
        assert result == {
            "choice": "deny",
            "reason": "no approver available (background context)",
        }
        assert "denied by user" not in result["reason"]
        # Never blocked, never enqueued: no pending approval anywhere.
        with approval_mod._lock:
            assert not approval_mod._gateway_queues

    def test_gateway_context_wires_can_use_tool(self, monkeypatch):
        approval_mod, token = self._gateway_ctx(monkeypatch, "tg:152:main")
        try:
            cb = approval_mod.build_sdk_gateway_approval_callback()
            assert cb is not None
            session, _ = _make_session(
                approval_callback=cb, permission_mode="default"
            )
            assert session.build_option_fields()["can_use_tool"] is not None
        finally:
            approval_mod.reset_current_session_key(token)

    def test_no_registered_notify_denies(self, monkeypatch, caplog):
        # UPDATED (W8): this test used to pin the bare-"deny" return — which
        # the SDK layer translated to "denied by user" for a prompt no user
        # ever saw, with no log line (the 2026-08-06 incident's approval
        # face). The no-approver deny is now structured with an honest
        # reason and logged at WARNING.
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-no-notify")
        try:
            cb = approval_mod.build_sdk_gateway_approval_callback()
            with caplog.at_level(logging.WARNING, logger="tools.approval"):
                result = cb("Bash(ls)", "Claude requests tool Bash")
            assert result == {
                "choice": "deny",
                "reason": "no approver available (background context)",
            }
            assert any(
                "NO approver available" in r.getMessage()
                and "Bash(ls)" in r.getMessage()
                and "sess-no-notify" in r.getMessage()
                for r in caplog.records
            ), "the silent deny must not stay silent"
        finally:
            approval_mod.reset_current_session_key(token)

    def test_background_turn_pages_operator_via_session_scoped_approver(
        self, monkeypatch,
    ):
        # The incident lane: deliver_background_results expects CLI-initiated
        # turns BETWEEN hermes turns — exactly when the turn-scoped
        # registration is gone. The session-scoped entry (refreshed by every
        # gateway turn, surviving its teardown) keeps a paging path alive.
        sk = "sess-bg-approver"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            notify, seen = self._resolve_with(approval_mod, sk, "once")
            # The gateway turn registers both; then the turn ends.
            approval_mod.register_gateway_notify(sk, notify)
            approval_mod.register_session_notify(sk, notify)
            approval_mod.unregister_gateway_notify(sk)
            try:
                cb = approval_mod.build_sdk_gateway_approval_callback()
                assert cb("Bash(ls)", "Claude requests tool Bash") == "once"
                assert len(seen) == 1  # the operator WAS paged
                assert seen[0]["command"] == "Bash(ls)"
            finally:
                approval_mod.unregister_session_notify(sk)
        finally:
            approval_mod.reset_current_session_key(token)

    def test_no_approver_deny_is_not_attributed_to_user(
        self, monkeypatch, caplog,
    ):
        # End to end across the widened channel: bridge (no approver) →
        # _make_can_use_tool → PermissionResultDeny carrying the honest
        # reason. "denied by user" is reserved for the plain user-deny path.
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-bg-honest")
        try:
            cb = approval_mod.build_sdk_gateway_approval_callback()
            session, _ = _make_session(
                approval_callback=cb, permission_mode="default"
            )
            fn = session._make_can_use_tool()
            with caplog.at_level(logging.WARNING, logger="tools.approval"):
                res = asyncio.run(fn("Bash", {"command": "ls"}, None))
            assert type(res).__name__ == "PermissionResultDeny"
            assert res.message == "no approver available (background context)"
            assert "denied by user" not in res.message
            assert any(
                "NO approver available" in r.getMessage()
                for r in caplog.records
            )

            # Back-compat: plain string returns keep their classic mapping.
            session2, _ = _make_session(
                approval_callback=lambda *a, **k: "deny",
                permission_mode="default",
            )
            res2 = asyncio.run(session2._make_can_use_tool()("Bash", {}, None))
            assert res2.message == "denied by user"
            session3, _ = _make_session(
                approval_callback=lambda *a, **k: "once",
                permission_mode="default",
            )
            res3 = asyncio.run(session3._make_can_use_tool()("Bash", {}, None))
            assert type(res3).__name__ == "PermissionResultAllow"
        finally:
            approval_mod.reset_current_session_key(token)

    def test_session_scoped_entry_lifecycle(self, monkeypatch):
        # Leak guard: the entry survives turn teardown (the feature), dies at
        # the conversation boundary (clear_session — the gateway's boundary
        # funnel) and at shutdown (clear_all_session_notify); re-register is
        # idempotent.
        sk = "sess-lifecycle"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            notify, _ = self._resolve_with(approval_mod, sk, "once")
            approval_mod.register_gateway_notify(sk, notify)
            approval_mod.register_session_notify(sk, notify)
            approval_mod.unregister_gateway_notify(sk)
            assert sk in approval_mod._session_notify_cbs  # survives the turn

            # Idempotent refresh: latest cb wins, still a single entry.
            def other(_data):
                pass

            approval_mod.register_session_notify(sk, other)
            assert approval_mod._session_notify_cbs[sk] is other

            # Conversation boundary removes it; the bridge then denies
            # honestly instead of paging a rotated-away session.
            approval_mod.clear_session(sk)
            assert sk not in approval_mod._session_notify_cbs
            cb = approval_mod.build_sdk_gateway_approval_callback()
            result = cb("Bash(ls)", "desc")
            assert result["reason"] == "no approver available (background context)"

            # Unknown-key unregister is a no-op; clear-all empties.
            approval_mod.unregister_session_notify("never-registered")
            approval_mod.register_session_notify(sk, notify)
            approval_mod.clear_all_session_notify()
            assert approval_mod._session_notify_cbs == {}
        finally:
            approval_mod.unregister_session_notify(sk)
            approval_mod.reset_current_session_key(token)

    def test_unanswered_and_failed_prompts_carry_honest_reasons(
        self, monkeypatch,
    ):
        # The model must never hear "denied by user" for a prompt no user
        # answered: timeout, notify-failure and /deny <reason> each carry
        # their own truth.
        sk = "sess-honest-reasons"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            cb = approval_mod.build_sdk_gateway_approval_callback()

            # Timeout: the operator was paged but never answered.
            monkeypatch.setattr(
                approval_mod, "_get_approval_timeout", lambda: 0.0
            )
            paged = []
            approval_mod.register_session_notify(sk, paged.append)
            result = cb("Bash(sleep)", "desc")
            assert result == {
                "choice": "deny",
                "reason": "approval timed out — no operator response",
            }
            assert len(paged) == 1

            # Notify failure: the prompt never reached the operator.
            def broken(_data):
                raise RuntimeError("adapter send failed")

            approval_mod.register_session_notify(sk, broken)
            result = cb("Bash(x)", "desc")
            assert result["choice"] == "deny"
            assert "notify failed" in result["reason"]
            assert "denied by user" not in result["reason"]

            # /deny <reason>: a REAL user deny — attributed, in their words.
            # Restore a sane timeout: the 0.0 above would hit the deadline
            # break before the wait ever observes the (already-set) event.
            monkeypatch.setattr(
                approval_mod, "_get_approval_timeout", lambda: 5.0
            )

            def deny_with_reason(_data):
                approval_mod.resolve_gateway_approval(
                    sk, "deny", reason="not now"
                )

            approval_mod.register_session_notify(sk, deny_with_reason)
            result = cb("Bash(y)", "desc")
            assert result == {
                "choice": "deny", "reason": "denied by user: not now",
            }
        finally:
            approval_mod.unregister_session_notify(sk)
            approval_mod.reset_current_session_key(token)

    def test_cron_born_session_approves_in_later_interactive_turn(
        self, monkeypatch,
    ):
        # Sticky-session freeze (incident defect 3): the SDK session and its
        # approval callback are frozen at creation; a session FIRST created
        # during a cron turn got callback=None forever — every
        # un-allowlisted tool silently denied even in later interactive
        # turns, until a session retire. Per-call resolution: the SAME
        # callback object denies honestly during cron turns and pages the
        # operator normally once an interactive turn refreshes the context.
        from tools import approval as approval_mod

        sk = "sess-cron-born"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        # What the cron turn's per-turn refresh writes into the holder.
        holder = {"gateway": False, "session_key": ""}
        cb = approval_mod.build_sdk_gateway_approval_callback(
            context_provider=lambda: dict(holder)
        )
        # RED pre-fix: the builder returned None for cron contexts, which
        # is exactly the freeze.
        assert cb is not None

        # Prompt during the cron turn: immediate honest deny — no paging,
        # no blocking, posture preserved.
        result = cb("Bash(ls)", "desc")
        assert result["reason"] == "no approver available (background context)"

        # Later INTERACTIVE turn on the SAME SDK session: the runtime's
        # per-turn refresh rewrites the holder; the gateway registers its
        # turn notify. No session retire happened.
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        holder.update({"gateway": True, "session_key": sk})
        notify, seen = self._resolve_with(approval_mod, sk, "once")
        approval_mod.register_gateway_notify(sk, notify)
        try:
            # Invoke from a FRESH thread — the SDK loop-thread reality:
            # contextvars invisible, so the holder must carry the context.
            out = {}
            t = threading.Thread(
                target=lambda: out.update(r=cb("Bash(uname)", "desc"))
            )
            t.start()
            t.join(timeout=10)
            assert out.get("r") == "once"
            assert len(seen) == 1
            assert seen[0]["command"] == "Bash(uname)"
        finally:
            approval_mod.unregister_gateway_notify(sk)

    def test_silent_denies_logged_with_tool_and_reason(
        self, monkeypatch, caplog,
    ):
        # P2.d: every deny that transits the SDK lane WITHOUT an operator
        # tap must be observable — the incident's silent denies had no log
        # line at all. The choke point is _make_can_use_tool; "denied by
        # user" is the trustworthy operator-attribution prefix (W8/W11)
        # and is deliberately NOT logged as silent.
        SILENT = "silent deny (no operator choice)"

        def _records(cl):
            return [r for r in cl.records if SILENT in r.getMessage()]

        def _deny_via(callback, cl):
            session, _ = _make_session(
                approval_callback=callback, permission_mode="default",
                hermes_session_id="sess-w13",
            )
            with cl.at_level(
                logging.INFO,
                logger="agent.transports.claude_agent_sdk_session",
            ):
                return asyncio.run(
                    session._make_can_use_tool()("Bash", {"command": "x"}, None)
                )

        # Class 1 — no-approver, via the REAL bridge (nothing registered).
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-w13-none")
        try:
            caplog.clear()
            res = _deny_via(
                approval_mod.build_sdk_gateway_approval_callback(), caplog,
            )
            assert res.message == "no approver available (background context)"
            recs = _records(caplog)
            assert len(recs) == 1
            msg = recs[0].getMessage()
            assert "tool=Bash" in msg
            assert "no approver available" in msg
            assert "session=sess-w13" in msg
        finally:
            approval_mod.reset_current_session_key(token)

        # Classes 2–3 — timeout and teardown-expiry reasons (the real
        # bridge produces these dicts; the choke point must log them).
        for reason in (
            "approval timed out — no operator response",
            "approval expired (turn ended)",
        ):
            caplog.clear()
            res = _deny_via(
                lambda *a, **k: {"choice": "deny", "reason": reason}, caplog,
            )
            assert res.message == reason
            recs = _records(caplog)
            assert len(recs) == 1
            assert reason in recs[0].getMessage()
            assert "tool=Bash" in recs[0].getMessage()

        # Class 4 — callback failure.
        caplog.clear()

        def _boom(*a, **k):
            raise RuntimeError("bridge exploded")

        res = _deny_via(_boom, caplog)
        assert res.message == "approval callback failed"
        recs = _records(caplog)
        assert len(recs) == 1
        assert "approval callback failed" in recs[0].getMessage()

        # Class 5 — the CLI thread-local callback's bare "timeout" string:
        # previously mapped to "denied by user" (fabricated attribution).
        caplog.clear()
        res = _deny_via(lambda *a, **k: "timeout", caplog)
        assert res.message == "approval timed out — no operator response"
        assert len(_records(caplog)) == 1

        # NEGATIVES — operator denies are NOT silent: no log line.
        for operator_deny in (
            lambda *a, **k: "deny",
            lambda *a, **k: {"choice": "deny", "reason": "denied by user: not now"},
        ):
            caplog.clear()
            res = _deny_via(operator_deny, caplog)
            assert res.message.startswith("denied by user")
            assert _records(caplog) == []

        # Allow path logs nothing either.
        caplog.clear()
        res = _deny_via(lambda *a, **k: "once", caplog)
        assert type(res).__name__ == "PermissionResultAllow"
        assert _records(caplog) == []

    def test_teardown_resolves_inflight_prompts_as_expired(self, monkeypatch):
        # Incident defect 2 (observed 08-04 and 08-06): turn teardown
        # signaled the blocked approval wait with an UNSET result; the
        # bridge read that as a deny and the model heard "denied by user"
        # for a prompt nobody answered. Teardown now stamps "expired" and
        # the SDK lane carries the honest reason.
        sk = "sess-teardown"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            # Paged but never answered — the prompt is in flight when the
            # turn tears down.
            approval_mod.register_gateway_notify(sk, lambda data: None)
            cb = approval_mod.build_sdk_gateway_approval_callback()
            out = {}
            t = threading.Thread(
                target=lambda: out.update(r=cb("Bash(x)", "desc"))
            )
            t.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with approval_mod._lock:
                    if approval_mod._gateway_queues.get(sk):
                        break
                time.sleep(0.01)
            approval_mod.unregister_gateway_notify(sk)
            t.join(timeout=10)
            assert out.get("r") == {
                "choice": "deny",
                "reason": "approval expired (turn ended)",
            }
            assert "denied by user" not in str(out.get("r"))
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_mod.reset_current_session_key(token)

    def test_tool_use_id_threads_from_context_to_approval_data(
        self, monkeypatch,
    ):
        # P2.a end to end: context.tool_use_id → callback kwarg (marker
        # opt-in) → approval_data → the pending entry the button resolves.
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-correlate")
        try:
            notify, seen = self._resolve_with(
                approval_mod, "sess-correlate", "once"
            )
            approval_mod.register_gateway_notify("sess-correlate", notify)
            try:
                cb = approval_mod.build_sdk_gateway_approval_callback()
                assert getattr(cb, "_accepts_tool_use_id", False) is True
                assert cb("Bash(a)", "desc", tool_use_id="toolu_T") == "once"
                assert seen[0]["tool_use_id"] == "toolu_T"
            finally:
                approval_mod.unregister_gateway_notify("sess-correlate")

            # Session layer: a marker-bearing callback receives the SDK
            # context's id...
            got = {}

            def marked(command, description, *, allow_permanent=False,
                       tool_use_id=""):
                got["tool_use_id"] = tool_use_id
                return "once"

            marked._accepts_tool_use_id = True
            session, _ = _make_session(
                approval_callback=marked, permission_mode="default"
            )
            ctx_obj = SimpleNamespace(tool_use_id="toolu_CTX")
            res = asyncio.run(
                session._make_can_use_tool()("Bash", {"command": "x"}, ctx_obj)
            )
            assert type(res).__name__ == "PermissionResultAllow"
            assert got["tool_use_id"] == "toolu_CTX"

            # ...and a marker-less (CLI-style) callback keeps its exact
            # signature — invoked without the kwarg, no TypeError.
            calls = {}

            def plain(command, description, *, allow_permanent=False):
                calls["ok"] = True
                return "deny"

            session2, _ = _make_session(
                approval_callback=plain, permission_mode="default"
            )
            res2 = asyncio.run(
                session2._make_can_use_tool()("Bash", {}, ctx_obj)
            )
            assert calls["ok"] is True
            assert type(res2).__name__ == "PermissionResultDeny"
        finally:
            approval_mod.reset_current_session_key(token)

    def test_no_context_deny_is_honest(self, monkeypatch, caplog):
        # A background prompt on a session whose latest turn context is
        # empty (no gateway, no key) must deny with the honest reason —
        # never the "denied by user" lie, never silently.
        from tools import approval as approval_mod

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        cb = approval_mod.build_sdk_gateway_approval_callback(
            context_provider=lambda: {}
        )
        assert cb is not None
        out = {}
        with caplog.at_level(logging.WARNING, logger="tools.approval"):
            t = threading.Thread(
                target=lambda: out.update(r=cb("Read(/x)", "desc"))
            )
            t.start()
            t.join(timeout=10)
        assert out.get("r") == {
            "choice": "deny",
            "reason": "no approver available (background context)",
        }
        assert any(
            "NO approver available" in r.getMessage() for r in caplog.records
        )

    def test_turn_refreshes_sdk_approval_context_snapshot(self, monkeypatch):
        # Runtime seam: the holder is rewritten at the TOP of
        # run_claude_agent_sdk_turn on every call — that per-turn refresh is
        # what un-freezes a cron-born session.
        import hermes_cli.config as cfg
        from tools import approval as approval_mod

        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input, **kw):
                return _make_turn()

            def close(self):
                pass

        monkeypatch.setattr(
            "agent.transports.claude_agent_sdk_session.ClaudeAgentSdkSession",
            SpySession,
        )
        monkeypatch.setattr(
            cfg, "load_config_readonly", lambda *a, **k: {}, raising=False
        )
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

        agent = _make_agent()
        agent._claude_sdk_session = None
        token = approval_mod.set_current_session_key("turn-key-1")
        try:
            run_claude_agent_sdk_turn(
                agent, user_message="hi", original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="t",
            )
        finally:
            approval_mod.reset_current_session_key(token)
        assert agent._sdk_approval_turn_ctx == {
            "gateway": True, "session_key": "turn-key-1",
        }
        assert captured.get("approval_callback") is not None  # bridge wired

        # Second call under a DIFFERENT key: the snapshot is rewritten (the
        # refresh runs before any session logic; forcing re-creation keeps
        # the spy simple — the refresh itself is call-scoped, not
        # creation-scoped).
        agent._claude_sdk_session = None
        token = approval_mod.set_current_session_key("turn-key-2")
        try:
            run_claude_agent_sdk_turn(
                agent, user_message="again", original_user_message="again",
                messages=[{"role": "user", "content": "again"}],
                effective_task_id="t",
            )
        finally:
            approval_mod.reset_current_session_key(token)
        assert agent._sdk_approval_turn_ctx == {
            "gateway": True, "session_key": "turn-key-2",
        }

    def _resolve_with(self, approval_mod, session_key, choice):
        seen = []

        def notify(data):
            seen.append(data)
            with approval_mod._lock:
                entry = approval_mod._gateway_queues[session_key][0]
            entry.result = choice
            entry.event.set()

        return notify, seen

    def test_approve_maps_to_once_and_clamps_durable_choices(self, monkeypatch):
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-mapped")
        try:
            # An older client button can still send "always" — the grant must
            # not outlive the single SDK permission request it answered.
            notify, seen = self._resolve_with(approval_mod, "sess-mapped", "always")
            approval_mod.register_gateway_notify("sess-mapped", notify)
            try:
                cb = approval_mod.build_sdk_gateway_approval_callback()
                assert cb("Bash(uname)", "Claude requests tool Bash") == "once"
            finally:
                approval_mod.unregister_gateway_notify("sess-mapped")
            assert seen[0]["allow_permanent"] is False
            assert seen[0]["allow_session"] is False
            assert seen[0]["command"] == "Bash(uname)"
        finally:
            approval_mod.reset_current_session_key(token)

    def test_deny_choice_denies(self, monkeypatch):
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-denied")
        try:
            notify, _ = self._resolve_with(approval_mod, "sess-denied", "deny")
            approval_mod.register_gateway_notify("sess-denied", notify)
            try:
                cb = approval_mod.build_sdk_gateway_approval_callback()
                assert cb("Bash(rm x)", "desc") == "deny"
            finally:
                approval_mod.unregister_gateway_notify("sess-denied")
        finally:
            approval_mod.reset_current_session_key(token)

    def test_create_session_falls_back_to_gateway_bridge(self, monkeypatch):
        # The runtime seam: no thread-local CLI callback + gateway context →
        # the session is constructed with the bridge callback, not None.
        from tools import approval as approval_mod
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}

        class _CapturingSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn(
                    projected_messages=[], final_text="ok",
                    token_usage_last=None,
                )

            def close(self):
                pass

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        token = approval_mod.set_current_session_key("tg:152:bridge")
        try:
            monkeypatch.setattr(
                session_mod, "ClaudeAgentSdkSession", _CapturingSession
            )
            agent = _make_agent()
            agent._claude_sdk_session = None
            run_claude_agent_sdk_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="task-1",
            )
            assert captured.get("approval_callback") is not None
        finally:
            approval_mod.reset_current_session_key(token)

    def test_sdk_tool_start_updates_shared_activity_before_progress(self, monkeypatch):
        """SDK lifecycle must drive the shared heartbeat activity contract."""
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}

        class _CapturingSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn(projected_messages=[], final_text="ok")

            def close(self):
                pass

        monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _CapturingSession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        seen_progress = []
        agent.tool_progress_callback = lambda *args: seen_progress.append(args)
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )

        captured["on_tool_started"]("Bash", "sqlite3 …", {"command": "sqlite3"})

        assert agent._current_tool == "Bash"
        agent._touch_activity.assert_called_once_with("executing tool: Bash")
        assert seen_progress == [("tool.started", "Bash", "sqlite3 …", {"command": "sqlite3"})]

    def test_sdk_tool_result_updates_isolated_live_iteration_counter(self, monkeypatch):
        """SDK result advances its guarded visibility count, not native state."""
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}

        class _CapturingSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn(projected_messages=[], final_text="ok")

            def close(self):
                pass

        monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _CapturingSession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._current_turn_id = "turn-one"
        agent._api_call_count = 0
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="task-1",
        )
        captured["on_tool_iteration"]()
        assert agent._sdk_visibility_iteration_count == 1
        assert agent._api_call_count == 0

    def test_late_sdk_callback_is_fenced_after_turn_replacement(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}
        class _CapturingSession:
            def __init__(self, **kwargs): captured.update(kwargs)
            def run_turn(self, user_input): return _make_turn(projected_messages=[], final_text="ok")
            def close(self): pass
        monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _CapturingSession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._current_turn_id = "turn-one"
        run_claude_agent_sdk_turn(agent, user_message="one", original_user_message="one", messages=[{"role": "user", "content": "one"}], effective_task_id="one")
        stale = captured["on_tool_iteration"]
        agent._current_turn_id = "turn-two"
        run_claude_agent_sdk_turn(agent, user_message="two", original_user_message="two", messages=[{"role": "user", "content": "two"}], effective_task_id="two")
        stale()
        assert agent._sdk_visibility_iteration_count == 0

    def test_sdk_interim_relay_scrubs_redacts_and_deduplicates(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as session_mod

        captured = {}
        class _CapturingSession:
            def __init__(self, **kwargs): captured.update(kwargs)
            def run_turn(self, user_input): return _make_turn(projected_messages=[], final_text="ok")
            def close(self): pass
        monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _CapturingSession)
        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._current_turn_id = "turn-one"
        agent._strip_think_blocks.side_effect = lambda text: text.replace("<think>private</think>", "")
        agent._delivered_interim_texts = set()
        agent._interim_text_was_delivered.side_effect = lambda text: text in agent._delivered_interim_texts
        agent._record_delivered_interim_text.side_effect = lambda text: agent._delivered_interim_texts.add(text)
        delivered = []
        agent.interim_assistant_callback = lambda text, **kw: delivered.append((text, kw))
        run_claude_agent_sdk_turn(agent, user_message="one", original_user_message="one", messages=[{"role": "user", "content": "one"}], effective_task_id="one")
        relay = captured["on_interim_assistant"]
        relay("<think>private</think> Checking token sk-ant-12345678901234567890")
        relay("<think>private</think> Checking token sk-ant-12345678901234567890")
        assert len(delivered) == 1
        assert "private" not in delivered[0][0]
        assert "12345678901234567890" not in delivered[0][0]
        assert delivered[0][1] == {"already_streamed": False}

    def test_create_session_without_gateway_context_keeps_none(self, monkeypatch):
        # CLI/bare-process posture unchanged: no context → callback stays None.
        import agent.transports.claude_agent_sdk_session as session_mod

        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        captured = {}

        class _CapturingSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input):
                return _make_turn(
                    projected_messages=[], final_text="ok",
                    token_usage_last=None,
                )

            def close(self):
                pass

        monkeypatch.setattr(
            session_mod, "ClaudeAgentSdkSession", _CapturingSession
        )
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        assert captured.get("approval_callback") is None


class TestAnthropicTokenGuard:
    """F1 follow-up (#65982 independent verification): ANTHROPIC_TOKEN alone
    authenticates hermes' native metered lane, so an API-key-shaped value is
    the same fail-closed class as ANTHROPIC_API_KEY — while an OAuth-shaped
    value is the subscription lane itself and must keep working."""

    def test_anthropic_token_api_key_shaped_refuses_startup(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-api03-fake")
        session = ClaudeAgentSdkSession(cwd="/tmp")  # no factory → real path
        turn = session.run_turn("hi")
        assert turn.should_retire
        assert "ANTHROPIC_TOKEN" in (turn.error or "")

    def test_anthropic_token_oauth_shaped_starts_normally(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-fake")
        session, _holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            turn = session.run_turn("ping")
        finally:
            session.close()
        assert turn.error is None

    def test_allow_metered_key_admits_api_key_shaped_token(self, monkeypatch):
        import hermes_cli.config as cfg

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-api03-fake")
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"allow_metered_key": True}}
            },
            raising=False,
        )
        session, _holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            turn = session.run_turn("ping")
        finally:
            session.close()
        assert turn.error is None



class TestModelAttribution:
    """F3 (#65982 independent verification): with model.default unset the
    usage rows carried model='unknown' while the SDK's own AssistantMessage
    knew the real id — capture it and back-fill the attribution."""

    def test_session_captures_model_last_from_assistant_message(self):
        script = [
            AssistantMessage(
                content=[TextBlock("hey")], model="claude-opus-4-8-20260115"
            ),
            ResultMessage(
                result="hey", usage={"input_tokens": 1, "output_tokens": 1}
            ),
        ]
        session, _holder = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.model_last == "claude-opus-4-8-20260115"

    def test_usage_row_backfills_model_from_turn(self):
        from agent.claude_sdk_runtime import _record_claude_sdk_usage

        agent = _make_agent()
        agent.model = ""
        db = MagicMock()
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = "sess-attr-1"
        turn = _make_turn(model_last="claude-opus-4-8-20260115")
        _record_claude_sdk_usage(agent, turn)
        kwargs = db.update_token_counts.call_args.kwargs
        assert kwargs["model"] == "claude-opus-4-8-20260115"

    def test_explicit_agent_model_still_wins(self):
        from agent.claude_sdk_runtime import _record_claude_sdk_usage

        agent = _make_agent()
        agent.model = "claude-sonnet-5"
        db = MagicMock()
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = "sess-attr-2"
        turn = _make_turn(model_last="claude-opus-4-8-20260115")
        _record_claude_sdk_usage(agent, turn)
        kwargs = db.update_token_counts.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-5"

    def test_explicit_metered_turn_is_not_recorded_as_subscription_included(self):
        from agent.claude_sdk_runtime import _record_claude_sdk_usage

        agent = _make_agent()
        db = MagicMock()
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = "sess-metered-1"
        turn = _make_turn(
            billing_mode="sdk_reported_metered",
            total_cost_usd=0.25,
        )
        result = _record_claude_sdk_usage(agent, turn)
        kwargs = db.update_token_counts.call_args.kwargs
        assert kwargs["billing_mode"] == "sdk_reported_metered"
        assert kwargs["actual_cost_usd"] == 0.25
        assert kwargs["cost_status"] == "reported"
        assert result["cost_status"] == "reported"
        assert result["actual_cost_usd"] == 0.25

    def test_missing_billing_evidence_is_not_recorded_as_included(self):
        from agent.claude_sdk_runtime import _record_claude_sdk_usage

        agent = _make_agent()
        db = MagicMock()
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = "sess-unverified-1"
        turn = _make_turn(billing_mode=None)
        result = _record_claude_sdk_usage(agent, turn)
        kwargs = db.update_token_counts.call_args.kwargs
        assert kwargs["billing_mode"] == "unknown"
        assert kwargs["cost_status"] == "unknown"
        assert kwargs["cost_source"] == "claude-agent-sdk-unverified"
        assert result["cost_status"] == "unknown"


class TestUnsolicitedDelivery:
    """The delivery half of the stream-ownership fix (dasbrow-hermes-coder#2):
    a finished background Agent task's answer must be CAPTURED and handed to
    the delivery callback — never served as a turn result (TestStreamOwnership
    pins that), and never silently discarded either (observed live 2026-07-29:
    14 dropped answers, 32-minute silences until the operator poked)."""

    @staticmethod
    def _wait(cond, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return True
            time.sleep(0.01)
        return False

    def test_callback_receives_unsolicited_result_text(self):
        got = []
        session, holder = _make_session(on_unsolicited_result=got.append)
        try:
            session.ensure_started()
            holder["client"].feed(
                AssistantMessage(content=[TextBlock("research done: Tupã wins")]),
                ResultMessage(result="research done: Tupã wins", uuid="bg-1"),
            )
            assert self._wait(lambda: got)
        finally:
            session.close()
        assert got == [["research done: Tupã wins"]]
        # Observability unchanged: the counter still ticks.
        assert session._unsolicited_results == 1

    def test_bg_burst_delivers_all_buffered_assistant_messages(self):
        # ×5 incident 2026-08-06: five out-of-turn AssistantMessages were
        # buffered, the terminal ResultMessage carried its own text, and the
        # intermediate "Research landed…" message was silently discarded —
        # only the terminal report reached the callback. The full burst must
        # arrive as an ordered list, result text deduped against the last
        # buffered entry.
        got = []
        session, holder = _make_session(on_unsolicited_result=got.append)
        try:
            session.ensure_started()
            holder["client"].feed(
                AssistantMessage(
                    content=[TextBlock("Research landed — writing up.")]
                ),
                AssistantMessage(content=[TextBlock("the full report")]),
                ResultMessage(result="the full report", uuid="burst-1"),
            )
            assert self._wait(lambda: got)
        finally:
            session.close()
        assert got == [["Research landed — writing up.", "the full report"]]

    def test_falls_back_to_buffered_assistant_text(self):
        # Some CLI results arrive with result=None; the assistant text blocks
        # of the unsolicited turn are the answer then.
        got = []
        session, holder = _make_session(on_unsolicited_result=got.append)
        try:
            session.ensure_started()
            holder["client"].feed(
                AssistantMessage(content=[TextBlock("the long answer body")]),
                ResultMessage(result=None, uuid="bg-2"),
            )
            assert self._wait(lambda: got)
        finally:
            session.close()
        assert got == [["the long answer body"]]

    def test_result_uuid_deduplicated(self):
        got = []
        session, holder = _make_session(on_unsolicited_result=got.append)
        try:
            session.ensure_started()
            holder["client"].feed(ResultMessage(result="answer", uuid="dup-1"))
            assert self._wait(lambda: got)
            holder["client"].feed(ResultMessage(result="answer", uuid="dup-1"))
            self._wait(lambda: len(got) >= 2, timeout=0.5)
        finally:
            session.close()
        assert got == [["answer"]]

    def test_subagent_text_excluded_from_buffer(self):
        # parent_tool_use_id set = subagent stream noise — same gate the
        # stream-delta forwarder uses. Only top-level text is the answer.
        got = []
        session, holder = _make_session(on_unsolicited_result=got.append)
        try:
            session.ensure_started()
            holder["client"].feed(
                AssistantMessage(
                    content=[TextBlock("sub noise")], parent_tool_use_id="t1"
                ),
                AssistantMessage(content=[TextBlock("top-level answer")]),
                ResultMessage(result=None, uuid="bg-3"),
            )
            assert self._wait(lambda: got)
        finally:
            session.close()
        assert got == [["top-level answer"]]

    def test_no_callback_keeps_drop_semantics(self):
        # Without a wired callback the historical WARN+counter drop stands
        # (TestStreamOwnership's pins rely on it).
        session, holder = _make_session()
        try:
            session.ensure_started()
            holder["client"].feed(ResultMessage(result="x", uuid="nc-1"))
            assert self._wait(
                lambda: getattr(session, "_unsolicited_results", 0) >= 1
            )
        finally:
            session.close()
        assert session._unsolicited_results == 1


class TestBackgroundDeliveryWiring:
    """Runtime glue: the session's delivery callback enqueues an
    sdk_background_result event for the gateway watcher's direct outbound
    send, config-gated."""

    def _spy_kwargs(self, monkeypatch):
        import agent.claude_sdk_runtime as runtime_mod

        captured = {}

        class SpySession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_turn(self, user_input, **kw):
                return _make_turn()

            def close(self):
                pass

        monkeypatch.setattr(
            "agent.transports.claude_agent_sdk_session.ClaudeAgentSdkSession",
            SpySession,
        )
        return captured

    def test_flag_on_wires_callback_and_queue_event(self, monkeypatch):
        import hermes_cli.config as cfg
        from tools.process_registry import process_registry

        captured = self._spy_kwargs(monkeypatch)
        events = []

        class _FakeQueue:
            def put(self, evt):
                events.append(evt)

        monkeypatch.setattr(process_registry, "completion_queue", _FakeQueue())
        monkeypatch.setattr(
            "tools.approval.get_current_session_key", lambda: "gw-key-7"
        )
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        # Opt-in flag (upstream-conservative default is OFF).
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"deliver_background_results": True}}
            },
            raising=False,
        )

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent.session_id = "sess-bg-1"
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        callback = captured.get("on_unsolicited_result")
        assert callback is not None, "flag defaults ON — callback must be wired"
        callback(["Research landed — writing up.", "background answer text"])
        assert len(events) == 1
        evt = events[0]
        # Direct-outbound event: the payload burst rides UNJOINED (each text
        # becomes its own outbound message) and no model-facing directive is
        # prepended — on a direct send it would leak to the user.
        assert evt["type"] == "sdk_background_result"
        assert evt["payloads"] == [
            "Research landed — writing up.", "background answer text",
        ]
        assert not any("[USER IS WAITING" in p for p in evt["payloads"])
        assert evt["session_key"] == "gw-key-7"
        assert evt["parent_session_id"] == "sess-bg-1"
        assert "delegation_id" not in evt

    def test_bg_parent_resolved_at_delivery_time_after_rotation(
        self, monkeypatch,
    ):
        # P0.g: the SDK session outlives hermes session rotations. The old
        # code snapshotted parent_session_id/session_key at SDK-session
        # CREATION, so a completion firing after rotation carried the dead
        # parent — the gateway classified it permanently gone and dropped
        # it. The callback must resolve the parent AT DELIVERY TIME, with
        # the creation-time snapshot only as a fallback for the SDK-loop
        # thread where the session-key contextvar is unset.
        import hermes_cli.config as cfg
        from tools.process_registry import process_registry

        captured = self._spy_kwargs(monkeypatch)
        events = []

        class _FakeQueue:
            def put(self, evt):
                events.append(evt)

        monkeypatch.setattr(process_registry, "completion_queue", _FakeQueue())
        monkeypatch.setattr(
            "tools.approval.get_current_session_key", lambda: "gw-key-7"
        )
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"deliver_background_results": True}}
            },
            raising=False,
        )

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent.session_id = "sess-before"
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        callback = captured.get("on_unsolicited_result")
        assert callback is not None

        # Hermes rotates the session between turns; the completion fires on
        # the SDK loop thread where the contextvar reads empty.
        agent.session_id = "sess-after-rotation"
        monkeypatch.setattr(
            "tools.approval.get_current_session_key", lambda: ""
        )
        callback(["late background report"])
        assert len(events) == 1
        assert events[0]["parent_session_id"] == "sess-after-rotation"
        # Empty live key -> creation-time snapshot fallback keeps the route.
        assert events[0]["session_key"] == "gw-key-7"

        # A live, non-empty contextvar read wins over the snapshot.
        monkeypatch.setattr(
            "tools.approval.get_current_session_key", lambda: "gw-key-LIVE"
        )
        callback(["second late report"])
        assert len(events) == 2
        assert events[1]["session_key"] == "gw-key-LIVE"
        assert events[1]["parent_session_id"] == "sess-after-rotation"

    def test_flag_off_leaves_callback_unwired(self, monkeypatch):
        import hermes_cli.config as cfg

        captured = self._spy_kwargs(monkeypatch)
        monkeypatch.delenv("HERMES_CLAUDE_SDK_DELIVER_BACKGROUND", raising=False)
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"deliver_background_results": False}}
            },
            raising=False,
        )
        agent = _make_agent()
        agent._claude_sdk_session = None
        run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert captured.get("on_unsolicited_result") is None


class TestBargeInInterruptHandoff:
    """W22 (2026-08-09 barge-in incident): a mid-turn user message interrupts
    the running turn; the CLI, aborted before any assistant content, returns
    is_error/error_during_execution ("[ede_diagnostic] result_type=user…").
    Two defects made that page the operator with a false "⚠️ Processing
    stopped … Try again": the runtime result dict omitted the "interrupted"
    key (codex_runtime and the native finalizer both return it — the gateway's
    queued-drain needs it to discard, not deliver, the abandoned turn), and
    the transport surfaced the CLI's interrupt-shaped error as a real error.
    The honest error path must NOT weaken: unrequested EDE, auth-hinted
    errors, and CLI death keep surfacing."""

    def test_sdk_result_dict_carries_interrupted_key(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._session_db = None

        class SpySession:
            def __init__(self, **kwargs):
                pass

            def run_turn(self, user_input):
                agent._interrupt_requested = True  # barge-in mid-turn
                return _make_turn(interrupted=True, final_text="",
                                  projected_messages=[])

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert result["interrupted"] is True
        assert result["partial"] is True

    def test_uninterrupted_turn_reports_interrupted_false(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._session_db = None

        class SpySession:
            def __init__(self, **kwargs):
                pass

            def run_turn(self, user_input):
                return _make_turn()

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        result = run_claude_agent_sdk_turn(
            agent, user_message="hi", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], effective_task_id="t",
        )
        assert result["interrupted"] is False

    def _ede_result(self):
        return ResultMessage(
            subtype="error_during_execution",
            is_error=True,
            errors=["[ede_diagnostic] result_type=user last_content_type=n/a "
                    "stop_reason=null"],
        )

    def test_requested_interrupt_ede_masks_error_with_log(self, caplog):
        import logging
        holder = {}

        class BargeInClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                holder["session"]._interrupt_event.set()
                self._pending.append(ResultMessage(
                    subtype="error_during_execution",
                    is_error=True,
                    errors=["[ede_diagnostic] result_type=user "
                            "last_content_type=n/a stop_reason=null"],
                ))

        def factory(options=None):
            client = BargeInClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        holder["session"] = session
        try:
            with caplog.at_level(
                logging.INFO, logger="agent.transports.claude_agent_sdk_session"
            ):
                turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.interrupted is True
        assert turn.error is None
        assert turn.should_retire is False
        assert any("masked error_during_execution" in r.getMessage()
                   for r in caplog.records)

    def test_unrequested_ede_stays_an_error(self):
        session, _ = _make_session(script=[self._ede_result()])
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert "error_during_execution" in turn.error

    def test_auth_hint_in_ede_outranks_interrupt_mask(self):
        holder = {}

        class AuthFailBargeIn(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                holder["session"]._interrupt_event.set()
                self._pending.append(ResultMessage(
                    subtype="error_during_execution",
                    is_error=True,
                    errors=["401 unauthorized: oauth token has expired"],
                ))

        def factory(options=None):
            client = AuthFailBargeIn(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        holder["session"] = session
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert turn.should_retire is True

    def test_stream_end_during_interrupt_stays_an_error(self):
        holder = {}

        class DyingClient(_FakeClient):
            async def query(self, text):
                self.queried.append(text)
                holder["session"]._interrupt_event.set()
                # No ResultMessage — the CLI dies; the reader sees the
                # stream END. (This _EOS is what makes the comment true:
                # without it the fake merely went silent and the test was
                # riding the old wall clock — 26s of suite time for the
                # wrong mechanism.)
                self._pending.append(_EOS)

        def factory(options=None):
            client = DyingClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        holder["session"] = session
        try:
            turn = session.run_turn("hi", turn_timeout=10.0)
        finally:
            session.close()
        assert turn.error is not None
        # Stream death retires (the poisoned-session fix), interrupt or not.
        assert turn.should_retire is True


# ---------- activity-aware turn lifetime (the 600s wall-clock fix) ----------
# Production forensics (24/7 gateway deployment, six "turn timed out after
# 600s" retires 2026-07-22 → 2026-08-09): four of six were ACTIVELY-WORKING
# turns — tool loops mid-execution, human approval taps counted as silence —
# killed by a hard wall clock over the whole turn. The lifetime is now evidence-based:
# outstanding tools and pending approvals suspend the rules; the budget only
# fires on a turn that is ALSO quiet; a post-tool quiet watchdog catches
# wedges early; a tripped turn that the CLI acks cleanly keeps its partial
# transcript and resume id instead of retiring.


class _HoldOpenClient(_FakeClient):
    """The wedge shape: the stream stays OPEN on silence. The base fake
    appends _EOS whenever the script has no ResultMessage (modeling a dead
    CLI) — but a wedged turn's stream is alive and silent, which is exactly
    the state the watchdogs exist to detect. Optionally acks interrupt()
    with scripted messages (the CLI's interrupt-ack ResultMessage)."""

    def __init__(self, *args, interrupt_ack=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._interrupt_ack = list(interrupt_ack or [])

    async def query(self, text):
        self.queried.append(text)
        self._pending.extend(self._script)

    async def interrupt(self):
        self.interrupted = True
        self._pending.extend(self._interrupt_ack)


def _make_hold_open_session(script=None, interrupt_ack=None, **kwargs):
    holder = {}

    def factory(options=None):
        holder["client"] = _HoldOpenClient(
            options=options, script=script, interrupt_ack=interrupt_ack
        )
        return holder["client"]

    session = ClaudeAgentSdkSession(
        cwd="/tmp", model="claude-opus-4-8", client_factory=factory, **kwargs
    )
    return session, holder


def _ede_interrupt_ack():
    """The CLI's honored-interrupt shape (the W22 mask target)."""
    return ResultMessage(
        subtype="error_during_execution",
        is_error=True,
        result=None,
        errors=["[ede_diagnostic] result_type=user"],
        uuid="uuid-ede-ack",
    )


def _wait_for_client(holder, timeout=5.0):
    """Feeder threads start before run_turn builds the client — block until
    the factory has run."""
    deadline = time.monotonic() + timeout
    while "client" not in holder:
        if time.monotonic() > deadline:
            raise AssertionError("client never created")
        time.sleep(0.01)
    return holder["client"]


class TestTurnLifetime:
    def test_active_turn_survives_past_turn_timeout(self):
        # RED on the pre-fix tree: the old hard wall clock kills this turn at
        # 0.5s with "turn timed out after 0s" even though tool results are
        # landing every 50ms. GREEN: activity extends the turn to completion.
        session, holder = _make_hold_open_session(script=[])
        stop = threading.Event()

        def feeder():
            client = _wait_for_client(holder)
            for i in range(18):  # ~0.9s of beats at 50ms, > 0.5s budget
                if stop.is_set():
                    return
                time.sleep(0.05)
                client.feed(
                    AssistantMessage(
                        content=[ToolUseBlock(id=f"t{i}", name="Read", input={})]
                    ),
                    UserMessage(
                        content=[ToolResultBlock(tool_use_id=f"t{i}", content="ok")]
                    ),
                )
            client.feed(
                AssistantMessage(content=[TextBlock("long job done")]),
                ResultMessage(result="long job done", uuid="uuid-long"),
            )

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            turn = session.run_turn(
                "big task",
                turn_timeout=0.5,
                post_tool_quiet_timeout=0.0,
                watch_poll_interval=0.02,
            )
        finally:
            stop.set()
            thread.join(timeout=5)
            session.close()
        assert turn.error is None
        assert turn.should_retire is False
        assert turn.final_text == "long job done"

    def test_outstanding_tool_suspends_budget(self):
        # A single long-running tool emits NOTHING on the stream. The issued
        # ToolUseBlock keeps the turn suspended past the budget until its
        # result lands. RED on the pre-fix tree (dies at 0.3s).
        session, holder = _make_hold_open_session(
            script=[
                AssistantMessage(
                    content=[ToolUseBlock(id="slow", name="Bash", input={})]
                ),
            ]
        )

        def feeder():
            client = _wait_for_client(holder)
            time.sleep(0.9)  # 3x the budget, tool still "running"
            client.feed(
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="slow", content="done")]
                ),
                AssistantMessage(content=[TextBlock("tool finished")]),
                ResultMessage(result="tool finished", uuid="uuid-slow"),
            )

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            turn = session.run_turn(
                "run the slow tool",
                turn_timeout=0.3,
                post_tool_quiet_timeout=0.0,
                watch_poll_interval=0.02,
            )
        finally:
            thread.join(timeout=5)
            session.close()
        assert turn.error is None
        assert turn.final_text == "tool finished"

    def test_post_tool_quiet_trips_on_wedge_clean_ack(self):
        # Wedge signature: a tool result lands, then the stream goes silent
        # (alive, no _EOS). The quiet watchdog trips fast, the CLI acks the
        # interrupt with the EDE shape — and the clean ack preserves the
        # partial transcript AND the resume id (no retire), while the trip
        # text WINS over the masked EDE ack (never "SDK result error ...").
        session, holder = _make_hold_open_session(
            script=[
                SystemMessage(session_id="sdk-wedge-1"),
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Grep", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="hits")]
                ),
            ],
            interrupt_ack=[_ede_interrupt_ack()],
        )
        try:
            turn = session.run_turn(
                "search",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.2,
                watch_poll_interval=0.05,
            )
        finally:
            session.close()
        assert turn.error is not None
        assert "turn timed out" in turn.error
        assert "after a tool result" in turn.error
        assert "SDK result error" not in turn.error  # W22 mask + trip wins
        assert turn.interrupted is True
        # should_retire=False is the load-bearing assertion: the runtime
        # persists the resume id ONLY for non-retiring turns. thread_id just
        # has to be present for it to persist (its exact value tracks the
        # last session_id-bearing message — here the ack's own default).
        assert turn.should_retire is False  # clean ack — resumable
        assert turn.thread_id
        assert holder["client"].interrupted is True
        # Partial transcript survived the trip.
        roles = [m["role"] for m in turn.projected_messages]
        assert "assistant" in roles and "tool" in roles

    def test_post_tool_watchdog_resets_on_activity(self):
        # Assistant output after the tool result DISARMS the quiet watchdog
        # (codex-parity reset semantics). The post-disarm silence here (0.7s)
        # EXCEEDS the quiet limit (0.25s) — with the disarm neutered this
        # turn trips; with it, the turn completes untouched.
        session, holder = _make_hold_open_session(script=[])

        def feeder():
            client = _wait_for_client(holder)
            client.feed(
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Read", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="data")]
                ),
            )
            time.sleep(0.1)  # armed, under the limit
            client.feed(AssistantMessage(content=[TextBlock("thinking done")]))
            time.sleep(0.7)  # SILENCE past the limit — trips iff still armed
            client.feed(ResultMessage(result="thinking done", uuid="uuid-r"))

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            turn = session.run_turn(
                "go",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.25,
                watch_poll_interval=0.02,
            )
        finally:
            thread.join(timeout=5)
            session.close()
        assert turn.error is None
        assert turn.final_text == "thinking done"
        # The load-bearing pin: a DISARMED watchdog never fires an interrupt
        # at all. (Without this, a broken disarm can hide behind the
        # completion-during-grace rescue, which still delivers the pre-trip
        # text — resilient, but the needless interrupt+reconnect cycle is
        # exactly what the disarm exists to avoid.)
        assert holder["client"].interrupted is False

    def test_stream_deltas_disarm_quiet_watchdog(self):
        # Streaming posture — the only posture where the quiet watchdog
        # defaults ON: partial deltas after a tool result prove the model
        # call is alive and DISARM the watchdog. The post-delta silence
        # (0.7s) exceeds the quiet limit (0.25s) — trips iff the StreamEvent
        # branch's disarm is broken.
        session, holder = _make_hold_open_session(script=[])
        session._streaming = True  # the __init__ snapshot, forced for the test

        def feeder():
            client = _wait_for_client(holder)
            client.feed(
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Bash", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="ok")]
                ),
            )
            for _ in range(3):
                time.sleep(0.05)
                client.feed(_text_delta_event("chunk "))
            time.sleep(0.7)  # silence past the limit — armed would trip
            client.feed(
                AssistantMessage(content=[TextBlock("streamed answer")]),
                ResultMessage(result="streamed answer", uuid="uuid-sd"),
            )

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            turn = session.run_turn(
                "stream it",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.25,
                watch_poll_interval=0.02,
            )
        finally:
            thread.join(timeout=5)
            session.close()
        assert turn.error is None
        assert turn.final_text == "streamed answer"
        assert holder["client"].interrupted is False  # watchdog never fired

    def test_hard_trip_retires_and_clears_interrupt_event(self):
        # The CLI ignores the interrupt for the whole grace: hard-cancel,
        # retire (today's shape, now the rare fallback) — and the interrupt
        # event must NOT leak into the next turn on this session object.
        # RED on the pre-fix tree: the old timeout branch left the event set.
        session, holder = _make_hold_open_session(script=[])  # total silence
        try:
            turn = session.run_turn(
                "hello?",
                turn_timeout=0.2,
                post_tool_quiet_timeout=0.0,
                watch_poll_interval=0.05,
                abort_grace=0.2,
            )
            assert turn.error is not None
            assert "turn timed out after" in turn.error
            assert turn.should_retire is True
            assert turn.interrupted is True
            assert holder["client"].interrupted is True
            assert session._interrupt_event.is_set() is False
        finally:
            session.close()

    def test_completion_during_grace_delivered_in_full(self):
        # The answer text streamed BEFORE the trip; the success ack's uuid
        # then completes the turn inside the grace. Completion wins: no trip
        # error, no retire, the PRE-TRIP text is delivered. (The ack's own
        # result= text is never projected — projection stops at the
        # interrupt — so final_text here is the earlier AssistantMessage's.)
        session, holder = _make_hold_open_session(
            script=[
                AssistantMessage(content=[TextBlock("the answer")]),
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Bash", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="ok")]
                ),
            ],
            interrupt_ack=[
                ResultMessage(result=None, uuid="uuid-late")
            ],
        )
        try:
            turn = session.run_turn(
                "answer then wedge",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.2,
                watch_poll_interval=0.05,
            )
        finally:
            session.close()
        assert turn.error is None
        assert turn.should_retire is False
        assert turn.final_text == "the answer"

    def test_success_ack_without_prior_text_stays_a_trip(self):
        # Negative control for the completion lane's final_text gate: a
        # SUCCESS ack with a uuid but NO answer text anywhere (pure
        # tool-work turn) must remain a trip — voiding it here would turn
        # the timeout into a silent empty delivery.
        session, holder = _make_hold_open_session(
            script=[
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Bash", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="ok")]
                ),
            ],
            interrupt_ack=[
                ResultMessage(result=None, uuid="uuid-empty")
            ],
        )
        try:
            turn = session.run_turn(
                "tool work then wedge",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.2,
                watch_poll_interval=0.05,
            )
        finally:
            session.close()
        assert turn.error is not None
        assert "turn timed out" in turn.error
        assert turn.interrupted is True
        assert turn.should_retire is False  # clean ack still preserves resume

    def test_coroutine_timeout_error_classified_not_spun(self):
        # py3.11 unifies the TimeoutError family: a TimeoutError RAISED BY
        # the turn coroutine (socket/pipe timeout under the CLI) must be
        # classified as a turn failure — the pre-fix tree misread it as the
        # turn hitting its own 600s wall ("turn timed out after 600s").
        class TimeoutRaisingClient(_FakeClient):
            async def query(self, text):
                raise TimeoutError("socket write timed out")

        holder = {}

        def factory(options=None):
            holder["client"] = TimeoutRaisingClient(options=options)
            return holder["client"]

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        try:
            turn = session.run_turn("hi", watch_poll_interval=0.05)
        finally:
            session.close()
        assert turn.error is not None
        assert "claude-agent-sdk turn failed" in turn.error
        assert "socket write timed out" in turn.error
        assert "turn timed out after" not in turn.error

    def test_pending_approval_suspends_watchdogs(self, monkeypatch):
        # A human being asked is not the turn being silent: while the
        # session's own can_use_tool bridge awaits the approval callback,
        # both rules stand down — the quiet watchdog (armed by the tool
        # result) must NOT trip during a 0.5s approval on a 0.15s quiet
        # limit. After the tap the turn completes.
        _plant_claude_agent_sdk_stand_in(monkeypatch)
        released = threading.Event()

        def slow_approval(preview, prompt, **kwargs):
            time.sleep(0.5)
            released.set()
            return "once"

        session, holder = _make_hold_open_session(
            script=[
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Write", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="ok")]
                ),
            ],
            approval_callback=slow_approval,
            permission_mode="default",
        )

        def feeder():
            # The approval prompt fires on the session loop while the turn
            # is in flight — the exact production shape (SDK invoking
            # can_use_tool mid-turn). Event-based sync: wait until the tool
            # result actually ARMED the watchdog, then fire the approval
            # immediately (a sleep here left a ~36ms margin against the
            # quiet limit — flake fuel on loaded CI).
            _wait_for_client(holder)
            deadline = time.monotonic() + 5
            while True:
                watch = session._turn_watch
                if watch is not None and watch.post_tool_armed:
                    break
                if time.monotonic() > deadline:
                    raise AssertionError("watchdog never armed")
                time.sleep(0.01)
            cb = session._make_can_use_tool()
            fut = asyncio.run_coroutine_threadsafe(
                cb("Write", {"file_path": "/x"}, SimpleNamespace(tool_use_id="t1")),
                session._loop,
            )
            fut.result(timeout=5)
            holder["client"].feed(
                AssistantMessage(content=[TextBlock("written")]),
                ResultMessage(result="written", uuid="uuid-appr"),
            )

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        try:
            turn = session.run_turn(
                "write it",
                turn_timeout=30.0,
                post_tool_quiet_timeout=0.15,
                watch_poll_interval=0.02,
            )
        finally:
            thread.join(timeout=5)
            session.close()
        assert released.is_set()  # the approval really took 0.5s
        assert turn.error is None
        assert turn.final_text == "written"

    def test_orphaned_approval_decrements_its_own_watch(self, monkeypatch):
        # The F9 shape: an approval wait that OUTLIVES its turn must
        # decrement the watch it suspended — never a later turn's. The
        # callback captures the watch object at entry; a stale decrement
        # lands on the dead watch.
        _plant_claude_agent_sdk_stand_in(monkeypatch)
        from agent.transports import claude_agent_sdk_session as session_mod

        release = threading.Event()

        def blocking_cb(preview, prompt, **kwargs):
            release.wait(5)
            return "once"

        session, holder = _make_hold_open_session(
            script=[], approval_callback=blocking_cb,
            permission_mode="default",
        )
        try:
            session.ensure_started()
            w1 = session_mod._TurnWatch()
            session._turn_watch = w1
            cb = session._make_can_use_tool()
            fut = asyncio.run_coroutine_threadsafe(
                cb("Bash", {}, SimpleNamespace(tool_use_id="t1")),
                session._loop,
            )
            deadline = time.monotonic() + 5
            while w1.approvals_pending != 1:
                if time.monotonic() > deadline:
                    raise AssertionError("approval never registered on w1")
                time.sleep(0.01)
            # Turn 1 ends; turn 2 installs a fresh watch while the approval
            # is still pending.
            w2 = session_mod._TurnWatch()
            session._turn_watch = w2
            release.set()
            fut.result(timeout=5)
            assert w1.approvals_pending == 0  # its OWN watch decremented
            assert w2.approvals_pending == 0  # the new turn's never touched
        finally:
            release.set()
            session.close()


class TestDeadStreamRetires:
    """A dead SDK stream is permanent on the session object (_stream_ended
    never resets, ensure_started returns early while _client is set) — so a
    non-retiring stream-death error poisons EVERY later turn into an instant
    zero-model-call failure. RED pre-fix: both shapes returned
    should_retire=False (retire fired only on auth hints)."""

    def test_stream_death_mid_turn_retires(self):
        # Base fake: a script with no ResultMessage models the CLI dying
        # mid-turn (_EOS ends the stream).
        session, holder = _make_session(
            script=[AssistantMessage(content=[TextBlock("partial")])]
        )
        try:
            turn = session.run_turn("hi", watch_poll_interval=0.02)
        finally:
            session.close()
        assert turn.error is not None
        assert "stream ended" in turn.error
        assert turn.should_retire is True

    def test_dead_stream_short_circuit_retires(self):
        # The POISONED-SESSION half: a second turn on the same object
        # short-circuits pre-query — it must retire too, or the session
        # errors instantly forever.
        session, holder = _make_session(
            script=[AssistantMessage(content=[TextBlock("partial")])]
        )
        try:
            first = session.run_turn("hi", watch_poll_interval=0.02)
            second = session.run_turn("again", watch_poll_interval=0.02)
        finally:
            session.close()
        assert first.should_retire is True
        assert second.error is not None
        assert "stream ended before this turn" in second.error
        assert second.should_retire is True

    def test_model_level_subtypes_still_do_not_retire(self):
        # The retire widening is stream-death-only: error_max_turns is a
        # model-level outcome on a HEALTHY stream and stays non-retiring.
        session, holder = _make_session(
            script=[
                AssistantMessage(content=[TextBlock("partial work")]),
                ResultMessage(subtype="error_max_turns", is_error=False),
            ]
        )
        try:
            turn = session.run_turn("hi", watch_poll_interval=0.02)
        finally:
            session.close()
        assert turn.error == "SDK turn ended: error_max_turns"
        assert turn.should_retire is False


class TestTurnLifetimeConfig:
    def _patch_block(self, monkeypatch, block):
        import hermes_cli.config as cfg

        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": block}},
            raising=False,
        )

    def test_turn_timeout_reader_validation(self, monkeypatch):
        from agent.transports.claude_agent_sdk_session import (
            _configured_turn_timeout,
        )

        self._patch_block(monkeypatch, {"turn_timeout": 1500})
        assert _configured_turn_timeout() == 1500.0
        self._patch_block(monkeypatch, {"turn_timeout": "900"})
        assert _configured_turn_timeout() == 900.0
        # 0 = unlimited does NOT exist for the budget; bools are not seconds;
        # garbage and negatives fall back — all with a warning.
        for bad in (0, -5, True, False, "plenty", [600]):
            self._patch_block(monkeypatch, {"turn_timeout": bad})
            assert _configured_turn_timeout() is None
        self._patch_block(monkeypatch, {})
        assert _configured_turn_timeout() is None

    def test_post_tool_quiet_reader_validation(self, monkeypatch):
        from agent.transports.claude_agent_sdk_session import (
            _configured_post_tool_quiet_timeout,
        )

        self._patch_block(monkeypatch, {"post_tool_quiet_timeout": 120})
        assert _configured_post_tool_quiet_timeout() == 120.0
        # 0 = explicitly disabled IS a valid value for the quiet watchdog.
        self._patch_block(monkeypatch, {"post_tool_quiet_timeout": 0})
        assert _configured_post_tool_quiet_timeout() == 0.0
        for bad in (-1, True, "off"):
            self._patch_block(monkeypatch, {"post_tool_quiet_timeout": bad})
            assert _configured_post_tool_quiet_timeout() is None

    def test_configured_turn_timeout_reaches_the_watchdog(self, monkeypatch):
        # End-to-end: config.yaml (not the signature default) is what the
        # budget rule enforces. A 0.2s configured budget kills a silent turn
        # fast even though run_turn was called with no explicit timeout.
        self._patch_block(monkeypatch, {"turn_timeout": 0.2})
        session, holder = _make_hold_open_session(script=[])
        try:
            turn = session.run_turn(
                "hi", watch_poll_interval=0.05, abort_grace=0.2
            )
        finally:
            session.close()
        assert turn.error is not None
        assert "turn timed out after" in turn.error

    def test_configured_quiet_reaches_the_watchdog(self, monkeypatch):
        # End-to-end for the second knob: with streaming OFF the quiet
        # watchdog defaults to disabled — a configured value must still
        # reach and arm it.
        self._patch_block(monkeypatch, {"post_tool_quiet_timeout": 0.2})
        session, holder = _make_hold_open_session(
            script=[
                AssistantMessage(
                    content=[ToolUseBlock(id="t1", name="Grep", input={})]
                ),
                UserMessage(
                    content=[ToolResultBlock(tool_use_id="t1", content="x")]
                ),
            ],
            interrupt_ack=[_ede_interrupt_ack()],
        )
        try:
            turn = session.run_turn(
                "hi", turn_timeout=30.0, watch_poll_interval=0.05
            )
        finally:
            session.close()
        assert turn.error is not None
        assert "after a tool result" in turn.error

    def test_turnwatch_check_semantics(self, monkeypatch):
        # Deterministic unit coverage of the verdict rules (no threads).
        # Module-LOCAL time shadow — patching stdlib time.monotonic
        # process-wide would freeze asyncio loop clocks in concurrent tests.
        from agent.transports import claude_agent_sdk_session as session_mod

        clock = {"now": 1000.0}
        monkeypatch.setattr(
            session_mod, "time", SimpleNamespace(monotonic=lambda: clock["now"])
        )
        watch = session_mod._TurnWatch()
        # Budget: needs elapsed >= budget AND idle >= min(30, budget).
        clock["now"] += 599.0
        assert watch.check(budget=600.0, quiet=0.0) is None
        clock["now"] += 2.0  # elapsed 601, idle 601
        assert watch.check(budget=600.0, quiet=0.0) == "budget"
        watch.tick()  # activity: idle 0 — over budget but alive
        assert watch.check(budget=600.0, quiet=0.0) is None
        # Outstanding tool suspends everything.
        clock["now"] += 700.0
        watch.note_tools_issued(1)
        assert watch.check(budget=600.0, quiet=0.0) is None
        watch.note_tools_resolved(1)
        assert watch.check(budget=600.0, quiet=0.0) == "budget"
        # Pending approval suspends everything.
        watch.approval_begin()
        clock["now"] += 700.0
        assert watch.check(budget=600.0, quiet=0.0) is None
        watch.approval_end()  # ticks: idle resets
        assert watch.check(budget=600.0, quiet=0.0) is None
        # Post-tool quiet: armed + idle >= quiet, before the budget.
        watch.arm_post_tool()
        clock["now"] += 91.0
        assert watch.check(budget=60000.0, quiet=90.0) == "post_tool_quiet"
        # Disabled quiet (0) never fires the post-tool rule.
        assert watch.check(budget=60000.0, quiet=0.0) is None
        watch.disarm_post_tool()
        assert watch.check(budget=60000.0, quiet=90.0) is None
        # Rebaseline absorbs a process stall.
        watch.arm_post_tool()
        clock["now"] += 500.0
        watch.rebaseline()
        assert watch.check(budget=60000.0, quiet=90.0) is None
