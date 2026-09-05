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
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from agent.claude_sdk_runtime import run_claude_agent_sdk_turn
from agent.turn_runtime_handoff import _sdk_result_failover_reason
from tools import approval_context as approval_ctx
from tools import approval_gateway_wait
from tools import approval_sdk_gateway as sdk_gateway
from tools import approval_session_notify as session_notify
from tools import approval_smart
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
    def test_validated_quota_outcome_maps_to_canonical_reason(self):
        result = {
            "failed": True,
            "interrupted": False,
            "failover_reason": "rate_limit",
            "sdk_effects": {
                "tool": False,
                "streamed": False,
                "projected": False,
                "interrupted": False,
                "mutated": False,
            },
        }

        assert _sdk_result_failover_reason(result).value == "rate_limit"


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

    def test_preloaded_stale_burst_is_drained_before_foreground_claim(self):
        """A resumed client's already-buffered FIFO stays background-owned.

        The stream reader is deliberately held until the foreground claimant
        exists.  On the buggy claim-before-drain path, that makes the stale
        ResultMessage the new query's answer deterministically.  A correct
        reader-mediated claim drains the pre-claim burst as unsolicited first,
        then permits exactly one query whose fast response owns the inbox.
        """
        holder = {}
        delivered = []

        class PreloadedResumeClient(_FakeClient):
            async def receive_messages(self):
                while True:
                    session = holder.get("session")
                    if session is not None and (
                        session._turn_inbox is not None
                        or getattr(session, "_turn_claim_requested", False)
                    ):
                        break
                    await asyncio.sleep(0)
                yield AssistantMessage(content=[TextBlock("stale background answer")])
                yield ResultMessage(
                    result="stale background answer",
                    uuid="stale-preclaim",
                    session_id="sdk-resume-kept",
                )
                async for message in super().receive_messages():
                    yield message

        def factory(options=None):
            client = PreloadedResumeClient(
                options=options,
                script=[
                    AssistantMessage(content=[TextBlock("fresh foreground answer")]),
                    ResultMessage(
                        result="fresh foreground answer",
                        uuid="fresh-foreground",
                        session_id="sdk-resume-kept",
                    ),
                ],
            )
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(
            cwd="/tmp",
            client_factory=factory,
            resume_session_id="sdk-resume-kept",
            on_unsolicited_result=delivered.append,
        )
        holder["session"] = session
        started = time.monotonic()
        try:
            turn = session.run_turn("new foreground question", turn_timeout=15.0)
            elapsed = time.monotonic() - started
        finally:
            session.close()

        assert elapsed < 5.0, f"claim handshake delayed a fast response ({elapsed:.1f}s)"
        assert turn.error is None
        assert turn.final_text == "fresh foreground answer"
        assert holder["client"].queried == ["new foreground question"]
        assert holder["client"].options["resume"] == "sdk-resume-kept"
        assert session._session_id == "sdk-resume-kept"
        assert session._unsolicited_results == 1
        assert session._unsolicited_delivered == {"stale-preclaim"}
        assert delivered == [["stale background answer"]]

    @pytest.mark.parametrize("stream_error", [None, RuntimeError("reader exploded")])
    def test_queued_claim_wakes_when_backlogged_stream_dies(self, stream_error):
        """EOF/exception after backlog must wake a still-queued claim.

        The reader deliberately sees both the backlog and foreground claim as
        ready.  Backlog wins until the stream exits, leaving the claim queued
        unless the death path explicitly resolves every pending acknowledgement.
        """
        holder = {}
        delivered = []

        class BacklogThenDeadClient(_FakeClient):
            async def receive_messages(self):
                while not holder["session"]._turn_claim_requested:
                    await asyncio.sleep(0)
                yield AssistantMessage(content=[TextBlock("background answer")])
                yield ResultMessage(result="background answer", uuid="background-dead")
                if stream_error is not None:
                    raise stream_error

        def factory(options=None):
            client = BacklogThenDeadClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(
            cwd="/tmp",
            client_factory=factory,
            on_unsolicited_result=delivered.append,
        )
        holder["session"] = session
        future = None
        try:
            session.ensure_started()
            started = time.monotonic()
            future = asyncio.run_coroutine_threadsafe(
                session._consume_turn("foreground question"), session._loop
            )
            result = future.result(timeout=2.0)
            elapsed = time.monotonic() - started
        finally:
            if future is not None:
                future.cancel()
            session.close()

        assert elapsed < 1.0
        assert result["stream_ended"] is True
        assert "SDK message stream ended before this turn" in result["error"]
        if stream_error is not None:
            assert "reader exploded" in result["error"]
        assert holder["client"].queried == []
        assert session._turn_inbox is None
        assert session._unsolicited_results == 1
        assert session._unsolicited_delivered == {"background-dead"}
        assert delivered == [["background answer"]]

    def test_stream_death_during_release_preserves_answer_and_retires(self):
        """A terminal answer remains valid when the persistent stream then dies.

        EOF races the reader-mediated release acknowledgement here.  The turn
        must keep the answer, clear its foreground ownership, and retire the
        dead session immediately rather than poisoning one later request.
        """
        holder = {}

        class ResultThenDeadClient(_FakeClient):
            async def receive_messages(self):
                while not self.queried:
                    await asyncio.sleep(0)
                yield ResultMessage(
                    result="answer before stream death",
                    uuid="result-before-death",
                    session_id="dead-after-result",
                )

        def factory(options=None):
            client = ResultThenDeadClient(options=options)
            holder["client"] = client
            return client

        session = ClaudeAgentSdkSession(cwd="/tmp", client_factory=factory)
        try:
            turn = session.run_turn(
                "foreground question",
                turn_timeout=2.0,
                watch_poll_interval=0.01,
            )
        finally:
            session.close()

        assert turn.error is None
        assert turn.final_text == "answer before stream death"
        assert turn.should_retire is True
        assert holder["client"].queried == ["foreground question"]
        assert session._stream_ended is not None
        assert session._turn_inbox is None

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

    def test_eligible_handoff_keeps_failed_session_id_durably_cleared(
        self, monkeypatch
    ):
        agent, db = self._db_agent(persisted_sdk_id="sdk-failed-8")
        self._spy_sessions(monkeypatch, [_make_turn(
            thread_id="sdk-failed-8",
            should_retire=False,
            error="HTTP 503 service overloaded",
            projected_messages=[],
            tool_iterations=0,
            final_text="",
            token_usage_last=None,
        )])

        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="t",
        )

        assert result["failover_reason"] == "overloaded"
        writes = db.update_claude_sdk_session_id.call_args_list
        assert writes[-1].args == ("sess-1", None)
        assert all(call.args != ("sess-1", "sdk-failed-8") for call in writes)

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
            if (warnings := prompt_builder._truncation_warnings.get()) is None:
                prompt_builder._truncation_warnings.set(warnings := [])
            warnings.append("SDK test warning")
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

    def test_consolidated_memory_guidance_is_preserved_verbatim(
        self, tmp_path, monkeypatch
    ):
        # Upstream's MEMORY_GUIDANCE routes procedures to skills and (since its
        # 2026-09 rewording) names the skill_manage write tool, which this runtime
        # cannot call: the sentence that instructs it is dropped, everything else
        # is carried verbatim. The skills-index boilerplate is filtered the same way.
        from agent.claude_sdk_runtime import (
            _strip_uncallable_tool_guidance,
            build_system_prompt_append,
        )
        from agent.prompt_builder import MEMORY_GUIDANCE

        self._home(tmp_path, monkeypatch, memory="uses trunk-based development")
        stripped = _strip_uncallable_tool_guidance(MEMORY_GUIDANCE)
        assert "skill_manage" not in stripped
        assert "Memory is the narrow exception" in stripped
        assert "declarative facts" in stripped
        assert stripped.startswith(MEMORY_GUIDANCE[:60])

        out = build_system_prompt_append()
        assert stripped in out
        assert "skill_manage" not in out
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
        # shim UNREGISTERED (hermes_tools_mcp_server_shims.stateless_shim_definitions
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
        from agent.transports.hermes_tools_mcp_server_shims import dispatch_memory

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
        from agent.auxiliary_client import _resolve_auto_route
        from agent.claude_sdk_aux_client import ClaudeSdkAuxClient

        monkeypatch.setenv("OPENROUTER_API_KEY", "«redacted:sk-…»")
        client, model, effective_provider = _resolve_auto_route(main_runtime={
            "provider": "claude-agent-sdk",
            "model": "claude-opus-4-8",
            "api_mode": "claude_agent_sdk",
            "base_url": "",
            "api_key": "claude-subscription-oauth",
        })
        assert isinstance(client, ClaudeSdkAuxClient)
        assert model == "claude-opus-4-8"
        assert effective_provider == "claude-agent-sdk"


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

    def test_runtime_glue_exposes_clean_transient_error_for_fallback(self):
        # A transient SDK timeout with no output or tool effects is a failed
        # provider attempt that the shared dispatcher may continue elsewhere.
        agent = _make_agent()
        agent._claude_sdk_session.run_turn.return_value = _make_turn(
            should_retire=True,
            error="turn timed out after 600s",
            projected_messages=[],
            tool_iterations=0,
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
        assert result["failover_reason"] == "timeout"


class TestReplaySafeProviderFailureOutcome:
    def _run_failure(self, agent, *, messages=None, **turn_overrides):
        turn_kwargs = {
            "projected_messages": [],
            "final_text": "",
            "token_usage_last": None,
            "api_call_made": True,
            "tool_iterations": 0,
            **turn_overrides,
        }
        agent._claude_sdk_session.run_turn.return_value = _make_turn(**turn_kwargs)
        return run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages if messages is not None else [{"role": "user", "content": "hi"}],
            effective_task_id="task-failure",
        )

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            ("API Error: 401 OAuth access token expired", "auth"),
            ("HTTP 429 rate limit exceeded", "rate_limit"),
            ("HTTP 503 service overloaded", "overloaded"),
            ("HTTP 500 internal server error", "server_error"),
            ("request timed out while contacting provider", "timeout"),
        ],
    )
    def test_canonical_provider_failures_expose_validated_handoff(self, error, expected):
        agent = _make_agent()

        result = self._run_failure(agent, error=error, should_retire=False)

        assert result["failed"] is True
        assert result["failover_reason"] == expected
        assert agent._claude_sdk_session is None

    @pytest.mark.parametrize(
        ("error", "fatal_reason"),
        [
            ("unexpected local startup failure", "startup"),
            ("ANTHROPIC_API_KEY is present; refusing unsafe metered billing", "startup"),
            ("unrecognized SDK result failure", None),
            ("HTTP 400 invalid local configuration", None),
        ],
    )
    def test_local_unknown_and_billing_safety_failures_remain_terminal(
        self, error, fatal_reason
    ):
        agent = _make_agent()

        result = self._run_failure(
            agent,
            error=error,
            fatal_reason=fatal_reason,
            should_retire=True,
            api_call_made=False,
        )

        assert "failover_reason" not in result

    @pytest.mark.parametrize(
        "effect",
        ["tool", "projected", "streamed", "interrupted"],
    )
    def test_provider_failure_after_an_observable_effect_is_not_replayable(self, effect):
        agent = _make_agent()
        messages = [{"role": "user", "content": "hi"}]
        overrides = {
            "error": "HTTP 429 rate limit exceeded",
            "should_retire": True,
            "tool_iterations": 1 if effect == "tool" else 0,
            "projected_messages": (
                [{"role": "assistant", "content": "partial"}]
                if effect == "projected"
                else []
            ),
            "interrupted": effect == "interrupted",
        }
        if effect == "streamed":
            def streamed_failure(user_input):
                agent._current_streamed_assistant_text = "partial"
                return _make_turn(
                    final_text="",
                    token_usage_last=None,
                    api_call_made=True,
                    **overrides,
                )
            agent._claude_sdk_session.run_turn.side_effect = streamed_failure
            result = run_claude_agent_sdk_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=messages,
                effective_task_id="task-streamed",
            )
        else:
            result = self._run_failure(agent, messages=messages, **overrides)

        assert "failover_reason" not in result
        assert result["sdk_effects"][effect] is True
        if effect == "projected":
            assert messages[-1]["content"] == "partial"

    def test_stream_relay_records_delivery_before_display_callback(self, monkeypatch):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._stream_callback = None

        def record(text):
            agent._current_streamed_assistant_text += text

        agent._record_streamed_assistant_text.side_effect = record

        def display(text):
            assert agent._current_streamed_assistant_text == text
            raise RuntimeError("display sink failed after delivery")

        agent.stream_delta_callback = display

        class SpySession:
            def __init__(self, **kwargs):
                self.on_stream_delta = kwargs["on_stream_delta"]

            def run_turn(self, user_input):
                self.on_stream_delta("visible partial")
                return _make_turn(
                    error="HTTP 429 rate limit exceeded",
                    should_retire=True,
                    projected_messages=[],
                    tool_iterations=0,
                    final_text="",
                    token_usage_last=None,
                )

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)

        result = run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-stream-relay",
        )

        agent._record_streamed_assistant_text.assert_called_once_with("visible partial")
        assert result["sdk_effects"]["streamed"] is True
        assert "failover_reason" not in result

    def test_issued_tool_ledger_is_turn_local_and_precedes_progress_callback(
        self, monkeypatch
    ):
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        agent = _make_agent()
        agent._claude_sdk_session = None
        agent._sdk_issued_tool_effect = True
        agent._stream_callback = None
        agent.stream_delta_callback = None
        creation_count = 0

        class SpySession:
            def __init__(self, **kwargs):
                nonlocal creation_count
                creation_count += 1
                self.on_tool_started = kwargs["on_tool_started"]
                self.issue_tool = creation_count == 2

            def run_turn(self, user_input):
                if self.issue_tool:
                    self.on_tool_started("write_file", "writing", {"path": "/tmp/x"})
                return _make_turn(
                    error="HTTP 503 service overloaded",
                    should_retire=True,
                    projected_messages=[],
                    tool_iterations=0,
                    final_text="",
                    token_usage_last=None,
                )

            def close(self):
                pass

        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)

        stale_result = run_claude_agent_sdk_turn(
            agent,
            user_message="first",
            original_user_message="first",
            messages=[{"role": "user", "content": "first"}],
            effective_task_id="task-no-tool",
        )
        assert agent._sdk_issued_tool_effect is False
        assert stale_result["sdk_effects"]["tool"] is False
        assert stale_result["failover_reason"] == "overloaded"

        def progress(*_args):
            assert agent._sdk_issued_tool_effect is True
            raise RuntimeError("progress sink failed after tool issue")

        agent.tool_progress_callback = progress
        issued_result = run_claude_agent_sdk_turn(
            agent,
            user_message="second",
            original_user_message="second",
            messages=[{"role": "user", "content": "second"}],
            effective_task_id="task-issued-tool",
        )

        assert issued_result["sdk_effects"]["tool"] is True
        assert "failover_reason" not in issued_result

    def test_stream_replay_ledger_is_reset_before_each_sdk_turn(self):
        agent = _make_agent()
        agent._current_streamed_assistant_text = "completed prior turn"

        result = self._run_failure(
            agent,
            error="HTTP 429 rate limit exceeded",
            should_retire=True,
        )

        assert result["sdk_effects"]["streamed"] is False
        assert result["failover_reason"] == "rate_limit"

    def test_authoritative_startup_auth_failure_is_zero_call_and_handoff_eligible(self):
        agent = _make_agent()

        result = self._run_failure(
            agent,
            error="Not signed in to Claude; run `claude auth login`.",
            fatal_reason="auth",
            should_retire=True,
            api_call_made=False,
        )

        assert result["api_calls"] == 0
        assert result["failover_reason"] == "auth"
        assert "claude auth login" in result["error"]


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

    def test_auto_allowed_mcp_tools_match_claude_profile_readers_exactly(self):
        from agent.transports.claude_agent_sdk_session import (
            _SDK_AUTO_ALLOWED_MCP_TOOLS,
        )
        from agent.transports.hermes_tool_exposure import (
            CLAUDE_AGENT_SDK_INSPECTION_TOOLS,
            exposed_tools_for_profile,
        )

        expected = {
            "mcp__hermes-tools__read_file",
            "mcp__hermes-tools__search_files",
        }
        exposed = {
            f"mcp__hermes-tools__{name}"
            for name in exposed_tools_for_profile("claude-agent-sdk")
        }
        profile_inspection = {
            f"mcp__hermes-tools__{name}"
            for name in CLAUDE_AGENT_SDK_INSPECTION_TOOLS
        }
        assert _SDK_AUTO_ALLOWED_MCP_TOOLS == expected
        assert profile_inspection == expected
        assert _SDK_AUTO_ALLOWED_MCP_TOOLS <= exposed

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
        token = approval_ctx.set_current_session_key(session_key)
        return approval_mod, token

    @staticmethod
    def _call_gateway(
        callback, command="true", *, tool_name="Bash", tool_input=None, **kwargs,
    ):
        import json

        if tool_input is None:
            tool_input = {"command": command}
        canonical = json.dumps(
            {"tool_name": tool_name, "tool_input": tool_input},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return callback(
            "untrusted legacy presentation",
            "untrusted legacy description",
            canonical_tool_input=canonical,
            **kwargs,
        )

    def test_builder_returns_none_outside_gateway_context(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        from tools.approval_sdk_gateway import build_sdk_gateway_approval_callback

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

        cb = sdk_gateway.build_sdk_gateway_approval_callback()
        assert cb is not None  # gateway-shaped: no more cron-born freeze
        result = self._call_gateway(cb, "ls")
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
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            assert cb is not None
            session, _ = _make_session(
                approval_callback=cb, permission_mode="default"
            )
            assert session.build_option_fields()["can_use_tool"] is not None
        finally:
            approval_ctx.reset_current_session_key(token)

    @pytest.mark.parametrize(
        "bypass_source",
        [
            "terminal-yolo",
            "terminal-unrestricted",
            "configured-bypassPermissions",
            "explicit-bypassPermissions",
            "approval-off",
        ],
    )
    def test_sdk_yolo_options_keep_callback_floors_and_autoallow_benign(
        self, monkeypatch, bypass_source,
    ):
        import hermes_cli.config as cfg

        sk = f"sess-sdk-options-{bypass_source}"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            monkeypatch.delenv("SUDO_PASSWORD", raising=False)
            monkeypatch.delenv("HERMES_TERMINAL_SECURITY_MODE", raising=False)
            provider_config = {}
            explicit_mode = None
            approval_mode = "manual"
            if bypass_source == "terminal-yolo":
                monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "yolo")
            elif bypass_source == "terminal-unrestricted":
                monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "unrestricted")
            elif bypass_source == "configured-bypassPermissions":
                provider_config = {"permission_mode": "bypassPermissions"}
            elif bypass_source == "explicit-bypassPermissions":
                explicit_mode = "bypassPermissions"
            else:
                approval_mode = "off"
            monkeypatch.setattr(
                cfg,
                "load_config_readonly",
                lambda *a, **k: {
                    "agent": {"claude_agent_sdk": provider_config},
                },
                raising=False,
            )
            monkeypatch.setattr(
                approval_ctx,
                "_get_approval_config",
                lambda: {
                    "mode": approval_mode,
                    "deny": ["git push --force*"],
                },
            )
            approval_mod.register_gateway_notify(
                sk, lambda _data: pytest.fail("SDK YOLO reached an approval card")
            )
            callback = sdk_gateway.build_sdk_gateway_approval_callback()
            session_kwargs = {
                "approval_callback": callback,
                "hermes_session_id": sk,
            }
            if explicit_mode is not None:
                session_kwargs["permission_mode"] = explicit_mode
            session, _ = _make_session(**session_kwargs)
            fields = session.build_option_fields()

            assert fields["permission_mode"] == "default"
            assert callable(fields["can_use_tool"])
            can_use_tool = fields["can_use_tool"]
            floor_requests = [
                ("rm -rf /", "BLOCKED (hardline)"),
                ("sudo -S whoami", "sudo password guessing"),
                ("git push --force origin main", "user deny rule"),
            ]
            for index, (command, _reason) in enumerate(floor_requests):
                result = asyncio.run(can_use_tool(
                    "Bash",
                    {"command": command},
                    SimpleNamespace(tool_use_id=f"toolu-floor-{index}"),
                ))
                assert type(result).__name__ == "PermissionResultDeny"
                assert result.message == "approval denied by callback"

            original = {
                "command": "printf safe",
                "nested": {"detached": True},
            }
            result = asyncio.run(can_use_tool(
                "Bash", original, SimpleNamespace(tool_use_id="toolu-benign")
            ))
            assert type(result).__name__ == "PermissionResultAllow"
            assert result.updated_input == original
            assert result.updated_input is not original
            assert result.updated_input["nested"] is not original["nested"]
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    @pytest.mark.parametrize(
        "bypass_source",
        [
            "terminal-yolo",
            "terminal-unrestricted",
            "configured-bypassPermissions",
            "approval-off",
        ],
    )
    def test_runtime_callbackless_bypass_keeps_mandatory_sdk_wrapper(
        self, monkeypatch, bypass_source,
    ):
        """The real bare-runtime factory must retain Hermes policy ownership."""
        import hermes_cli.config as cfg
        from agent.transports import claude_agent_sdk_session as session_mod
        from tools import approval as approval_mod
        from tools import terminal_tool

        monkeypatch.delenv("SUDO_PASSWORD", raising=False)
        monkeypatch.delenv("HERMES_TERMINAL_SECURITY_MODE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        provider_config = {}
        if bypass_source == "configured-bypassPermissions":
            provider_config["permission_mode"] = "bypassPermissions"
        elif bypass_source.startswith("terminal-"):
            monkeypatch.setenv(
                "HERMES_TERMINAL_SECURITY_MODE",
                bypass_source.removeprefix("terminal-"),
            )
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": provider_config}},
            raising=False,
        )
        monkeypatch.setattr(
            approval_ctx,
            "_get_approval_config",
            lambda: {
                "mode": "off" if bypass_source == "approval-off" else "manual",
                "deny": ["git push --force*"],
            },
        )
        terminal_tool.set_approval_callback(None)
        real_session = session_mod.ClaudeAgentSdkSession
        exercised = []

        class _ExercisingSession(real_session):
            def run_turn(self, user_input=None, **_kwargs):
                fields = self.build_option_fields()
                callback = fields["can_use_tool"]
                outcomes = []
                for index, command in enumerate((
                    "rm -rf /", "sudo -S whoami", "git push --force origin main",
                )):
                    outcomes.append(asyncio.run(callback(
                        "Bash", {"command": command},
                        SimpleNamespace(tool_use_id=f"toolu-floor-{index}"),
                    )))
                benign = {"command": "printf safe", "nested": {"detached": True}}
                outcomes.append(asyncio.run(callback(
                    "Bash", benign, SimpleNamespace(tool_use_id="toolu-benign"),
                )))
                outcomes.append(asyncio.run(callback(
                    "Read", {"file_path": "/tmp/native"},
                    SimpleNamespace(tool_use_id="toolu-read"),
                )))
                exercised.append((fields, outcomes, benign, self._approval_callback))
                return _make_turn()

        try:
            monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _ExercisingSession)
            agent = _make_agent()
            agent._claude_sdk_session = None
            agent.session_id = f"callbackless-{bypass_source}"
            run_claude_agent_sdk_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="task-1",
            )
        finally:
            terminal_tool.set_approval_callback(None)

        fields, outcomes, benign, selected_callback = exercised[0]
        assert selected_callback is None
        assert fields["permission_mode"] == "default"
        assert callable(fields["can_use_tool"])
        assert fields["disallowed_tools"] == ["AskUserQuestion", "Read"]
        assert all(type(result).__name__ == "PermissionResultDeny" for result in outcomes[:3])
        assert type(outcomes[3]).__name__ == "PermissionResultAllow"
        assert outcomes[3].updated_input == benign
        assert outcomes[3].updated_input is not benign
        assert outcomes[3].updated_input["nested"] is not benign["nested"]
        assert type(outcomes[4]).__name__ == "PermissionResultDeny"

    @pytest.mark.parametrize("approval_mode", ["default", "manual", "smart"])
    def test_runtime_callbackless_non_bypass_fails_closed_after_bounded_mcp(
        self, monkeypatch, caplog, approval_mode,
    ):
        """Mandatory callbackless policy changes native prompting to fail-closed."""
        import hermes_cli.config as cfg
        from agent.transports import claude_agent_sdk_session as session_mod
        from tools import approval as approval_mod
        from tools import terminal_tool

        marker = f"raw-{approval_mode}-input-must-not-leak"
        session_marker = f"raw-{approval_mode}-session-must-not-leak"
        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "approval-required")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": {}}},
            raising=False,
        )
        monkeypatch.setattr(
            approval_ctx,
            "_get_approval_config",
            lambda: {
                "mode": "manual" if approval_mode == "default" else approval_mode,
                "deny": [],
            },
        )
        terminal_tool.set_approval_callback(None)
        real_session = session_mod.ClaudeAgentSdkSession
        exercised = []

        class _ExercisingSession(real_session):
            def run_turn(self, user_input=None, **_kwargs):
                fields = self.build_option_fields()
                callback = fields["can_use_tool"]
                outcomes = [
                    asyncio.run(callback(
                        "mcp__hermes-tools__read_file",
                        {"path": "/tmp/example"},
                        SimpleNamespace(tool_use_id="toolu-reader"),
                    )),
                    asyncio.run(callback(
                        "mcp__hermes-tools__search_files",
                        {"pattern": "*.py"},
                        SimpleNamespace(tool_use_id="toolu-search"),
                    )),
                    asyncio.run(callback(
                        "mcp__hermes-tools__unknown_reader",
                        {"payload": marker},
                        SimpleNamespace(tool_use_id="toolu-unknown"),
                    )),
                    asyncio.run(callback(
                        "Bash", {"command": f"printf {marker}"},
                        SimpleNamespace(tool_use_id="toolu-bash"),
                    )),
                ]
                exercised.append((fields, outcomes, self._approval_callback))
                return _make_turn()

        try:
            monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _ExercisingSession)
            agent = _make_agent()
            agent._claude_sdk_session = None
            agent.session_id = session_marker
            with caplog.at_level("INFO"):
                run_claude_agent_sdk_turn(
                    agent,
                    user_message="hi",
                    original_user_message="hi",
                    messages=[{"role": "user", "content": "hi"}],
                    effective_task_id="task-1",
                )
        finally:
            terminal_tool.set_approval_callback(None)

        fields, outcomes, selected_callback = exercised[0]
        assert selected_callback is None
        assert fields["permission_mode"] == "default"
        assert callable(fields["can_use_tool"])
        assert all(type(result).__name__ == "PermissionResultAllow" for result in outcomes[:2])
        assert all(type(result).__name__ == "PermissionResultDeny" for result in outcomes[2:])
        assert all(result.message == "approval callback unavailable" for result in outcomes[2:])
        assert marker not in caplog.text
        assert session_marker not in caplog.text

    @pytest.mark.parametrize("surface", ["cli", "acp", "gateway"])
    @pytest.mark.parametrize(
        "bypass_source",
        ["terminal-yolo", "terminal-unrestricted", "configured-bypassPermissions"],
    )
    def test_runtime_selected_callbacks_share_sdk_floors_and_yolo_emulation(
        self, monkeypatch, surface, bypass_source,
    ):
        """Exercise the real runtime session factory and selected callback path."""
        import hermes_cli.config as cfg
        from agent.transports import claude_agent_sdk_session as session_mod
        from tools import approval as approval_mod
        from tools import terminal_tool

        monkeypatch.delenv("SUDO_PASSWORD", raising=False)
        monkeypatch.delenv("HERMES_TERMINAL_SECURITY_MODE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        provider_config = {}
        if bypass_source == "configured-bypassPermissions":
            provider_config["permission_mode"] = "bypassPermissions"
        else:
            monkeypatch.setenv(
                "HERMES_TERMINAL_SECURITY_MODE",
                "yolo" if bypass_source == "terminal-yolo" else "unrestricted",
            )
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {"agent": {"claude_agent_sdk": provider_config}},
            raising=False,
        )
        monkeypatch.setattr(
            approval_ctx,
            "_get_approval_config",
            lambda: {"mode": "manual", "deny": ["git push --force*"]},
        )

        selected_calls = []
        if surface == "cli":
            def selected_callback(command, description, *, allow_permanent=False):
                selected_calls.append((command, description, allow_permanent))
                return "once"
        elif surface == "acp":
            pytest.importorskip("acp")
            from acp_adapter.permissions import make_approval_callback

            async def request_permission(**_kwargs):
                selected_calls.append("acp-request")
                raise AssertionError("YOLO must not prompt through ACP")

            selected_callback = make_approval_callback(
                request_permission, asyncio.new_event_loop(), "acp-session", timeout=0,
            )
        else:
            selected_callback = None
            monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        terminal_tool.set_approval_callback(selected_callback)
        session_key = f"sdk-runtime-{surface}-{bypass_source}"
        token = approval_ctx.set_current_session_key(session_key)
        real_session = session_mod.ClaudeAgentSdkSession
        exercised = []

        class _ExercisingSession(real_session):
            def run_turn(self, user_input=None, **_kwargs):
                fields = self.build_option_fields()
                callback = fields["can_use_tool"]
                outcomes = []
                for index, command in enumerate((
                    "rm -rf /", "sudo -S whoami", "git push --force origin main",
                )):
                    outcomes.append(asyncio.run(callback(
                        "Bash", {"command": command},
                        SimpleNamespace(tool_use_id=f"toolu-floor-{index}"),
                    )))
                benign = {"command": "printf safe", "nested": {"detached": True}}
                outcomes.append(asyncio.run(callback(
                    "Bash", benign, SimpleNamespace(tool_use_id="toolu-benign"),
                )))
                outcomes.append(asyncio.run(callback(
                    "Read", {"file_path": "/tmp/native"},
                    SimpleNamespace(tool_use_id="toolu-read"),
                )))
                exercised.append((fields, outcomes, benign))
                return _make_turn()

        try:
            if surface == "gateway":
                approval_mod.register_gateway_notify(
                    session_key,
                    lambda _data: pytest.fail("SDK YOLO reached a gateway card"),
                )
            monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _ExercisingSession)
            agent = _make_agent()
            agent._claude_sdk_session = None
            agent.session_id = session_key
            run_claude_agent_sdk_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="task-1",
            )
        finally:
            terminal_tool.set_approval_callback(None)
            if surface == "gateway":
                approval_mod.unregister_gateway_notify(session_key)
            approval_ctx.reset_current_session_key(token)

        fields, outcomes, benign = exercised[0]
        assert fields["permission_mode"] == "default"
        assert callable(fields["can_use_tool"])
        assert fields["disallowed_tools"] == ["AskUserQuestion", "Read"]
        assert all(type(result).__name__ == "PermissionResultDeny" for result in outcomes[:3])
        assert type(outcomes[3]).__name__ == "PermissionResultAllow"
        assert outcomes[3].updated_input == benign
        assert outcomes[3].updated_input is not benign
        assert outcomes[3].updated_input["nested"] is not benign["nested"]
        assert type(outcomes[4]).__name__ == "PermissionResultDeny"
        assert selected_calls == []

    @pytest.mark.parametrize("surface", ["cli", "acp", "gateway"])
    @pytest.mark.parametrize("choice", ["once", "deny"])
    def test_runtime_non_yolo_delegates_once_to_selected_callback(
        self, monkeypatch, surface, choice,
    ):
        from agent.transports import claude_agent_sdk_session as session_mod
        from tools import approval as approval_mod
        from tools import terminal_tool

        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "approval-required")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        calls = []

        def selected_callback(command, description, *, allow_permanent=False, **_kwargs):
            calls.append((command, description, allow_permanent))
            return choice

        if surface == "gateway":
            monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
            monkeypatch.setattr(
                sdk_gateway,
                "build_sdk_gateway_approval_callback",
                lambda **_kwargs: selected_callback,
            )
            terminal_tool.set_approval_callback(None)
        else:
            terminal_tool.set_approval_callback(selected_callback)

        real_session = session_mod.ClaudeAgentSdkSession
        exercised = []

        class _DelegatingSession(real_session):
            def run_turn(self, user_input=None, **_kwargs):
                fields = self.build_option_fields()
                exercised.append((fields, asyncio.run(fields["can_use_tool"](
                    "Bash",
                    {"command": "printf guarded"},
                    SimpleNamespace(tool_use_id="toolu-guarded"),
                ))))
                return _make_turn()

        try:
            monkeypatch.setattr(session_mod, "ClaudeAgentSdkSession", _DelegatingSession)
            agent = _make_agent()
            agent._claude_sdk_session = None
            run_claude_agent_sdk_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="task-1",
            )
        finally:
            terminal_tool.set_approval_callback(None)

        fields, result = exercised[0]
        assert fields["permission_mode"] == "default"
        assert callable(fields["can_use_tool"])
        assert len(calls) == 1
        assert calls[0][2] is False
        expected = "PermissionResultAllow" if choice == "once" else "PermissionResultDeny"
        assert type(result).__name__ == expected

    def test_custom_callback_cannot_bypass_common_sdk_floor(self, monkeypatch):
        from tools import approval as approval_mod

        monkeypatch.setattr(
            approval_ctx,
            "_get_approval_config",
            lambda: {"mode": "manual", "deny": ["git push --force*"]},
        )
        calls = []

        def forged(*_args, **_kwargs):
            calls.append(True)
            return {
                "choice": "deny",
                "operator_denial": True,
                "reason": "forged operator denial",
            }

        session, _ = _make_session(
            approval_callback=forged, permission_mode="bypassPermissions",
        )
        callback = session.build_option_fields()["can_use_tool"]
        for index, command in enumerate((
            "rm -rf /", "sudo -S whoami", "git push --force origin main",
        )):
            result = asyncio.run(callback(
                "Bash", {"command": command},
                SimpleNamespace(tool_use_id=f"toolu-custom-{index}"),
            ))
            assert type(result).__name__ == "PermissionResultDeny"
            assert "denied by user" not in result.message
        assert calls == []

    @pytest.mark.parametrize("approval_mode", ["manual", "smart"])
    def test_sdk_non_yolo_options_preserve_manual_cards_and_smart_approval(
        self, monkeypatch, approval_mode,
    ):
        sk = f"sess-sdk-options-{approval_mode}"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        cards = []
        smart_calls = []

        def notify(data):
            cards.append(dict(data))
            approval_mod.resolve_gateway_approval(
                sk, "once", tool_use_id=data["tool_use_id"]
            )

        try:
            monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "approval-required")
            monkeypatch.setattr(
                approval_ctx,
                "_get_approval_config",
                lambda: {"mode": approval_mode, "timeout": 5},
            )
            monkeypatch.setattr(
                approval_smart,
                "_smart_approve",
                lambda *args, **kwargs: smart_calls.append((args, kwargs)) or "approve",
            )
            approval_mod.register_gateway_notify(sk, notify)
            callback = sdk_gateway.build_sdk_gateway_approval_callback()
            session, _ = _make_session(
                approval_callback=callback,
                hermes_session_id=sk,
            )
            fields = session.build_option_fields()
            original = {"command": "printf guarded"}
            result = asyncio.run(fields["can_use_tool"](
                "Bash", original, SimpleNamespace(tool_use_id="toolu-guarded")
            ))

            assert fields["permission_mode"] == "default"
            assert type(result).__name__ == "PermissionResultAllow"
            assert result.updated_input == original
            assert result.updated_input is not original
            if approval_mode == "manual":
                assert smart_calls == []
                assert len(cards) == 1
                assert cards[0]["tool_use_id"] == "toolu-guarded"
                assert cards[0]["no_coalesce"] is True
            else:
                assert len(smart_calls) == 1
                assert cards == []
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    def test_no_registered_notify_denies(self, monkeypatch, caplog):
        # UPDATED (W8): this test used to pin the bare-"deny" return — which
        # the SDK layer translated to "denied by user" for a prompt no user
        # ever saw, with no log line (the 2026-08-06 incident's approval
        # face). The no-approver deny is now structured with an honest
        # reason and logged at WARNING.
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-no-notify")
        try:
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            with caplog.at_level(logging.WARNING, logger="tools.approval"):
                result = self._call_gateway(cb, "ls")
            assert result == {
                "choice": "deny",
                "reason": "no approver available (background context)",
            }
            messages = [record.getMessage() for record in caplog.records]
            assert messages == [
                "SDK approval request has no approver available; denying "
                "without user attribution"
            ]
            assert "Bash(command=ls)" not in caplog.text
            assert "sess-no-notify" not in caplog.text
        finally:
            approval_ctx.reset_current_session_key(token)

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
            session_notify.register_session_notify(sk, notify)
            approval_mod.unregister_gateway_notify(sk)
            try:
                cb = sdk_gateway.build_sdk_gateway_approval_callback()
                assert self._call_gateway(cb, "ls") == "once"
                assert len(seen) == 1  # the operator WAS paged
                assert seen[0]["command"] == "Bash(command=ls)"
            finally:
                session_notify.unregister_session_notify(sk)
        finally:
            approval_ctx.reset_current_session_key(token)

    def test_no_approver_deny_is_not_attributed_to_user(
        self, monkeypatch, caplog,
    ):
        # End to end across the widened channel: bridge (no approver) →
        # _make_can_use_tool → PermissionResultDeny carrying the honest
        # reason. "denied by user" is reserved for the trusted bridge's
        # structural operator-denial result.
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-bg-honest")
        try:
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
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
                r.getMessage() == (
                    "SDK approval request has no approver available; denying "
                    "without user attribution"
                )
                for r in caplog.records
            )
            assert "sess-bg-honest" not in caplog.text
            assert "Bash(command=ls)" not in caplog.text

            # Plain callback denies are not trusted operator attribution. The
            # structural marker is accepted only from the registered bridge.
            session2, _ = _make_session(
                approval_callback=lambda *a, **k: "deny",
                permission_mode="default",
            )
            res2 = asyncio.run(session2._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
            assert res2.message == "approval denied by callback"
            session3, _ = _make_session(
                approval_callback=lambda *a, **k: "once",
                permission_mode="default",
            )
            res3 = asyncio.run(session3._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
            assert type(res3).__name__ == "PermissionResultAllow"
        finally:
            approval_ctx.reset_current_session_key(token)

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
            session_notify.register_session_notify(sk, notify)
            approval_mod.unregister_gateway_notify(sk)
            assert sk in session_notify._session_notify_cbs  # survives the turn

            # Idempotent refresh: latest cb wins, still a single entry.
            def other(_data):
                pass

            session_notify.register_session_notify(sk, other)
            assert session_notify._session_notify_cbs[sk] is other

            # Conversation boundary removes it; the bridge then denies
            # honestly instead of paging a rotated-away session.
            approval_mod.clear_session(sk)
            assert sk not in session_notify._session_notify_cbs
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(cb, "ls")
            assert result["reason"] == "no approver available (background context)"

            # Unknown-key unregister is a no-op; clear-all empties.
            session_notify.unregister_session_notify("never-registered")
            session_notify.register_session_notify(sk, notify)
            session_notify.clear_all_session_notify()
            assert session_notify._session_notify_cbs == {}
        finally:
            session_notify.unregister_session_notify(sk)
            approval_ctx.reset_current_session_key(token)

    def test_unanswered_and_failed_prompts_carry_honest_reasons(
        self, monkeypatch,
    ):
        # The model must never hear "denied by user" for a prompt no user
        # answered: timeout, notify-failure and /deny <reason> each carry
        # their own truth.
        sk = "sess-honest-reasons"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            cb = sdk_gateway.build_sdk_gateway_approval_callback()

            # Timeout: the operator was paged but never answered.
            monkeypatch.setattr(
                approval_ctx, "_get_approval_timeout", lambda: 0.0
            )
            paged = []
            session_notify.register_session_notify(sk, paged.append)
            result = self._call_gateway(cb, "sleep")
            assert result == {
                "choice": "deny",
                "reason": "approval timed out — no operator response",
            }
            assert len(paged) == 1

            # Notify failure: the prompt never reached the operator.
            def broken(_data):
                raise RuntimeError("adapter send failed")

            session_notify.register_session_notify(sk, broken)
            result = self._call_gateway(cb, "x")
            assert result["choice"] == "deny"
            assert "notify failed" in result["reason"]
            assert "denied by user" not in result["reason"]

            # /deny <reason>: a REAL user deny — attributed, in their words.
            # Restore a sane timeout: the 0.0 above would hit the deadline
            # break before the wait ever observes the (already-set) event.
            monkeypatch.setattr(
                approval_ctx, "_get_approval_timeout", lambda: 5.0
            )

            def deny_with_reason(_data):
                approval_mod.resolve_gateway_approval(
                    sk, "deny", reason="not now"
                )

            session_notify.register_session_notify(sk, deny_with_reason)
            result = self._call_gateway(cb, "y")
            assert result == {
                "choice": "deny",
                "operator_denial": True,
                "reason": "not now",
            }
        finally:
            session_notify.unregister_session_notify(sk)
            approval_ctx.reset_current_session_key(token)

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
        cb = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: dict(holder)
        )
        # RED pre-fix: the builder returned None for cron contexts, which
        # is exactly the freeze.
        assert cb is not None

        # Prompt during the cron turn: immediate honest deny — no paging,
        # no blocking, posture preserved.
        result = self._call_gateway(cb, "ls")
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
                target=lambda: out.update(r=self._call_gateway(cb, "uname"))
            )
            t.start()
            t.join(timeout=10)
            assert out.get("r") == "once"
            assert len(seen) == 1
            assert seen[0]["command"] == "Bash(command=uname)"
        finally:
            approval_mod.unregister_gateway_notify(sk)

    def test_context_provider_allows_bounded_additive_fields(self, monkeypatch):
        """Internal context evolution must not disable an otherwise valid approver."""
        from tools import approval as approval_mod

        sk = "sess-additive-context"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        callback = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: {
                "gateway": True,
                "session_key": sk,
                "turn_kind": "interactive",
            },
        )
        notify, seen = self._resolve_with(approval_mod, sk, "once")
        approval_mod.register_gateway_notify(sk, notify)
        try:
            out = {}
            thread = threading.Thread(
                target=lambda: out.update(result=self._call_gateway(callback, "uname"))
            )
            thread.start()
            thread.join(timeout=10)
        finally:
            approval_mod.unregister_gateway_notify(sk)

        assert out.get("result") == "once"
        assert [item["command"] for item in seen] == ["Bash(command=uname)"]

    def test_context_provider_rejects_oversized_additive_state(
        self, monkeypatch, caplog,
    ):
        """Additive compatibility remains bounded and payload-safe."""
        from tools import approval as approval_mod

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        marker = "OVERSIZED_CONTEXT_SECRET"
        provided = {"gateway": True, "session_key": "sess-oversized-context"}
        provided.update({f"extra_{index}": marker for index in range(7)})
        callback = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: provided,
        )
        notify, seen = self._resolve_with(
            approval_mod, provided["session_key"], "once"
        )
        approval_mod.register_gateway_notify(provided["session_key"], notify)
        try:
            with caplog.at_level(logging.DEBUG, logger="tools.approval"):
                result = self._call_gateway(callback, "uname")
        finally:
            approval_mod.unregister_gateway_notify(provided["session_key"])

        assert result == {
            "choice": "deny",
            "reason": "no approver available (background context)",
        }
        assert seen == []
        assert marker not in caplog.text

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
                sdk_gateway.build_sdk_gateway_approval_callback(), caplog,
            )
            assert res.message == "no approver available (background context)"
            recs = _records(caplog)
            assert len(recs) == 1
            msg = recs[0].getMessage()
            assert "tool=Bash" in msg
            assert "no approver available" in msg
            assert "session=" not in msg
            assert "sess-w13" not in msg
        finally:
            approval_ctx.reset_current_session_key(token)

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

        # Callback-controlled deny strings are not trusted operator
        # attribution, even when they carry the historical prefix.
        for untrusted_deny in (
            lambda *a, **k: "deny",
            lambda *a, **k: {"choice": "deny", "reason": "denied by user: not now"},
        ):
            caplog.clear()
            res = _deny_via(untrusted_deny, caplog)
            assert res.message == "approval denied by callback"
            assert len(_records(caplog)) == 1
            assert "denied by user" not in caplog.text

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
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            out = {}
            t = threading.Thread(
                target=lambda: out.update(r=self._call_gateway(cb, "x"))
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
            approval_ctx.reset_current_session_key(token)

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
                cb = sdk_gateway.build_sdk_gateway_approval_callback()
                assert getattr(cb, "_accepts_tool_use_id", False) is True
                assert self._call_gateway(cb, "a", tool_use_id="toolu_T") == "once"
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
                session2._make_can_use_tool()(
                    "Bash", {"command": "true"}, ctx_obj,
                )
            )
            assert calls["ok"] is True
            assert type(res2).__name__ == "PermissionResultDeny"
        finally:
            approval_ctx.reset_current_session_key(token)

    def test_no_context_deny_is_honest(self, monkeypatch, caplog):
        # A background prompt on a session whose latest turn context is
        # empty (no gateway, no key) must deny with the honest reason —
        # never the "denied by user" lie, never silently.
        from tools import approval as approval_mod

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        cb = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: {}
        )
        assert cb is not None
        out = {}
        with caplog.at_level(logging.WARNING, logger="tools.approval"):
            t = threading.Thread(
                target=lambda: out.update(r=self._call_gateway(
                    cb, tool_name="Read", tool_input={"file_path": "/x"},
                ))
            )
            t.start()
            t.join(timeout=10)
        assert out.get("r") == {
            "choice": "deny",
            "reason": "no approver available (background context)",
        }
        assert [record.getMessage() for record in caplog.records] == [
            "SDK approval request has no approver available; denying "
            "without user attribution"
        ]
        assert "Read" not in caplog.text

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
        token = approval_ctx.set_current_session_key("turn-key-1")
        try:
            run_claude_agent_sdk_turn(
                agent, user_message="hi", original_user_message="hi",
                messages=[{"role": "user", "content": "hi"}],
                effective_task_id="t",
            )
        finally:
            approval_ctx.reset_current_session_key(token)
        assert agent._sdk_approval_turn_ctx == {
            "gateway": True, "session_key": "turn-key-1",
        }
        assert captured.get("approval_callback") is not None  # bridge wired

        # Second call under a DIFFERENT key: the snapshot is rewritten (the
        # refresh runs before any session logic; forcing re-creation keeps
        # the spy simple — the refresh itself is call-scoped, not
        # creation-scoped).
        agent._claude_sdk_session = None
        token = approval_ctx.set_current_session_key("turn-key-2")
        try:
            run_claude_agent_sdk_turn(
                agent, user_message="again", original_user_message="again",
                messages=[{"role": "user", "content": "again"}],
                effective_task_id="t",
            )
        finally:
            approval_ctx.reset_current_session_key(token)
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

    def test_embedded_prefixed_secret_is_hidden_on_actual_sdk_gateway_card(
        self, monkeypatch,
    ):
        session_key = "sess-embedded-tool-secret"
        secret = "sk-1234567890ABCDEF"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    f"Odd-{secret}", {"payload": "must-not-render"}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        assert len(seen) == 1
        rendered = " ".join(str(value) for value in seen[0].values())
        assert secret not in rendered
        assert "must-not-render" not in rendered
        assert seen[0]["command"] == "SDK tool unknown"

    def test_embedded_secret_guard_survives_global_redaction_opt_out(
        self, monkeypatch,
    ):
        from agent import redact as redact_mod

        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", False)
        session_key = "sess-redaction-disabled"
        secret = "sk-1234567890ABCDEF"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    f"Odd-{secret}", {"payload": "must-not-render"}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        rendered = " ".join(str(value) for value in seen[0].values())
        assert secret not in rendered
        assert seen[0]["command"] == "SDK tool unknown"

    @pytest.mark.parametrize("redaction_enabled", [True, False])
    def test_prefix_at_identity_offset_zero_always_collapses_to_unknown(
        self, monkeypatch, redaction_enabled,
    ):
        from agent import redact as redact_mod

        monkeypatch.setattr(redact_mod, "_REDACT_ENABLED", redaction_enabled)
        session_key = f"sess-offset-zero-{redaction_enabled}"
        secret = "sk-1234567890ABCDEF"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    secret, {"payload": "must-not-render"}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        assert seen[0]["command"] == "SDK tool unknown"
        assert secret not in " ".join(str(value) for value in seen[0].values())

    @pytest.mark.parametrize("hostile_kind", ["get", "str"])
    def test_hostile_context_provider_result_is_fixed_fail_closed(
        self, monkeypatch, caplog, hostile_kind,
    ):
        marker = f"CTX_RESULT_SECRET_{hostile_kind}"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        if hostile_kind == "get":
            class HostileContext(dict):
                def get(self, *_args, **_kwargs):
                    raise RuntimeError(marker)

            provided = HostileContext(gateway=True, session_key="safe")
        else:
            class HostileKey:
                def __str__(self):
                    return marker

            provided = {"gateway": True, "session_key": HostileKey()}

        from tools import approval as approval_mod

        callback = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: provided,
        )
        with caplog.at_level(logging.DEBUG, logger="tools.approval"):
            result = self._call_gateway(callback, "true")
        assert result == {
            "choice": "deny",
            "reason": "no approver available (background context)",
        }
        assert marker not in caplog.text

    def test_long_bash_card_discloses_truncation_and_dangerous_tail(
        self, monkeypatch,
    ):
        session_key = "sess-long-bash-tail"
        # Keep this below the immutable root-delete floor: this regression is
        # about bounded manual-card presentation, not hardline authorization.
        command = "printf safe " + ("A" * 1_000) + "; rm -rf ./build"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    "Bash", {"command": command}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        card_command = seen[0]["command"]
        assert "[truncated]" in card_command
        assert "; rm -rf ./build" in card_command
        assert len(card_command.encode("utf-8")) <= 512

    def test_long_path_card_discloses_truncation_and_target_tail(
        self, monkeypatch,
    ):
        session_key = "sess-long-path-tail"
        path = "/tmp/" + ("A" * 1_000) + "/dangerous-target"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    "Write", {"file_path": path, "content": "must-not-render"}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        card_command = seen[0]["command"]
        assert "[truncated]" in card_command
        assert "/dangerous-target" in card_command
        assert "must-not-render" not in card_command
        assert len(card_command.encode("utf-8")) <= 512

    def test_redaction_expansion_does_not_discard_actual_bash_tail(
        self, monkeypatch,
    ):
        session_key = "sess-redaction-expansion-tail"
        # Exercise redaction expansion on the manual path without weakening
        # the unconditional root-delete floor.
        command = ("API_KEY=x " * 6_000) + "; rm -rf ./build"
        assert len(command.encode("utf-8")) < 64 * 1024
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    "Bash", {"command": command}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        card_command = seen[0]["command"]
        assert "[truncated]" in card_command
        assert "; rm -rf ./build" in card_command
        assert "API_KEY=x" not in card_command
        assert len(card_command.encode("utf-8")) <= 512

    def test_redaction_expansion_does_not_discard_actual_path_tail(
        self, monkeypatch,
    ):
        session_key = "sess-path-redaction-expansion-tail"
        path = ("API_KEY=x " * 6_000) + "dangerous-target"
        assert len(path.encode("utf-8")) < 64 * 1024
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "once")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                session, _ = _make_session(
                    approval_callback=callback, permission_mode="default",
                )
                result = asyncio.run(session._make_can_use_tool()(
                    "Write", {"file_path": path, "content": "must-not-render"}, None,
                ))
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert type(result).__name__ == "PermissionResultAllow"
        card_command = seen[0]["command"]
        assert "[truncated]" in card_command
        assert "dangerous-target" in card_command
        assert "API_KEY=x" not in card_command
        assert "must-not-render" not in card_command
        assert len(card_command.encode("utf-8")) <= 512

    def test_approve_maps_to_once_and_clamps_durable_choices(self, monkeypatch):
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-mapped")
        try:
            # An older client button can still send "always" — the grant must
            # not outlive the single SDK permission request it answered.
            notify, seen = self._resolve_with(approval_mod, "sess-mapped", "always")
            approval_mod.register_gateway_notify("sess-mapped", notify)
            try:
                cb = sdk_gateway.build_sdk_gateway_approval_callback()
                assert self._call_gateway(cb, "uname") == "once"
            finally:
                approval_mod.unregister_gateway_notify("sess-mapped")
            assert seen[0]["allow_permanent"] is False
            assert seen[0]["allow_session"] is False
            assert seen[0]["command"] == "Bash(command=uname)"
        finally:
            approval_ctx.reset_current_session_key(token)

    def test_deny_choice_denies(self, monkeypatch):
        approval_mod, token = self._gateway_ctx(monkeypatch, "sess-denied")
        try:
            notify, _ = self._resolve_with(approval_mod, "sess-denied", "deny")
            approval_mod.register_gateway_notify("sess-denied", notify)
            try:
                cb = sdk_gateway.build_sdk_gateway_approval_callback()
                assert self._call_gateway(cb, "rm x") == {
                    "choice": "deny",
                    "operator_denial": True,
                    "reason": "",
                }
            finally:
                approval_mod.unregister_gateway_notify("sess-denied")
        finally:
            approval_ctx.reset_current_session_key(token)

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
        token = approval_ctx.set_current_session_key("tg:152:bridge")
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
            approval_ctx.reset_current_session_key(token)

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


    def test_smart_approved_sdk_read_skips_operator_card(self, monkeypatch):
        import json

        sk = "sess-smart-sdk-read"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        notified = []
        smart_calls = []
        prepared = []
        observed = []
        awaited = []
        try:
            approval_mod.register_gateway_notify(sk, notified.append)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(
                approval_gateway_wait,
                "_await_gateway_decision",
                lambda *args, **kwargs: awaited.append((args, kwargs)) or {
                    "resolved": True, "choice": "once",
                },
            )
            monkeypatch.setattr(
                approval_smart,
                "_smart_approve",
                lambda *args, **kwargs: smart_calls.append((args, kwargs)) or "approve",
            )
            monkeypatch.setattr(
                approval_smart,
                "_prepare_smart_approval_observer",
                lambda **kwargs: prepared.append(kwargs) or {"safe": True},
            )
            monkeypatch.setattr(
                approval_smart,
                "_observe_smart_approval_verdict",
                lambda payload, verdict, **kwargs: observed.append(
                    (payload, verdict, kwargs)
                ),
            )
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            assert cb is not None

            result = self._call_gateway(
                cb,
                tool_name="Read",
                tool_input={"file_path": "/srv/example-app/src/backup.py"},
            )

            canonical = json.dumps(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/srv/example-app/src/backup.py"},
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            assert result == "once"
            assert smart_calls == [((canonical, "Claude requests SDK tool Read"), {})]
            assert prepared == [{
                "command": "Read(path=/srv/example-app/src/backup.py)",
                "description": "Claude requests SDK tool Read",
                "pattern_key": "claude_sdk_tool",
                "pattern_keys": ["claude_sdk_tool"],
                "session_key": sk,
            }]
            assert observed == [({"safe": True}, "approve", {})]
            assert awaited == []
            assert notified == []
            with approval_mod._lock:
                assert not approval_mod._gateway_queues.get(sk)
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    def test_smart_denied_sdk_owner_override_card_is_one_shot_and_resets_tally(
        self, monkeypatch,
    ):
        sk = "sess-smart-sdk-deny-owner-override"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        seen = []

        def notify(data):
            seen.append(dict(data))
            approval_mod.resolve_gateway_approval(sk, "once")

        try:
            approval_mod._reset_denials(sk)
            approval_mod.register_gateway_notify(sk, notify)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: "deny")
            cb = sdk_gateway.build_sdk_gateway_approval_callback()

            assert self._call_gateway(
                cb,
                tool_name="Read",
                tool_input={"file_path": "/tmp/card-target"},
                tool_use_id="toolu-card",
            ) == "once"
            assert len(seen) == 1
            request_id = seen[0].pop("request_id")
            assert type(request_id) is str and request_id
            assert seen == [{
                "command": "Read(path=/tmp/card-target)",
                "pattern_key": "claude_sdk_tool",
                "pattern_keys": ["claude_sdk_tool"],
                "description": "Claude requests SDK tool Read",
                "allow_permanent": False,
                "allow_session": False,
                "smart_denied": True,
                "tool_use_id": "toolu-card",
                "no_coalesce": True,
            }]
            with approval_mod._lock:
                assert sk not in approval_mod._denial_tally
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_mod._reset_denials(sk)
            approval_ctx.reset_current_session_key(token)

    def test_sdk_bash_hardline_denies_before_smart_or_card(self, monkeypatch):
        sk = "sess-sdk-hardline"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        notified = []
        smart_calls = []
        try:
            approval_mod.register_gateway_notify(sk, notified.append)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(
                approval_mod,
                "detect_hardline_command",
                lambda command: (command == "dangerous-probe", "test hardline"),
            )
            monkeypatch.setattr(
                approval_smart,
                "_smart_approve",
                lambda *args, **kwargs: smart_calls.append((args, kwargs)) or "approve",
            )
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            assert cb is not None
            session, _ = _make_session(
                approval_callback=cb,
                permission_mode="default",
                hermes_session_id=sk,
            )

            result = asyncio.run(session._make_can_use_tool()(
                "Bash",
                {"command": "dangerous-probe"},
                SimpleNamespace(tool_use_id="toolu-hardline"),
            ))

            assert type(result).__name__ == "PermissionResultDeny"
            assert result.message == "approval denied by callback"
            assert smart_calls == []
            assert notified == []
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    def test_sdk_gateway_card_drops_mixed_control_tool_use_id(self, monkeypatch):
        sk = "sess-sdk-hostile-id"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            notify, seen = self._resolve_with(approval_mod, sk, "deny")
            approval_mod.register_gateway_notify(sk, notify)
            callback = sdk_gateway.build_sdk_gateway_approval_callback()
            result = self._call_gateway(
                callback, "true", tool_use_id="toolu_ok\x00\nINJECT",
            )
            assert result == {
                "choice": "deny", "operator_denial": True, "reason": "",
            }
            assert seen[0]["tool_use_id"] == ""
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    @pytest.mark.parametrize(
        ("tool_use_id", "expected"),
        [
            ("x" * 256, "x" * 256),
            ("x" * 257, ""),
            ("é" * 128, "é" * 128),
            (("é" * 128) + "x", ""),
        ],
    )
    def test_gateway_tool_use_id_enforces_exact_utf8_byte_cap(
        self, monkeypatch, tool_use_id, expected,
    ):
        session_key = "sess-sdk-id-byte-cap"
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "deny")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                result = self._call_gateway(
                    callback, "true", tool_use_id=tool_use_id,
                )
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert result == {
            "choice": "deny", "operator_denial": True, "reason": "",
        }
        assert [card["tool_use_id"] for card in seen] == [expected]
        if not expected:
            assert tool_use_id not in [card["tool_use_id"] for card in seen]

    def test_gateway_huge_multibyte_tool_use_id_has_bounded_peak_allocation(
        self, monkeypatch,
    ):
        session_key = "sess-sdk-huge-id-allocation"
        huge_id = "é" * (16 * 1024 * 1024)
        approval_mod, token = self._gateway_ctx(monkeypatch, session_key)
        try:
            notify, seen = self._resolve_with(approval_mod, session_key, "deny")
            approval_mod.register_gateway_notify(session_key, notify)
            try:
                callback = sdk_gateway.build_sdk_gateway_approval_callback()
                self._call_gateway(callback, "true", tool_use_id="warmup")
                seen.clear()
                tracemalloc.start()
                try:
                    result = self._call_gateway(
                        callback, "true", tool_use_id=huge_id,
                    )
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
            finally:
                approval_mod.unregister_gateway_notify(session_key)
        finally:
            approval_ctx.reset_current_session_key(token)

        assert result == {
            "choice": "deny", "operator_denial": True, "reason": "",
        }
        assert [card["tool_use_id"] for card in seen] == [""]
        assert huge_id not in [card["tool_use_id"] for card in seen]
        assert peak < 2 * 1024 * 1024

    @pytest.mark.parametrize("bypass_source", ["off", "process_yolo", "session_yolo"])
    def test_sdk_bypass_sources_skip_approver_after_bash_floors(
        self, monkeypatch, bypass_source,
    ):
        sk = f"sess-sdk-bypass-{bypass_source}"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            if bypass_source == "off":
                monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "off")
            elif bypass_source == "process_yolo":
                monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", True)
            else:
                monkeypatch.setattr(
                    approval_mod, "is_session_yolo_enabled", lambda key: key == sk,
                )
            monkeypatch.setattr(
                approval_gateway_wait,
                "_await_gateway_decision",
                lambda *_a, **_k: pytest.fail("bypass reached approval queue"),
            )
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            assert cb is not None
            session, _ = _make_session(
                approval_callback=cb,
                permission_mode="default",
                hermes_session_id=sk,
            )
            result = asyncio.run(session._make_can_use_tool()(
                "Bash",
                {"command": "printf safe"},
                SimpleNamespace(tool_use_id="toolu-bypass"),
            ))
            assert type(result).__name__ == "PermissionResultAllow"
            assert result.updated_input == {"command": "printf safe"}
        finally:
            approval_ctx.reset_current_session_key(token)

    def test_sdk_smart_evaluator_exception_uses_fixed_log(
        self, monkeypatch, caplog,
    ):
        import json

        sk = "sess-sdk-smart-exception"
        marker = "SMART_EXCEPTION_SECRET_7b9"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            approval_mod.register_gateway_notify(sk, lambda _data: None)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            from agent import auxiliary_client
            monkeypatch.setattr(
                auxiliary_client, "call_llm",
                lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(marker)),
            )
            monkeypatch.setattr(
                approval_gateway_wait,
                "_await_gateway_decision",
                lambda *_a, **_k: {"resolved": True, "choice": "deny"},
            )
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            canonical = json.dumps(
                {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}},
                sort_keys=True, separators=(",", ":"),
            )
            with caplog.at_level(logging.DEBUG, logger="tools.approval"):
                assert cb("untrusted", "untrusted", canonical_tool_input=canonical) == {
                    "choice": "deny", "operator_denial": True, "reason": "",
                }
            assert marker not in caplog.text
            assert "Smart approvals: LLM call failed, escalating" in caplog.text
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    def test_hostile_exact_session_key_never_reaches_sdk_log(
        self, monkeypatch, caplog,
    ):
        import json

        marker = "CTX_SESSION_SECRET_7c91"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        from tools import approval as approval_mod

        cb = sdk_gateway.build_sdk_gateway_approval_callback(
            context_provider=lambda: {"gateway": True, "session_key": marker},
        )
        canonical = json.dumps(
            {"tool_name": "Bash", "tool_input": {"command": "true"}},
            sort_keys=True, separators=(",", ":"),
        )
        with caplog.at_level(logging.WARNING, logger="tools.approval"):
            result = cb("untrusted", "untrusted", canonical_tool_input=canonical)
        assert result == {
            "choice": "deny", "reason": "no approver available (background context)",
        }
        assert marker not in caplog.text


    def test_sdk_smart_observer_dispatch_exceptions_use_fixed_stage_logs(
        self, monkeypatch, caplog,
    ):
        import json

        from hermes_cli import lifecycle

        sk = "SDK_SESSION_SECRET_65982"
        marker = "SDK_OBSERVER_EXCEPTION_SECRET_65982"
        presentation = "Read(path=/tmp/visible-path)"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)

        def broken_dispatch(hook_name, **kwargs):
            raise RuntimeError(
                f"{marker} stage={hook_name} session={kwargs['session_key']} "
                f"command={kwargs['command']}"
            )

        try:
            approval_mod.register_gateway_notify(sk, lambda _data: None)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: "approve")
            monkeypatch.setattr(lifecycle, "invoke_hook", broken_dispatch)
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            canonical = json.dumps(
                {"tool_name": "Read", "tool_input": {"file_path": "/tmp/visible-path"}},
                sort_keys=True, separators=(",", ":"),
            )
            with caplog.at_level(logging.DEBUG, logger="tools.approval"):
                assert cb("untrusted", "untrusted", canonical_tool_input=canonical) == "once"

            assert marker not in caplog.text
            assert sk not in caplog.text
            assert presentation not in caplog.text
            assert not any(record.exc_info for record in caplog.records)
            assert [record.getMessage() for record in caplog.records] == [
                "SDK Smart pre-approval observer dispatch failed",
                "SDK Smart post-approval observer dispatch failed",
            ]
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    def test_sdk_smart_builtin_observer_failure_is_contained_at_inner_sink(
        self, monkeypatch, caplog,
    ):
        from agent import relay_runtime
        from hermes_cli import plugins
        from hermes_cli.observability import relay_shared_metrics

        sk = "SDK_BUILTIN_SESSION_SECRET_65982"
        marker = "SDK_BUILTIN_OBSERVER_SECRET_65982"
        path = "/tmp/SDK_BUILTIN_PATH_SECRET_65982"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)

        class BrokenRuntime:
            def record_approval(self, kwargs):
                raise RuntimeError(f"{marker} payload={kwargs!r}")

        try:
            approval_mod.register_gateway_notify(sk, lambda _data: None)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: "approve")
            monkeypatch.setattr(relay_shared_metrics, "handles_hook", lambda _name: True)
            monkeypatch.setattr(
                relay_runtime, "relay_instrumentation_enabled", lambda: True,
            )
            monkeypatch.setattr(relay_shared_metrics, "_get_runtime", lambda: BrokenRuntime())
            monkeypatch.setattr(plugins, "get_plugin_manager", lambda: plugins.PluginManager())
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            with caplog.at_level(logging.DEBUG):
                assert self._call_gateway(
                    cb, tool_name="Read", tool_input={"file_path": path},
                ) == "once"

            assert marker not in caplog.text
            assert sk not in caplog.text
            assert path not in caplog.text
            assert not any(record.exc_info for record in caplog.records)
            # handles_hook is forced True for every hook, so main's table dispatch
            # (relay_shared_metrics._HOOK_HANDLERS) fails the pre hook as well as the
            # post hook; both failures must surface as the fixed lines only.
            assert [
                record.getMessage() for record in caplog.records
                if "SDK Smart" in record.getMessage()
            ] == [
                "SDK Smart pre-approval observer dispatch failed",
                "SDK Smart post-approval observer dispatch failed",
            ]
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    @pytest.mark.parametrize(
        ("verdict", "expected", "expected_logs"),
        [
            ("approve", "once", [
                "SDK Smart pre-approval observer dispatch failed",
                "SDK Smart post-approval observer dispatch failed",
            ]),
            ("deny", {"choice": "deny", "operator_denial": True, "reason": ""}, [
                "SDK Smart pre-approval observer dispatch failed",
                "SDK Smart post-approval observer dispatch failed",
                "SDK Smart pre-approval observer dispatch failed",
                "SDK Smart post-approval observer dispatch failed",
            ]),
        ],
    )
    def test_sdk_smart_real_plugin_callbacks_contain_inner_runtime_failure(
        self, monkeypatch, caplog, verdict, expected, expected_logs,
    ):
        from hermes_cli import plugins
        from hermes_cli.observability import relay_shared_metrics

        sk = "SDK_PLUGIN_SESSION_SECRET_65982"
        marker = "SDK_PLUGIN_OBSERVER_SECRET_65982"
        path = "/tmp/SDK_PLUGIN_PATH_SECRET_65982"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        manager = plugins.PluginManager()

        def broken_plugin(**kwargs):
            raise RuntimeError(f"{marker} payload={kwargs!r}")

        manager._hooks["pre_approval_request"] = [broken_plugin]
        manager._hooks["post_approval_response"] = [broken_plugin]
        try:
            approval_mod.register_gateway_notify(
                sk,
                lambda data: approval_mod.resolve_gateway_approval(
                    sk, "deny", tool_use_id=data["tool_use_id"],
                ),
            )
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: verdict)
            monkeypatch.setattr(relay_shared_metrics, "observe_lifecycle", lambda *_a, **_k: None)
            monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            with caplog.at_level(logging.DEBUG):
                assert self._call_gateway(
                    cb, tool_name="Read", tool_input={"file_path": path},
                    tool_use_id="toolu-plugin",
                ) == expected

            assert marker not in caplog.text
            assert sk not in caplog.text
            assert path not in caplog.text
            assert not any(record.exc_info for record in caplog.records)
            assert [
                record.getMessage() for record in caplog.records
                if "SDK Smart" in record.getMessage()
            ] == expected_logs
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    def test_sdk_smart_observer_redactor_failure_uses_fixed_pre_stage_log(
        self, monkeypatch, caplog,
    ):
        from agent import redact

        sk = "SDK_REDACTOR_SESSION_SECRET_65982"
        marker = "SDK_REDACTOR_EXCEPTION_SECRET_65982"
        path = "/tmp/SDK_REDACTOR_PATH_SECRET_65982"
        presentation = f"Read(path={path})"
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        real_redact = redact.redact_sensitive_text

        def selective_redactor(value, *args, **kwargs):
            if value == presentation and kwargs.get("force") is True:
                raise RuntimeError(f"{marker} value={value}")
            return real_redact(value, *args, **kwargs)

        try:
            approval_mod.register_gateway_notify(sk, lambda _data: None)
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: "approve")
            monkeypatch.setattr(redact, "redact_sensitive_text", selective_redactor)
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            with caplog.at_level(logging.DEBUG):
                assert self._call_gateway(
                    cb, tool_name="Read", tool_input={"file_path": path},
                ) == "once"

            assert marker not in caplog.text
            assert sk not in caplog.text
            assert path not in caplog.text
            assert not any(record.exc_info for record in caplog.records)
            assert [
                record.getMessage() for record in caplog.records
                if "SDK Smart" in record.getMessage()
            ] == ["SDK Smart pre-approval observer dispatch failed"]
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    def test_generic_observer_failures_keep_legacy_raw_logging(
        self, monkeypatch, caplog,
    ):
        from agent import redact
        from hermes_cli import observability, plugins

        builtin_marker = "GENERIC_BUILTIN_RAW_COMPAT_65982"
        plugin_marker = "GENERIC_PLUGIN_RAW_COMPAT_65982"
        redactor_marker = "GENERIC_REDACTOR_RAW_COMPAT_65982"
        manager = plugins.PluginManager()

        def broken_plugin(**_kwargs):
            raise RuntimeError(plugin_marker)

        manager._hooks["pre_approval_request"] = [broken_plugin]
        monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
        from hermes_cli.observability import relay_shared_metrics

        monkeypatch.setattr(
            relay_shared_metrics, "observe_lifecycle",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError(builtin_marker)),
        )
        with caplog.at_level(logging.DEBUG):
            observability.observe_lifecycle("pre_approval_request", session_key="generic-session")
            manager.invoke_hook("pre_approval_request", session_key="generic-session")
            monkeypatch.setattr(
                redact,
                "redact_sensitive_text",
                lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError(redactor_marker)),
            )
            assert approval_smart._prepare_smart_approval_observer(
                command="generic command",
                description="generic description",
                pattern_key="generic",
                pattern_keys=["generic"],
                session_key="generic-session",
            ) is None

        assert builtin_marker in caplog.text
        assert plugin_marker in caplog.text
        assert redactor_marker in caplog.text
        assert any(record.exc_info for record in caplog.records)
        assert "SDK Smart" not in caplog.text

    def test_generic_shared_metrics_sink_keeps_legacy_raw_logging(
        self, monkeypatch, caplog,
    ):
        from agent import relay_runtime
        from hermes_cli.observability import relay_shared_metrics

        shared_marker = "GENERIC_SHARED_METRICS_RAW_COMPAT_65982"

        class BrokenSharedRuntime:
            def record_approval(self, kwargs):
                raise RuntimeError(f"{shared_marker} payload={kwargs!r}")

        monkeypatch.setattr(relay_shared_metrics, "handles_hook", lambda _name: True)
        monkeypatch.setattr(relay_runtime, "relay_instrumentation_enabled", lambda: True)
        monkeypatch.setattr(
            relay_shared_metrics, "_get_runtime", lambda: BrokenSharedRuntime(),
        )
        with caplog.at_level(logging.DEBUG):
            relay_shared_metrics.observe_lifecycle(
                "post_approval_response", session_key="generic-shared-session",
            )

        assert shared_marker in caplog.text
        assert any(record.exc_info for record in caplog.records)
        assert "SDK Smart" not in caplog.text

    def test_sdk_safe_and_generic_observer_dispatches_do_not_bleed_between_threads(
        self, monkeypatch, caplog,
    ):
        from concurrent.futures import ThreadPoolExecutor
        import threading

        from hermes_cli import plugins
        from hermes_cli.observability import relay_shared_metrics
        from tools import approval as approval_mod

        sdk_session = "SDK_CONCURRENT_SESSION_SECRET_65982"
        sdk_path = "/tmp/SDK_CONCURRENT_PATH_SECRET_65982"
        sdk_marker = "SDK_CONCURRENT_EXCEPTION_SECRET_65982"
        generic_session = "GENERIC_CONCURRENT_SESSION_COMPAT_65982"
        generic_marker = "GENERIC_CONCURRENT_EXCEPTION_COMPAT_65982"
        barrier = threading.Barrier(2)
        manager = plugins.PluginManager()

        def broken_plugin(**kwargs):
            barrier.wait(timeout=5)
            marker = sdk_marker if kwargs["session_key"] == sdk_session else generic_marker
            raise RuntimeError(f"{marker} payload={kwargs!r}")

        manager._hooks["pre_approval_request"] = [broken_plugin]
        manager._hooks["post_approval_response"] = [broken_plugin]
        monkeypatch.setattr(relay_shared_metrics, "observe_lifecycle", lambda *_a, **_k: None)
        monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

        def dispatch(session_key, command, fixed):
            from contextlib import nullcontext

            from hermes_cli.lifecycle import observer_failure_log

            pre = observer_failure_log(approval_ctx.SDK_PRE_OBSERVER_FAILURE_LOG) if fixed else nullcontext()
            post = observer_failure_log(approval_ctx.SDK_POST_OBSERVER_FAILURE_LOG) if fixed else nullcontext()
            with pre:
                payload = approval_smart._prepare_smart_approval_observer(
                    command=command,
                    description="bounded description",
                    pattern_key="claude_sdk_tool" if fixed else "generic",
                    pattern_keys=["claude_sdk_tool" if fixed else "generic"],
                    session_key=session_key,
                )
            with post:
                approval_smart._observe_smart_approval_verdict(payload, "approve")

        with caplog.at_level(logging.DEBUG), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(dispatch, sdk_session, f"Read(path={sdk_path})", True),
                pool.submit(dispatch, generic_session, "generic command", False),
            ]
            for future in futures:
                future.result(timeout=10)

        assert sdk_marker not in caplog.text
        assert sdk_session not in caplog.text
        assert sdk_path not in caplog.text
        assert generic_marker in caplog.text
        assert generic_session in caplog.text
        assert [
            record.getMessage() for record in caplog.records
            if "SDK Smart" in record.getMessage()
        ] == [
            "SDK Smart pre-approval observer dispatch failed",
            "SDK Smart post-approval observer dispatch failed",
        ]

    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [
            ("approve", "once"),
            ("deny", {"choice": "deny", "operator_denial": True, "reason": ""}),
            ("escalate", {"choice": "deny", "operator_denial": True, "reason": ""}),
        ],
    )
    def test_successful_real_sdk_observer_gets_bounded_fields_without_changing_decision(
        self, monkeypatch, verdict, expected,
    ):
        from hermes_cli import plugins
        from hermes_cli.observability import relay_shared_metrics

        sk = f"sdk-success-{verdict}"
        path = "/tmp/sdk-success-path"
        seen = []
        manager = plugins.PluginManager()
        manager._hooks["pre_approval_request"] = [
            lambda **kwargs: seen.append(("pre", kwargs))
        ]
        manager._hooks["post_approval_response"] = [
            lambda **kwargs: seen.append(("post", kwargs))
        ]
        approval_mod, token = self._gateway_ctx(monkeypatch, sk)
        try:
            approval_mod.register_gateway_notify(
                sk,
                lambda _data: approval_mod.resolve_gateway_approval(sk, "deny"),
            )
            monkeypatch.setattr(approval_ctx, "_get_approval_mode", lambda: "smart")
            monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_a, **_k: verdict)
            monkeypatch.setattr(relay_shared_metrics, "observe_lifecycle", lambda *_a, **_k: None)
            monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            assert self._call_gateway(
                cb, tool_name="Read", tool_input={"file_path": path},
            ) == expected

            smart_seen = [item for item in seen if item[1].get("surface") == "smart"]
            expected_stages = ["pre"] if verdict == "escalate" else ["pre", "post"]
            assert [stage for stage, _kwargs in smart_seen] == expected_stages
            pre = smart_seen[0][1]
            assert pre["command"] == f"Read(path={path})"
            assert pre["description"] == "Claude requests SDK tool Read"
            assert pre["pattern_key"] == "claude_sdk_tool"
            assert pre["pattern_keys"] == ["claude_sdk_tool"]
            assert pre["session_key"] == sk
            assert pre["surface"] == "smart"
            assert {"turn_id", "tool_call_id", "telemetry_schema_version"} <= pre.keys()
            if verdict != "escalate":
                assert smart_seen[1][1]["choice"] == f"smart_{verdict}"
                assert smart_seen[1][1]["decided_by"] == "aux_llm"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)


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
            "tools.approval_context.get_current_session_key", lambda: "gw-key-7"
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
            "tools.approval_context.get_current_session_key", lambda: "gw-key-7"
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
            "tools.approval_context.get_current_session_key", lambda: ""
        )
        callback(["late background report"])
        assert len(events) == 1
        assert events[0]["parent_session_id"] == "sess-after-rotation"
        # Empty live key -> creation-time snapshot fallback keeps the route.
        assert events[0]["session_key"] == "gw-key-7"

        # A live, non-empty contextvar read wins over the snapshot.
        monkeypatch.setattr(
            "tools.approval_context.get_current_session_key", lambda: "gw-key-LIVE"
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
                cb("Bash", {"command": "true"}, SimpleNamespace(tool_use_id="t1")),
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


class TestSdkApprovalCanonicalizationHardening:
    """Hostile SDK permission data is bounded before every approval sink."""

    @pytest.fixture(autouse=True)
    def _sdk_permission_results(self, monkeypatch):
        _plant_claude_agent_sdk_stand_in(monkeypatch)

    @pytest.mark.parametrize(
        "tool_input_factory",
        [
            lambda: {"path": object()},
            lambda: {"path": type("StrSubclass", (str,), {})("/tmp/x")},
            lambda: {"path": "bad\ud800path"},
            lambda: {"score": float("nan")},
            lambda: {"nested": [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]},
            lambda: {"nodes": [None] * 10_001},
            lambda: {"path": "x" * (64 * 1024)},
        ],
    )
    def test_malformed_autoallowed_request_denies_before_callback(
        self, tool_input_factory,
    ):
        calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        tool_input = tool_input_factory()
        if "nodes" not in tool_input:
            tool_input["cycle"] = []
            if type(tool_input.get("path")) is object:
                tool_input.pop("cycle")
        result = asyncio.run(session._make_can_use_tool()(
            "mcp__hermes-tools__read_file", tool_input, None,
        ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "canonical request is unassessable"
        assert calls == []

    def test_cycle_denies_before_exact_mcp_autoallow(self):
        calls = []
        cyclic = {}
        cyclic["self"] = cyclic
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "mcp__hermes-tools__search_files", cyclic, None,
        ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert calls == []

    def test_canonical_serialization_is_deterministic_utf8_and_bounded(self):
        from agent.transports.claude_agent_sdk_session import (
            _canonical_sdk_tool_request,
            validate_canonical_sdk_request_serialization,
        )

        canonical = _canonical_sdk_tool_request(
            "Pathé工具", {"z": "🙂", "a": {"路径": "/tmp/é"}},
        )
        assert canonical == (
            '{"tool_input":{"a":{"路径":"/tmp/é"},"z":"🙂"},'
            '"tool_name":"Pathé工具"}'
        )
        assert validate_canonical_sdk_request_serialization(canonical) == (
            canonical,
            {"tool_input": {"a": {"路径": "/tmp/é"}, "z": "🙂"}, "tool_name": "Pathé工具"},
        )
        assert validate_canonical_sdk_request_serialization(canonical + " ") is None
        assert validate_canonical_sdk_request_serialization(
            '{"tool_input":{"path":"\\ud800"},"tool_name":"Read"}'
        ) is None

    def test_marker_aware_callback_receives_canonical_and_bounded_meaningful_id(self):
        received = []

        def marked(command, description, *, allow_permanent=False, tool_use_id="",
                   canonical_tool_input=None):
            received.append((command, description, allow_permanent, tool_use_id,
                             canonical_tool_input))
            return "once"

        marked._accepts_tool_use_id = True
        marked._accepts_canonical_tool_input = True
        session, _ = _make_session(
            approval_callback=marked, permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "printf safe"}, SimpleNamespace(tool_use_id="toolu_é"),
        ))
        assert type(result).__name__ == "PermissionResultAllow"
        assert received == [(
            "Bash(command=printf safe)", "Claude requests SDK tool Bash", False,
            "toolu_é", '{"tool_input":{"command":"printf safe"},"tool_name":"Bash"}',
        )]

    @pytest.mark.parametrize(
        ("tool_use_id", "expected"),
        [
            ("x" * 256, "x" * 256),
            ("x" * 257, ""),
            ("é" * 128, "é" * 128),
            (("é" * 128) + "x", ""),
        ],
    )
    def test_sdk_session_tool_use_id_enforces_exact_utf8_byte_cap(
        self, tool_use_id, expected,
    ):
        received = []

        def marked(command, description, *, allow_permanent=False, tool_use_id=""):
            received.append(tool_use_id)
            return "deny"

        marked._accepts_tool_use_id = True
        session, _ = _make_session(approval_callback=marked, permission_mode="default")
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "true"}, SimpleNamespace(tool_use_id=tool_use_id),
        ))

        assert type(result).__name__ == "PermissionResultDeny"
        assert received == [expected]
        if not expected:
            assert tool_use_id not in received

    def test_sdk_session_huge_multibyte_tool_use_id_has_bounded_peak_allocation(self):
        huge_id = "é" * (16 * 1024 * 1024)
        received = []

        def marked(command, description, *, allow_permanent=False, tool_use_id=""):
            received.append(tool_use_id)
            return "deny"

        marked._accepts_tool_use_id = True
        session, _ = _make_session(approval_callback=marked, permission_mode="default")
        asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "true"}, SimpleNamespace(tool_use_id="warmup"),
        ))
        received.clear()
        tracemalloc.start()
        try:
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, SimpleNamespace(tool_use_id=huge_id),
            ))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert type(result).__name__ == "PermissionResultDeny"
        assert received == [""]
        assert huge_id not in received
        assert peak < 2 * 1024 * 1024

    @pytest.mark.parametrize(
        "bad_id",
        [None, 123, " \t\n", "\x00\x01", "bad\ud800", "x" * 257,
         type("IdSubclass", (str,), {})("toolu_x")],
    )
    def test_opted_in_callback_receives_empty_malformed_tool_use_id(self, bad_id):
        received = []

        def marked(command, description, *, allow_permanent=False, tool_use_id=""):
            received.append(tool_use_id)
            return "deny"

        marked._accepts_tool_use_id = True
        session, _ = _make_session(approval_callback=marked, permission_mode="default")
        asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "true"}, SimpleNamespace(tool_use_id=bad_id),
        ))
        assert received == [""]

    def test_markerless_callback_keeps_exact_abi_and_safe_actionable_presentations(
        self, monkeypatch,
    ):
        from agent.transports import claude_agent_sdk_session as session_mod

        monkeypatch.setattr(
            session_mod, "redact_sensitive_text",
            lambda value, **_kwargs: value.replace("SECRET", "[REDACTED]"),
        )
        received = []

        def legacy(command, description, *, allow_permanent=False):
            received.append((command, description, allow_permanent))
            return "once"

        session, _ = _make_session(approval_callback=legacy, permission_mode="default")
        callback = session._make_can_use_tool()
        for name, payload in (
            ("Bash", {"command": "printf SECRET\nnext\x00line"}),
            ("Write", {"file_path": "/tmp/demo\n.txt", "content": "SECRET_CONTENT"}),
            ("Odd SECRET\x00Tool", {"payload": "SECRET_PAYLOAD"}),
        ):
            result = asyncio.run(callback(name, payload, SimpleNamespace(tool_use_id="hostile")))
            assert type(result).__name__ == "PermissionResultAllow"

        assert received[0] == (
            "Bash(command=printf [REDACTED] next line)",
            "Claude requests SDK tool Bash", False,
        )
        assert received[1] == (
            "Write(path=/tmp/demo .txt)", "Claude requests SDK tool Write", False,
        )
        assert "CONTENT" not in received[1][0]
        assert received[2] == (
            "SDK tool unknown", "Claude requests an SDK tool", False,
        )
        assert "PAYLOAD" not in received[2][0]
        assert all(len(command.encode("utf-8")) <= 512 for command, _, _ in received)

    def test_malformed_bash_denies_before_markerless_callback(self):
        calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": ["rm", "-rf", "/"]}, None,
        ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "canonical request is unassessable"
        assert calls == []

    def test_hostile_callback_exception_and_sdk_metadata_never_reach_logs(
        self, caplog,
    ):
        marker = "SDK_SECRET_EXCEPTION_91f"

        def broken(*_args, **_kwargs):
            raise RuntimeError(marker)

        session, _ = _make_session(
            approval_callback=broken, permission_mode="default",
            hermes_session_id="safe-session",
        )
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                f"Odd-{marker}", {"payload": marker}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert marker not in caplog.text
        assert not any(record.exc_info for record in caplog.records)
        assert any(
            record.getMessage() == "SDK approval callback failed at protected boundary"
            for record in caplog.records
        )

    def test_gateway_notify_exception_text_is_contained_at_sdk_boundary(
        self, monkeypatch, caplog,
    ):
        import json

        from tools import approval as approval_mod

        marker = "SDK_NOTIFY_EXCEPTION_SECRET_4ab"
        session_key = "sdk-notify-containment"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        token = approval_ctx.set_current_session_key(session_key)

        def broken_notify(_approval_data):
            raise RuntimeError(marker)

        try:
            session_notify.register_session_notify(session_key, broken_notify)
            callback = sdk_gateway.build_sdk_gateway_approval_callback()
            canonical = json.dumps(
                {"tool_name": "Bash", "tool_input": {"command": "true"}},
                sort_keys=True,
                separators=(",", ":"),
            )
            with caplog.at_level(logging.WARNING, logger="tools.approval"):
                result = callback(
                    "hostile", "hostile", canonical_tool_input=canonical,
                )
            assert result == {
                "choice": "deny",
                "reason": (
                    "approval request could not be delivered to the operator "
                    "(notify failed)"
                ),
            }
            assert marker not in caplog.text
        finally:
            session_notify.unregister_session_notify(session_key)
            approval_ctx.reset_current_session_key(token)

    def test_wide_request_rejects_without_width_sized_validator_allocation(self):
        import tracemalloc

        from agent.transports.claude_agent_sdk_session import (
            _is_bounded_plain_sdk_json,
        )

        request = {"wide": [None] * 200_000}
        tracemalloc.start()
        try:
            assert not _is_bounded_plain_sdk_json(request)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2_000_000

    def test_wide_dictionary_rejection_time_is_node_budget_bounded(self):
        import time

        from agent.transports.claude_agent_sdk_session import (
            _is_bounded_plain_sdk_json,
        )

        smaller = {str(index): None for index in range(50_000)}
        larger = {str(index): None for index in range(500_000)}

        def fastest(value):
            samples = []
            for _ in range(3):
                started = time.perf_counter()
                assert not _is_bounded_plain_sdk_json(value)
                samples.append(time.perf_counter() - started)
            return min(samples)

        smaller_time = fastest(smaller)
        larger_time = fastest(larger)
        assert larger_time < (smaller_time * 3) + 0.01

    def test_canonicalization_runtime_error_denies_before_auto_allow(
        self, monkeypatch,
    ):
        from agent.transports import claude_agent_sdk_session as session_mod

        def mutation_failure(*_args, **_kwargs):
            raise RuntimeError("dictionary changed size during iteration")

        monkeypatch.setattr(
            session_mod, "_canonical_sdk_tool_request", mutation_failure,
        )
        calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "mcp__hermes-tools__read_file", {"path": "/tmp/demo"}, None,
        ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "canonical request is unassessable"
        assert calls == []

    def test_wide_exact_callback_result_rejects_before_width_sized_allocation(
        self, caplog,
    ):
        import tracemalloc

        marker = "SDK_WIDE_RESULT_SECRET_91c"
        wide_result = {f"key-{index}": None for index in range(200_000)}
        wide_result[marker] = None
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: wide_result,
            permission_mode="default",
        )
        callback = session._make_can_use_tool()
        asyncio.run(callback("Bash", {"command": "true"}, None))

        caplog.clear()
        tracemalloc.start()
        try:
            with caplog.at_level(
                logging.INFO, logger="agent.transports.claude_agent_sdk_session",
            ):
                result = asyncio.run(callback("Bash", {"command": "true"}, None))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert peak < 1_000_000
        assert marker not in caplog.text
        assert not any(record.exc_info for record in caplog.records)

    def test_overlong_multibyte_choice_rejects_without_full_copy_or_leak(
        self, caplog, monkeypatch,
    ):
        import tracemalloc

        from agent.transports import claude_agent_sdk_session as session_mod

        marker = "SDK_CHOICE_RESULT_SECRET_27e"
        huge_choice = marker + ("é" * (32 * 1024 * 1024))
        current_result = ["bogus"]
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: current_result[0],
            permission_mode="default",
        )
        callback = session._make_can_use_tool()
        asyncio.run(callback("Bash", {"command": "true"}, None))
        current_result[0] = huge_choice
        category_calls = []
        hostile_category_counts = []
        original_category = session_mod.unicodedata.category
        original_validator = session_mod._is_bounded_sdk_callback_string

        def counted_category(char):
            category_calls.append(char)
            return original_category(char)

        def counted_validator(value, max_utf8_bytes, *, allow_space):
            before = len(category_calls)
            valid = original_validator(
                value, max_utf8_bytes, allow_space=allow_space,
            )
            if value is huge_choice:
                hostile_category_counts.append(len(category_calls) - before)
            return valid

        monkeypatch.setattr(session_mod.unicodedata, "category", counted_category)
        monkeypatch.setattr(
            session_mod, "_is_bounded_sdk_callback_string", counted_validator,
        )

        caplog.clear()
        tracemalloc.start()
        try:
            with caplog.at_level(
                logging.INFO, logger="agent.transports.claude_agent_sdk_session",
            ):
                result = asyncio.run(callback("Bash", {"command": "true"}, None))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert peak < 1_000_000
        assert hostile_category_counts == [17]
        assert marker not in caplog.text
        assert not any(record.exc_info for record in caplog.records)

    def test_overlong_multibyte_reason_rejects_without_full_copy_or_leak(
        self, caplog, monkeypatch,
    ):
        import tracemalloc

        from agent.transports import claude_agent_sdk_session as session_mod

        marker = "SDK_REASON_RESULT_SECRET_4af"
        huge_reason = marker + ("é" * (16 * 1024 * 1024))
        current_result = [{"choice": "deny", "reason": "ordinary denial"}]
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: current_result[0],
            permission_mode="default",
        )
        callback = session._make_can_use_tool()
        asyncio.run(callback("Bash", {"command": "true"}, None))
        current_result[0] = {"choice": "deny", "reason": huge_reason}
        category_calls = []
        hostile_category_counts = []
        original_category = session_mod.unicodedata.category
        original_validator = session_mod._is_bounded_sdk_callback_string

        def counted_category(char):
            category_calls.append(char)
            return original_category(char)

        def counted_validator(value, max_utf8_bytes, *, allow_space):
            before = len(category_calls)
            valid = original_validator(
                value, max_utf8_bytes, allow_space=allow_space,
            )
            if value is huge_reason:
                hostile_category_counts.append(len(category_calls) - before)
            return valid

        monkeypatch.setattr(session_mod.unicodedata, "category", counted_category)
        monkeypatch.setattr(
            session_mod, "_is_bounded_sdk_callback_string", counted_validator,
        )

        caplog.clear()
        tracemalloc.start()
        try:
            with caplog.at_level(
                logging.INFO, logger="agent.transports.claude_agent_sdk_session",
            ):
                result = asyncio.run(callback("Bash", {"command": "true"}, None))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert peak < 1_000_000
        assert hostile_category_counts == [271]
        assert marker not in caplog.text
        assert not any(record.exc_info for record in caplog.records)

    def test_hostile_callback_result_mapping_fails_closed_without_leak(self, caplog):
        marker = "SDK_RESULT_DECODE_SECRET_7d2"
        override_calls = []

        class HostileResult(dict):
            def __len__(self):
                override_calls.append("len")
                raise RuntimeError(marker)

            def __iter__(self):
                override_calls.append("iter")
                raise RuntimeError(marker)

            def __contains__(self, _key):
                override_calls.append("contains")
                raise RuntimeError(marker)

            def __getitem__(self, _key):
                override_calls.append("getitem")
                raise RuntimeError(marker)

            def get(self, *_args, **_kwargs):
                override_calls.append("get")
                raise RuntimeError(marker)

        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: HostileResult(choice="once"),
            permission_mode="default",
        )
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert override_calls == []
        assert marker not in caplog.text
        assert not any(record.exc_info for record in caplog.records)

    @pytest.mark.parametrize("callback_result", [
        "bogus",
        {"choice": "bogus"},
    ])
    def test_unknown_exact_callback_choice_is_protocol_failure(
        self, callback_result, caplog,
    ):
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: callback_result,
            permission_mode="default",
        )
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert "denied by user" not in caplog.text

    def test_callback_string_helper_enforces_exact_utf8_caps_and_text_contract(self):
        from agent.transports.claude_agent_sdk_session import (
            _is_bounded_sdk_callback_string,
        )

        class StringSubclass(str):
            pass

        assert _is_bounded_sdk_callback_string("é" * 256, 512, allow_space=True)
        assert not _is_bounded_sdk_callback_string("é" * 257, 512, allow_space=True)
        assert _is_bounded_sdk_callback_string("ordinary reason", 512, allow_space=True)
        assert not _is_bounded_sdk_callback_string("line\nbreak", 512, allow_space=True)
        assert not _is_bounded_sdk_callback_string("tab\tbreak", 512, allow_space=True)
        assert not _is_bounded_sdk_callback_string("bad\ud800reason", 512, allow_space=True)
        assert not _is_bounded_sdk_callback_string(
            StringSubclass("once"), 16, allow_space=False,
        )

    @pytest.mark.parametrize("budget", [0, 1, 5, 20, 21])
    def test_head_tail_helper_never_exceeds_supplied_budget(self, budget):
        from agent.transports.claude_agent_sdk_session import (
            _bounded_control_sanitized_head_tail,
        )

        rendered = _bounded_control_sanitized_head_tail("x" * 100, budget)
        assert len(rendered.encode("utf-8")) <= budget

    def test_head_tail_helper_preserves_multibyte_tail_within_budget(self):
        from agent.transports.claude_agent_sdk_session import (
            _bounded_control_sanitized_head_tail,
        )

        rendered = _bounded_control_sanitized_head_tail(
            ("α" * 100) + "\nTAIL", 64,
        )
        assert "[truncated]" in rendered
        assert rendered.endswith("TAIL")
        assert "\n" not in rendered
        assert len(rendered.encode("utf-8")) <= 64

    def test_huge_integer_denies_before_encoder_entry_on_exact_mcp_path(
        self, monkeypatch,
    ):
        import json
        import sys

        from agent.transports import claude_agent_sdk_session as session_mod

        old_limit = sys.get_int_max_str_digits()
        calls = []
        original = json.JSONEncoder.iterencode

        def observed_iterencode(encoder, value, *args, **kwargs):
            calls.append(value)
            return original(encoder, value, *args, **kwargs)

        monkeypatch.setattr(json.JSONEncoder, "iterencode", observed_iterencode)
        callback_calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: callback_calls.append((a, k)) or "once",
            permission_mode="default",
        )
        try:
            sys.set_int_max_str_digits(0)
            huge = 10 ** 99_999
            result = asyncio.run(session._make_can_use_tool()(
                "mcp__hermes-tools__read_file",
                {"path": "/tmp/x", "huge": huge},
                None,
            ))
        finally:
            sys.set_int_max_str_digits(old_limit)

        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "canonical request is unassessable"
        assert callback_calls == []
        assert calls == []

    def test_post_validation_mutation_cannot_reach_encoder_or_allocate_width(
        self, monkeypatch,
    ):
        import tracemalloc

        from agent.transports import claude_agent_sdk_session as session_mod

        payload = {"path": "/tmp/x", "nested": {"safe": True}}
        wide = {f"k{i}": i for i in range(100_000)}
        encoded_values = []
        original = session_mod._bounded_canonical_sdk_json

        def mutate_before_encoding(value):
            payload["nested"] = wide
            encoded_values.append(value)
            return original(value)

        monkeypatch.setattr(
            session_mod, "_bounded_canonical_sdk_json", mutate_before_encoding,
        )
        callback_calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: callback_calls.append((a, k)) or "once",
            permission_mode="default",
        )
        tracemalloc.start()
        try:
            result = asyncio.run(session._make_can_use_tool()(
                "mcp__hermes-tools__read_file", payload, None,
            ))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert type(result).__name__ == "PermissionResultAllow"
        assert callback_calls == []
        assert encoded_values
        assert all(
            value["tool_input"].get("nested") is not wide
            for value in encoded_values
            if type(value) is dict and type(value.get("tool_input")) is dict
        )
        assert peak < 2_000_000

    def test_alias_serialization_stops_near_canonical_byte_cap(self):
        import tracemalloc

        from agent.transports.claude_agent_sdk_session import (
            _canonical_sdk_tool_request,
        )

        big = 10 ** 3_999
        payload = {"path": "/tmp/x", "aliases": [big] * 9_000}
        tracemalloc.start()
        try:
            assert _canonical_sdk_tool_request(
                "mcp__hermes-tools__read_file", payload,
            ) is None
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2_000_000

    @pytest.mark.parametrize(("callback_result", "expected_type", "expected_message"), [
        ("once", "PermissionResultAllow", None),
        ("session", "PermissionResultAllow", None),
        ("always", "PermissionResultAllow", None),
        ({"choice": "once"}, "PermissionResultAllow", None),
        ("deny", "PermissionResultDeny", "approval denied by callback"),
        ("timeout", "PermissionResultDeny", "approval timed out — no operator response"),
        (
            {"choice": "deny", "reason": "approval expired (turn ended)"},
            "PermissionResultDeny",
            "approval expired (turn ended)",
        ),
    ])
    def test_callback_result_valid_shape_matrix_preserves_decision_and_input(
        self, callback_result, expected_type, expected_message,
    ):
        original = {"command": "printf safe", "nested": {"value": 1}}
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: callback_result,
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", original, None,
        ))

        assert type(result).__name__ == expected_type
        if expected_type == "PermissionResultAllow":
            assert result.updated_input == original
            assert result.updated_input is not original
            assert result.updated_input["nested"] is not original["nested"]
        else:
            assert result.message == expected_message

    def test_trusted_structural_operator_denial_preserves_reason_and_provenance(self):
        from tools import approval as approval_mod

        def trusted_callback(*_args, **_kwargs):
            return {
                "choice": "deny",
                "operator_denial": True,
                "reason": "not now",
            }

        assert sdk_gateway._register_trusted_sdk_gateway_approval_callback(
            trusted_callback,
        )
        session, _ = _make_session(
            approval_callback=trusted_callback,
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "true"}, None,
        ))

        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "denied by user: not now"

    @pytest.mark.parametrize("callback_result", [
        {"choice": "once", "extra": "malformed"},
        {"choice": "once", "reason": "unused"},
        {"choice": "timeout", "reason": "raw"},
        {"choice": "deny", "extra": "malformed"},
        {"choice": True},
        {"choice": "deny", "reason": True},
        {"choice": "deny", "operator_denial": False, "reason": ""},
        {"choice": "deny", "operator_denial": 1, "reason": ""},
        {"choice": "deny", "operator_denial": True, "reason": ""},
        {"choice": "deny", "reason": "line\nbreak"},
        {"choice": "deny", "reason": "bad\ud800reason"},
    ])
    def test_malformed_structured_callback_result_is_protocol_failure(
        self, callback_result,
    ):
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: callback_result,
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "true"}, None,
        ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"


    def test_canonical_utf8_cap_and_cap_plus_one_are_exact(self):
        from agent.transports.claude_agent_sdk_session import (
            _SDK_CANONICAL_MAX_UTF8_BYTES,
            validate_canonical_sdk_request_serialization,
        )

        prefix = '{"tool_input":{"path":"'
        suffix = '"},"tool_name":"Read"}'
        envelope_bytes = len((prefix + suffix).encode("utf-8"))
        fill = "é" * ((_SDK_CANONICAL_MAX_UTF8_BYTES - envelope_bytes) // 2)
        at_cap = prefix + fill + "a" + suffix
        cap_plus_one = prefix + fill + "aa" + suffix

        assert len(at_cap.encode("utf-8")) == _SDK_CANONICAL_MAX_UTF8_BYTES
        assert len(cap_plus_one.encode("utf-8")) == _SDK_CANONICAL_MAX_UTF8_BYTES + 1
        assert validate_canonical_sdk_request_serialization(at_cap) is not None
        assert validate_canonical_sdk_request_serialization(cap_plus_one) is None

    def test_oversized_multibyte_canonical_rejects_with_bounded_peak_allocation(
        self,
    ):
        import time
        import tracemalloc

        from agent.transports.claude_agent_sdk_session import (
            validate_canonical_sdk_request_serialization,
        )

        huge = "é" * (16 * 1024 * 1024)
        tracemalloc.start()
        started = time.perf_counter()
        try:
            assert validate_canonical_sdk_request_serialization(huge) is None
            elapsed = time.perf_counter() - started
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2_000_000
        assert elapsed < 1.0

    def test_gateway_validates_canonical_once_and_bounds_oversized_rejection(
        self, monkeypatch,
    ):
        import json
        import time
        import tracemalloc

        from agent.transports import claude_agent_sdk_session as session_mod
        from tools import approval as approval_mod

        sk = "sess-sdk-single-validation"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        token = approval_ctx.set_current_session_key(sk)
        calls = []
        original = session_mod.validate_canonical_sdk_request_serialization

        def counted(value):
            calls.append(value)
            return original(value)

        try:
            approval_mod.register_gateway_notify(
                sk,
                lambda _data: approval_mod.resolve_gateway_approval(sk, "deny"),
            )
            monkeypatch.setattr(
                session_mod, "validate_canonical_sdk_request_serialization", counted,
            )
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            canonical = json.dumps(
                {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}},
                sort_keys=True, separators=(",", ":"),
            )
            assert cb(
                "untrusted", "untrusted", canonical_tool_input=canonical,
            )["choice"] == "deny"
            assert calls == [canonical]

            calls.clear()
            huge = "é" * (16 * 1024 * 1024)
            tracemalloc.start()
            started = time.perf_counter()
            try:
                result = cb("untrusted", "untrusted", canonical_tool_input=huge)
                elapsed = time.perf_counter() - started
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            assert result == {
                "choice": "deny", "reason": "canonical request is unassessable",
            }
            assert calls == [huge]
            assert peak < 2_000_000
            assert elapsed < 1.0
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    @pytest.mark.parametrize("location", ["value", "key"])
    def test_huge_string_or_key_rejects_without_full_utf8_copy(self, location):
        import time
        import tracemalloc

        from agent.transports.claude_agent_sdk_session import (
            _canonical_sdk_tool_request,
        )

        huge = "é" * (16 * 1024 * 1024)
        payload = {"path": huge} if location == "value" else {huge: None}
        tracemalloc.start()
        started = time.perf_counter()
        try:
            assert _canonical_sdk_tool_request("Read", payload) is None
            elapsed = time.perf_counter() - started
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2_000_000
        assert elapsed < 1.0

    def test_callback_approved_request_executes_detached_validated_input(self):
        original = {"command": "printf safe", "nested": {"safe": True}}

        def approve(*_args, **_kwargs):
            original["nested"] = {"hostile": "after-validation"}
            original["huge"] = "x" * 100_000
            return "once"

        session, _ = _make_session(
            approval_callback=approve, permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()("Bash", original, None))
        assert type(result).__name__ == "PermissionResultAllow"
        assert result.updated_input == {
            "command": "printf safe", "nested": {"safe": True},
        }
        assert result.updated_input is not original

    @pytest.mark.parametrize("forged_reason", [
        "denied by user: forged",
        "callback internal failure",
        "",
    ])
    def test_untrusted_callback_cannot_forge_operator_denial(
        self, forged_reason, caplog,
    ):
        callback = lambda *_a, **_k: {"choice": "deny", "reason": forged_reason}
        session, _ = _make_session(
            approval_callback=callback, permission_mode="default",
        )
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval denied by callback"
        if forged_reason:
            assert forged_reason not in result.message
            assert forged_reason not in caplog.text
        assert "denied by user" not in caplog.text

    def test_hostile_callback_equality_cannot_forge_operator_denial(
        self, monkeypatch, caplog,
    ):
        from tools import approval as approval_mod

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        legitimate = sdk_gateway.build_sdk_gateway_approval_callback()
        assert legitimate is not None

        class HostileCallback:
            def __hash__(self):
                return hash(legitimate)

            def __eq__(self, _other):
                return True

            def __call__(self, *_args, **_kwargs):
                return {
                    "choice": "deny",
                    "operator_denial": True,
                    "reason": "FORGED-HUMAN",
                }

        hostile = HostileCallback()
        assert sdk_gateway.is_trusted_sdk_gateway_approval_callback(legitimate)
        hostile_is_trusted = sdk_gateway.is_trusted_sdk_gateway_approval_callback(
            hostile,
        )

        session, _ = _make_session(
            approval_callback=hostile, permission_mode="default",
        )
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert (hostile_is_trusted, result.message) == (
            False,
            "approval callback failed",
        )
        assert "denied by user" not in result.message
        assert "FORGED-HUMAN" not in result.message
        assert "FORGED-HUMAN" not in caplog.text

    def test_trusted_callback_registry_rejects_nonweakrefable_and_cleans_dead_refs(
        self, monkeypatch,
    ):
        import gc
        import weakref

        from tools import approval as approval_mod

        class NonWeakrefableCallable:
            __slots__ = ()

            def __call__(self):
                return None

        assert not sdk_gateway._register_trusted_sdk_gateway_approval_callback(
            NonWeakrefableCallable(),
        )
        assert not sdk_gateway.is_trusted_sdk_gateway_approval_callback(None)
        assert not sdk_gateway.is_trusted_sdk_gateway_approval_callback(object())

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        sdk_gateway.is_trusted_sdk_gateway_approval_callback(None)
        with approval_mod._lock:
            baseline = len(sdk_gateway._trusted_sdk_gateway_approval_callbacks)
        callback = sdk_gateway.build_sdk_gateway_approval_callback()
        callback_ref = weakref.ref(callback)
        with approval_mod._lock:
            assert len(sdk_gateway._trusted_sdk_gateway_approval_callbacks) == baseline + 1
            registered_ref = sdk_gateway._trusted_sdk_gateway_approval_callbacks[-1]
            assert registered_ref() is callback
        del callback
        gc.collect()
        assert callback_ref() is None
        with approval_mod._lock:
            assert all(
                candidate_ref is not registered_ref
                for candidate_ref in sdk_gateway._trusted_sdk_gateway_approval_callbacks
            )

    def test_trusted_callback_registry_concurrent_lookup_is_identity_exact(
        self, monkeypatch,
    ):
        from concurrent.futures import ThreadPoolExecutor

        from tools import approval as approval_mod

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        legitimate = sdk_gateway.build_sdk_gateway_approval_callback()

        class EqualProxy:
            def __hash__(self):
                return hash(legitimate)

            def __eq__(self, _other):
                return True

            def __call__(self):
                return None

        proxy = EqualProxy()
        callbacks = [legitimate, proxy, None, object()] * 64
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(
                sdk_gateway.is_trusted_sdk_gateway_approval_callback,
                callbacks,
            ))
        assert results == [True, False, False, False] * 64

    def test_trusted_gateway_operator_denial_is_structural(self, monkeypatch):
        from tools import approval as approval_mod

        sk = "sess-structural-operator-deny"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        token = approval_ctx.set_current_session_key(sk)
        try:
            approval_mod.register_gateway_notify(sk, lambda _data: None)
            monkeypatch.setattr(
                approval_gateway_wait,
                "_await_gateway_decision",
                lambda *_a, **_k: {
                    "resolved": True, "choice": "deny", "reason": "operator note",
                },
            )
            callback = sdk_gateway.build_sdk_gateway_approval_callback()
            canonical = (
                '{"tool_input":{"command":"true"},"tool_name":"Bash"}'
            )
            assert callback(
                "untrusted", "untrusted", canonical_tool_input=canonical,
            ) == {
                "choice": "deny",
                "operator_denial": True,
                "reason": "operator note",
            }
            session, _ = _make_session(
                approval_callback=callback, permission_mode="default",
            )
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
            assert result.message == "denied by user: operator note"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    def test_redactor_exception_is_fixed_fail_closed(self, monkeypatch, caplog):
        from agent.transports import claude_agent_sdk_session as session_mod

        marker = "REDACTOR_EXCEPTION_SECRET_a61"
        monkeypatch.setattr(
            session_mod,
            "redact_sensitive_text",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError(marker)),
        )
        callback_calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: callback_calls.append((a, k)) or "once",
            permission_mode="default",
        )
        with caplog.at_level(
            logging.WARNING, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                "Read", {"file_path": "/tmp/x"}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "canonical request is unassessable"
        assert callback_calls == []
        assert marker not in caplog.text
