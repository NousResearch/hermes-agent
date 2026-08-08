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

    monkeypatch.setattr(cfg, "load_config_readonly", lambda *a, **k: {}, raising=False)


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

    def test_hint_preserves_underlying_error(self):
        # A hit RETIRES the session, so the true error must survive in the
        # message — a misclassification that also swallows the evidence is
        # undebuggable.
        hint = classify_auth_failure("HTTP 401 unauthorized: oauth token expired")
        assert "401 unauthorized" in hint

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
        self.queried: list[str] = []
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
        self.queried.append(text)
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

    def test_sdk_error_result_surfaces(self):
        script = [ResultMessage(subtype="error_max_turns", is_error=False)]
        session, _ = _make_session(script=script)
        try:
            turn = session.run_turn("hi")
        finally:
            session.close()
        assert turn.error is not None
        assert "error_max_turns" in turn.error

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
        assert mcp["args"] == ["-m", "agent.transports.hermes_tools_mcp_server"]
        # Hard rule: a metered key never reaches any child of this runtime.
        assert "ANTHROPIC_API_KEY" not in (mcp.get("env") or {})
        assert options["permission_mode"] in {
            "acceptEdits", "default", "bypassPermissions",
        }
        # Explicit SDK isolation: None would load ALL of ~/.claude and
        # .claude/settings*, letting ambient settings shadow the gateway's
        # approval posture. The empty list is the SDK's isolation mode.
        assert options["setting_sources"] == []

    def test_askuserquestion_in_disallowed_tools(self):
        # No answer channel for AskUserQuestion in hermes — the model must
        # ask in plain text; the tool is removed from its context.
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        fields = session.build_option_fields()
        assert fields["disallowed_tools"] == ["AskUserQuestion"]

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
        # A typo must never silently change the posture — the env mapping
        # stands (default env → acceptEdits).
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
        assert session.build_option_fields()["permission_mode"] == "acceptEdits"

    def test_empty_config_permission_mode_keeps_env_mapping(self, monkeypatch):
        # "" (the canonical default) = current behavior: the
        # HERMES_TERMINAL_SECURITY_MODE mapping stands.
        monkeypatch.setenv("HERMES_TERMINAL_SECURITY_MODE", "approval-required")
        session, _ = _make_session(script=[ResultMessage(result="ok")])
        assert session.build_option_fields()["permission_mode"] == "default"

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


# ---------- background review must not spawn on this runtime ----------


class TestBackgroundReviewSuppressed:
    """The review fork inherits ``api_mode="claude_agent_sdk"`` and lands in
    a fresh SDK session whose tool surface has no ``memory``/``skill_manage``
    — it burns a subscription turn and cannot write anything. The runtime
    must therefore never spawn it, while the nudge counters keep ticking so
    a bounded replacement pass can reuse them. (#25267)"""

    def test_memory_nudge_does_not_spawn_review(self):
        agent = _make_agent()
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
            should_review_memory=True,
        )
        agent._spawn_background_review.assert_not_called()

    def test_skill_nudge_does_not_spawn_review_but_counter_still_ticks(self):
        agent = _make_agent()
        agent._skill_nudge_interval = 1
        agent.valid_tool_names = {"skill_manage"}
        run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}],
            effective_task_id="task-1",
        )
        agent._spawn_background_review.assert_not_called()
        # Counter machinery stays intact: the interval crossing still resets
        # it, exactly as before — only the spawn is suppressed.
        assert agent._iters_since_skill == 0


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
        # so the session line and platform hint reflect the live session.
        import agent.claude_sdk_runtime as rt
        import agent.transports.claude_agent_sdk_session as sdk_session_mod

        captured = {}

        def fake_append(**kwargs):
            captured.update(kwargs)
            return "APPEND-UNDER-TEST"

        class SpySession:
            def __init__(self, **kwargs):
                pass

            def run_turn(self, user_input):
                return _make_turn()

        monkeypatch.setattr(rt, "build_system_prompt_append", fake_append)
        monkeypatch.setattr(sdk_session_mod, "ClaudeAgentSdkSession", SpySession)
        agent = _make_agent()
        agent._claude_sdk_session = None
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
        seen = []
        agent.stream_delta_callback = seen.append  # assigned AFTER creation
        relay("delta-text")
        assert seen == ["delta-text"]
        agent.stream_delta_callback = None  # cleared between turns → no crash
        relay("dropped")
        assert seen == ["delta-text"]


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
    def _home(tmp_path, monkeypatch, *, soul=None, memory=None, user=None):
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
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"append_file": append_file}}
            },
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


class TestAuxLaneFailClosed:
    def test_aux_auto_detect_disabled_under_claude_sdk(self, monkeypatch):
        # Validator C7 (HIGH): with the main provider on the subscription
        # lane, aux tasks (title-gen, compression) silently fell through to
        # the metered OpenRouter/Nous auto-detect chain. Auto-detect must
        # fail closed; explicit aux config remains the operator's opt-in.
        from agent.auxiliary_client import _resolve_auto

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake-key")
        client, model = _resolve_auto(main_runtime={
            "provider": "claude-agent-sdk",
            "model": "claude-opus-4-8",
            "api_mode": "claude_agent_sdk",
            "base_url": "",
            "api_key": "claude-subscription-oauth",
        })
        assert client is None and model is None


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
        check_claude_sdk_available()
        assert called == {"feature": "provider.claude_agent_sdk", "prompt": False}

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
