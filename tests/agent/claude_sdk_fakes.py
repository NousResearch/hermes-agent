"""Tests for the claude-agent-sdk runtime (#25267).

Covers the three new modules end-to-end without requiring the optional
``claude-agent-sdk`` extra: the projector and session duck-type on class
NAMES, so local stand-in classes named like the SDK's types are the fixture.

Plant-the-failure discipline: every guard here is exercised RED first —
the auth classifier has a negative control (an ordinary error must NOT
produce the re-auth hint), and the session's error path is asserted to
retire the client rather than silently continue.

Shared stand-ins, fake clients and builders for the split
``tests/agent/test_claude_sdk_*.py`` modules (Push 3 of #65982). Not a
conftest on purpose: every module opts in explicitly.
"""

import asyncio
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from tools import approval_context as approval_ctx
from tools import approval_smart
from agent.transports.claude_agent_sdk_session import (
    ClaudeAgentSdkSession,
)


def isolate_provider_config(monkeypatch):
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

class ApprovalBridgeCase:
    """Shared helpers of the gateway approval bridge suite (the former
    ``TestGatewayApprovalBridge``): the planted SDK permission results, the
    gateway context, the canonical callback caller, the smart-approval
    recorder and the operator-choice resolver."""

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
    # --- Smart-approval Bash pre-filter -------------------------------------
    #
    # The SDK bridge calls the smart evaluator for every tool call that reaches
    # it, with a fixed pattern key and no screen in front of it. The evaluator's
    # two other entry points both screen first: the native shell path returns
    # approved when neither tirith nor the dangerous-pattern detector flagged
    # anything (approval.py's `if not warnings` early return), and the
    # execute_code path consults the approval cache (#39275, added precisely
    # because every call re-prompted). Both reach the evaluator through the
    # single call site in approval_smart.
    #
    # Measured on one live gateway session: 446 evaluator calls, ~432k tokens.
    #
    # The trade: a command both static screens clear no longer reaches the
    # guardian, so one the guardian would have denied is granted with no
    # operator card. That is exactly what the native early return already does.

    def _smart_recorder(self, monkeypatch, verdict="approve"):
        calls = []
        monkeypatch.setattr(
            approval_smart,
            "_smart_approve",
            lambda *a, **k: calls.append((a, k)) or verdict,
        )
        return calls
    def _resolve_with(self, approval_mod, session_key, choice):
        seen = []

        def notify(data):
            seen.append(data)
            with approval_mod._lock:
                entry = approval_mod._gateway_queues[session_key][0]
            entry.result = choice
            entry.event.set()

        return notify, seen
