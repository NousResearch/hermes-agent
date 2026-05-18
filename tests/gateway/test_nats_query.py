"""Phase 6 (T6.4): mid-stream query & approval round-trip for the NATS adapter.

Covers, per docs/nats-gateway-design.md §7:

* :meth:`NatsAdapter.request_interaction` — resolves the active
  :class:`PromptStream` via the T5.0 contextvar-first lookup, forwards
  the prompt to ``stream.ask(timeout=…)``, and maps ``QueryTimeout`` /
  no-stream / arbitrary exceptions to ``None``.
* Module-level helpers in :mod:`gateway.platforms.base`:
  ``_format_approval_prompt``, ``_parse_approval_reply``,
  ``adapter_supports_request_interaction``, and
  ``dispatch_approval_via_request_interaction`` — the shared plumbing
  that both ``run.py:_approval_notify_sync`` and
  ``nats.py:_run_agent_sync`` route through.
* End-to-end: register a gateway notify callback on a live session,
  invoke it with a synthetic approval request, and verify the adapter's
  ``stream.ask`` fires AND ``resolve_gateway_approval`` is called with
  the normalized choice — i.e. the agent thread unblock path is wired.

The SDK is still mocked via ``tests/gateway/conftest.py::_ensure_natsagent_mock``.
No real NATS broker is touched.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import base as base_mod
from gateway.platforms.base import (
    BasePlatformAdapter,
    _format_approval_prompt,
    _parse_approval_reply,
    adapter_supports_request_interaction,
    dispatch_approval_via_request_interaction,
)
from gateway.platforms.nats import NatsAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_extra(**overrides) -> dict:
    base = {
        "servers": ["nats://127.0.0.1:4222"],
        "owner": "rene",
        "name": "gateway",
        "ack_keepalive_interval_s": 1,
    }
    base.update(overrides)
    return base


def _build_adapter(**extra_overrides) -> NatsAdapter:
    return NatsAdapter(PlatformConfig(enabled=True, extra=_valid_extra(**extra_overrides)))


def _fake_stream(reply_text: str | None = None, *, raises=None) -> MagicMock:
    """Build a PromptStream-shaped MagicMock with an async ``ask``.

    ``reply_text`` becomes the returned envelope's ``.prompt`` attribute.
    If ``raises`` is set, ``ask`` raises that instead.
    """
    stream = MagicMock()
    stream.send = AsyncMock()
    if raises is not None:
        stream.ask = AsyncMock(side_effect=raises)
    else:
        reply = MagicMock()
        reply.prompt = reply_text
        stream.ask = AsyncMock(return_value=reply)
    return stream


# ---------------------------------------------------------------------------
# _parse_approval_reply
# ---------------------------------------------------------------------------


class TestParseApprovalReply:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("once", "once"),
            ("ONCE", "once"),
            ("yes", "once"),
            ("y", "once"),
            ("ok", "once"),
            ("approve", "once"),
            ("o", "once"),
            ("session", "session"),
            ("S", "session"),
            ("always", "always"),
            ("A", "always"),
            ("permanent", "always"),
            ("deny", "deny"),
            ("no", "deny"),
            ("cancel", "deny"),
            ("reject", "deny"),
        ],
    )
    def test_canonical_mappings(self, reply, expected):
        assert _parse_approval_reply(reply) == expected

    def test_first_token_wins(self):
        # Casual user replies — we take the first whitespace-split token
        # so "yes please" and "approve this one" still classify.
        assert _parse_approval_reply("yes please") == "once"
        assert _parse_approval_reply("approve this one") == "once"
        assert _parse_approval_reply("session thanks") == "session"
        assert _parse_approval_reply("deny immediately") == "deny"

    def test_none_defaults_to_deny(self):
        assert _parse_approval_reply(None) == "deny"

    def test_empty_string_defaults_to_deny(self):
        assert _parse_approval_reply("") == "deny"
        assert _parse_approval_reply("    ") == "deny"

    def test_unknown_token_defaults_to_deny(self):
        # Fail-safe: anything we don't recognize is a deny, not an "once".
        assert _parse_approval_reply("maybe") == "deny"
        assert _parse_approval_reply("hmm") == "deny"
        assert _parse_approval_reply("whatever") == "deny"

    def test_non_string_defaults_to_deny(self):
        assert _parse_approval_reply(42) == "deny"  # type: ignore[arg-type]
        assert _parse_approval_reply(True) == "deny"  # type: ignore[arg-type]
        assert _parse_approval_reply([]) == "deny"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _format_approval_prompt
# ---------------------------------------------------------------------------


class TestFormatApprovalPrompt:
    def test_includes_command_and_description(self):
        out = _format_approval_prompt({
            "command": "rm -rf /tmp/foo",
            "description": "recursive delete",
        })
        assert "rm -rf /tmp/foo" in out
        assert "recursive delete" in out
        assert "once" in out and "session" in out and "always" in out and "deny" in out

    def test_truncates_long_commands(self):
        big = "x" * 2000
        out = _format_approval_prompt({"command": big, "description": "desc"})
        # Must not balloon the prompt with the entire 2 KB command —
        # callers have to read this line and act on it.
        assert len(out) < 1000
        assert "…" in out

    def test_missing_fields_use_safe_defaults(self):
        out = _format_approval_prompt({})
        # Description falls back to something generic; empty command is fine.
        assert "dangerous command" in out


# ---------------------------------------------------------------------------
# adapter_supports_request_interaction
# ---------------------------------------------------------------------------


class TestAdapterCapabilityCheck:
    def test_nats_adapter_is_detected(self):
        adapter = _build_adapter()
        assert adapter_supports_request_interaction(adapter) is True

    def test_base_adapter_is_not_detected(self):
        # Build a minimal subclass that inherits the base default so we
        # don't have to spin up a concrete adapter class.
        class _BareAdapter(BasePlatformAdapter):
            async def connect(self) -> bool:
                return True

            async def disconnect(self) -> None:
                return None

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                from gateway.platforms.base import SendResult
                return SendResult(success=True)

            async def get_chat_info(self, chat_id):
                return {"name": chat_id, "type": "dm"}

        from gateway.config import Platform
        config = PlatformConfig(enabled=True, extra={})
        adapter = _BareAdapter(config, Platform.TELEGRAM)
        assert adapter_supports_request_interaction(adapter) is False


# ---------------------------------------------------------------------------
# NatsAdapter.request_interaction
# ---------------------------------------------------------------------------


class TestRequestInteraction:
    @pytest.mark.asyncio
    async def test_resolves_stream_and_returns_reply_prompt(self, monkeypatch):
        adapter = _build_adapter()
        stream = _fake_stream(reply_text="yes please")
        adapter._active_streams[("alice", id(stream))] = stream

        reply = await adapter.request_interaction(
            chat_id="alice",
            prompt="approve?",
            kind="approval",
            timeout=10.0,
        )

        assert reply == "yes please"
        stream.ask.assert_awaited_once()
        # Positional first arg is the prompt text; timeout is kw.
        call = stream.ask.await_args
        assert call.args[0] == "approve?"
        assert call.kwargs["timeout"] == 10.0

    @pytest.mark.asyncio
    async def test_prefers_contextvar_over_dict(self):
        adapter = _build_adapter()
        dict_stream = _fake_stream(reply_text="dict")
        ctx_stream = _fake_stream(reply_text="ctx")
        # Register dict_stream under "alice" — but contextvar should win.
        adapter._active_streams[("alice", id(dict_stream))] = dict_stream

        import gateway.platforms.nats as nats_mod
        token = nats_mod._current_stream.set(ctx_stream)
        try:
            reply = await adapter.request_interaction(
                chat_id="alice",
                prompt="approve?",
                kind="approval",
                timeout=1.0,
            )
        finally:
            nats_mod._current_stream.reset(token)

        assert reply == "ctx"
        ctx_stream.ask.assert_awaited_once()
        dict_stream.ask.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_stream_returns_none(self):
        adapter = _build_adapter()
        # No registrations, no contextvar → graceful None so approvals
        # fail safe rather than raising from the agent's worker thread.
        reply = await adapter.request_interaction(
            chat_id="ghost",
            prompt="approve?",
            kind="approval",
            timeout=1.0,
        )
        assert reply is None

    @pytest.mark.asyncio
    async def test_query_timeout_maps_to_none(self):
        adapter = _build_adapter()
        query_timeout_cls = sys.modules["natsagent"].QueryTimeout
        stream = _fake_stream(raises=query_timeout_cls("no reply"))
        adapter._active_streams[("alice", id(stream))] = stream

        reply = await adapter.request_interaction(
            chat_id="alice",
            prompt="approve?",
            kind="approval",
            timeout=1.0,
        )
        assert reply is None

    @pytest.mark.asyncio
    async def test_generic_exception_maps_to_none(self):
        adapter = _build_adapter()
        stream = _fake_stream(raises=RuntimeError("broken pipe"))
        adapter._active_streams[("alice", id(stream))] = stream

        reply = await adapter.request_interaction(
            chat_id="alice",
            prompt="approve?",
            kind="approval",
            timeout=1.0,
        )
        assert reply is None


# ---------------------------------------------------------------------------
# dispatch_approval_via_request_interaction
# ---------------------------------------------------------------------------


class TestDispatchApproval:
    @pytest.mark.asyncio
    async def test_returns_false_when_adapter_inherits_base_default(self):
        class _BareAdapter(BasePlatformAdapter):
            async def connect(self) -> bool:
                return True

            async def disconnect(self) -> None:
                return None

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                from gateway.platforms.base import SendResult
                return SendResult(success=True)

            async def get_chat_info(self, chat_id):
                return {"name": chat_id, "type": "dm"}

        from gateway.config import Platform
        config = PlatformConfig(enabled=True, extra={})
        adapter = _BareAdapter(config, Platform.TELEGRAM)
        loop = asyncio.get_running_loop()
        approval = {"command": "rm -rf /", "description": "recursive delete"}

        scheduled = dispatch_approval_via_request_interaction(
            adapter,
            "alice",
            "agent:main:telegram:dm:alice",
            approval,
            loop,
            timeout=10.0,
        )
        assert scheduled is False

    @pytest.mark.asyncio
    async def test_schedules_request_interaction_and_resolves_approval(self, monkeypatch):
        adapter = _build_adapter()
        stream = _fake_stream(reply_text="session")
        adapter._active_streams[("alice", id(stream))] = stream

        # Spy on resolve_gateway_approval so we can assert the helper
        # actually unblocks the waiting agent thread after the reply.
        resolved: list[tuple[str, str]] = []

        def _fake_resolve(session_key, choice, resolve_all=False):
            resolved.append((session_key, choice))
            return 1

        import tools.approval as approval_mod
        monkeypatch.setattr(approval_mod, "resolve_gateway_approval", _fake_resolve)

        loop = asyncio.get_running_loop()
        approval = {"command": "rm -rf /tmp/foo", "description": "recursive delete"}

        scheduled = dispatch_approval_via_request_interaction(
            adapter,
            "alice",
            "agent:main:nats:dm:alice",
            approval,
            loop,
            timeout=10.0,
        )
        assert scheduled is True

        # The helper scheduled a coroutine on `loop`. Yield so it runs.
        # One ``await`` isn't enough because ``run_coroutine_threadsafe``
        # wraps the coroutine in a task that needs to be dispatched.
        for _ in range(10):
            await asyncio.sleep(0)
            if resolved:
                break

        assert resolved == [("agent:main:nats:dm:alice", "session")]
        stream.ask.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_timeout_reply_resolves_as_deny(self, monkeypatch):
        adapter = _build_adapter()
        query_timeout_cls = sys.modules["natsagent"].QueryTimeout
        stream = _fake_stream(raises=query_timeout_cls("no reply"))
        adapter._active_streams[("bob", id(stream))] = stream

        resolved: list[tuple[str, str]] = []

        def _fake_resolve(session_key, choice, resolve_all=False):
            resolved.append((session_key, choice))
            return 1

        import tools.approval as approval_mod
        monkeypatch.setattr(approval_mod, "resolve_gateway_approval", _fake_resolve)

        loop = asyncio.get_running_loop()
        approval = {"command": "kill -9 1", "description": "kill init"}

        scheduled = dispatch_approval_via_request_interaction(
            adapter,
            "bob",
            "agent:main:nats:dm:bob",
            approval,
            loop,
            timeout=0.5,
        )
        assert scheduled is True

        for _ in range(10):
            await asyncio.sleep(0)
            if resolved:
                break

        # Timeout → None → "deny"
        assert resolved == [("agent:main:nats:dm:bob", "deny")]

    @pytest.mark.asyncio
    async def test_unknown_reply_resolves_as_deny(self, monkeypatch):
        # The caller replied but with gibberish — safer to treat as deny
        # than to auto-approve on anything-not-explicit-deny.
        adapter = _build_adapter()
        stream = _fake_stream(reply_text="maybe later")
        adapter._active_streams[("carol", id(stream))] = stream

        resolved: list[tuple[str, str]] = []

        def _fake_resolve(session_key, choice, resolve_all=False):
            resolved.append((session_key, choice))
            return 1

        import tools.approval as approval_mod
        monkeypatch.setattr(approval_mod, "resolve_gateway_approval", _fake_resolve)

        loop = asyncio.get_running_loop()
        approval = {"command": "dd if=/dev/zero", "description": "disk copy"}

        scheduled = dispatch_approval_via_request_interaction(
            adapter,
            "carol",
            "agent:main:nats:dm:carol",
            approval,
            loop,
            timeout=1.0,
        )
        assert scheduled is True

        for _ in range(10):
            await asyncio.sleep(0)
            if resolved:
                break

        assert resolved == [("agent:main:nats:dm:carol", "deny")]


# ---------------------------------------------------------------------------
# Integration — gateway notify callback → request_interaction → resolve
# ---------------------------------------------------------------------------


class TestGatewayApprovalIntegration:
    """Stand-in for the end-to-end wiring inside ``_run_agent_sync``.

    We don't spin up a real ``AIAgent`` — instead we imitate its notify
    path: register a callback that uses the dispatch helper, then invoke
    it as ``check_all_command_guards`` would (from the agent worker
    thread via ``run_in_executor``) and verify the approval entry gets
    resolved on the caller's reply.
    """

    @pytest.mark.asyncio
    async def test_notify_callback_resolves_pending_approval_on_reply(
        self, monkeypatch
    ):
        adapter = _build_adapter()
        stream = _fake_stream(reply_text="always")
        adapter._active_streams[("alice", id(stream))] = stream

        session_key = "agent:main:nats:dm:alice"
        resolved: list[tuple[str, str]] = []

        def _fake_resolve(sk, choice, resolve_all=False):
            resolved.append((sk, choice))
            return 1

        import tools.approval as approval_mod
        monkeypatch.setattr(approval_mod, "resolve_gateway_approval", _fake_resolve)

        loop = asyncio.get_running_loop()

        def _notify(approval_data):
            dispatch_approval_via_request_interaction(
                adapter,
                "alice",
                session_key,
                approval_data,
                loop,
                timeout=5.0,
            )

        # Simulate check_all_command_guards firing the callback from a
        # worker thread. The callback schedules the coroutine and returns
        # immediately — the agent thread would then wait on ApprovalEntry.
        _notify({"command": "rm -rf /tmp/foo", "description": "recursive delete"})

        for _ in range(10):
            await asyncio.sleep(0)
            if resolved:
                break

        assert resolved == [(session_key, "always")]
        stream.ask.assert_awaited_once()
        # Prompt must have made it through the formatter.
        ask_prompt = stream.ask.await_args.args[0]
        assert "rm -rf /tmp/foo" in ask_prompt
        assert "recursive delete" in ask_prompt


# ---------------------------------------------------------------------------
# Sanity: run.py imports the helpers we export
# ---------------------------------------------------------------------------


class TestModuleSurface:
    def test_base_exports_helpers(self):
        # Guards against a rename silently breaking run.py's import.
        assert hasattr(base_mod, "dispatch_approval_via_request_interaction")
        assert hasattr(base_mod, "adapter_supports_request_interaction")
        assert hasattr(base_mod, "_parse_approval_reply")
        assert hasattr(base_mod, "_format_approval_prompt")

    def test_base_adapter_default_raises(self):
        # Default must raise so the capability check's identity comparison
        # accurately distinguishes overriders from non-overriders.
        import inspect
        src = inspect.getsource(BasePlatformAdapter.request_interaction)
        assert "NotImplementedError" in src
