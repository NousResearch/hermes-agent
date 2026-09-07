"""Interface-convergence regression tests for exec-approval request binding (#104915).

The gateway's single ``send_exec_approval(...)`` call site passes ``request_id``
to every adapter that accepts it. Before the convergence, the widened call raised
``TypeError`` on the Slack / Teams / Matrix / Feishu / Telegram signatures before
any approval prompt rendered. These tests pin, AST-level (no platform SDK imports):

1. every built-in adapter's ``send_exec_approval`` declares ``request_id``;
2. the shared call site gates the keyword through ``_accepts_keyword`` so
   external/plugin adapters predating the converged signature keep the legacy
   call shape instead of crashing.
"""

import ast
from pathlib import Path

import pytest

from agent.interrupt_compat import _accepts_keyword

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ADAPTER_FILES = [
    "gateway/platforms/qqbot/adapter.py",
    "gateway/platforms/whatsapp_cloud.py",
    "gateway/relay/adapter.py",
    "plugins/platforms/discord/adapter.py",
    "plugins/platforms/feishu/adapter.py",
    "plugins/platforms/matrix/adapter.py",
    "plugins/platforms/slack/adapter.py",
    "plugins/platforms/teams/adapter.py",
    "plugins/platforms/telegram/adapter.py",
]


def _send_exec_approval_args(rel_path: str) -> set[str]:
    tree = ast.parse((_REPO_ROOT / rel_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "send_exec_approval"
        ):
            return {a.arg for a in node.args.args + node.args.kwonlyargs}
    raise AssertionError(f"send_exec_approval not found in {rel_path}")


class TestConvergedSignature:
    @pytest.mark.parametrize("rel_path", _ADAPTER_FILES)
    def test_send_exec_approval_declares_request_id(self, rel_path):
        """The gateway call site passes request_id unconditionally to built-ins;
        each adapter must declare it (pre-convergence this raised TypeError)."""
        assert "request_id" in _send_exec_approval_args(rel_path), (
            f"{rel_path}: send_exec_approval must accept request_id (#104915)"
        )


class TestExternalAdapterCompat:
    def test_accepts_keyword_distinguishes_legacy_from_converged(self):
        """The call-site gate must keep legacy (external/plugin) adapters on the
        old call shape while passing request_id to converged ones."""

        async def legacy_send(chat_id, command, session_key, description="d",
                              metadata=None, allow_permanent=True, allow_session=True,
                              smart_denied=False):
            ...

        class Converged:
            async def send_exec_approval(self, chat_id, command, session_key,
                                         description="d", metadata=None,
                                         allow_permanent=True, allow_session=True,
                                         smart_denied=False, request_id=None):
                ...

        assert not _accepts_keyword(legacy_send, "request_id")
        assert _accepts_keyword(Converged.send_exec_approval, "request_id")


class TestLegacyAdapterFailsClosed:
    """Review follow-up (#104915): an adapter whose ``send_exec_approval`` predates the
    converged signature must not render an interactive approval at all. Its taps can only
    resolve through the session FIFO, so a stale/overlapping control could settle a
    different pending request — the notify path fails closed to the typed-text prompt
    (a clearly separate resolution path) instead of rendering an unbound card."""

    @staticmethod
    def _make_runner(adapter):
        from types import SimpleNamespace

        from gateway.run_turn_runner import TurnRunner

        runner = TurnRunner.__new__(TurnRunner)
        runner._ctx = SimpleNamespace(
            _status_adapter=adapter,
            _status_chat_id="chat-1",
            session_key="sess-1",
            _status_thread_metadata=None,
        )
        runner._close_native_stream_boundary = lambda *args, **kwargs: None
        return runner

    @pytest.mark.asyncio
    async def test_legacy_adapter_renders_typed_text_not_unbound_card(self):
        """A legacy adapter's send_exec_approval must never be called: the typed-text
        prompt (with the typed prefix) is the only approval surface it gets."""
        import asyncio

        class LegacyAdapter:
            typed_command_prefix = "!"

            def pause_typing_for_chat(self, chat_id): ...

            async def send_exec_approval(self, chat_id, command, session_key, description="d",
                                         metadata=None, allow_permanent=True,
                                         allow_session=True, smart_denied=False):
                raise AssertionError(
                    "legacy adapter must not render an interactive approval (#104915)"
                )

            sent = []

            async def send(self, chat_id, msg, metadata=None):
                from gateway.platforms.base import SendResult

                self.sent.append((msg, metadata))
                return SendResult(success=True)

        adapter = LegacyAdapter()
        runner = self._make_runner(adapter)
        loop = asyncio.get_running_loop()
        runner._schedule = lambda coro, log_message, loop_arg=None: asyncio.run_coroutine_threadsafe(coro, loop)

        def notify():
            runner._approval_notify_sync(
                {"command": "rm -rf /tmp/probe", "description": "probe", "request_id": "req-legacy-1"}
            )

        await asyncio.to_thread(notify)

        assert adapter.sent, "the typed-text fallback must fire for a legacy adapter"
        msg, metadata = adapter.sent[0]
        assert "`!approve`" in msg, "the prompt must use the adapter's typed prefix"
        assert metadata.get("is_approval_prompt") is True

    @pytest.mark.asyncio
    async def test_converged_adapter_still_receives_request_id(self):
        """Counterpart: a converged adapter keeps the interactive card and the card is
        bound to its request generation."""
        import asyncio

        class ConvergedAdapter:
            typed_command_prefix = "/"

            def pause_typing_for_chat(self, chat_id): ...

            approvals = []
            sent = []

            async def send_exec_approval(self, chat_id, command, session_key, description="d",
                                         metadata=None, allow_permanent=True,
                                         allow_session=True, smart_denied=False,
                                         request_id=None):
                from gateway.platforms.base import SendResult

                self.approvals.append(request_id)
                return SendResult(success=True)

            async def send(self, chat_id, msg, metadata=None):
                from gateway.platforms.base import SendResult

                self.sent.append((msg, metadata))
                return SendResult(success=True)

        adapter = ConvergedAdapter()
        runner = self._make_runner(adapter)
        loop = asyncio.get_running_loop()
        runner._schedule = lambda coro, log_message, loop_arg=None: asyncio.run_coroutine_threadsafe(coro, loop)

        await asyncio.to_thread(
            lambda: runner._approval_notify_sync(
                {"command": "probe", "description": "probe", "request_id": "req-conv-1"}
            )
        )

        assert adapter.approvals == ["req-conv-1"], "converged card must receive the request id"
        assert not adapter.sent, "interactive path delivered; no text fallback expected"
