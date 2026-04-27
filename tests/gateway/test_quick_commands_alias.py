"""Regression tests for #16094: gateway quick_commands `type: alias`
must dispatch to built-in slash commands, not return "Unknown command".

The CLI implementation in `cli.py` already does this correctly via
`process_command(aliased_command)`. The gateway implementation in
`gateway/run.py` previously stored the whole compound target string into
`command` and "fell through" to the plugin/skill checks below — none of
which can dispatch a built-in command, so the user got a confusing
"Unknown command `/model claude-... --provider ...`" reply.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _build_runner(monkeypatch, tmp_path, quick_commands):
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = GatewayRunner(GatewayConfig())
    # The dispatcher accepts dict OR object-with-attribute for self.config.
    # A dict carrying just `quick_commands` is the smallest fixture that
    # exercises the alias branch.
    runner.config = {"quick_commands": quick_commands}
    return runner


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="user-1",
        ),
        message_id="m1",
    )


@pytest.mark.asyncio
async def test_alias_to_built_in_dispatches_to_built_in_handler(monkeypatch, tmp_path):
    """An alias targeting /help must invoke _handle_help_command."""
    runner = _build_runner(
        monkeypatch, tmp_path,
        {"sonnet": {"type": "alias", "target": "/help"}},
    )
    monkeypatch.setattr(GatewayRunner, "_is_user_authorized", lambda self, src: True)

    help_handler = AsyncMock(return_value="HELP RESPONSE")
    monkeypatch.setattr(GatewayRunner, "_handle_help_command", help_handler)

    result = await runner._handle_message(_make_event("/sonnet"))

    assert help_handler.await_count == 1, (
        "Without #16094 fix, alias falls through to plugin/skill checks "
        "and never reaches the built-in canonical dispatch."
    )
    assert result == "HELP RESPONSE"


@pytest.mark.asyncio
async def test_alias_forwards_user_args_to_target(monkeypatch, tmp_path):
    """Args after the alias must be appended to the resolved target text."""
    runner = _build_runner(
        monkeypatch, tmp_path,
        {"m": {"type": "alias", "target": "/model"}},
    )
    monkeypatch.setattr(GatewayRunner, "_is_user_authorized", lambda self, src: True)

    captured = []

    async def _capture_model(self, event):
        captured.append(event.text)
        return "ok"

    monkeypatch.setattr(GatewayRunner, "_handle_model_command", _capture_model)

    await runner._handle_message(
        _make_event("/m claude-sonnet-4-6 --provider anthropic --global"),
    )

    assert captured == [
        "/model claude-sonnet-4-6 --provider anthropic --global",
    ], (
        "The alias's user args must be appended to the target command, and "
        "the recursive _handle_message call must see the full target text."
    )


@pytest.mark.asyncio
async def test_alias_target_with_embedded_args_dispatches_correctly(monkeypatch, tmp_path):
    """Aliases that pre-bake args (the original repro from #16094) must work."""
    runner = _build_runner(
        monkeypatch, tmp_path,
        {
            "sonnet": {
                "type": "alias",
                "target": "/model claude-sonnet-4-6 --provider anthropic --global",
            },
        },
    )
    monkeypatch.setattr(GatewayRunner, "_is_user_authorized", lambda self, src: True)

    captured = []

    async def _capture_model(self, event):
        captured.append(event.text)
        return "model switched"

    monkeypatch.setattr(GatewayRunner, "_handle_model_command", _capture_model)

    result = await runner._handle_message(_make_event("/sonnet"))

    assert captured == [
        "/model claude-sonnet-4-6 --provider anthropic --global",
    ]
    assert result == "model switched"


@pytest.mark.asyncio
async def test_alias_depth_guard_breaks_recursive_loops(monkeypatch, tmp_path):
    """Alias-of-alias-of-... must terminate, not infinitely recurse."""
    # a → b → a → b → ... (mutual recursion).
    runner = _build_runner(
        monkeypatch, tmp_path,
        {
            "a": {"type": "alias", "target": "/b"},
            "b": {"type": "alias", "target": "/a"},
        },
    )
    monkeypatch.setattr(GatewayRunner, "_is_user_authorized", lambda self, src: True)

    result = await runner._handle_message(_make_event("/a"))

    assert result is not None
    assert "alias depth exceeded" in result.lower(), (
        "Mutual-recursion aliases must terminate with an error, not exhaust "
        "Python's recursion limit."
    )


@pytest.mark.asyncio
async def test_alias_with_no_target_returns_error(monkeypatch, tmp_path):
    runner = _build_runner(
        monkeypatch, tmp_path,
        {"broken": {"type": "alias", "target": ""}},
    )
    monkeypatch.setattr(GatewayRunner, "_is_user_authorized", lambda self, src: True)

    result = await runner._handle_message(_make_event("/broken"))
    assert result is not None
    assert "no target defined" in result.lower()


# ---------------------------------------------------------------------------
# Helper unit tests: resolve_quick_command_alias_text
# ---------------------------------------------------------------------------


def _patch_quick_commands(monkeypatch, quick_commands):
    """Make hermes_cli.commands.read_raw_config return a config dict
    carrying the supplied quick_commands."""
    from hermes_cli import config as config_mod

    monkeypatch.setattr(
        config_mod, "read_raw_config",
        lambda: {"quick_commands": quick_commands},
    )


class TestResolveQuickCommandAliasText:
    def test_resolves_simple_alias(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch, {"halt": {"type": "alias", "target": "/stop"}},
        )
        assert resolve_quick_command_alias_text("/halt") == "/stop"

    def test_resolves_alias_with_user_args_appended(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch, {"m": {"type": "alias", "target": "/model"}},
        )
        assert (
            resolve_quick_command_alias_text("/m claude-sonnet-4-6 --provider anthropic")
            == "/model claude-sonnet-4-6 --provider anthropic"
        )

    def test_resolves_alias_with_pre_baked_target_args(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch,
            {"sonnet": {
                "type": "alias",
                "target": "/model claude-sonnet-4-6 --provider anthropic --global",
            }},
        )
        assert (
            resolve_quick_command_alias_text("/sonnet")
            == "/model claude-sonnet-4-6 --provider anthropic --global"
        )

    def test_normalizes_target_without_leading_slash(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch, {"halt": {"type": "alias", "target": "stop"}},
        )
        assert resolve_quick_command_alias_text("/halt") == "/stop"

    def test_returns_none_for_exec_quick_commands(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch, {"dn": {"type": "exec", "command": "echo daily-note"}},
        )
        assert resolve_quick_command_alias_text("/dn") is None

    def test_returns_none_for_unknown_command(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch, {"halt": {"type": "alias", "target": "/stop"}},
        )
        assert resolve_quick_command_alias_text("/unknown-thing") is None

    def test_returns_none_for_non_slash_text(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch, {"halt": {"type": "alias", "target": "/stop"}},
        )
        assert resolve_quick_command_alias_text("hello world") is None

    def test_returns_none_for_alias_with_empty_target(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch, {"broken": {"type": "alias", "target": "  "}},
        )
        assert resolve_quick_command_alias_text("/broken") is None

    def test_resolves_uppercase_alias(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch, {"halt": {"type": "alias", "target": "/stop"}},
        )
        assert resolve_quick_command_alias_text("/HALT") == "/stop"
        assert resolve_quick_command_alias_text("/Halt") == "/stop"

    def test_resolves_alias_with_bot_suffix(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch, {"halt": {"type": "alias", "target": "/stop"}},
        )
        assert resolve_quick_command_alias_text("/halt@HermesBot") == "/stop"
        assert resolve_quick_command_alias_text("/HALT@HermesBot") == "/stop"

    def test_resolves_uppercase_alias_with_bot_suffix_and_args(self, monkeypatch):
        from hermes_cli.commands import resolve_quick_command_alias_text
        _patch_quick_commands(
            monkeypatch, {"m": {"type": "alias", "target": "/model"}},
        )
        assert (
            resolve_quick_command_alias_text("/M@HermesBot claude-sonnet-4-6")
            == "/model claude-sonnet-4-6"
        )


# ---------------------------------------------------------------------------
# Adapter-level tests: aliased commands bypass active-session guard
# ---------------------------------------------------------------------------


class _StubAdapter(BasePlatformAdapter):
    """Concrete adapter with abstract methods stubbed out."""

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _make_stub_adapter():
    config = PlatformConfig(enabled=True, token="test-token")
    adapter = _StubAdapter(config, Platform.TELEGRAM)
    adapter.sent_responses = []

    async def _mock_handler(event):
        cmd = event.get_command()
        return f"handled:{cmd}:{event.text}" if cmd else f"handled:text:{event.text}"

    adapter._message_handler = _mock_handler

    async def _mock_send_retry(chat_id, content, **kwargs):
        adapter.sent_responses.append(content)

    adapter._send_with_retry = _mock_send_retry
    return adapter


def _make_adapter_event(text, chat_id="12345"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm",
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


def _adapter_session_key(chat_id="12345"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm",
    )
    return build_session_key(source)


class TestAliasBypassesActiveSessionGuard:
    """Aliases that resolve to bypass-eligible built-ins must inherit
    the bypass behavior — otherwise /halt → /stop is queued/interrupted
    while /stop sails through, defeating the alias."""

    @pytest.mark.asyncio
    async def test_aliased_stop_bypasses_guard(self, monkeypatch):
        # /halt is configured as an alias for /stop. With an active session,
        # /halt must reach the message handler via the bypass path, NOT be
        # queued as ordinary text.
        _patch_quick_commands(
            monkeypatch, {"halt": {"type": "alias", "target": "/stop"}},
        )

        adapter = _make_stub_adapter()
        sk = _adapter_session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_adapter_event("/halt"))

        assert sk not in adapter._pending_messages, (
            "/halt was queued as a pending message instead of being "
            "dispatched — alias bypass missing"
        )
        assert any(
            "handled:stop" in r for r in adapter.sent_responses
        ), (
            "Aliased /halt did not reach the message handler as resolved "
            "/stop. sent_responses=%r" % (adapter.sent_responses,)
        )

    @pytest.mark.asyncio
    async def test_aliased_new_bypasses_guard(self, monkeypatch):
        _patch_quick_commands(
            monkeypatch, {"fresh": {"type": "alias", "target": "/new"}},
        )

        adapter = _make_stub_adapter()
        sk = _adapter_session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_adapter_event("/fresh"))

        assert sk not in adapter._pending_messages
        assert any("handled:new" in r for r in adapter.sent_responses)

    @pytest.mark.asyncio
    async def test_aliased_status_bypasses_guard_with_args(self, monkeypatch):
        # /st aliased to /status — bypass-eligible; user appends args.
        _patch_quick_commands(
            monkeypatch, {"st": {"type": "alias", "target": "/status"}},
        )

        adapter = _make_stub_adapter()
        sk = _adapter_session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_adapter_event("/st verbose"))

        assert sk not in adapter._pending_messages
        # Dispatched message text must reflect the resolved target plus
        # the user's args.
        assert any(
            "handled:status:" in r and "/status verbose" in r
            for r in adapter.sent_responses
        ), adapter.sent_responses

    @pytest.mark.asyncio
    async def test_aliased_to_non_bypass_built_in_still_dispatches(self, monkeypatch):
        # /m aliased to /model — /model isn't in the dedicated handoff
        # list (not stop/new/reset) but IS resolvable, so it must
        # bypass via the direct-dispatch path.
        _patch_quick_commands(
            monkeypatch,
            {"m": {"type": "alias", "target": "/model claude-sonnet-4-6 --provider anthropic --global"}},
        )

        adapter = _make_stub_adapter()
        sk = _adapter_session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_adapter_event("/m"))

        assert sk not in adapter._pending_messages
        assert any(
            "handled:model" in r and "claude-sonnet-4-6" in r
            for r in adapter.sent_responses
        ), adapter.sent_responses

    @pytest.mark.asyncio
    async def test_exec_quick_command_does_not_bypass(self, monkeypatch):
        # /dn (exec) is NOT a slash-command alias. It should NOT bypass
        # the active-session guard — exec quick_commands run on the
        # synchronous CLI path and the runner's quick_commands block
        # handles them when no session is busy. With an active session,
        # they should be queued like any normal message.
        _patch_quick_commands(
            monkeypatch,
            {"dn": {"type": "exec", "command": "echo daily-note"}},
        )

        adapter = _make_stub_adapter()
        sk = _adapter_session_key()
        adapter._active_sessions[sk] = asyncio.Event()

        await adapter.handle_message(_make_adapter_event("/dn"))

        # Exec aliases get queued (or interrupt-handled) — the message
        # handler must NOT be hit via the bypass path.
        assert not any(
            "handled:dn" in r for r in adapter.sent_responses
        ), (
            "Exec quick_command unexpectedly bypassed the active-session "
            "guard. sent_responses=%r" % (adapter.sent_responses,)
        )
