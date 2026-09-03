"""Focused ordinary-gateway turn to delegate binding coverage."""

import queue
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import Platform
from gateway.run import GatewayRunner, TurnRunner, _format_delegation_binding_line
from gateway.session import SessionContext, SessionSource
from gateway.session_context import reset_session_vars
from tools.delegate_tool import delegate_task


class _Store:
    def __init__(self, key, record):
        import threading

        self._lock = threading.Lock()
        self._entries = {key: record}

    def _ensure_loaded_locked(self):
        return None


def test_gateway_turn_binds_lazily_at_delegate_boundary():
    key = "agent:main:telegram:dm:chat:user"
    record = object()
    adapter = object()
    runner = object.__new__(GatewayRunner)
    runner.session_store = _Store(key, record)
    runner._adapter_for_source = lambda source: adapter
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat",
        user_id="user",
        profile="default",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={}, session_key=key)
    tokens = runner._set_session_env(context)
    try:
        from gateway.session_context import current_gateway_run_authority

        authority = current_gateway_run_authority()
        assert authority is not None
        assert authority.record is record
        assert authority.transport is adapter

        parent = SimpleNamespace(
            _delegate_depth=0,
            _active_children=[],
            _active_children_lock=__import__("threading").Lock(),
            tool_progress_callback=None,
            _session_db=None,
            platform="telegram",
        )
        binding = SimpleNamespace(
            ui_session_id="",
            owner_generation=str(id(record)),
            transport_generation=str(id(adapter)),
            session_key=key,
            profile="default",
            cwd="/workspace",
            head="a" * 40,
            repo_root="/workspace",
            worktree_root="/workspace",
            git_common_dir="/workspace/.git",
            branch="main",
            ref="refs/heads/main",
            short_head="aaaaaaaaaaaa",
            differences=lambda other: (),
            display=lambda: "repo=/workspace branch=main sha=aaaaaaaaaaaa",
        )
        child = MagicMock()
        with (
            patch("tui_gateway.git_probe.capture_run_binding", return_value=binding),
            patch("tui_gateway.git_probe.run_git", return_value=""),
            patch("tools.delegate_tool._get_worktree_isolation", return_value=False),
            patch("tools.delegate_tool._build_child_preserving_parent_tools", return_value=child) as build,
            patch("tools.delegate_tool._run_single_child", return_value={"status": "completed"}),
        ):
            result = delegate_task(goal="Inspect the selected repository", parent_agent=parent)

        assert '"status": "completed"' in result
        assert '"delegation_status": "completed"' in result
        build.assert_called_once()
        assert build.call_args.kwargs["run_binding"] is binding
    finally:
        runner._clear_session_env(tokens)
        reset_session_vars()


def test_gateway_progress_keeps_bound_checkout_visible():
    progress = queue.Queue()
    ctx = MagicMock()
    ctx.progress_queue = progress
    ctx._run_still_current.return_value = True
    runner = object.__new__(GatewayRunner)
    TurnRunner(runner, ctx).progress_callback(
        "subagent.start",
        preview="Inspect the selected repository",
        run_binding={"repo": "/workspace", "branch": "main", "sha": "abc123456789"},
    )
    assert progress.get_nowait() == (
        "🔀 delegation started · repo=/workspace branch=main sha=abc123456789"
    )


def test_final_adapter_delivery_keeps_binding_when_progress_is_off():
    runner = object.__new__(GatewayRunner)
    adapter = MagicMock()
    adapter.extract_media.side_effect = lambda text: ([], text)
    adapter.send = AsyncMock()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat")
    line = _format_delegation_binding_line(
        {"repo": "/workspace", "branch": "main", "sha": "abc123456789"},
        "completed",
    )
    response = f"The child completed the requested work.\n\n{line}"

    # This is the existing final adapter path. No progress queue is involved;
    # progress display is off, but the final same-channel reply remains intact.
    asyncio.run(
        runner._deliver_queued_first_response(
            response,
            source,
            adapter,
            deliver_media=False,
        )
    )

    adapter.send.assert_awaited_once()
    assert response in adapter.send.await_args.args
    assert adapter.send.await_args.args[1].count("🔀 delegation") == 1


def test_streaming_binding_trailing_delivery_is_exactly_once_when_progress_is_off():
    runner = object.__new__(GatewayRunner)
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner._adapter_for_source = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat")
    event = SimpleNamespace()
    line = _format_delegation_binding_line(
        {"repo": "/workspace", "branch": "main", "sha": "abc123456789"},
        "completed",
    )

    # This is the already_sent=True trailing adapter path.  There is no
    # progress queue; the terminal binding is still delivered once.
    asyncio.run(runner._send_delegation_binding_line(source, event, line))

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.args[1] == line
    assert adapter.send.await_args.args[1].count("🔀 delegation") == 1
