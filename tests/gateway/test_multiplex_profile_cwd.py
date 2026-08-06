import json
import os
from pathlib import Path

import pytest
import yaml

from agent.runtime_cwd import get_session_cwd_override, resolve_agent_cwd
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from tools.terminal_tool import (
    clear_task_env_overrides,
    cleanup_all_environments,
    clear_session_cwd,
    get_session_cwd,
    register_task_env_overrides,
    terminal_tool,
)


def _context(
    profile: str = "secondary",
    session_key: str = "agent:secondary:telegram:group:-1001234567890:101",
) -> SessionContext:
    return SessionContext(
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001234567890",
            chat_type="group",
            profile=profile,
        ),
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key=session_key,
        session_id="test-session",
    )


def _runner(profile_home: Path) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._resolve_profile_home_for_source = lambda source: profile_home
    return runner


def test_multiplex_profile_terminal_cwd_is_bound_to_session(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "secondary"
    workspace = tmp_path / "workspace"
    profile_home.mkdir(parents=True)
    workspace.mkdir()
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": "local", "cwd": str(workspace)}})
    )
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "default-workspace"))

    runner = _runner(profile_home)
    tokens = runner._set_session_env(_context())
    try:
        assert resolve_agent_cwd() == workspace
        assert get_session_cwd(_context().session_key) is None
        result = json.loads(
            terminal_tool(command="pwd", task_id="test-session", session_id="test-session")
        )
        assert result["exit_code"] == 0
        assert result["output"].strip() == str(workspace)
        assert get_session_cwd(_context().session_key) == str(workspace)
    finally:
        runner._clear_session_env(tokens)
        cleanup_all_environments()
        clear_session_cwd(_context().session_key)
        clear_task_env_overrides("test-session")


def test_multiplex_profile_placeholder_does_not_inherit_process_cwd(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "secondary"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": "local", "cwd": "."}})
    )
    process_cwd = tmp_path / "default-workspace"
    process_cwd.mkdir()
    profile_launch_home = tmp_path / "launch-home"
    profile_launch_home.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(process_cwd))
    monkeypatch.setenv("HOME", str(profile_launch_home))

    runner = _runner(profile_home)
    tokens = runner._set_session_env(_context())
    try:
        assert resolve_agent_cwd() == profile_launch_home
        result = json.loads(
            terminal_tool(command="pwd", task_id="test-session", session_id="test-session")
        )
        assert result["exit_code"] == 0
        assert result["output"].strip() == str(profile_launch_home)
        assert get_session_cwd(_context().session_key) == str(profile_launch_home)
    finally:
        runner._clear_session_env(tokens)
        cleanup_all_environments()
        clear_session_cwd(_context().session_key)
        clear_task_env_overrides("test-session")


def test_existing_session_cwd_is_not_overwritten(tmp_path):
    profile_home = tmp_path / "profiles" / "secondary"
    profile_home.mkdir(parents=True)
    configured_workspace = tmp_path / "configured-workspace"
    configured_workspace.mkdir()
    changed_workspace = tmp_path / "changed-workspace"
    changed_workspace.mkdir()
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"terminal": {"backend": "local", "cwd": str(configured_workspace)}}
        )
    )
    register_task_env_overrides("test-session", {"cwd": str(changed_workspace)})

    runner = _runner(profile_home)
    tokens = runner._set_session_env(_context())
    try:
        result = json.loads(
            terminal_tool(command="pwd", task_id="test-session", session_id="test-session")
        )
        assert result["exit_code"] == 0
        assert result["output"].strip() == str(changed_workspace)
        assert get_session_cwd(_context().session_key) == str(changed_workspace)
    finally:
        runner._clear_session_env(tokens)
        cleanup_all_environments()
        clear_session_cwd(_context().session_key)
        clear_task_env_overrides("test-session")


def test_single_profile_gateway_keeps_process_cwd(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "secondary"
    profile_home.mkdir(parents=True)
    configured_workspace = tmp_path / "configured-workspace"
    configured_workspace.mkdir()
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"terminal": {"backend": "local", "cwd": str(configured_workspace)}}
        )
    )
    process_cwd = tmp_path / "process-workspace"
    process_cwd.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(process_cwd))

    runner = _runner(profile_home)
    runner.config.multiplex_profiles = False
    tokens = runner._set_session_env(_context())
    try:
        assert resolve_agent_cwd() == process_cwd
        assert get_session_cwd(_context().session_key) is None
    finally:
        runner._clear_session_env(tokens)
        cleanup_all_environments()
        clear_session_cwd(_context().session_key)
        clear_task_env_overrides("test-session")


def test_multiplex_nonlocal_backend_keeps_existing_process_cwd(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "secondary"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": "ssh", "cwd": "/remote/project"}})
    )
    process_cwd = tmp_path / "process-workspace"
    process_cwd.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(process_cwd))

    runner = _runner(profile_home)
    tokens = runner._set_session_env(_context())
    try:
        assert resolve_agent_cwd() == process_cwd
        assert get_session_cwd(_context().session_key) is None
    finally:
        runner._clear_session_env(tokens)
        cleanup_all_environments()
        clear_session_cwd(_context().session_key)
        clear_task_env_overrides("test-session")


def test_multiplex_sessions_do_not_share_cached_local_environment(tmp_path):
    profile_a = tmp_path / "profiles" / "alpha"
    profile_b = tmp_path / "profiles" / "beta"
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    for path in (profile_a, profile_b, workspace_a, workspace_b):
        path.mkdir(parents=True)
    (profile_a / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": "local", "cwd": str(workspace_a)}})
    )
    (profile_b / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": "local", "cwd": str(workspace_b)}})
    )
    key_a = "agent:alpha:telegram:group:-1001234567890:101"
    key_b = "agent:beta:telegram:group:-1001234567890:202"

    runner_a = _runner(profile_a)
    tokens_a = runner_a._set_session_env(_context("alpha", key_a))
    try:
        result_a = json.loads(
            terminal_tool(command="pwd", task_id="session-a", session_id="session-a")
        )
        assert result_a["exit_code"] == 0
        assert result_a["output"].strip() == str(workspace_a)
    finally:
        runner_a._clear_session_env(tokens_a)

    runner_b = _runner(profile_b)
    tokens_b = runner_b._set_session_env(_context("beta", key_b))
    try:
        result_b = json.loads(
            terminal_tool(command="pwd", task_id="session-b", session_id="session-b")
        )
        assert result_b["exit_code"] == 0
        assert result_b["output"].strip() == str(workspace_b)
    finally:
        runner_b._clear_session_env(tokens_b)
        cleanup_all_environments()
        for session_key in (key_a, key_b):
            clear_session_cwd(session_key)
        for task_id in ("session-a", "session-b"):
            clear_task_env_overrides(task_id)


def test_multiplex_at_reference_expansion_uses_profile_cwd(monkeypatch, tmp_path):
    """The inbound @-reference preprocessor must expand against the routed
    profile's cwd, not the primary profile's process-global TERMINAL_CWD.

    Pre-fix, the preprocessor read ``os.environ["TERMINAL_CWD"]`` directly
    before session binding, so a routed secondary profile resolved @file:
    references against the primary workspace. This pins the early session-cwd
    pin in ``_prepare_profile_scoped_inbound_message_text``.
    """
    import asyncio

    import gateway.run as gateway_run
    from agent.context_references import ContextReferenceResult
    from gateway.platforms.base import MessageEvent

    profile_home = tmp_path / "profiles" / "secondary"
    workspace = tmp_path / "workspace"
    profile_home.mkdir(parents=True)
    workspace.mkdir()
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": "local", "cwd": str(workspace)}})
    )
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "default-workspace"))

    captured: dict = {}

    async def _fake_preprocess(message, *, cwd, context_length, url_fetcher=None, allowed_root=None):
        captured["cwd"] = cwd
        captured["allowed_root"] = allowed_root
        return ContextReferenceResult(
            message="[expanded body]",
            original_message=message,
            expanded=True,
        )

    import agent.context_references as ctx_mod

    monkeypatch.setattr(ctx_mod, "preprocess_context_references_async", _fake_preprocess)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda cfg=None: "openai/gpt-4.1-mini",
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "model": {"default": "openai/gpt-4.1-mini", "context_length": 128000},
            "terminal": {"backend": "local", "cwd": str(workspace)},
        },
    )

    runner = _runner(profile_home)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001234567890",
        chat_type="group",
        profile="secondary",
    )
    event = MessageEvent(text="please look at @file:notes.txt", source=source)

    # Real code flow: _set_session_env binds the ContextVar first, then
    # _prepare_profile_scoped_inbound_message_text reads it. The test must
    # simulate that ordering.
    context = _context()
    tokens = runner._set_session_env(context)
    try:
        asyncio.run(
            runner._prepare_profile_scoped_inbound_message_text(
                event=event,
                source=source,
                history=[],
            )
        )
    finally:
        runner._clear_session_env(tokens)

    assert captured.get("cwd") == str(workspace), (
        f"@-reference expansion should use profile cwd {workspace!r}, got {captured.get('cwd')!r}"
    )
    assert captured.get("allowed_root") == str(workspace)


def test_multiplex_runtime_footer_uses_profile_cwd(monkeypatch, tmp_path):
    """The runtime-metadata footer must show the routed profile's cwd, not the
    primary profile's process-global TERMINAL_CWD.

    Pre-fix, the footer read ``os.environ["TERMINAL_CWD"]`` directly, so a
    routed secondary profile displayed the primary workspace in its status
    line. This pins the footer's session-cwd override read.
    """
    profile_home = tmp_path / "profiles" / "secondary"
    workspace = tmp_path / "workspace"
    profile_home.mkdir(parents=True)
    workspace.mkdir()
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": "local", "cwd": str(workspace)}})
    )
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "default-workspace"))
    monkeypatch.setenv("HOME", str(tmp_path))

    runner = _runner(profile_home)
    tokens = runner._set_session_env(_context())
    try:
        assert get_session_cwd_override() == str(workspace)
        from gateway.runtime_footer import build_footer_line

        footer = build_footer_line(
            user_config={"display": {"runtime_footer": {"enabled": True}}},
            platform_key="telegram",
            model="openai/gpt-5.4",
            context_tokens=68000,
            context_length=100000,
            cwd=get_session_cwd_override() or os.environ.get("TERMINAL_CWD", ""),
        )
        # Footer home-relativizes the path; the routed profile's cwd is
        # ``tmp_path/workspace`` so it renders as ``~/workspace``.
        assert "~/workspace" in footer
        assert "default-workspace" not in footer
    finally:
        runner._clear_session_env(tokens)
        cleanup_all_environments()
        clear_session_cwd(_context().session_key)
