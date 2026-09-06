"""Production-path invariants for gateway session workspace routing."""

import asyncio
import json

import pytest

from agent.runtime_cwd import resolve_agent_cwd
from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_context
from gateway.workspace import (
    WorkspaceUnavailable, configured_gateway_workspace, normalize_gateway_workspace,
)
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.terminal_tool import (
    _plan_execution, _resolve_command_cwd, clear_session_cwd, clear_task_env_overrides,
    get_session_cwd, record_session_cwd, terminal_tool,
)


@pytest.mark.asyncio
async def test_gateway_creation_routes_dm_channels_and_threads_without_cross_session_leakage(
    tmp_path, monkeypatch,
):
    default = tmp_path / "orchestrator"
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    for path in (default, project_a, project_b):
        path.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(default))
    monkeypatch.setenv("TERMINAL_ENV", "local")

    config = GatewayConfig(
        sessions_dir=tmp_path / "sessions",
        platforms={
            Platform.SLACK: PlatformConfig(channel_overrides={
                "C-A": ChannelOverride(cwd=str(project_a)),
                "C-B": ChannelOverride(cwd=str(project_b)),
            }),
        },
    )
    runner = GatewayRunner(config)
    runner.adapters = {}
    sources = [
        SessionSource(platform=Platform.SLACK, chat_id="D-DM", chat_type="dm", user_id="owner"),
        SessionSource(platform=Platform.SLACK, chat_id="C-A", chat_type="channel", user_id="owner"),
        SessionSource(
            platform=Platform.SLACK, chat_id="C-B", chat_type="thread", thread_id="T-B",
            parent_chat_id="C-B", user_id="owner",
        ),
    ]

    resolved = await asyncio.gather(*(
        runner._hmwa_resolve_session(MessageEvent(text="hello", source=source), source)
        for source in sources
    ))
    expected = [str(default), str(project_a), str(project_b)]
    assert [item[1].cwd for item in resolved] == expected

    for (_, entry, _), cwd in zip(resolved, expected):
        row = runner.session_store._db_for_key(entry.session_key).get_session(entry.session_id)
        assert (row["cwd"], row["git_repo_root"]) == (cwd, None)

    async def observe(item, delay):
        source, entry, _ = item
        tokens = runner._set_session_env(build_session_context(source, config, entry))
        try:
            await asyncio.sleep(delay)
            return str(resolve_agent_cwd()), get_session_cwd(entry.session_id)
        finally:
            runner._clear_session_env(tokens)

    observed = await asyncio.gather(*(observe(item, delay) for item, delay in zip(resolved, (.03, .02, .01))))
    assert observed == [(cwd, cwd) for cwd in expected]


@pytest.mark.asyncio
async def test_workspace_override_clear_rotation_and_backend_paths_use_one_durable_contract(
    tmp_path, monkeypatch,
):
    default = tmp_path / "orchestrator"
    old = tmp_path / "old"
    selected = tmp_path / "selected"
    selected_child = selected / "child"
    for path in (default, old, selected, selected_child):
        path.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(default))
    monkeypatch.setenv("TERMINAL_ENV", "local")

    config = GatewayConfig(sessions_dir=tmp_path / "sessions")
    runner = GatewayRunner(config)
    runner.adapters = {}
    source = SessionSource(
        platform=Platform.SLACK, chat_id="D-WORKSPACE", chat_type="dm", user_id="owner",
    )
    _, entry, _ = await runner._hmwa_resolve_session(MessageEvent(text="hello", source=source), source)
    db = runner.session_store._db_for_key(entry.session_key)

    def terminal_pwd(current_entry):
        tokens = runner._set_session_env(build_session_context(source, config, current_entry))
        approval_token = set_current_session_key(current_entry.session_key)
        try:
            result = json.loads(terminal_tool(
                "pwd", task_id=current_entry.session_id, session_id=current_entry.session_id,
            ))
            assert result["exit_code"] == 0
            return result["output"].strip()
        finally:
            reset_current_session_key(approval_token)
            runner._clear_session_env(tokens)

    assert terminal_pwd(entry) == str(default)
    db.update_session_cwd(
        entry.session_id, str(old), git_branch="dirty", git_repo_root=str(old),
        replace_git_meta=True,
    )

    reply = await runner._handle_workspace_command(
        MessageEvent(text=f"/workspace {selected}", source=source),
    )
    assert str(selected) in reply
    row = db.get_session(entry.session_id)
    assert (row["cwd"], row["git_branch"], row["git_repo_root"]) == (str(selected), None, None)
    assert get_session_cwd(entry.session_key) == str(selected)
    assert terminal_pwd(entry) == str(selected)

    default.rmdir()
    reply = await runner._handle_workspace_command(
        MessageEvent(text=f"/workspace {selected}", source=source),
    )
    assert str(selected) in reply
    assert (await runner.async_session_store.get_or_create_session(source)).session_id == entry.session_id
    record_session_cwd(entry.session_key, str(selected_child))
    tokens = runner._set_session_env(build_session_context(source, config, entry))
    try:
        assert str(resolve_agent_cwd()) == str(selected_child)
        assert get_session_cwd(entry.session_key) == str(selected_child)
        assert get_session_cwd(entry.session_id) == str(selected_child)
    finally:
        runner._clear_session_env(tokens)
    default.mkdir()

    child_id = f"{entry.session_id}_compression"
    db.create_session(child_id, source="slack", parent_session_id=entry.session_id)
    assert db.get_session(child_id)["cwd"] == str(selected)
    await runner.async_session_store.advance_compression_session(
        entry.session_key, entry.session_id, child_id,
    )
    _, entry, _ = await runner._hmwa_resolve_session(
        MessageEvent(text="after compression", source=source), source,
    )
    assert (entry.session_id, entry.cwd) == (child_id, str(selected))
    assert db.get_session(child_id)["git_repo_root"] is None

    resumed_id = f"{entry.session_id}_resumed"
    db.create_session(resumed_id, source="slack", cwd=str(default))
    record_session_cwd(entry.session_key, str(selected_child))
    entry = await runner.async_session_store.switch_session(entry.session_key, resumed_id)
    assert entry is not None
    assert get_session_cwd(entry.session_key) is None
    _, entry, _ = await runner._hmwa_resolve_session(
        MessageEvent(text="after resume", source=source), source,
    )
    assert (entry.session_id, entry.cwd, terminal_pwd(entry)) == (
        resumed_id, str(default), str(default),
    )

    reply = await runner._handle_workspace_command(MessageEvent(text="/workspace clear", source=source))
    assert str(default) in reply
    cleared = db.get_session(entry.session_id)
    assert (cleared["cwd"], cleared["git_branch"], cleared["git_repo_root"]) == (None, None, None)
    assert get_session_cwd(entry.session_key) is None
    assert get_session_cwd(entry.session_id) is None

    _, rebound, _ = await runner._hmwa_resolve_session(MessageEvent(text="again", source=source), source)
    assert rebound.cwd == str(default)
    assert db.get_session(entry.session_id)["cwd"] == str(default)
    assert terminal_pwd(rebound) == str(default)

    assert normalize_gateway_workspace("/workspace/project", backend="docker") == "/workspace/project"
    assert normalize_gateway_workspace(str(selected), backend="docker") == str(selected)
    with pytest.raises(WorkspaceUnavailable):
        normalize_gateway_workspace("relative/project", backend="docker")
    with pytest.raises(WorkspaceUnavailable):
        normalize_gateway_workspace("/Users/owner/project", backend="docker")
    with pytest.raises(WorkspaceUnavailable):
        normalize_gateway_workspace("/workspace/../home/owner/project", backend="docker")

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setenv("TERMINAL_CWD", str(default))
    assert configured_gateway_workspace(config, source) == str(default)
    runner._register_workspace_task_cwd(entry.session_key, entry.session_id, str(default))
    tokens = runner._set_session_env(build_session_context(source, config, entry))
    try:
        plan = _plan_execution(
            "pwd", task_id=entry.session_id, timeout=None, background=False, _host_local=False,
        )
        assert (plan.cwd, plan.host_cwd) == ("/workspace", str(default))
        assert _resolve_command_cwd(
            workdir=None, default_cwd=plan.cwd, session_key=entry.session_key,
            env_type=plan.env_type,
        ) == "/workspace"
        runner._clear_session_env(tokens)
        record_session_cwd(entry.session_key, "/workspace")
        tokens = runner._set_session_env(build_session_context(source, config, entry))
        recreated = _plan_execution(
            "pwd", task_id=entry.session_id, timeout=None, background=False, _host_local=False,
        )
        assert (recreated.cwd, recreated.host_cwd) == ("/workspace", str(default))
    finally:
        runner._clear_session_env(tokens)

    monkeypatch.delenv("TERMINAL_CWD")
    clear_task_env_overrides(entry.session_id)
    clear_session_cwd(entry.session_key)
    entry.cwd = "/root"
    tokens = runner._set_session_env(build_session_context(source, config, entry))
    try:
        unattached = _plan_execution(
            "pwd", task_id=entry.session_id, timeout=None, background=False, _host_local=False,
        )
        assert (unattached.cwd, unattached.host_cwd) == ("/root", None)
    finally:
        runner._clear_session_env(tokens)
