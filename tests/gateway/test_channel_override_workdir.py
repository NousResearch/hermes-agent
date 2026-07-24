"""Channel-override workspace routing and isolation contracts."""

from __future__ import annotations

import asyncio
import json
import pytest

from agent.prompt_builder import build_context_files_prompt
from agent.runtime_cwd import resolve_context_cwd
from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, _get_channel_override
from gateway.session import SessionContext, SessionSource
from tools.file_tools import read_file_tool
from tools.terminal_tool import terminal_tool


def _config(overrides: dict[str, ChannelOverride]) -> GatewayConfig:
    return GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                channel_overrides=overrides,
            ),
            Platform.SLACK: PlatformConfig(
                enabled=True,
                channel_overrides=overrides,
            ),
        }
    )


def _runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    return runner


def _source(
    platform: Platform = Platform.DISCORD,
    *,
    chat_id: str = "333",
    parent_id: str | None = "222",
    ancestors: tuple[str, ...] = ("222", "111"),
) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type="thread",
        thread_id=chat_id,
        parent_chat_id=parent_id,
        ancestor_chat_ids=ancestors,
        user_id="user-1",
    )


def test_generic_direct_channel_workdir_resolution(tmp_path):
    config = _config({"222": ChannelOverride(workdir=str(tmp_path))})
    source = SessionSource(platform=Platform.SLACK, chat_id="222")

    assert _runner(config)._resolve_workdir_for_session(source, "session-1") == str(
        tmp_path.resolve()
    )


def test_discord_category_channel_thread_precedence(tmp_path):
    category = tmp_path / "category"
    channel = tmp_path / "channel"
    thread = tmp_path / "thread"
    for path in (category, channel, thread):
        path.mkdir()
    source = _source()

    category_only = _config({"111": ChannelOverride(workdir=str(category))})
    assert _runner(category_only)._resolve_workdir_for_session(source, "s1") == str(category)

    channel_override = _config(
        {
            "111": ChannelOverride(workdir=str(category)),
            "222": ChannelOverride(workdir=str(channel)),
        }
    )
    assert _runner(channel_override)._resolve_workdir_for_session(source, "s2") == str(channel)

    thread_override = _config(
        {
            "111": ChannelOverride(workdir=str(category)),
            "222": ChannelOverride(workdir=str(channel)),
            "333": ChannelOverride(workdir=str(thread)),
        }
    )
    assert _runner(thread_override)._resolve_workdir_for_session(source, "s3") == str(thread)


def test_override_lookup_is_deterministic_deduplicated_and_preserves_other_fields():
    category = ChannelOverride(
        model="model-category",
        provider="provider-category",
        system_prompt="category prompt",
    )
    channel = ChannelOverride(
        model="model-channel",
        provider="provider-channel",
        system_prompt="channel prompt",
    )
    config = _config({"111": category, "222": channel})

    selected = _get_channel_override(
        config,
        Platform.DISCORD,
        "333",
        thread_id="333",
        parent_id="222",
        ancestor_ids=("222", "111", "111"),
    )

    assert selected is channel
    assert selected.model == "model-channel"
    assert selected.provider == "provider-channel"
    assert selected.system_prompt == "channel prompt"


@pytest.mark.parametrize("configured", ["relative/project", "/missing/hermes-workdir"])
def test_invalid_workdir_fails_clearly(configured):
    runner = _runner(_config({"222": ChannelOverride(workdir=configured)}))
    source = SessionSource(platform=Platform.SLACK, chat_id="222")

    with pytest.raises(ValueError, match="absolute path|does not exist"):
        runner._resolve_workdir_for_session(source, "session-invalid")


def test_existing_file_is_not_accepted_as_workdir(tmp_path):
    configured = tmp_path / "not-a-directory"
    configured.write_text("content", encoding="utf-8")
    runner = _runner(_config({"222": ChannelOverride(workdir=str(configured))}))
    source = SessionSource(platform=Platform.SLACK, chat_id="222")

    with pytest.raises(ValueError, match="not a directory"):
        runner._resolve_workdir_for_session(source, "session-file")


def test_workdir_mapping_is_pinned_until_conversation_scope_clears(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    override = ChannelOverride(workdir=str(first))
    runner = _runner(_config({"222": override}))
    source = SessionSource(platform=Platform.SLACK, chat_id="222")

    assert runner._resolve_workdir_for_session(source, "session-1") == str(first)
    override.workdir = str(second)
    assert runner._resolve_workdir_for_session(source, "session-1") == str(first)

    runner._clear_session_boundary_security_state = lambda _key: None
    runner._clear_conversation_scope("session-1", reason="test reset")
    assert runner._resolve_workdir_for_session(source, "session-1") == str(second)


def test_conversation_boundary_clears_only_owned_workspace_cwd(tmp_path, monkeypatch):
    import tools.terminal_tool as terminal_module

    workspace = str(tmp_path.resolve())
    monkeypatch.setattr(
        terminal_module,
        "_task_env_overrides",
        {"task-1": {"docker_image": "neutral:latest", "cwd": workspace}},
    )
    monkeypatch.setattr(
        terminal_module,
        "_session_cwd",
        {"task-1": workspace},
    )
    runner = _runner(_config({}))
    runner._session_workdirs = {"session-1": workspace}
    runner._session_workdir_task_overrides = {
        "session-1": {"task-1": workspace},
    }
    runner._clear_session_boundary_security_state = lambda _key: None

    runner._clear_conversation_scope("session-1", reason="test boundary")

    assert terminal_module._task_env_overrides == {
        "task-1": {"docker_image": "neutral:latest"}
    }
    assert terminal_module.get_session_cwd("task-1") is None
    assert "session-1" not in runner._session_workdirs
    assert "session-1" not in runner._session_workdir_task_overrides


def test_session_source_ancestor_ids_roundtrip():
    restored = SessionSource.from_dict(_source().to_dict())
    assert restored.ancestor_chat_ids == ("222", "111")


def test_relay_wire_preserves_ancestor_ids():
    from gateway.relay.ws_transport import _event_from_wire

    event = _event_from_wire(
        {
            "text": "hello",
            "source": _source().to_dict(),
        }
    )

    assert event.source.ancestor_chat_ids == ("222", "111")


def test_concurrent_workdirs_isolate_context_file_terminal_and_file_tools(tmp_path):
    workspaces = []
    for name in ("alpha", "docs"):
        workspace = tmp_path / name
        workspace.mkdir()
        (workspace / "AGENTS.md").write_text(f"workspace-context:{name}", encoding="utf-8")
        (workspace / "marker.txt").write_text(f"workspace-file:{name}", encoding="utf-8")
        workspaces.append((name, workspace))

    async def observe(name: str, workspace) -> tuple[str, str, str]:
        runner = _runner(_config({}))
        source = SessionSource(platform=Platform.SLACK, chat_id=name)
        context = SessionContext(
            source=source,
            connected_platforms=[Platform.SLACK],
            home_channels={},
            session_key=f"session-{name}",
            session_id=f"id-{name}",
        )
        tokens = runner._set_session_env(context, cwd=str(workspace))
        task_id = f"id-{name}"
        try:
            await asyncio.sleep(0)
            prompt = build_context_files_prompt(cwd=resolve_context_cwd())
            from tools.terminal_tool import register_task_env_overrides

            register_task_env_overrides(task_id, {"cwd": str(workspace)})
            file_result = read_file_tool("marker.txt", task_id=task_id)
            from tools.code_execution_tool import _resolve_child_cwd

            code_cwd = _resolve_child_cwd(
                "project",
                str(tmp_path),
                task_id=task_id,
            )
            terminal_result = await asyncio.to_thread(
                terminal_tool,
                "pwd",
                task_id=task_id,
            )
            terminal_data = json.loads(terminal_result)
            assert code_cwd == str(workspace)
            return prompt, file_result, terminal_data["output"].strip()
        finally:
            from tools.terminal_tool import clear_task_env_overrides

            clear_task_env_overrides(task_id)
            runner._clear_session_env(tokens)

    async def run_both():
        return await asyncio.gather(
            *(observe(name, workspace) for name, workspace in workspaces)
        )

    alpha, docs = asyncio.run(run_both())

    assert "workspace-context:alpha" in alpha[0]
    assert "workspace-context:docs" not in alpha[0]
    assert "workspace-file:alpha" in alpha[1]
    assert alpha[2] == str(workspaces[0][1])
    assert "workspace-context:docs" in docs[0]
    assert "workspace-context:alpha" not in docs[0]
    assert "workspace-file:docs" in docs[1]
    assert docs[2] == str(workspaces[1][1])
