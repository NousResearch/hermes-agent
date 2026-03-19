import json
from pathlib import Path

import pytest

from agent.prompt_builder import load_soul_md
from cron.jobs import create_job, load_jobs
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.profiles import GatewayRuntimeRegistry
from gateway.session import SessionSource, build_session_key
from runtime_context import build_runtime_context, use_runtime
from tools.memory_tool import MemoryStore
from tools.file_tools import read_file_tool, search_tool, write_file_tool
from tools.skills_tool import skills_list
from tools.terminal_tool import terminal_tool


def _skill_md(name: str, description: str = "Test skill") -> str:
    return f"""---
name: {name}
description: {description}
---

# {name}

Follow this skill.
"""


def test_profile_runtime_memory_store_uses_profile_home(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    runtime = build_runtime_context(
        global_config={"profiles": {"alice": {"config": {}}}},
        profile_name="alice",
    )

    with use_runtime(runtime):
        store = MemoryStore()
        store.load_from_disk()
        result = store.add("memory", "Alice-specific fact")

    assert result["success"] is True
    assert (runtime.memories_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert not (hermes_home / "memories" / "MEMORY.md").exists()


def test_profile_runtime_defaults_workspace_to_profile_home(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    runtime = build_runtime_context(
        global_config={"profiles": {"alice": {"config": {}}}},
        profile_name="alice",
    )

    assert runtime.workspace == str(runtime.effective_home / "workspace")
    assert (runtime.effective_home / "workspace").is_dir()


def test_profile_runtime_skills_merge_shared_private_and_profile_disables(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    shared_skill = hermes_home / "skills" / "shared-skill" / "SKILL.md"
    shared_skill.parent.mkdir(parents=True, exist_ok=True)
    shared_skill.write_text(_skill_md("shared-skill"), encoding="utf-8")

    runtime = build_runtime_context(
        global_config={
            "profiles": {
                "alice": {
                    "config": {
                        "skills": {
                            "disabled": ["shared-skill"],
                        }
                    }
                }
            }
        },
        profile_name="alice",
    )
    private_skill = runtime.private_skills_dir / "private-skill" / "SKILL.md"
    private_skill.parent.mkdir(parents=True, exist_ok=True)
    private_skill.write_text(_skill_md("private-skill"), encoding="utf-8")

    with use_runtime(runtime):
        payload = json.loads(skills_list())

    assert payload["success"] is True
    assert [skill["name"] for skill in payload["skills"]] == ["private-skill"]


def test_profile_runtime_soul_prefers_profile_and_falls_back_to_global(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "SOUL.md").write_text("global soul", encoding="utf-8")

    runtime = build_runtime_context(
        global_config={"profiles": {"alice": {"config": {}}}},
        profile_name="alice",
    )

    with use_runtime(runtime):
        assert load_soul_md() == "global soul"

    runtime.soul_path.write_text("profile soul", encoding="utf-8")
    with use_runtime(runtime):
        assert load_soul_md() == "profile soul"


def test_gateway_profile_registry_resolves_profile_and_prefixes_session_key(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    registry = GatewayRuntimeRegistry(
        gateway_config=GatewayConfig(),
        global_config={
            "profiles": {
                "alice": {
                    "users": {
                        "telegram": ["12345"],
                    },
                    "config": {},
                }
            }
        },
        global_home=hermes_home,
        has_active_processes_fn=lambda _: False,
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="12345",
    )

    with registry.use_runtime(source=source) as runtime_state:
        session_key = build_session_key(source)

    assert runtime_state.profile_name == "alice"
    assert session_key.startswith("profile:alice:")


def test_cron_jobs_store_under_profile_runtime(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    runtime = build_runtime_context(
        global_config={"profiles": {"alice": {"config": {}}}},
        profile_name="alice",
    )

    with use_runtime(runtime):
        create_job(prompt="Check profile", schedule="30m", name="Profile job")
        jobs = load_jobs()

    assert len(jobs) == 1
    assert (runtime.cron_dir / "jobs.json").exists()
    assert not (hermes_home / "cron" / "jobs.json").exists()


def test_profile_runtime_file_tools_block_access_outside_isolated_workspace(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    workspace.mkdir(parents=True, exist_ok=True)
    outside.mkdir(parents=True, exist_ok=True)
    inside_file = workspace / "inside.txt"
    outside_file = outside / "outside.txt"
    inside_file.write_text("inside\n", encoding="utf-8")
    outside_file.write_text("outside\n", encoding="utf-8")

    runtime = build_runtime_context(
        global_config={
            "profiles": {
                "alice": {
                    "config": {
                        "workspace": str(workspace),
                        "isolation": {"enabled": True},
                    }
                }
            }
        },
        profile_name="alice",
    )

    with use_runtime(runtime):
        inside_payload = json.loads(read_file_tool(str(inside_file)))
        outside_payload = json.loads(read_file_tool(str(outside_file)))
        write_payload = json.loads(write_file_tool(str(outside / "new.txt"), "nope"))
        search_payload = json.loads(search_tool("outside", path=str(outside)))

    assert "inside" in inside_payload["content"]
    assert "outside the isolated workspace root" in outside_payload["error"]
    assert "outside the isolated workspace root" in write_payload["error"]
    assert "outside the isolated workspace root" in search_payload["error"]


def test_profile_runtime_terminal_blocks_paths_outside_isolated_workspace(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    workspace = tmp_path / "workspace"
    outside_file = tmp_path / "secret.txt"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    workspace.mkdir(parents=True, exist_ok=True)
    outside_file.write_text("secret\n", encoding="utf-8")

    runtime = build_runtime_context(
        global_config={
            "profiles": {
                "alice": {
                    "config": {
                        "workspace": str(workspace),
                        "isolation": {"enabled": True},
                    }
                }
            }
        },
        profile_name="alice",
    )

    with use_runtime(runtime):
        allowed = json.loads(terminal_tool("pwd", task_id="allowed", workdir=str(workspace)))
        blocked = json.loads(
            terminal_tool(f"cat {outside_file}", task_id="blocked", workdir=str(workspace))
        )

    assert allowed["exit_code"] == 0
    assert "cannot access paths outside" in blocked["error"]


@pytest.mark.asyncio
async def test_profile_runtime_disables_configured_slash_commands(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from gateway.run import GatewayRunner

    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    registry = GatewayRuntimeRegistry(
        gateway_config=GatewayConfig(),
        global_config={
            "profiles": {
                "alice": {
                    "users": {"telegram": ["12345"]},
                    "config": {
                        "messaging": {
                            "disabled_slash_commands": ["model", "my-skill"],
                        }
                    },
                }
            }
        },
        global_home=hermes_home,
        has_active_processes_fn=lambda _: False,
    )

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.config = {}
    runner._session_db = None
    runner.session_store = MagicMock()
    runner._is_user_authorized = lambda source: True
    runner._get_unauthorized_dm_behavior = lambda platform: "ignore"
    runner._session_key_for_source = lambda source: build_session_key(source)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="12345",
    )
    model_event = MessageEvent(text="/model", source=source)
    help_event = MessageEvent(text="/help", source=source)

    with registry.use_runtime(source=source):
        blocked = await runner._handle_message_impl(model_event)
        with patch("agent.skill_commands.get_skill_commands", return_value={"/my-skill": {"description": "Skill"}}):
            help_text = await runner._handle_help_command(help_event)

    assert "disabled" in blocked.lower()
    assert "`/model" not in help_text
    assert "`/my-skill`" not in help_text
