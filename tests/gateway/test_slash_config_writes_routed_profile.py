"""Slash-command config writes must land in the routed profile's config.yaml.

Regression for #87939 / #75684: the multiplexed inbound handler already runs
every slash handler inside ``_profile_runtime_scope`` (routed HERMES_HOME
override), but several handlers built their write path from the module
constant ``gateway.run._hermes_home`` — the LAUNCH home — so ``/reasoning
--global``, ``/fast``, ``/memory approval``, ``/skills approval``, ``/verbose``
and ``/footer`` persisted into the default profile's config.yaml. They now go
through ``_gateway_config_home()`` like the reads do.
"""

from __future__ import annotations

import pytest
import yaml

import gateway.run as gateway_run
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.slash_commands import GatewaySlashCommandsMixin
from tools import write_approval as wa


class _Runner(GatewaySlashCommandsMixin):
    _run_in_executor_with_context = GatewayRunner._run_in_executor_with_context
    _get_executor = GatewayRunner._get_executor

    def _session_key_for_source(self, _source):
        return "k"

    def _evict_cached_agent(self, _session_key):
        pass

    def _resolve_profile_home_for_source(self, _source):
        return getattr(self, "routed_home", gateway_run._gateway_config_home())


class _Event:
    def __init__(self, args: str = ""):
        self._args = args
        self.source = object()

    def get_command_args(self) -> str:
        return self._args


@pytest.fixture
def homes(tmp_path, monkeypatch):
    default_home = tmp_path / "default"
    routed_home = tmp_path / "profiles" / "beta"
    default_home.mkdir()
    routed_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text("agent:\n  reasoning_effort: medium\n")
    (routed_home / "config.yaml").write_text("agent:\n  reasoning_effort: none\n")
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home, routed_home


@pytest.mark.asyncio
async def test_slash_config_writes_hit_routed_profile_and_leave_default_untouched(homes):
    default_home, routed_home = homes
    default_before = (default_home / "config.yaml").read_bytes()
    runner = _Runner()

    with _profile_runtime_scope(routed_home):
        assert runner._save_gateway_config_key("agent.reasoning_effort", "high")
        await runner._handle_memory_command(_Event("approval on"))
        await runner._handle_skills_command(_Event("approval on"))

    routed = yaml.safe_load((routed_home / "config.yaml").read_text())
    assert routed["agent"]["reasoning_effort"] == "high"
    assert routed["memory"]["write_approval"] is True
    assert routed["skills"]["write_approval"] is True
    assert (default_home / "config.yaml").read_bytes() == default_before


@pytest.mark.asyncio
async def test_memory_and_skills_review_commands_use_routed_profile_without_caller_scope(homes):
    """The handler itself scopes routed review commands, including destructive actions."""
    default_home, routed_home = homes
    default_before = (default_home / "config.yaml").read_bytes()
    runner = _Runner()
    runner.routed_home = routed_home

    with _profile_runtime_scope(routed_home):
        memory_approve = wa.stage_write(
            wa.MEMORY, {"action": "add", "target": "memory", "content": "routed approved"},
            summary="routed-memory-approve", origin="foreground")
        memory_reject = wa.stage_write(
            wa.MEMORY, {"action": "add", "target": "memory", "content": "routed rejected"},
            summary="routed-memory-reject", origin="foreground")
        skill_reject = wa.stage_write(
            wa.SKILLS, {"action": "create", "name": "routed-skill", "content": "---\nname: routed-skill\n---\n"},
            summary="routed-skill-reject", origin="foreground")
        skill_approve = wa.stage_write(
            wa.SKILLS, {"action": "create", "name": "routed-approved-skill",
                        "content": "---\nname: routed-approved-skill\ndescription: Use when testing routed approval.\n---\n\nVerify the routed profile.\n"},
            summary="routed-skill-approve", origin="foreground")

    assert "routed-memory-approve" in await runner._handle_memory_command(_Event("pending"))
    assert "Approved 1 memory write(s)." in await runner._handle_memory_command(
        _Event(f"approve {memory_approve['id']}"))
    assert "routed approved" in (routed_home / "memories" / "MEMORY.md").read_text()
    assert "Rejected pending memory write" in await runner._handle_memory_command(
        _Event(f"reject {memory_reject['id']}"))
    assert "routed-skill-reject" in await runner._handle_skills_command(_Event("pending"))
    assert "Pending skill write" in await runner._handle_skills_command(
        _Event(f"diff {skill_reject['id']}"))
    assert "Rejected pending skills write" in await runner._handle_skills_command(
        _Event(f"reject {skill_reject['id']}"))
    assert "Approved 1 skills write(s)." in await runner._handle_skills_command(
        _Event(f"approve {skill_approve['id']}"))
    assert (routed_home / "skills" / "routed-approved-skill" / "SKILL.md").exists()
    assert "set to 'on'" in await runner._handle_memory_command(_Event("approval on"))
    assert "set to 'on'" in await runner._handle_skills_command(_Event("approval on"))

    routed = yaml.safe_load((routed_home / "config.yaml").read_text())
    assert routed["memory"]["write_approval"] is True
    assert routed["skills"]["write_approval"] is True
    assert not (default_home / "pending").exists()
    assert not (default_home / "memories").exists()
    assert not (default_home / "skills").exists()
    assert (default_home / "config.yaml").read_bytes() == default_before
