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


class _Runner(GatewaySlashCommandsMixin):
    _run_in_executor_with_context = GatewayRunner._run_in_executor_with_context
    _get_executor = GatewayRunner._get_executor

    def _session_key_for_source(self, _source):
        return "k"

    def _evict_cached_agent(self, _session_key):
        pass


class _Event:
    def __init__(self, args: str = ""):
        self._args = args
        self.source = None

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
