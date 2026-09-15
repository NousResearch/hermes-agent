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

from types import SimpleNamespace

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
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
    (default_home / "config.yaml").write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")
    (routed_home / "config.yaml").write_text("agent:\n  reasoning_effort: none\n", encoding="utf-8")
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

    routed = yaml.safe_load((routed_home / "config.yaml").read_text(encoding="utf-8"))
    assert routed["agent"]["reasoning_effort"] == "high"
    assert routed["memory"]["write_approval"] is True
    assert routed["skills"]["write_approval"] is True
    assert (default_home / "config.yaml").read_bytes() == default_before


def test_plugin_admin_gate_reads_routed_profile_without_handler_context(
    homes, monkeypatch
):
    default_home, routed_home = homes
    (default_home / "config.yaml").write_text(
        "gateway:\n  platforms:\n    buzz:\n      extra:\n"
        "        group_allow_admin_from: [default-admin]\n"
    )
    (routed_home / "config.yaml").write_text(
        "gateway:\n  platforms:\n    buzz:\n      extra:\n"
        "        group_allow_admin_from: [routed-admin]\n"
    )
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(platforms={})
    runner._plugin_source_identity_candidates = lambda _source: ("routed-admin",)
    runner._plugin_routed_profile = lambda _source: "beta"
    source = SimpleNamespace(
        platform=Platform("buzz"),
        profile="beta",
        chat_type="group",
        user_id="routed-admin",
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda _name: routed_home
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_command",
        lambda _name: {"with_context": False},
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.plugin_command_access_level",
        lambda _entry, _args, _context: "admin",
    )

    assert runner._check_slash_access(source, "buzz", "listen always") is None

    runner._plugin_source_identity_candidates = lambda _source: ("default-admin",)
    denial = runner._check_slash_access(source, "buzz", "listen always")
    assert denial is not None
    assert "admin-only" in denial
