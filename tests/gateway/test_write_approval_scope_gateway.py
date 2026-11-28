"""Messaging-gateway ``/skills approval background`` and ``/memory approval background`` persist the
boolean gate and the scope into the routed profile's config.yaml, and ``/skills`` keeps answering."""

from __future__ import annotations

import pytest

import gateway.run as gateway_run
import hermes_yaml as yaml
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, _profile_runtime_scope


class _Runner(GatewayRunner):
    def __init__(self):  # skip GatewayRunner.__init__: only the dispatcher + slash mixins are exercised
        self.config = GatewayConfig(multiplex_profiles=True)

    def _session_key_for_source(self, _source):
        return "k"

    def _evict_cached_agent(self, _session_key):
        pass

    def _resolve_profile_home_for_source(self, _source):
        return getattr(self, "routed_home", gateway_run._gateway_config_home())

    async def slash(self, name: str, args: str) -> str:
        event = _Event(args)
        handled, reply = await self._hm_dispatch_canonical_command(event, event.source, "k", name)
        assert handled
        return reply


class _Event:
    def __init__(self, args: str = ""):
        self._args = args
        self.source = object()

    def get_command_args(self) -> str:
        return self._args


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("agent:\n  reasoning_effort: medium\n")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.mark.asyncio
async def test_gateway_background_sets_gate_and_scope_and_keeps_skills_reachable(home):
    runner = _Runner()
    runner.routed_home = home
    with _profile_runtime_scope(home):
        reply = await runner.slash("skills", "approval background")
        assert "set to 'on' (background writes only" in reply
        await runner.slash("memory", "approval background")
        # The boolean gate is on, so /skills review keeps answering (no "approval is off" detour).
        assert "No pending skills writes" in await runner.slash("skills", "pending")
    saved = yaml.safe_load((home / "config.yaml").read_text())
    for section in ("skills", "memory"):
        assert saved[section]["write_approval"] is True
        assert saved[section].get("write_approval_scope") == "background"
    assert saved["agent"]["reasoning_effort"] == "medium"  # unrelated keys untouched


@pytest.mark.asyncio
async def test_gateway_on_resets_scope_and_off_keeps_it(home):
    runner = _Runner()
    runner.routed_home = home
    with _profile_runtime_scope(home):
        await runner.slash("skills", "approval background")
        await runner.slash("skills", "approval on")
        assert yaml.safe_load((home / "config.yaml").read_text())["skills"].get("write_approval_scope") == "all"
        await runner.slash("skills", "approval background")
        await runner.slash("skills", "approval off")
    skills = yaml.safe_load((home / "config.yaml").read_text())["skills"]
    assert skills["write_approval"] is False
    assert skills.get("write_approval_scope") == "background"
