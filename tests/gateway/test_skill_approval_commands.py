"""Skills approval UX through the routed messaging dispatcher."""

from pathlib import Path

import pytest
import hermes_yaml as yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, _profile_runtime_scope
from tools import write_approval as wa


class Runner(GatewayRunner):
    def __init__(self, home):
        self.config = GatewayConfig(multiplex_profiles=True)
        self.home = home
        self.evictions = []

    def _session_key_for_source(self, source):
        return "approval-session"

    def _resolve_profile_home_for_source(self, source):
        return self.home

    def _evict_cached_agent(self, key):
        self.evictions.append(key)

    async def slash(self, args, subsystem="skills"):
        class Event:
            source = object()
            def get_command_args(self):
                return args
        event = Event()
        handled, reply = await self._hm_dispatch_canonical_command(
            event, event.source, "approval-session", subsystem)
        assert handled
        return reply


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    routed = tmp_path / "profiles" / "beta"
    launch.mkdir()
    routed.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(gateway_run, "_hermes_home", launch)
    (launch / "config.yaml").write_text("agent:\n  reasoning_effort: medium\n")
    (routed / "config.yaml").write_text(
        "# policy\nskills:\n  write_approval: false # gate\n  write_approval_mode: create # scope\n"
        "model:\n  default: 'unchanged' # keep\n")
    return launch, routed


@pytest.mark.asyncio
async def test_gateway_scope_roundtrips_are_profile_local_and_preserve_pending(profiles):
    launch, home = profiles
    runner = Runner(home)
    before = (launch / "config.yaml").read_bytes()
    with _profile_runtime_scope(home):
        record = wa.stage_write("skills", {"action": "create", "name": "queued"},
                                summary="old queued request", origin="foreground")
    pending = home / "pending" / "skills" / (record["id"] + ".json")
    pending_before = pending.read_bytes()
    for command, enabled, scope in [
        ("approval all", True, "all"), ("approval off", False, "all"),
        ("mode on", True, "all"), ("mode create", True, "create"),
    ]:
        assert "set to" in await runner.slash(command)
        config = yaml.safe_load((home / "config.yaml").read_text())
        assert config["skills"] == {"write_approval": enabled, "write_approval_mode": scope}
        assert pending.read_bytes() == pending_before
        assert record["id"] in await runner.slash("pending")
        assert scope in await runner.slash("approval status")
    assert (launch / "config.yaml").read_bytes() == before
    assert not (home / "skills" / "queued").exists()
    text = (home / "config.yaml").read_text()
    for comment in ["# policy", "# gate", "# scope", "'unchanged' # keep"]:
        assert comment in text


@pytest.mark.asyncio
async def test_gateway_failure_invalid_values_and_memory_boolean_semantics(profiles, monkeypatch):
    import utils
    _launch, home = profiles
    runner = Runner(home)
    for args, subsystem in [("approval typo", "skills"), ("approval all extra", "skills"),
                            ("approval create", "memory"), ("mode all", "memory")]:
        before = (home / "config.yaml").read_bytes()
        assert "Invalid value" in await runner.slash(args, subsystem)
        assert (home / "config.yaml").read_bytes() == before
    for arg, enabled in [("on", True), ("off", False)]:
        assert "set to" in await runner.slash("approval " + arg, "memory")
        assert yaml.safe_load((home / "config.yaml").read_text())["memory"] == {"write_approval": enabled}
    before = (home / "config.yaml").read_bytes()
    evictions = list(runner.evictions)
    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(utils.os, "replace", fail_replace)
    assert "Failed to set" in await runner.slash("approval all")
    assert (home / "config.yaml").read_bytes() == before
    assert runner.evictions == evictions
