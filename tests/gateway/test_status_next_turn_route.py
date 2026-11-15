"""/status with no resident agent and no /model pin names the route the NEXT turn resolves.

That route is ``_resolve_session_agent_runtime``'s, inside the owning profile's scope: a secondary
runtime profile's config, a provider-only ``channel_overrides`` entry, a LOCAL session's frozen launch
policy. A parallel "default" lookup that skipped any of them replaced a correct recorded route.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway.test_status_command import _make_event
from tests.gateway.test_status_model_truth import _runner_with_old_usage


async def _multiplexed_secondary_profile(tmp_path, monkeypatch):
    from gateway import run
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    secondary = home / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(run, "_hermes_home", home)
    (home / "config.yaml").write_text(json.dumps({"model": {"default": "launch-model", "provider": "openrouter"}}))
    (secondary / "config.yaml").write_text(json.dumps({"model": {"default": "secondary-model",
                                                                 "provider": "openrouter"}}))
    runner, db = _runner_with_old_usage(tmp_path)
    runner.config.multiplex_profiles = True
    db.update_token_counts("sess-1", model="secondary-model", billing_provider="openrouter",
                           input_tokens=5, api_call_count=1)
    runner._resolve_route_context = AsyncMock(return_value=None)
    event = _make_event("/status")
    event.source.profile = "secondary"
    try:
        return await runner._handle_status_command(event), "**Model:** `secondary-model` (openrouter)"
    finally:
        db.close()


async def _provider_only_channel_override(tmp_path, monkeypatch):
    from gateway.config import ChannelOverride, Platform
    runner, db = _runner_with_old_usage(tmp_path)
    runner._resolve_route_context = AsyncMock(return_value=None)
    runner.config.platforms[Platform.TELEGRAM].channel_overrides = {"c1": ChannelOverride(provider="anthropic")}
    db.update_token_counts("sess-1", model="shared-model", billing_provider="anthropic", input_tokens=5,
                           api_call_count=1)
    try:
        with patch("gateway.run._load_gateway_config",
                   return_value={"model": {"default": "shared-model", "provider": "openrouter"}}), \
                patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"provider": "openrouter"}), \
                patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                      return_value={"provider": "anthropic"}):
            out = await runner._handle_status_command(_make_event("/status"))
            assert runner._resolve_session_agent_runtime(source=_make_event("/status").source)[0] == "shared-model"
        return out, "**Model:** `shared-model` (anthropic)"
    finally:
        db.close()


async def _frozen_local_launch_policy(tmp_path, monkeypatch):
    """A real LOCAL session (authority, frozen policy, slash.exec) whose profile default moved on."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_state import AsyncSessionDB, SessionDB
    monkeypatch.setattr(run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({
        "model": {"provider": "custom", "default": "session-model", "base_url": "http://127.0.0.1:9/v1",
                  "api_key": "loopback-only"},
        "auxiliary": {"title_generation": {"enabled": False}}}))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    db = SessionDB(tmp_path / "state.db")
    runner.session_store._db = runner._session_db = db
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id="owner", db=db)
    connection = AuthorityConnection(authority, object(), {"user_id": "owner"})
    try:
        sid = create_local_session(authority, connection.actor, {
            "request_id": "pinned", "source": "cli", "cwd": str(tmp_path), "toolsets": [],
            "model": "pinned-model", "provider": "custom", "base_url": "http://127.0.0.1:9/v1",
        }).session_id
        await connection.dispatch({"id": "r", "method": "session.resume", "params": {"session_id": sid}})
        db.update_token_counts(sid, model="pinned-model", billing_provider="custom", input_tokens=5,
                               api_call_count=1)
        runner._session_db = AsyncSessionDB(db)  # production's slash reads go through the async facade
        reply = await connection.dispatch({"id": "s", "method": "slash.exec",
                                           "params": {"session_id": sid, "command": "/status"}})
        assert runner._cached_agent_for(authority.sessions[sid].route) is None
        return reply["result"]["output"], "**Model:** `pinned-model`"
    finally:
        runner._session_db = db
        await connection.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("build", [
    _multiplexed_secondary_profile, _provider_only_channel_override, _frozen_local_launch_policy,
], ids=lambda f: f.__name__.strip("_"))
async def test_cold_status_names_the_route_the_next_turn_resolves(tmp_path, monkeypatch, build):
    out, expected = await build(tmp_path, monkeypatch)
    model_line = next(line for line in out.splitlines() if "**Model:**" in line)
    assert expected in model_line, model_line
