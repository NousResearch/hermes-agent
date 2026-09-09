"""Same-profile credentials, persistence and adapter busy policy without touching real profiles."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_secondary_profile_busy_policy_credentials_and_database(tmp_path, monkeypatch):
    from hermes_cli.commands import should_bypass_active_session
    from hermes_cli.plugins import get_plugin_manager
    from hermes_cli.plugin_side_runs import SideRunConfig
    from gateway.side_runs import SideRunService
    from gateway.session_context import get_session_env

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    secondary = root / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text("plugins:\n  enabled: []\n")
    (secondary / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": ["fixture"]},
        "providers": {"fixture": {"base_url": "http://route.invalid/v1", "key_env": "FIXTURE_ROUTE_KEY"}},
    }))
    (secondary / ".env").write_text("FIXTURE_ROUTE_KEY=secondary-fixture\n")
    plugin_dir = secondary / "plugins" / "fixture"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: fixture\nversion: 0.1.0\n")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n    ctx.register_command('side-secondary', lambda args: 'ok', busy_policy='noninterrupting')\n")
    with _profile_runtime_scope(secondary):
        manager = get_plugin_manager()
        manager.discover_and_load()
    assert not should_bypass_active_session("side-secondary")
    assert should_bypass_active_session("side-secondary", profile="secondary")
    assert get_hermes_home() == root
    primary_db = SessionDB(root / "state.db")
    secondary_db = SessionDB(secondary / "state.db")
    observed = []
    class Agent:
        def __init__(self, **kwargs):
            self.provider, self.model = kwargs["provider"], kwargs["model"]
            observed.append((get_hermes_home(), kwargs["api_key"], get_session_env("HERMES_SESSION_PROFILE")))
        def run_conversation(self, **kwargs):
            return {"final_response": "secondary answer"}
        def close(self):
            pass
        def interrupt(self, *args, **kwargs):
            pass
    monkeypatch.setattr("run_agent.AIAgent", Agent)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    def open_profile_db():
        assert get_hermes_home() == secondary, "side-run admission opened the primary profile DB"
        return SimpleNamespace(_db=secondary_db)
    runner._open_session_db_for_active_scope = open_profile_db
    runner._resolve_profile_home_for_source = lambda source: secondary
    runner._draining = False
    runner.session_store = SimpleNamespace(_entries={})
    runner._session_key_for_source = lambda source: "secondary-parent"
    runner._reply_anchor_for_event = lambda event: "trigger"
    runner._thread_metadata_for_source = lambda source, *args: {"hermes_profile": source.profile}
    secondary_adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(send=AsyncMock())}
    runner._adapter_for_source = lambda source: secondary_adapter
    service = SideRunService(runner)
    source = SessionSource(Platform.TELEGRAM, "chat", user_id="owner", profile="secondary")
    sid = service.start(MessageEvent(text="/side-secondary prompt", source=source), "fixture", "prompt",
                        SideRunConfig.from_mapping({"provider": "custom:fixture", "model": "fixture-model"}))
    try:
        await asyncio.wait_for(service.wait(), 10)
        assert observed == [(secondary, "secondary-fixture", "secondary")]
        assert primary_db.get_session(sid) is None
        assert secondary_db.get_session(sid)["end_reason"] == "side_run_completed"
        assert secondary_adapter.send.await_args.kwargs["metadata"]["hermes_profile"] == "secondary"
        runner.adapters[Platform.TELEGRAM].send.assert_not_awaited()
    finally:
        service.shutdown()
        await service.wait()
        runner._shutdown_executor()
        manager.unload()
        secondary_db.close()
        primary_db.close()
