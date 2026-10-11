"""Stored completion origins use their runtime profile under a named launcher."""

from collections import OrderedDict
from pathlib import Path

import pytest


@pytest.mark.parametrize("transport", [None, "default"])
@pytest.mark.asyncio
async def test_stored_process_origins_use_their_runtime_store(
    tmp_path, monkeypatch, transport
):
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from agent.secret_scope import get_secret
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from gateway.session import SessionSource, SessionStore
    from gateway.session_identity import resolve_identity
    from hermes_constants import get_hermes_home

    home = tmp_path / ".hermes"
    launcher = home / "profiles" / "launcher"
    secondary = home / "profiles" / "worker"
    for path in (home, launcher, secondary):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.yaml").write_text("model: {default: probe}\n")
        (path / ".env").write_text(f"STORED_ORIGIN_TOKEN={path.name}\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launcher))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._primary_profile_name = "default"
    runner.adapters = {}
    runner._profile_adapters = {"worker": {}}
    runner._agent_cache = OrderedDict()
    token = set_multiplex_context(True)
    observed = []
    stores = []
    try:
        pinned = SessionSource(
            platform=Platform.TELEGRAM, chat_id="pinned", chat_type="dm"
        )
        resolve_identity(pinned, runner=runner, primary_home=home)
        assert runner._resolve_profile_home_for_source(pinned) == home
        for profile, path in ((None, home), ("worker", secondary), (None, home)):
            with _profile_runtime_scope(path):
                store = SessionStore(path / "sessions", runner.config)
                stores.append(store)
                source = SessionSource(
                    platform=Platform.TELEGRAM,
                    chat_id="stored",
                    chat_type="dm",
                    profile=profile,
                )
                entry = store.get_or_create_session(source)
                entry.transport_profile = transport
                store.update_session(entry.session_key, touch_activity=False)
                restored = SessionStore(path / "sessions", runner.config)
                stores.append(restored)
                restored._ensure_loaded()
                restored_entry = restored._entries[entry.session_key]
                assert restored_entry.origin is not source
            runner.session_store = restored
            event = {"session_key": entry.session_key}
            recovered_source = runner._build_process_event_source(event)
            async with runner._completion_event_scope(event):
                observed.append((
                    runner._resolve_profile_home_for_source(recovered_source),
                    get_hermes_home(),
                    get_secret("STORED_ORIGIN_TOKEN"),
                ))
        assert observed == [(path, path, path.name) for path in (home, secondary, home)]
    finally:
        for store in stores:
            store.close_all_db_handles()
        reset_multiplex_context(token)


def test_standalone_default_source_keeps_the_named_launch_profile(
    tmp_path, monkeypatch
):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_identity import restore_identity

    launcher = tmp_path / ".hermes" / "profiles" / "launcher"
    launcher.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launcher))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="stored", chat_type="dm")

    identity = restore_identity(source, runner=runner, transport_profile="default")

    assert (identity, runner._resolve_profile_home_for_source(source)) == (
        None,
        launcher,
    )
