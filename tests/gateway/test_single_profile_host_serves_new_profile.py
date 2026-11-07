"""A host gateway that booted with ONE profile comes up standalone (``SINGLE_PROFILE_REASON``), and
multiplexing is settled once per boot. A profile created afterwards (Desktop's New Agent, ``hermes
profile create``) was then never served: every ticket the Desktop minted for it through the host's
control socket was refused, so its sessions, chat and Delete all failed with "Invalid gateway ticket
response" (#141623). The creator's ``rescan-profiles`` signal (and the 30 s watcher) must start serving
it from the SAME process: no restart, no drain, and unscoped secret reads fail closed from then on.
"""
import asyncio
import os
import shutil
from types import SimpleNamespace

import pytest

import hermes_constants
from agent.secret_scope import is_multiplex_active
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.status import flush_runtime_status, write_runtime_status
from hermes_cli import gateway_migrate as gm
from hermes_cli.gateway_multiplex_mode import SINGLE_PROFILE_REASON


@pytest.fixture
def standalone_host(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True)
    (home / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    # The migration preflight's service-manager probes: nothing else on this host runs a gateway.
    monkeypatch.setattr(gm, "_installed_services", lambda _home: [])
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda _home: None)
    monkeypatch.setattr(gm, "_host_supports_migration", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "launch-key")
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=False))
    runner._running = True
    runner.session_runtime_descriptor = {"state": "ready", "served_profiles": [], "capabilities": []}
    runner.session_ticket_store = None
    restarts = []
    runner.request_restart = lambda **kw: restarts.append(kw) or True
    write_runtime_status(gateway_state="running", multiplex_standalone_reason=SINGLE_PROFILE_REASON,
                         served_profiles=[])
    flush_runtime_status()
    yield runner, home, restarts
    from gateway.runtime_ownership import process_ownership
    for profile in (home / "profiles").iterdir():
        process_ownership.release(profile.resolve())


def _mkprofile(home, name):
    d = home / "profiles" / name
    d.mkdir()
    (d / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    return d


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_profile_created_after_a_single_profile_boot_is_served_without_a_restart(standalone_host):
    """End to end through the creator's real signal: the control socket's ``rescan-profiles`` verb."""
    from gateway import control_socket
    from gateway.run_bootstrap import _start_gateway_start_control_socket

    runner, home, restarts = standalone_host
    server = await _start_gateway_start_control_socket(runner)
    assert server is not None
    try:
        answer = await asyncio.to_thread(control_socket.rescan_gateway_profiles, home)
        assert answer["multiplex"] is False and not is_multiplex_active()

        bot = _mkprofile(home, "daily-news-bot")
        (bot / ".env").write_text("OPENAI_API_KEY=bot-own-key\n", encoding="utf-8")
        answer = await asyncio.to_thread(control_socket.rescan_gateway_profiles, home)
        assert answer["multiplex"] is True
        assert answer["added"] == ["daily-news-bot"]
        assert answer["served_profiles"] == ["default", "daily-news-bot"]
        assert restarts == [] and not runner._draining
        # The new profile reads its own credentials and never borrows the launch profile's env.
        assert is_multiplex_active()
        from agent.secret_scope import UnscopedSecretError, get_secret
        from gateway.run import _profile_runtime_scope
        with _profile_runtime_scope(bot.resolve()):
            assert get_secret("OPENAI_API_KEY") == "bot-own-key"
        # The default profile's next message goes through the production handler the switch re-wired.
        seen = []

        async def fake_handle(event):
            seen.append(get_secret("OPENAI_API_KEY"))

        runner._handle_message = fake_handle
        source = SimpleNamespace(platform=None, profile=None, chat_id="1", user_id="u")
        await runner._primary_message_handler()(SimpleNamespace(source=source))
        assert seen == ["launch-key"]
        with pytest.raises(UnscopedSecretError):
            get_secret("OPENAI_API_KEY")

        # Deleting it again unserves it in place; the host keeps running.
        shutil.rmtree(bot)
        answer = await asyncio.to_thread(control_socket.rescan_gateway_profiles, home)
        assert answer["removed"] == ["daily-news-bot"] and answer["served_profiles"] == ["default"]
        assert restarts == []
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_a_blocked_new_profile_is_served_once_the_blocker_clears(standalone_host, monkeypatch):
    """A preflight blocker (the new profile still runs its own gateway) keeps the host standalone. It
    is not latched: the next rescan after the blocker clears serves the profile."""
    runner, home, restarts = standalone_host
    _mkprofile(home, "coder")
    own = {"coder": True}
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda h: os.getpid() if own.get(
        hermes_constants.profile_name_for_home(h)) else None)
    result = await runner.reconcile_served_profiles(reason="watcher")
    assert result["multiplex"] is False and not is_multiplex_active()

    own["coder"] = False
    result = await runner.reconcile_served_profiles(reason="watcher")
    assert result["added"] == ["coder"] and restarts == []


@pytest.mark.asyncio
async def test_auto_multiplex_migration_false_keeps_a_single_profile_boot_standalone(standalone_host):
    """``gateway.auto_multiplex_migration: false`` on the default profile opts out of the in-place
    switch: a second profile stays unserved by this process on the creator's signal and the watcher, and
    setting the key back to true converges on the next rescan without a restart."""
    runner, home, restarts = standalone_host
    (home / "config.yaml").write_text(
        "model: {default: m}\ngateway:\n  auto_multiplex_migration: false\n", encoding="utf-8")
    _mkprofile(home, "coder")
    for reason in ("control-socket", "watcher"):
        result = await runner.reconcile_served_profiles(reason=reason)
        assert result["multiplex"] is False and result["served_profiles"] == ["default"]
    assert not is_multiplex_active() and not runner.config.multiplex_profiles

    (home / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    result = await runner.reconcile_served_profiles(reason="watcher")
    assert result["added"] == ["coder"] and is_multiplex_active() and restarts == []


def test_launch_profile_work_on_a_standalone_host_binds_its_scope_once_a_second_home_is_hosted(standalone_host):
    """Desktop's ``profiles.create`` hosts the new home before any rescan: from then on unscoped
    secret reads fail closed, so the launch profile's own ``session.create`` (owner dispatch on a
    still-standalone gateway) must bind the launch scope, not crash with UnscopedSecretError."""
    from types import SimpleNamespace

    from agent.secret_scope import get_secret
    from gateway.session_authorities import owner_scope
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting

    runner, home, _restarts = standalone_host
    activate_multi_profile_hosting()
    with owner_scope(SimpleNamespace(runner=runner, profile_id=str(home))):
        assert get_secret("XAI_API_KEY") is None
