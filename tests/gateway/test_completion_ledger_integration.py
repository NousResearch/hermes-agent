"""Ledger provenance reaches the real API adapter and preserves launch-profile scope."""

import asyncio
import contextlib
from pathlib import Path

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state, scope_for_profile
from agent.secret_scope import current_secret_scope_home, get_secret, set_multiplex_active
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.session_authorities import SessionAuthorities
from gateway.session_authority import SessionAuthority
from hermes_constants import get_hermes_home, get_hermes_home_override, hermes_home_key
from persistence_assembly import close_stores_under, open_persistence, profile_scope
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_session
from tests.gateway.test_completion_ledger_owner import row, seed
from tools import async_delegation as ledger
from tools.terminal_scope import terminal_env


@pytest.fixture
def api_profiles(tmp_path, monkeypatch):
    launch = tmp_path / ".hermes"
    homes = {"default": launch, "worker": launch / "profiles" / "worker",
             "reviewer": launch / "profiles" / "reviewer"}
    for home in homes.values():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("ENV_ONLY_MARKER", "launch-only")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "launch.invalid")
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True, sessions_dir=launch / "sessions"))
    runner._served_profile_homes = homes
    runner._primary_profile_name = "default"
    runner.session_authorities = SessionAuthorities(launch, multiplexed=True)
    stores = {}
    for name, home in homes.items():
        store = stores[name] = open_persistence(profile_scope(home=home))
        authority = SessionAuthority(runner, profile_id=str(home), instance_id="completion-test",
                                     persistence=store, epoch=store.runtime.begin_epoch(instance_id="completion-test"))
        runner.session_authorities.add(home, authority, name=name)
        if name == "default":
            runner.session_authority = authority
        seed_session(store, "raw-shared", "api_server")
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting
    activate_multi_profile_hosting()
    ledger._reset_for_tests()
    try:
        yield runner, adapter, homes, stores
    finally:
        ledger._reset_for_tests()
        set_multiplex_active(False)
        adapter._close_cached_session_dbs()
        runner.session_store.close_all_db_handles()
        runner.close_all_session_db_handles()
        close_stores_under(launch)


@pytest.mark.asyncio
async def test_raw_completion_persists_only_in_owning_profile_and_retries_without_transport(api_profiles):
    runner, adapter, homes, stores = api_profiles
    events = {name: seed(home, "deleg-shared", parent="raw-shared") for name, home in homes.items()}
    original = {name: row(home, "deleg-shared")[2] for name, home in homes.items()}
    runner.adapters = {}
    for name, event in events.items():
        async with runner._completion_event_scope(event):
            assert terminal_env("TERMINAL_ENV") == ("ssh" if name == "default" else "local")
            assert get_secret("ENV_ONLY_MARKER") == ("launch-only" if name == "default" else None)
        # Real readiness refuses a missing transport without claiming or exhausting the result.
        assert await runner._deliver_async_delegation_group([event]) is False
        assert row(homes[name], "deleg-shared")[:2] == ("pending", 0)
    runner.adapters[Platform.API_SERVER] = adapter
    for name, event in events.items():
        assert await runner._deliver_async_delegation_group([event]) is True
        saved = row(homes[name], "deleg-shared")
        assert saved == ("delivered", 1, original[name])
        delivered = transcript_rows(stores[name], "raw-shared")
        assert len(delivered) == 1
        assert delivered[0]["display_kind"] == "async_delegation_complete"
        assert delivered[0]["display_metadata"]["delegation_id"] == "deleg-shared"
        # A duplicate queue copy is settled without appending a second transcript row.
        await runner._deliver_async_delegation_group([dict(event)])
        assert transcript_rows(stores[name], "raw-shared") == delivered
        for other in set(homes) - {name}:
            expected = 1 if row(homes[other], "deleg-shared")[0] == "delivered" else 0
            assert len(transcript_rows(stores[other], "raw-shared")) == expected
    assert get_hermes_home() == homes["default"]


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["unscoped", "launch", "prepared", "foreign"])
async def test_single_profile_stamp_keeps_launch_environment_and_restores_scope(tmp_path, monkeypatch, caller):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("PROFILE_MARKER=profile-value\n", encoding="utf-8")
    (home / "config.yaml").write_text("terminal:\n  timeout: 17\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("PROFILE_MARKER", "ambient-value")
    monkeypatch.setenv("ENV_ONLY_MARKER", "injected-value")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "fixture.invalid")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "17")
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=False, sessions_dir=home / "sessions"))
    foreign = home / "profiles" / "foreign"
    foreign.mkdir(parents=True)
    (foreign / ".env").write_text("PROFILE_MARKER=foreign-value\n", encoding="utf-8")
    (foreign / "config.yaml").write_text("terminal:\n  timeout: 3\n", encoding="utf-8")
    if caller == "prepared":
        outer = bind_state(scope_for_profile(home),
            secrets={"PROFILE_MARKER": "prepared-value", "ENV_ONLY_MARKER": "injected-value"},
            terminal_policy={"TERMINAL_ENV": "docker", "TERMINAL_TIMEOUT": "31",
                             "TERMINAL_SSH_HOST": "prepared.invalid"})
    else:
        outer = (contextlib.nullcontext() if caller == "unscoped"
                 else bind_profile_scope(home if caller == "launch" else foreign))
    try:
        with outer:
            before = (get_hermes_home_override(), current_secret_scope_home(), get_secret("PROFILE_MARKER"),
                      terminal_env("TERMINAL_ENV"), terminal_env("TERMINAL_TIMEOUT"), terminal_env("TERMINAL_SSH_HOST"))
            expected_marker = "profile-value" if caller == "foreign" else before[2]
            expected_terminal = ("ssh", "17", "fixture.invalid") if caller == "foreign" else before[3:]
            event = {"type": "async_delegation", "_completion_ledger_home": hermes_home_key(home)}
            async with runner._completion_event_scope(event):
                assert get_hermes_home() == home
                assert get_secret("PROFILE_MARKER") == expected_marker
                assert (terminal_env("TERMINAL_ENV"), terminal_env("TERMINAL_TIMEOUT"),
                        terminal_env("TERMINAL_SSH_HOST")) == expected_terminal
                assert get_secret("ENV_ONLY_MARKER") == "injected-value"
                assert await asyncio.to_thread(get_hermes_home) == home
                assert await asyncio.to_thread(get_secret, "PROFILE_MARKER") == expected_marker
            assert (get_hermes_home_override(), current_secret_scope_home(), get_secret("PROFILE_MARKER"),
                    terminal_env("TERMINAL_ENV"), terminal_env("TERMINAL_TIMEOUT"), terminal_env("TERMINAL_SSH_HOST")) == before
    finally:
        runner.session_store.close_all_db_handles()
        runner.close_all_session_db_handles()
        close_stores_under(home)


@pytest.mark.asyncio
async def test_pinned_launch_owner_survives_a_mirrored_process_home(api_profiles, monkeypatch):
    runner, _adapter, homes, stores = api_profiles
    event = seed(homes["default"], "deleg-pinned", parent="raw-shared")
    runner.adapters[Platform.API_SERVER] = _adapter
    with bind_profile_scope(homes["default"]):
        # Embedding hosts may mirror their active profile into the process env;
        # the runtime's launch identity was pinned before that mirror changed.
        monkeypatch.setenv("HERMES_HOME", str(homes["worker"]))
        async with runner._completion_event_scope(event):
            assert get_hermes_home() == homes["default"]
            assert get_secret("ENV_ONLY_MARKER") == "launch-only"
        assert await runner._deliver_async_delegation_group([event]) is True
        assert row(homes["default"], "deleg-pinned")[:2] == ("delivered", 1)
        assert len(transcript_rows(stores["default"], "raw-shared")) == 1
        assert transcript_rows(stores["worker"], "raw-shared") == []
        assert transcript_rows(stores["reviewer"], "raw-shared") == []
        assert get_hermes_home() == homes["default"]


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["default", "worker"])
@pytest.mark.parametrize("multiplexed", [False, True])
async def test_hosted_scope_cannot_enter_a_local_completion_owner(api_profiles, owner, multiplexed):
    from hermes_scope import Principal, Scope, ScopeSwitchRefused, current_scope

    runner, _adapter, homes, stores = api_profiles
    set_multiplex_active(multiplexed)
    hosted = Scope("remote-tenant", "remote-agent", Principal("remote-user"),
                   owner=Principal("remote-user"), can_administer=True)
    with bind_state(hosted):
        with pytest.raises(ScopeSwitchRefused):
            async with runner._completion_event_scope({"_completion_ledger_home": hermes_home_key(homes[owner])}):
                pytest.fail("hosted activity entered a local profile")
        assert current_scope() == hosted
    assert all(transcript_rows(store, "raw-shared") == [] for store in stores.values())
