"""Startup Bot Chat recovery of one served profile can never stop the gateway: a secondary whose
durable leftovers cannot be recovered is parked like one whose store cannot be opened, and every
other profile still starts (#140018)."""
from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing, contextmanager
from types import SimpleNamespace

import pytest

from gateway import run_runtime
from gateway.config import Platform


def _runtime(tmp_path, monkeypatch, failing):
    from gateway.session_authorities import SessionAuthorities
    launch = (tmp_path / "root").resolve()
    homes = {n: (tmp_path / "profiles" / n).resolve() for n in ("beta", "gamma")}
    for home in (launch, *homes.values()):
        home.mkdir(parents=True)
    registry = SessionAuthorities(launch, multiplexed=True)
    runner = SimpleNamespace(
        session_authorities=registry, session_control_server=None, _running_agents={},
        config=SimpleNamespace(multiplex_profiles=True, _runtime_profile_homes=tuple(homes.items())),
        session_runtime_descriptor={"served_profiles": []},
        session_ticket_store=SimpleNamespace(profile_ids=frozenset()),
        session_store=SimpleNamespace(list_sessions=list))

    async def no_native_sessions(bindings):
        return {}

    for name, home in (("default", launch), *homes.items()):
        registry.add(home, SimpleNamespace(runner=runner, profile_id=str(home), sessions={}, pending_stops={},
                                           adopted={}, retiring=False, hosted_room_service=None,
                                           waiters={}, native_waiters=set(),
                                           db=SimpleNamespace(_read_all=lambda *_a: []),
                                           logical_owner=lambda sid: sid,
                                           recover_native_sessions=no_native_sessions), name=name)
    recovered, parked = [], {}

    async def recover_bot(authority):
        recovered.append(authority.profile_id)
        if authority.profile_id == str(failing):
            raise RuntimeError("leftover receipt the store cannot answer")

    async def fake_api(runner_):
        return SimpleNamespace(api_origin="http://127.0.0.1:1")

    monkeypatch.setattr("gateway.run_api.start_gateway_api", fake_api)
    monkeypatch.setattr("gateway.session_bot.recover_bot_deliveries", recover_bot)
    monkeypatch.setattr("gateway.session_cron.unbind_owner", lambda authority: None)
    monkeypatch.setattr(run_runtime, "_record_parked_profiles", lambda value: parked.update(value))
    return runner, launch, homes, recovered, parked


def test_a_secondary_whose_recovery_fails_is_parked_and_the_rest_start(tmp_path, monkeypatch):
    from gateway.runtime_ownership import process_ownership
    runner, launch, homes, recovered, parked = _runtime(tmp_path, monkeypatch, failing=tmp_path / "profiles" / "beta")
    process_ownership.reserve([homes["beta"]])
    try:
        asyncio.run(run_runtime.start_gateway_runtime_api(runner))
        assert runner.session_runtime_descriptor["api_origin"] == "http://127.0.0.1:1"
        assert recovered == [str(launch), str(homes["beta"]), str(homes["gamma"])]
        assert "beta" in parked and "leftover receipt" in parked["beta"]
        assert runner.session_authorities.for_home(homes["beta"]) is None
        assert not process_ownership.owns(homes["beta"]), "a parked profile keeps its reservation"
        assert [n for n, _ in runner.config._runtime_profile_homes] == ["gamma"]
        served = {p["profile_id"] for p in runner.session_runtime_descriptor["served_profiles"]}
        assert served == {str(launch), str(homes["gamma"])}
    finally:
        process_ownership.release(homes["beta"])


def test_a_secondary_whose_retirement_times_out_keeps_its_lock_but_is_never_started(tmp_path, monkeypatch):
    """A writer that outlives Stop keeps the profile's lock and retiring authority (shutdown still
    settles it), but nothing later in boot may serve it: no adapters, no served status, no cron
    ticks, no native-session recovery, no hosted-room service."""
    from gateway.run import GatewayRunner, _cron_tick_profile_homes
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import all_authorities
    runner, launch, homes, _recovered, parked = _runtime(
        tmp_path, monkeypatch, failing=tmp_path / "profiles" / "beta")
    beta = runner.session_authorities.for_home(homes["beta"])

    async def writer_outlives_stop(authority, timeout):
        raise TimeoutError("Profile workers did not stop; ownership retained")

    monkeypatch.setattr("gateway.session_runtime_workers.join_authority_work", writer_outlives_stop)
    process_ownership.reserve([homes["beta"]])
    try:
        asyncio.run(run_runtime.start_gateway_runtime_api(runner))
        assert "ownership retained" in parked["beta"]
        assert process_ownership.owns(homes["beta"]), "the live writer's home lost its lock"
        assert runner.session_authorities.for_home(homes["beta"]) is beta and beta.retiring
        assert beta in all_authorities(runner), "shutdown must still settle the retained authority"

        # Adapter startup and the adapter-level served status (the real runner methods).
        monkeypatch.setenv("HERMES_HOME", str(launch))
        started, published = [], {}
        host = object.__new__(GatewayRunner)
        host.config, host.adapters, host._profile_adapters = runner.config, {}, {}
        host.pairing_store, host.pairing_stores, host._served_profile_signatures = None, {}, {}
        host._primary_profile_name = "default"

        async def start_one(name, home, claimed):
            started.append(name)
            return 0

        host._start_one_profile_adapters = start_one
        host._restore_secondary_completion_ledgers = lambda profile_homes: None
        monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda *a, **k: [])
        monkeypatch.setattr("gateway.status.publish_runtime_status", lambda **kw: published.update(kw))
        monkeypatch.setattr("gateway.pairing.PairingStore", lambda profile: SimpleNamespace(profile=profile))
        asyncio.run(host._start_secondary_profile_adapters())
        assert started == ["gamma"]
        assert published["served_profiles"] == ["default", "gamma"]
        assert "beta" not in {name for name, _home in _cron_tick_profile_homes(runner.config)}

        # Native-session recovery and hosted-room service creation cover only serving profiles.
        local, hosted = [], []

        async def ensure_hosted(runner_, authority):
            hosted.append(authority.profile_id)

        async def no_webhooks(authority):
            return None

        monkeypatch.setattr("gateway.session_hosted_service._ensure_hosted_service", ensure_hosted)
        monkeypatch.setattr("gateway.session_local_recovery.recover_local_sessions",
                            lambda authority, schedule=False: local.append(authority.profile_id))
        monkeypatch.setattr("gateway.platforms.webhook_ingress.recover_webhook_finalizations", no_webhooks)
        asyncio.run(run_runtime.recover_gateway_native_sessions(runner))
        assert str(homes["beta"]) not in local
        assert str(homes["beta"]) not in hosted
        assert set(local) == set(hosted) == {str(launch), str(homes["gamma"])}

        # The retained store may still fail every read: neither the launch API adapter's
        # publication (API-turn recovery, the real publish path) nor the startup auto-resume guard
        # may let that error stop the boot.
        def unreadable(*_a, **_k):
            raise sqlite3.OperationalError("disk I/O error")

        @contextmanager
        def no_admissions():
            with closing(sqlite3.connect(":memory:")) as conn:
                conn.execute("CREATE TABLE session_admissions (target_session_id, principal_id, status)")
                yield conn

        for authority in all_authorities(runner):
            authority.db._read_ctx = unreadable if authority is beta else no_admissions
            authority.db._read_one = unreadable if authority is beta else (lambda *_a: None)
        api_read = []

        def collect(db):
            if db is beta.db:
                unreadable()
            api_read.append(db)

        monkeypatch.setattr("gateway.session_ingress_media.collect_unheld_api_images", collect)
        monkeypatch.setattr("gateway.session_api_media.compact_settled_api_payloads", lambda db: None)
        host._sync_voice_mode_state_to_adapter = host._bind_voice_input_callback = lambda adapter: None
        api_adapter = SimpleNamespace(gateway_runner=runner)
        try:
            host._publish_primary_adapter(Platform.API_SERVER, api_adapter)
        except sqlite3.OperationalError as exc:
            pytest.fail(f"the retained store's read error stopped API adapter publication: {exc}")
        assert host.adapters[Platform.API_SERVER] is api_adapter
        assert api_read == [a.db for a in all_authorities(runner) if a is not beta]

        # Auto-resume fails closed on the retained store: the entry is held back, never resumed.
        fresh_checks = []
        monkeypatch.setattr("gateway.run._is_fresh_gateway_interruption",
                            lambda *a, **k: fresh_checks.append(a) or False)
        host.session_authorities = runner.session_authorities
        for authority in all_authorities(runner):
            if authority is not beta:
                authority.logical_owner = lambda sid: ""  # no lineage: the bare id is the owner
        entry = SimpleNamespace(session_id="s1", session_key="k1", last_resume_marked_at=None, updated_at=None)
        host._resume_pending_candidates = lambda platform=None: [entry]
        try:
            assert host._schedule_resume_pending_sessions() == 0
        except sqlite3.OperationalError as exc:
            pytest.fail(f"the retained store's read error stopped startup auto-resume: {exc}")
        assert fresh_checks == [], "an entry the retained store may own went on toward resume"
    finally:
        process_ownership.release(homes["beta"])


def test_the_launch_profile_recovery_failure_still_stops_the_start(tmp_path, monkeypatch):
    runner, _launch, _homes, _recovered, parked = _runtime(tmp_path, monkeypatch, failing=tmp_path / "root")
    with pytest.raises(RuntimeError, match="leftover receipt"):
        asyncio.run(run_runtime.start_gateway_runtime_api(runner))
    assert parked == {}
