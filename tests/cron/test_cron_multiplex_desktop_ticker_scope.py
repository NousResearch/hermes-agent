"""Regression tests for #100489 — desktop multiplex ticker must not deliver a
secondary profile's cron output through the default profile's identity.

Two halves:

1. ``_deliver_result``'s standalone fallback pool (taken when the caller has a
   RUNNING event loop — the desktop dashboard shape) spawns a fresh thread that
   did not inherit the profile ContextVars; it must run inside a copy of the
   active context so the sender reads THIS profile's home + secrets.
2. The desktop ticker must stand down, per tick, for a profile whose OWN
   gateway is running — that gateway ticks it with live adapters, and racing it
   on the tick lock lets the adapter-less desktop ticker deliver standalone.
"""
import asyncio
import threading
from unittest.mock import patch

import pytest



def test_standalone_fallback_pool_keeps_profile_scope(tmp_path, monkeypatch):
    from agent.secret_scope import (
        get_secret,
        set_multiplex_active,
        set_secret_scope,
    )
    from hermes_constants import get_hermes_home, set_hermes_home_override
    import cron.scheduler as sched
    import tools.send_message_tool as smt

    default_home = tmp_path / "default"
    sec_home = tmp_path / "profiles" / "ops"
    for home in (default_home, sec_home):
        (home / "cron").mkdir(parents=True)
        (home / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "DEFAULT-TOKEN")
    set_multiplex_active(True)

    seen = {}

    async def fake_send(platform, pconfig, chat_id, message, **kwargs):
        seen["home"] = str(get_hermes_home())
        seen["token"] = get_secret("TELEGRAM_BOT_TOKEN", None)
        return {"success": True, "message_id": "1"}

    job = {"id": "j1", "name": "probe", "deliver": "telegram:12345", "schedule": {"kind": "cron"}}

    async def _inside_running_loop():
        # Emulate the multiplex ticker's per-profile scope on the caller.
        set_hermes_home_override(str(sec_home))
        set_secret_scope({"TELEGRAM_BOT_TOKEN": "OPS-TOKEN"})
        return sched._deliver_result(job, "hello", adapters={}, loop=None)

    try:
        with patch.object(smt, "_send_to_platform", fake_send):
            err = asyncio.run(_inside_running_loop())
    finally:
        set_multiplex_active(False)

    assert err is None, err
    assert seen["home"] == str(sec_home.resolve())
    assert seen["token"] == "OPS-TOKEN"


def test_multiplex_ticker_profile_gate_skips_rejected_profile(tmp_path):
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_constants import get_hermes_home

    own_gateway = tmp_path / "own-gateway"
    orphan = tmp_path / "orphan"
    for home in (own_gateway, orphan):
        (home / "cron").mkdir(parents=True)

    stop = threading.Event()
    ticked: list[str] = []

    def _tick(*args, **kwargs):
        ticked.append(str(get_hermes_home()))
        if len(ticked) >= 3:
            stop.set()
        return 0

    provider = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=_tick):
        thread = threading.Thread(
            target=provider.start,
            args=(stop,),
            kwargs={
                "interval": 0,
                "profile_homes": [("own-gateway", own_gateway), ("orphan", orphan)],
                "profile_gate": lambda name, home: name != "own-gateway",
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5)
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert set(ticked) == {str(orphan)}
    # The gated profile gets no tick-loop success marker either: its own
    # gateway owns that status surface.
    assert not (own_gateway / "cron" / "ticker_last_success").exists()
    assert (orphan / "cron" / "ticker_last_success").exists()


def test_multiplex_ticker_recheck_gate_under_tick_for_each_profile(tmp_path):
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_constants import get_hermes_home

    own_gateway = tmp_path / "own-gateway"
    orphan = tmp_path / "orphan"
    for home in (own_gateway, orphan):
        (home / "cron").mkdir(parents=True)

    stop = threading.Event()
    gate_calls = {"own-gateway": 0, "orphan": 0}
    rechecks: list[tuple[str, bool | None]] = []
    tick_counts = {str(own_gateway): 0, str(orphan): 0}

    def profile_gate(name, _home):
        gate_calls[name] += 1
        return name != "own-gateway" or gate_calls[name] % 2 == 1

    def _tick(*args, **kwargs):
        home = str(get_hermes_home())
        gate = kwargs.get("can_dispatch")
        rechecks.append((home, gate() if gate is not None else None))
        tick_counts[home] += 1
        if all(count >= 2 for count in tick_counts.values()):
            stop.set()
        return 0

    provider = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=_tick):
        thread = threading.Thread(
            target=provider.start,
            args=(stop,),
            kwargs={
                "interval": 0,
                "profile_homes": [("own-gateway", own_gateway), ("orphan", orphan)],
                "profile_gate": profile_gate,
            },
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5)
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    own_gateway_checks = [
        allowed for home, allowed in rechecks if home == str(own_gateway)
    ]
    orphan_checks = [allowed for home, allowed in rechecks if home == str(orphan)]
    assert len(own_gateway_checks) >= 2
    assert len(orphan_checks) >= 2
    assert all(allowed is False for allowed in own_gateway_checks)
    assert all(allowed is True for allowed in orphan_checks)

    passthrough_stop = threading.Event()
    received_gates = []

    def passthrough():
        return True

    def _capture_gate(*args, **kwargs):
        received_gates.append(kwargs.get("can_dispatch"))
        passthrough_stop.set()
        return 0

    provider = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=_capture_gate):
        passthrough_thread = threading.Thread(
            target=provider.start,
            args=(passthrough_stop,),
            kwargs={
                "interval": 0,
                "profile_homes": [("orphan", orphan)],
                "can_dispatch": passthrough,
                "profile_gate": None,
            },
            daemon=True,
        )
        passthrough_thread.start()
        passthrough_thread.join(timeout=5)
        passthrough_stop.set()
        passthrough_thread.join(timeout=5)

    assert not passthrough_thread.is_alive()
    assert len(received_gates) == 1
    assert received_gates[0] is passthrough


@pytest.mark.parametrize("profile_count", [1, 2])
def test_desktop_ticker_gates_on_profile_gateway_running(tmp_path, monkeypatch, profile_count):
    """Desktop yields to each live gateway, including a single-profile install."""
    from hermes_cli import web_server

    homes = [("default", tmp_path / "default"), ("ops", tmp_path / "ops")][:profile_count]
    running = {homes[-1][1]}
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve", lambda multiplex=False: list(homes)
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._check_gateway_running", lambda home: home in running
    )
    monkeypatch.setattr("hermes_cli.profiles._served_by_running_multiplexer", lambda name: False)
    captured = {}

    class _Provider:
        name = "fake"

        def start(self, stop_event, **kwargs):
            captured.update(kwargs)

    from cron import scheduler_provider as sp

    monkeypatch.setattr(web_server, "resolve_cron_scheduler", lambda: _Provider(), raising=False)
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: _Provider())
    monkeypatch.setattr(sp, "InProcessCronScheduler", _Provider)
    monkeypatch.setattr("hermes_logging.enable_profile_log_routing", lambda homes: None)

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    # The Desktop hands the scheduler a live enumerator, not a startup snapshot,
    # so profiles created or deleted while the app runs are picked up per tick.
    profile_homes = captured.get("profile_homes")
    assert callable(profile_homes)
    assert profile_homes() == homes
    gate = captured.get("profile_gate")
    assert gate is not None, "desktop ticker did not install a profile gate"
    for name, home in homes:
        assert gate(name, home) is (home not in running)

    # Re-evaluate on each tick: Desktop resumes fallback after gateway exit
    # and stands down again if a gateway starts later.
    running.clear()
    assert all(gate(name, home) for name, home in homes)
    running.update(home for _, home in homes)
    assert not any(gate(name, home) for name, home in homes)


def test_a_routed_profile_fire_runs_under_multiplex_semantics_for_exactly_its_scope(tmp_path, monkeypatch):
    """The desktop ticker fires a SIBLING profile's job from a process that is not a multiplexer.
    The tick only MARKS the fire as routed; multiplex semantics switch on where run_one_job
    installs the profile's secret scope and off with it — so the routed .env stays out of the
    shared os.environ, a scope miss never falls back to the launch profile's credentials, and
    the parent process env is byte-identical after the tick (#107692)."""
    import os

    from cron import scheduler
    from agent import secret_scope
    from cron.scheduler_provider import _profile_cron_scope, routed_profile_fire
    from hermes_cli.env_loader import load_hermes_dotenv

    launch, routed = tmp_path / "launch", tmp_path / "launch" / "profiles" / "ops"
    for home in (launch, routed):
        (home / "cron").mkdir(parents=True)
    (launch / ".env").write_text("XAI_API_KEY=launch-key\nDISCORD_BOT_TOKEN=launch-bot\n", encoding="utf-8")
    (routed / ".env").write_text("XAI_API_KEY=routed-key\nROUTED_ONLY=routed-only\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("XAI_API_KEY", "launch-key")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "launch-bot")
    monkeypatch.delenv("ROUTED_ONLY", raising=False)
    secret_scope.set_multiplex_active(False)  # the desktop backend never sets the process flag
    environ_before = dict(os.environ)

    with _profile_cron_scope(routed):
        # Before the scope exists — where run_one_job's restart-safe handoff runs — the marker is
        # set, the semantics are not.
        assert routed_profile_fire() is True
        assert secret_scope.is_multiplex_active() is False

        tokens = scheduler._install_fire_secret_scope()
        try:
            assert secret_scope.is_multiplex_active() is True
            # The job's per-run dotenv reload, exactly as cron/scheduler does it: hydrate-only.
            assert load_hermes_dotenv(hermes_home=routed, load_external_secrets=False) == []
            assert secret_scope.get_secret("XAI_API_KEY") == "routed-key"
            assert secret_scope.get_secret("DISCORD_BOT_TOKEN") is None  # never the launch bot
        finally:
            scheduler._reset_fire_secret_scope(tokens)
        assert secret_scope.is_multiplex_active() is False  # off with the scope, not later

    assert routed_profile_fire() is False
    assert dict(os.environ) == environ_before


@pytest.mark.parametrize(
    ("routed_env_line", "expected"),
    [
        ("", "managed-key"),  # managed-only credential: resolvable, not absent
        ("ORG_API_KEY=user-key\n", "managed-key"),  # managed-vs-user collision: policy wins
    ],
    ids=["managed-only", "managed-beats-user"],
)
def test_routed_fire_scope_carries_managed_env_authority(tmp_path, monkeypatch, routed_env_line, expected):
    """Under multiplex semantics get_secret never falls back to os.environ, so the routed fire's
    scope itself must carry the administrator-managed .env with the precedence _apply_managed_env
    gives it in the launch process: a managed-only key is present and a managed value beats the
    routed profile's own (#111187 review)."""
    from cron import scheduler
    from agent import secret_scope
    from cron.scheduler_provider import _profile_cron_scope
    from hermes_cli import managed_scope

    launch, routed = tmp_path / "launch", tmp_path / "launch" / "profiles" / "ops"
    managed = tmp_path / "managed"
    for home in (launch, routed, managed):
        (home / "cron").mkdir(parents=True)
    (routed / ".env").write_text(routed_env_line, encoding="utf-8")
    (managed / ".env").write_text("ORG_API_KEY=managed-key\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("ORG_API_KEY", "managed-key")  # what _apply_managed_env left in the launch env
    managed_scope.invalidate_managed_cache()
    secret_scope.set_multiplex_active(False)

    with _profile_cron_scope(routed):
        tokens = scheduler._install_fire_secret_scope()
        try:
            assert secret_scope.is_multiplex_active() is True
            assert secret_scope.get_secret("ORG_API_KEY") == expected
        finally:
            scheduler._reset_fire_secret_scope(tokens)


def test_dashboard_run_now_isolates_a_sibling_profile_fire_like_the_ticker(tmp_path, monkeypatch):
    """The dashboard's manual Run now binds the sibling profile through ``_cron_store_scope``, not the
    ticker's ``_profile_cron_scope`` (#126982 review). Routed-fire detection follows the fire's HOME,
    so this entry point gets the same isolation: the routed ``.env`` reaches the job but never the
    shared ``os.environ``, the launch profile's own credential never reaches the routed job, and the
    launch profile's own fire afterwards (A -> B -> A) keeps single-profile semantics."""
    import os

    from cron import scheduler
    from hermes_cli import web_server_cron
    from agent import secret_scope
    from cron.jobs import create_job
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_cli import profiles
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch = tmp_path / ".hermes"
    routed = launch / "profiles" / "ops"
    probe = tmp_path / "probe"
    probe.mkdir()
    for home in (launch, routed):
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "scripts").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
    (launch / ".env").write_text("LAUNCH_ONLY_SECRET=launch-secret\n", encoding="utf-8")
    (routed / ".env").write_text("ROUTED_ONLY_SECRET=routed-secret\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("LAUNCH_ONLY_SECRET", "launch-secret")
    monkeypatch.delenv("ROUTED_ONLY_SECRET", raising=False)
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: launch)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: launch / "profiles")
    monkeypatch.setattr(scheduler, "_hermes_home", None)
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: InProcessCronScheduler())
    secret_scope.set_multiplex_active(False)  # the desktop backend never sets the process flag

    def _job_for(home, name):
        (home / "scripts" / f"{name}.sh").write_text(
            "#!/usr/bin/env bash\n"
            f'echo "${{ROUTED_ONLY_SECRET:-<unset>}}|${{LAUNCH_ONLY_SECRET:-<unset>}}" > "{probe / name}"\n',
            encoding="utf-8")
        token = set_hermes_home_override(str(home))
        try:
            return create_job(prompt=None, schedule="every 1h", script=f"{name}.sh",
                              no_agent=True, deliver="local")["id"]
        finally:
            reset_hermes_home_override(token)

    routed_job, launch_job = _job_for(routed, "routed"), _job_for(launch, "launch")
    environ_before = dict(os.environ)

    # B: the sibling profile, fired from the launch process's dashboard.
    assert web_server_cron._fire_cron_job_for_profile("ops", routed_job) is True
    assert (probe / "routed").read_text().strip() == "routed-secret|<unset>"
    assert "ROUTED_ONLY_SECRET" not in os.environ
    assert dict(os.environ) == environ_before
    assert secret_scope.is_multiplex_active() is False

    # A again: the launch profile's own fire is not routed and still sees its own env.
    assert web_server_cron._fire_cron_job_for_profile("default", launch_job) is True
    assert (probe / "launch").read_text().strip() == "<unset>|launch-secret"
    assert "ROUTED_ONLY_SECRET" not in os.environ
