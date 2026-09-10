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

    assert captured.get("profile_homes") == homes
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
    nothing that runs before the scope (the restart-safe handoff) can be fail-closed (#107692)."""
    import contextvars
    import os

    import cron.scheduler as scheduler
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

    with _profile_cron_scope(routed):
        # Before the scope exists — where run_one_job's restart-safe handoff runs — nothing is
        # multiplex: the marker is set, the semantics are not.
        assert routed_profile_fire() is True
        assert secret_scope.is_multiplex_active() is False
        ctx = contextvars.copy_context()  # what _submit_with_guard hands the pool worker

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
    assert os.environ["XAI_API_KEY"] == "launch-key"
    assert "ROUTED_ONLY" not in os.environ

    # The marker reaches the worker that performs the write; the worker's own scope install is
    # what turns multiplex semantics on there.
    seen = {}

    def _worker():
        tokens = scheduler._install_fire_secret_scope()
        try:
            seen["v"] = secret_scope.is_multiplex_active()
        finally:
            scheduler._reset_fire_secret_scope(tokens)

    worker = threading.Thread(target=lambda: ctx.run(_worker))
    worker.start()
    worker.join()
    assert seen["v"] is True


def test_the_process_own_profile_fire_keeps_single_profile_semantics(tmp_path, monkeypatch):
    """The launch profile's own fire is not routed: its scope install leaves the process in
    single-profile semantics, so its .env keeps loading as today and a scope miss still reads
    the process env (systemd / ``op run`` injected keys)."""
    import cron.scheduler as scheduler
    from agent import secret_scope
    from cron.scheduler_provider import _profile_cron_scope, routed_profile_fire

    launch = tmp_path / "launch"
    (launch / "cron").mkdir(parents=True)
    (launch / ".env").write_text("XAI_API_KEY=launch-key\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("SHELL_INJECTED_TOKEN", "from-systemd")
    secret_scope.set_multiplex_active(False)

    with _profile_cron_scope(launch):
        assert routed_profile_fire() is False
        tokens = scheduler._install_fire_secret_scope()
        try:
            assert secret_scope.is_multiplex_active() is False
            assert secret_scope.get_secret("SHELL_INJECTED_TOKEN") == "from-systemd"
        finally:
            scheduler._reset_fire_secret_scope(tokens)


def test_the_restart_safe_handoff_is_not_fail_closed_by_a_routed_tick(tmp_path, monkeypatch):
    """run_one_job hands a fire to the external worker BEFORE the body installs the profile scope.
    A routed tick must not make that handoff read secrets fail-closed: with a passthrough key
    registered, building the worker env there raises UnscopedSecretError if multiplex semantics
    are on with no scope (#107399's path). The tick only marks the fire; the handoff keeps its
    current semantics (its own scope is #107413 / #106050's seam)."""
    from agent import secret_scope
    from cron.scheduler_provider import _profile_cron_scope
    from tools.env_passthrough import clear_env_passthrough, is_env_passthrough, register_env_passthrough
    from tools.environments.local import build_subprocess_env

    launch, routed = tmp_path / "launch", tmp_path / "launch" / "profiles" / "ops"
    for home in (launch, routed):
        (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("SERVICE_TOKEN", "A")
    secret_scope.set_multiplex_active(False)
    register_env_passthrough(["SERVICE_TOKEN"])
    try:
        assert is_env_passthrough("SERVICE_TOKEN")
        with _profile_cron_scope(routed):
            assert secret_scope.is_multiplex_active() is False
            build_subprocess_env(scrub_secrets=True)  # the handoff's child env, pre-scope: must not raise
    finally:
        clear_env_passthrough()
