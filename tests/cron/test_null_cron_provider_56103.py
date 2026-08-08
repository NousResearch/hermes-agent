"""NullCronScheduler — the explicit no-trigger provider (#56103).

``cron.provider: none`` (aliases ``off``/``disabled``) selects a provider that
never ticks, so a gateway can run purely as an inbound/serving process — the
standby half of an active/standby HA pair, where the built-in flock tick-lock
cannot coordinate across hosts. These tests pin the full contract:

  - the resolver maps the reserved names (case-insensitively, plus the YAML-1.1
    boolean footguns: bare ``off``→False→disabled, bare ``on``→True→builtin,
    and ``null``/``~``→None→builtin *by design*) to the right provider WITHOUT
    consulting plugin discovery, so a plugin directory named "none" can never
    shadow the disable switch;
  - a config read failure falls back to the built-in ticker LOUDLY (a silent
    fallback would re-arm the scheduler on a deliberately disabled standby);
  - start() returns immediately, never calls cron.scheduler.tick, and never
    records a ticker heartbeat (no false "alive" signal for `cron status`);
  - fire_due() refuses inbound fires with a logged warning — a disabled
    instance must not execute jobs off the /api/cron/fire webhook;
  - manual execution stays available BY DESIGN: `hermes cron run` (via
    _execute_job_now) and `hermes cron tick` are explicit human actions, not
    the automatic trigger the null provider suppresses — tick warns first;
  - `hermes cron status` reports the scheduler as disabled instead of the
    external-provider "jobs fire via the managed scheduler" line or the
    ticker-staleness heuristics;
  - job create/list still warns that jobs won't fire on this instance (the
    #51038 silent-never-fires failure mode).
"""
import logging
import threading
from unittest.mock import patch

import pytest


# ── Resolver: reserved names select the null provider ────────────────────────


@pytest.mark.parametrize("value", ["none", "off", "disabled", "None", "OFF", "Disabled"])
def test_resolve_reserved_names_return_null_provider(monkeypatch, caplog, value):
    """Every reserved spelling resolves to the null provider (name 'none'),
    and the operator sees an INFO line saying the scheduler is disabled."""
    import hermes_cli.config as cfg
    from cron import scheduler_provider as sp

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": value}})
    with caplog.at_level(logging.INFO, logger="cron.scheduler_provider"):
        prov = sp.resolve_cron_scheduler()
    assert isinstance(prov, sp.NullCronScheduler)
    assert prov.name == "none"
    assert "Cron scheduler disabled" in caplog.text


def test_resolve_yaml_boolean_false_means_disabled(monkeypatch):
    """YAML 1.1 parses a bare `off` / `no` as boolean False. A user who wrote
    `provider: off` asked for "no scheduler" — that must select the null
    provider, not silently fall back to the built-in ticker."""
    import hermes_cli.config as cfg
    from cron import scheduler_provider as sp

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": False}})
    prov = sp.resolve_cron_scheduler()
    assert isinstance(prov, sp.NullCronScheduler)


def test_resolve_yaml_boolean_true_means_builtin(monkeypatch):
    """The symmetric footgun: a bare `on` / `yes` parses as boolean True —
    "scheduler on" is the default built-in ticker, mapped explicitly rather
    than via a swallowed AttributeError."""
    import hermes_cli.config as cfg
    from cron import scheduler_provider as sp

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": True}})
    assert sp.resolve_cron_scheduler().name == "builtin"


def test_resolve_yaml_null_means_default_builtin(monkeypatch):
    """DELIBERATE: a YAML `null` / `~` parses to Python None, which is
    indistinguishable from a bare `provider:` (key present, empty value) and
    therefore keeps meaning "unset → built-in". Disabling requires the quoted
    string "none" — documented in the config comment and cron-internals.md."""
    import hermes_cli.config as cfg
    from cron import scheduler_provider as sp

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": None}})
    assert sp.resolve_cron_scheduler().name == "builtin"


def test_resolve_empty_still_builtin(monkeypatch):
    """The default (empty) is untouched by the reserved names — built-in."""
    import hermes_cli.config as cfg
    from cron import scheduler_provider as sp

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": ""}})
    assert sp.resolve_cron_scheduler().name == "builtin"


def test_resolve_config_error_warns_and_falls_back_builtin(monkeypatch, caplog):
    """A config read failure falls back to the built-in ticker (the historical
    fail-open contract) but must do so LOUDLY: on a `none`-configured standby
    that fallback re-arms the scheduler, so silence would hide a double-fire
    hazard (#56103)."""
    import hermes_cli.config as cfg
    from cron import scheduler_provider as sp

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(cfg, "load_config", _boom)
    with caplog.at_level(logging.WARNING, logger="cron.scheduler_provider"):
        prov = sp.resolve_cron_scheduler()
    assert prov.name == "builtin"
    assert "Could not read cron.provider" in caplog.text
    assert "WILL tick" in caplog.text


def test_reserved_names_bypass_plugin_discovery(monkeypatch):
    """'none' is resolved BEFORE plugin discovery: a plugin named 'none' can
    never shadow the disable switch, and no plugin code runs at all.

    The loader mock RECORDS calls rather than only raising — the resolver's
    plugin path is wrapped in `except Exception`, so a raise alone would be
    swallowed there and the ordering regression could go undetected."""
    import hermes_cli.config as cfg
    import plugins.cron_providers as pc
    from cron import scheduler_provider as sp

    calls = []

    def _record(name):
        calls.append(name)
        raise AssertionError("plugin discovery must not run for reserved names")

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": "none"}})
    monkeypatch.setattr(pc, "load_cron_scheduler", _record)
    prov = sp.resolve_cron_scheduler()
    assert isinstance(prov, sp.NullCronScheduler)
    assert calls == [], "plugin discovery was consulted for a reserved name"


def test_null_is_available_true():
    """'none' is an explicit operator choice, never a degraded state the
    resolver should fall back from — is_available() must stay True."""
    from cron.scheduler_provider import NullCronScheduler

    assert NullCronScheduler().is_available() is True


# ── start(): no ticking, no heartbeat, immediate return ─────────────────────


def test_null_start_never_ticks_and_never_heartbeats(caplog):
    """start() must not call cron.scheduler.tick and must not record a ticker
    heartbeat — a heartbeat would make `cron status` claim a live ticker on an
    instance that intentionally has none. Run in a thread with a join timeout
    so a regression to a blocking loop FAILS instead of hanging the test."""
    from cron.scheduler_provider import NullCronScheduler

    stop = threading.Event()  # NOT set — start() must return regardless
    with patch("cron.scheduler.tick") as tick, \
         patch("cron.jobs.record_ticker_heartbeat") as heartbeat, \
         caplog.at_level(logging.INFO, logger="cron.scheduler_provider"):
        t = threading.Thread(
            target=NullCronScheduler().start, args=(stop,),
            kwargs={"interval": 0}, daemon=True,
        )
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "start() blocked instead of returning"

    tick.assert_not_called()
    heartbeat.assert_not_called()
    assert "Cron scheduler disabled" in caplog.text


def test_null_start_thread_exits_without_stop_event():
    """Gateway contract: the provider runs in a daemon thread and shutdown
    awaits that thread. With the null provider the thread must exit on its own,
    immediately — never blocking shutdown on a stop_event nobody needs."""
    from cron.scheduler_provider import NullCronScheduler

    stop = threading.Event()
    t = threading.Thread(
        target=NullCronScheduler().start, args=(stop,), daemon=True
    )
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "null provider start() did not return immediately"


def test_null_stop_is_noop():
    from cron.scheduler_provider import NullCronScheduler

    assert NullCronScheduler().stop() is None


def test_null_inherits_jobs_changed_and_reconcile_noops():
    from cron.scheduler_provider import NullCronScheduler

    p = NullCronScheduler()
    assert p.on_jobs_changed() is None
    assert p.reconcile() is None


# ── fire_due(): a disabled instance refuses inbound fires ────────────────────


def test_null_fire_due_refuses_and_never_runs(monkeypatch, caplog):
    """The /api/cron/fire webhook resolves the provider and calls fire_due.
    The inherited default would claim and EXECUTE the job — silently
    re-enabling remote cron execution on a disabled instance. The null
    provider must return False without claiming or running anything, and log
    a warning naming the refused job (the webhook has already returned 202,
    so this log line is the only operator-visible trace)."""
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.scheduler_provider import NullCronScheduler

    claimed, ran = [], []
    monkeypatch.setattr(
        jobs, "claim_job_for_fire", lambda jid: claimed.append(jid) or True,
        raising=False,
    )
    monkeypatch.setattr(
        sched, "run_one_job", lambda job, **kw: ran.append(job["id"]) or True
    )

    with caplog.at_level(logging.WARNING, logger="cron.scheduler_provider"):
        assert NullCronScheduler().fire_due("j1") is False
    assert claimed == [], "disabled instance claimed a fire"
    assert ran == [], "disabled instance ran a job"
    assert "refused" in caplog.text
    assert "j1" in caplog.text


# ── manual execution stays available by design ───────────────────────────────


def test_manual_run_still_executes_on_disabled_instance(monkeypatch):
    """Docs promise (`hermes cron run` row, NullCronScheduler docstring,
    config comment, cron-internals.md): manual run still executes with
    cron.provider: none. The manual path (_execute_job_now, #41037) must NOT
    route through the provider's fire_due, which refuses on a null instance."""
    import hermes_cli.config as cfg
    import cron.scheduler as sched
    from tools import cronjob_tools as ct

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": "none"}})

    job = {"id": "j1", "name": "manual", "prompt": "hi", "enabled": True}
    ran = []
    monkeypatch.setattr(ct, "claim_job_for_fire", lambda jid: True)
    monkeypatch.setattr(ct, "get_job", lambda jid: dict(job, last_status="ok"))
    monkeypatch.setattr(sched, "run_one_job", lambda j, **kw: ran.append(j["id"]) or True)

    result = ct._execute_job_now(job)
    assert result["claimed"] is True
    assert result["success"] is True
    assert ran == ["j1"], "manual run did not execute on the disabled instance"


def test_cron_tick_warns_but_still_ticks_on_disabled_instance(monkeypatch, capsys):
    """`hermes cron tick` is a deliberate manual override — like `cron run`,
    an explicit human action. On a disabled instance it must still execute,
    but say so first, so an operator poking a "disabled" standby is not
    surprised that jobs fire."""
    from hermes_cli import cron as cron_cli

    ticks = []
    monkeypatch.setattr(cron_cli, "_active_cron_provider_name", lambda: "none")
    with patch("cron.scheduler.tick", side_effect=lambda **kw: ticks.append(kw) or 0):
        cron_cli.cron_tick()

    out = capsys.readouterr().out
    assert "manual override" in out
    assert len(ticks) == 1, "tick did not run under the manual override"


# ── ABC guard: the null provider satisfies the unchanged required surface ────


def test_required_abc_surface_unchanged():
    """Adding the null provider must not grow the ABC's required surface
    (name + start) — see test_abc_growth_stays_additive."""
    from cron.scheduler_provider import CronScheduler, NullCronScheduler

    assert set(CronScheduler.__abstractmethods__) == {"name", "start"}
    NullCronScheduler()  # instantiable → satisfies the ABC as-is


# ── `hermes cron status` reports "disabled", not "managed" or "stalled" ─────


def test_cron_status_reports_disabled(monkeypatch, capsys):
    """With provider 'none' (resolved through the REAL resolver from config),
    status must say the scheduler is disabled — not the external-provider
    'jobs fire via the managed scheduler' line, not 'will fire automatically',
    and not the ticker-staleness warnings."""
    import hermes_cli.config as cfg
    import cron.jobs as jobs
    from hermes_cli import cron as cron_cli

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": "none"}})
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [4321])
    # Stale ages would trip the STALLED heuristics if the none-branch leaked
    # through to them.
    monkeypatch.setattr(jobs, "get_ticker_heartbeat_age", lambda: 9_999.0)
    monkeypatch.setattr(jobs, "get_ticker_success_age", lambda: 9_999.0)
    monkeypatch.setattr("cron.jobs.list_jobs", lambda **k: [])

    cron_cli.cron_status()
    out = capsys.readouterr().out
    assert "DISABLED" in out
    assert "will NOT fire here" in out
    assert "will fire automatically" not in out
    assert "managed scheduler" not in out
    assert "STALLED" not in out


def test_cron_status_still_summarizes_jobs_when_disabled(monkeypatch, capsys):
    """The active-jobs summary still prints (job state is intact — only the
    trigger is off)."""
    import hermes_cli.config as cfg
    from hermes_cli import cron as cron_cli

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": "none"}})
    monkeypatch.setattr(
        "cron.jobs.list_jobs",
        lambda **k: [{"id": "j1", "next_run_at": "2099-01-01T00:00:00Z"}],
    )

    cron_cli.cron_status()
    out = capsys.readouterr().out
    assert "1 active job(s)" in out
    assert "Next run: 2099-01-01T00:00:00Z" in out


# ── create/list warning: jobs created here will not fire here ────────────────


def test_warn_on_disabled_instance_even_with_gateway_running(monkeypatch, capsys):
    """Creating/listing jobs on a 'none' instance warns that they won't fire
    here — even though the gateway process is up (the process heuristic says
    nothing about a scheduler that intentionally never ticks). Same
    silent-never-fires class as #51038."""
    from hermes_cli import cron as cron_cli

    monkeypatch.setattr(cron_cli, "_active_cron_provider_name", lambda: "none")
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [4321])

    cron_cli._warn_if_gateway_not_running()
    out = capsys.readouterr().out
    assert "disabled on this instance" in out
    assert "jobs won't fire here" in out


def test_warn_stays_quiet_for_external_provider(monkeypatch, capsys):
    """Contrast: a real external provider (e.g. chronos) still gets the
    historical silent treatment — its jobs DO fire, via webhook."""
    from hermes_cli import cron as cron_cli

    monkeypatch.setattr(cron_cli, "_active_cron_provider_name", lambda: "chronos")
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])

    cron_cli._warn_if_gateway_not_running()
    assert capsys.readouterr().out == ""
