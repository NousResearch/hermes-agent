"""Characterization tests for the cron trigger before/after the provider refactor.

These lock the CURRENT in-process-ticker contract (Phase 0 of the pluggable
CronScheduler plan, .hermes/plans/cron-scheduler-provider-interface.md). They
must pass unchanged on `main` now, and after every subsequent phase of the
refactor — they are the regression harness that proves the built-in firing
behavior is byte-for-byte preserved when the ticker is moved behind the
CronScheduler provider interface.

No production code is exercised beyond the two ticker entry points:
  - gateway/run.py::_start_cron_ticker        (production gateway ticker)
  - hermes_cli/web_server.py::_start_desktop_cron_ticker  (desktop fallback)

Both call `cron.scheduler.tick(...)` on a loop and exit when their stop_event
is set. We patch `cron.scheduler.tick` (both tickers import it locally as
`cron_tick`, so the module-attribute patch is observed) and assert the loop
drives it and stops promptly.
"""
import threading
import time
from unittest.mock import patch


def _wait_until(predicate, timeout=10.0, interval=0.005):
    """Block until ``predicate()`` is truthy or ``timeout`` elapses.

    Returns the predicate's final value. Used instead of a fixed
    ``time.sleep`` before asserting that a background ticker thread has called
    tick()/heartbeat() at least N times — under loaded CI the worker thread may
    not be scheduled within a short fixed sleep, which made these tests flake
    (``assert 0 >= 1`` / ``provider never called tick()``).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def test_ticker_calls_tick_at_least_once_then_stops():
    """The gateway in-process ticker loop calls cron.scheduler.tick repeatedly
    and exits promptly once the stop_event is set."""
    from gateway.run import _start_cron_ticker

    calls = []
    stop = threading.Event()

    def fake_tick(*args, **kwargs):
        calls.append(kwargs)
        return 0

    with patch("cron.scheduler.tick", side_effect=fake_tick):
        # interval=0 keeps the loop tight; stop after the first observed tick.
        t = threading.Thread(
            target=_start_cron_ticker,
            args=(stop,),
            kwargs={"interval": 0},
            daemon=True,
        )
        t.start()
        assert _wait_until(lambda: len(calls) >= 1), "ticker never called tick()"
        stop.set()
        t.join(timeout=5)

    assert not t.is_alive(), "ticker did not exit after stop_event was set"
    assert len(calls) >= 1, "ticker never called tick()"
    # Contract: the ticker invokes tick with sync=False (fire-and-forget from
    # the background thread, never the synchronous CLI path).
    assert calls[0].get("sync") is False


def test_desktop_ticker_calls_tick_then_stops():
    """The desktop dashboard ticker loop calls cron.scheduler.tick and exits
    once the stop_event is set. Desktop has no live adapters, so it ticks with
    no adapters/loop."""
    from hermes_cli.web_server import _start_desktop_cron_ticker

    calls = []
    stop = threading.Event()

    def fake_tick(*args, **kwargs):
        calls.append(kwargs)
        return 0

    with patch("cron.scheduler.tick", side_effect=fake_tick):
        t = threading.Thread(
            target=_start_desktop_cron_ticker,
            args=(stop,),
            kwargs={"interval": 0},
            daemon=True,
        )
        t.start()
        assert _wait_until(lambda: len(calls) >= 1), "desktop ticker never called tick()"
        stop.set()
        t.join(timeout=5)

    assert not t.is_alive(), "desktop ticker did not exit after stop_event was set"
    assert len(calls) >= 1, "desktop ticker never called tick()"
    assert calls[0].get("sync") is False


# ── Phase 1: CronScheduler ABC + InProcessCronScheduler ──────────────────────


def test_cronscheduler_is_abstract():
    """name + start are abstract — the bare ABC can't be instantiated."""
    import pytest
    from cron.scheduler_provider import CronScheduler

    with pytest.raises(TypeError):
        CronScheduler()


def test_cronscheduler_default_is_available_true():
    """is_available defaults to True (no-network) for a minimal subclass."""
    from cron.scheduler_provider import CronScheduler

    class Dummy(CronScheduler):
        @property
        def name(self):
            return "dummy"

        def start(self, stop_event, **kw):
            pass

    assert Dummy().is_available() is True


def test_abc_growth_stays_additive():
    """Forward-compat guard: the ABC's REQUIRED surface is exactly name+start.

    Any optional hook added later for the external provider
    (on_jobs_changed/fire_due/reconcile) must be NON-abstract (carry a default),
    so the built-in keeps satisfying the ABC without overriding them. This test
    fails loudly if someone makes a future hook abstract (a breaking change that
    would force every provider — including the built-in — to implement it).
    """
    from cron.scheduler_provider import CronScheduler

    abstract = set(getattr(CronScheduler, "__abstractmethods__", set()))
    assert abstract == {"name", "start"}, (
        f"CronScheduler abstractmethods changed to {abstract}; growth must be "
        "additive (optional methods with defaults), not new abstract methods."
    )


def test_inprocess_provider_ticks_and_stops():
    """The built-in provider drives cron.scheduler.tick(sync=False) on a loop
    and exits promptly when stop_event is set — same contract as the raw
    ticker characterized above."""
    from cron.scheduler_provider import InProcessCronScheduler

    calls = []
    stop = threading.Event()
    prov = InProcessCronScheduler()
    assert prov.name == "builtin"

    with patch("cron.scheduler.tick", side_effect=lambda *a, **k: calls.append(k) or 0):
        t = threading.Thread(
            target=prov.start, args=(stop,), kwargs={"interval": 0}, daemon=True
        )
        t.start()
        # Wait for the loop to actually call tick() at least once rather than
        # sleeping a fixed window — under loaded CI the worker thread may not be
        # scheduled within a short sleep, which made this flake (assert 0 >= 1).
        assert _wait_until(lambda: len(calls) >= 1), "provider never called tick()"
        stop.set()
        t.join(timeout=5)

    assert not t.is_alive(), "provider did not exit after stop_event was set"
    assert len(calls) >= 1, "provider never called tick()"
    assert calls[0].get("sync") is False


def test_inprocess_provider_stop_is_noop():
    """The default stop() hook is a safe no-op (the stop_event is the real
    stop signal for the built-in)."""
    from cron.scheduler_provider import InProcessCronScheduler

    assert InProcessCronScheduler().stop() is None


# ── Phase 2: config key, discovery, resolver ─────────────────────────────────


def test_default_config_cron_provider_is_empty():
    """The new cron.provider key defaults to empty (= built-in)."""
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["cron"]["provider"] == ""


def test_discover_cron_schedulers_returns_list():
    """Discovery returns bundled non-default providers.

    The built-in is core, not discovered here.
    """
    from plugins.cron_providers import discover_cron_schedulers

    result = discover_cron_schedulers()
    assert isinstance(result, list)
    assert any(name == "chronos" for name, _desc, _available in result)


def test_load_unknown_cron_scheduler_returns_none():
    from plugins.cron_providers import load_cron_scheduler

    assert load_cron_scheduler("does-not-exist-xyz") is None


def test_cron_provider_package_does_not_shadow_core_cron_package(monkeypatch):
    """Putting plugins/ first on sys.path must not hide the core cron package."""
    from importlib.machinery import PathFinder
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]

    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.syspath_prepend(str(repo_root / "plugins"))

    cron_spec = PathFinder.find_spec("cron")
    assert cron_spec is not None
    assert Path(cron_spec.origin).resolve() == repo_root / "cron" / "__init__.py"

    jobs_spec = PathFinder.find_spec("cron.jobs", [str(repo_root / "cron")])
    assert jobs_spec is not None
    assert Path(jobs_spec.origin).resolve() == repo_root / "cron" / "jobs.py"


def test_resolve_defaults_to_builtin(monkeypatch):
    """Empty cron.provider → built-in."""
    import hermes_cli.config as cfg
    from cron import scheduler_provider as sp

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": ""}})
    prov = sp.resolve_cron_scheduler()
    assert prov.name == "builtin"


def test_resolve_no_cron_section_falls_back_to_builtin(monkeypatch):
    """Config with no cron section at all → built-in (cfg_get returns default)."""
    import hermes_cli.config as cfg
    from cron import scheduler_provider as sp

    monkeypatch.setattr(cfg, "load_config", lambda: {})
    prov = sp.resolve_cron_scheduler()
    assert prov.name == "builtin"


def test_resolve_unknown_provider_falls_back_to_builtin(monkeypatch):
    """A named provider that doesn't exist → built-in (cron never dies)."""
    import hermes_cli.config as cfg
    from cron import scheduler_provider as sp

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": "nope-not-real"}})
    prov = sp.resolve_cron_scheduler()
    assert prov.name == "builtin"


def test_resolve_unavailable_provider_falls_back(monkeypatch):
    """A provider that loads but reports is_available()==False → built-in."""
    import hermes_cli.config as cfg
    import plugins.cron_providers as pc
    from cron import scheduler_provider as sp
    from cron.scheduler_provider import CronScheduler

    class Unavailable(CronScheduler):
        @property
        def name(self):
            return "unavailable"

        def is_available(self):
            return False

        def start(self, stop_event, **kw):
            pass

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": "unavailable"}})
    monkeypatch.setattr(pc, "load_cron_scheduler", lambda n: Unavailable())
    prov = sp.resolve_cron_scheduler()
    assert prov.name == "builtin"


def test_resolve_available_provider_is_used(monkeypatch):
    """A provider that loads and is available is returned (not the fallback)."""
    import hermes_cli.config as cfg
    import plugins.cron_providers as pc
    from cron import scheduler_provider as sp
    from cron.scheduler_provider import CronScheduler

    class Fake(CronScheduler):
        @property
        def name(self):
            return "fake"

        def is_available(self):
            return True

        def start(self, stop_event, **kw):
            pass

    monkeypatch.setattr(cfg, "load_config", lambda: {"cron": {"provider": "fake"}})
    monkeypatch.setattr(pc, "load_cron_scheduler", lambda n: Fake())
    prov = sp.resolve_cron_scheduler()
    assert prov.name == "fake"


# ── Phase 4B: additive hooks (on_jobs_changed / fire_due / reconcile) ────────


def test_hooks_did_not_change_required_surface():
    """The additive hooks must NOT become abstractmethods — the Phase-1 guard
    still holds (required surface is exactly name + start)."""
    from cron.scheduler_provider import CronScheduler

    assert set(CronScheduler.__abstractmethods__) == {"name", "start"}


def test_builtin_inherits_hook_defaults():
    """The built-in inherits no-op defaults for the new hooks (it never needs
    to override them)."""
    from cron.scheduler_provider import InProcessCronScheduler

    p = InProcessCronScheduler()
    assert p.on_jobs_changed() is None
    assert p.reconcile() is None
    # built-in does not override fire_due; it simply isn't called for built-in.
    assert hasattr(p, "fire_due")


def test_fire_due_default_claims_then_runs(monkeypatch):
    """The default fire_due claims via the store CAS, fetches the job, and runs
    it through the shared run_one_job body."""
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.scheduler_provider import InProcessCronScheduler

    ran = []
    monkeypatch.setattr(jobs, "claim_job_for_fire", lambda jid: True, raising=False)
    monkeypatch.setattr(jobs, "get_job", lambda jid: {"id": jid, "name": "t"})
    monkeypatch.setattr(sched, "run_one_job", lambda job, **kw: ran.append(job["id"]) or True)

    assert InProcessCronScheduler().fire_due("j1") is True
    assert ran == ["j1"]


def test_fire_due_lost_claim_does_not_run(monkeypatch):
    """If the CAS claim is lost (another machine/retry won), fire_due returns
    False and never runs the job."""
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.scheduler_provider import InProcessCronScheduler

    ran = []
    monkeypatch.setattr(jobs, "claim_job_for_fire", lambda jid: False, raising=False)
    monkeypatch.setattr(sched, "run_one_job", lambda job, **kw: ran.append(job["id"]) or True)

    assert InProcessCronScheduler().fire_due("j1") is False
    assert ran == []


def test_fire_due_missing_job_does_not_run(monkeypatch):
    """If the job vanished between arm and fire (e.g. repeat-N exhausted),
    fire_due returns False without running."""
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.scheduler_provider import InProcessCronScheduler

    ran = []
    monkeypatch.setattr(jobs, "claim_job_for_fire", lambda jid: True, raising=False)
    monkeypatch.setattr(jobs, "get_job", lambda jid: None)
    monkeypatch.setattr(sched, "run_one_job", lambda job, **kw: ran.append(job["id"]) or True)

    assert InProcessCronScheduler().fire_due("gone") is False
    assert ran == []


# ── F2a: ticker liveness — survival, heartbeat, honest status (#32612, #32895) ──


def test_ticker_survives_baseexception_from_tick():
    """A BaseException (e.g. SystemExit from a provider SDK) raised by tick()
    must NOT kill the ticker loop — it logs and keeps looping (#32612)."""
    from cron.scheduler_provider import InProcessCronScheduler

    calls = []

    def _boom(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise SystemExit("provider SDK called sys.exit")
        return 0

    stop = threading.Event()
    prov = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=_boom), \
         patch("cron.jobs.record_ticker_heartbeat"):
        t = threading.Thread(target=prov.start, args=(stop,), kwargs={"interval": 0}, daemon=True)
        t.start()
        # Survive the BaseException AND keep ticking: wait for ≥2 calls.
        assert _wait_until(lambda: len(calls) >= 2), \
            "ticker did not keep ticking after the BaseException"
        stop.set()
        t.join(timeout=5)

    assert not t.is_alive(), "ticker thread died on BaseException instead of surviving"
    assert len(calls) >= 2, "ticker did not keep ticking after the BaseException"


def test_ticker_records_heartbeat_each_iteration():
    """The loop records a liveness heartbeat on start and after each tick,
    bumping the success marker only on a clean tick."""
    from cron.scheduler_provider import InProcessCronScheduler

    beats = []  # (success,) per call
    stop = threading.Event()
    prov = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=lambda *a, **k: 0), \
         patch("cron.jobs.record_ticker_heartbeat",
               side_effect=lambda success=False: beats.append(success)):
        t = threading.Thread(target=prov.start, args=(stop,), kwargs={"interval": 0}, daemon=True)
        t.start()
        # Wait for the pre-loop liveness beat AND at least one successful
        # post-tick beat before stopping (was a fixed 0.2s sleep → flaky).
        assert _wait_until(lambda: any(b is True for b in beats[1:])), \
            "successful tick did not bump success marker"
        stop.set()
        t.join(timeout=5)

    # one pre-loop liveness beat (success=False) + post-tick beats with success=True
    assert len(beats) >= 2, "ticker did not record heartbeats"
    assert beats[0] is False, "pre-loop beat should be liveness-only"
    assert any(b is True for b in beats[1:]), "successful tick did not bump success marker"


def test_failing_tick_records_liveness_but_not_success():
    """A tick that raises bumps the liveness heartbeat but NOT the success
    marker — so status can distinguish 'alive but failing' from 'firing'."""
    from cron.scheduler_provider import InProcessCronScheduler

    beats = []
    stop = threading.Event()
    prov = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=RuntimeError("every tick fails")), \
         patch("cron.jobs.record_ticker_heartbeat",
               side_effect=lambda success=False: beats.append(success)):
        t = threading.Thread(target=prov.start, args=(stop,), kwargs={"interval": 0}, daemon=True)
        t.start()
        # Wait for the pre-loop beat + at least one post-tick beat (was flaky
        # with a fixed 0.2s sleep under loaded CI).
        assert _wait_until(lambda: len(beats) >= 2), "ticker did not record heartbeats"
        stop.set()
        t.join(timeout=5)

    # every post-tick beat must be success=False (ticks always failed)
    assert len(beats) >= 2
    assert all(b is False for b in beats), "a failing tick wrongly bumped the success marker"


def test_heartbeat_roundtrip_and_age(tmp_path, monkeypatch):
    """record_ticker_heartbeat writes fresh timestamps atomically; the age
    getters read them back as small positive ages."""
    import cron.jobs as jobs

    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr(jobs, "TICKER_HEARTBEAT_FILE", cron_dir / "ticker_heartbeat")
    monkeypatch.setattr(jobs, "TICKER_SUCCESS_FILE", cron_dir / "ticker_last_success")

    # No files yet -> unknown (None), NOT "dead"
    assert jobs.get_ticker_heartbeat_age() is None
    assert jobs.get_ticker_success_age() is None

    # liveness-only: heartbeat set, success still unknown
    jobs.record_ticker_heartbeat(success=False)
    hb = jobs.get_ticker_heartbeat_age()
    assert hb is not None and 0.0 <= hb < 5.0
    assert jobs.get_ticker_success_age() is None

    # success: both set
    jobs.record_ticker_heartbeat(success=True)
    ok = jobs.get_ticker_success_age()
    assert ok is not None and 0.0 <= ok < 5.0


def test_heartbeat_age_detects_staleness(tmp_path, monkeypatch):
    """A heartbeat written far in the past reads back as a large age."""
    import cron.jobs as jobs

    cron_dir = tmp_path / "cron"
    cron_dir.mkdir(parents=True)
    hb = cron_dir / "ticker_heartbeat"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "TICKER_HEARTBEAT_FILE", hb)

    import time as _t
    hb.write_text(str(_t.time() - 10_000), encoding="utf-8")
    age = jobs.get_ticker_heartbeat_age()
    assert age is not None and age > 9_000


def test_heartbeat_write_failure_is_silent(tmp_path, monkeypatch):
    """A real atomic-write failure must be swallowed AND leave no temp file.

    Point CRON_DIR at a path that cannot be created (its parent is a regular
    file), so ensure_dirs()/mkstemp inside _atomic_write_epoch genuinely fail.
    record_ticker_heartbeat must not raise, and no stray .hb_*.tmp may leak.
    """
    import cron.jobs as jobs

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file, not a directory")
    bad_cron_dir = blocker / "cron"  # parent is a file -> mkdir/mkstemp fail
    monkeypatch.setattr(jobs, "CRON_DIR", bad_cron_dir)
    monkeypatch.setattr(jobs, "OUTPUT_DIR", bad_cron_dir / "output")
    monkeypatch.setattr(jobs, "TICKER_HEARTBEAT_FILE", bad_cron_dir / "ticker_heartbeat")
    monkeypatch.setattr(jobs, "TICKER_SUCCESS_FILE", bad_cron_dir / "ticker_last_success")

    jobs.record_ticker_heartbeat(success=True)  # must not raise

    # The write never succeeded, so no heartbeat is recorded...
    assert jobs.get_ticker_heartbeat_age() is None
    # ...and no stray temp file leaked anywhere under tmp_path.
    assert not list(tmp_path.rglob(".hb_*.tmp")), "atomic write leaked a temp file on failure"


def test_cron_status_reports_alive_but_failing(tmp_path, monkeypatch, capsys):
    """cron_status warns when the ticker is alive (fresh heartbeat) but no tick
    has succeeded recently (#32612: alive-but-failing must not look healthy)."""
    import cron.jobs as jobs
    from hermes_cli import cron as cron_cli

    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [4321])
    monkeypatch.setattr(jobs, "get_ticker_heartbeat_age", lambda: 5.0)      # fresh
    monkeypatch.setattr(jobs, "get_ticker_success_age", lambda: 9_999.0)    # stale
    monkeypatch.setattr("cron.jobs.list_jobs", lambda **k: [])

    cron_cli.cron_status()
    out = capsys.readouterr().out
    assert "no tick has succeeded" in out
    assert "will fire automatically" not in out


def test_cron_status_healthy_when_both_fresh(tmp_path, monkeypatch, capsys):
    import cron.jobs as jobs
    from hermes_cli import cron as cron_cli

    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [4321])
    monkeypatch.setattr(jobs, "get_ticker_heartbeat_age", lambda: 5.0)
    monkeypatch.setattr(jobs, "get_ticker_success_age", lambda: 5.0)
    monkeypatch.setattr("cron.jobs.list_jobs", lambda **k: [])

    cron_cli.cron_status()
    out = capsys.readouterr().out
    assert "will fire automatically" in out


def test_cron_status_reports_stalled_when_no_heartbeat(tmp_path, monkeypatch, capsys):
    import cron.jobs as jobs
    from hermes_cli import cron as cron_cli

    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [4321])
    monkeypatch.setattr(jobs, "get_ticker_heartbeat_age", lambda: 9_999.0)  # dead
    monkeypatch.setattr(jobs, "get_ticker_success_age", lambda: 9_999.0)
    monkeypatch.setattr("cron.jobs.list_jobs", lambda **k: [])

    cron_cli.cron_status()
    out = capsys.readouterr().out
    assert "STALLED" in out
    assert "will fire automatically" not in out


# ── F8: runtime backstop — never resolve a stored pair that exfiltrates a key ──


class TestGuardJobCredentialExfil:
    """run_job() must fail closed before provider resolution when a job's stored
    provider/base_url pair would ship a named provider's stored credential to an
    off-host endpoint — covering jobs persisted before the create/update guard
    or written directly to the store (F8 stored-job path; CWE-200/CWE-522)."""

    def test_named_registry_provider_offhost_is_blocked(self):
        import pytest
        from cron.scheduler import _guard_job_credential_exfil

        job = {"id": "j1", "provider": "anthropic",
               "base_url": "https://evil.example/v1"}
        with pytest.raises(RuntimeError) as exc:
            _guard_job_credential_exfil(job)
        assert "blocked for safety" in str(exc.value)

    def test_named_custom_offhost_is_blocked(self, monkeypatch):
        import pytest
        import hermes_cli.runtime_provider as rp
        from cron.scheduler import _guard_job_credential_exfil

        monkeypatch.setattr(rp, "has_named_custom_provider", lambda n: True)
        monkeypatch.setattr(
            rp, "_get_named_custom_provider",
            lambda n: {"name": "legit", "base_url": "https://legit.example/v1",
                       "api_key": "sk-legit"},
        )
        job = {"id": "j2", "provider": "custom:legit",
               "base_url": "https://evil.example/v1"}
        with pytest.raises(RuntimeError):
            _guard_job_credential_exfil(job)

    def test_named_custom_matching_host_is_allowed(self, monkeypatch):
        import hermes_cli.runtime_provider as rp
        from cron.scheduler import _guard_job_credential_exfil

        monkeypatch.setattr(rp, "has_named_custom_provider", lambda n: True)
        monkeypatch.setattr(
            rp, "_get_named_custom_provider",
            lambda n: {"name": "legit", "base_url": "https://legit.example/v1",
                       "api_key": "sk-legit"},
        )
        job = {"id": "j3", "provider": "custom:legit",
               "base_url": "https://legit.example/v1"}
        assert _guard_job_credential_exfil(job) is None

    def test_bare_custom_is_allowed(self):
        from cron.scheduler import _guard_job_credential_exfil

        job = {"id": "j4", "provider": "custom",
               "base_url": "https://anything.example/v1"}
        assert _guard_job_credential_exfil(job) is None

    def test_no_base_url_is_allowed(self):
        from cron.scheduler import _guard_job_credential_exfil

        assert _guard_job_credential_exfil({"id": "j5", "provider": "anthropic"}) is None
        assert _guard_job_credential_exfil({"id": "j6"}) is None
