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


def test_abc_growth_stays_additive():
    """The provider interface stays source-compatible with existing plugins.

    ``start`` must be the only required implementation hook: future optional
    behavior belongs in non-abstract default methods so custom plugins do not
    break on import after an upgrade.
    """
    from cron.scheduler_provider import CronScheduler

    abstract = set(getattr(CronScheduler, "__abstractmethods__", set()))
    assert abstract == {"name", "start"}, (
        f"CronScheduler abstractmethods changed to {abstract}; growth must be "
        "additive (optional methods with defaults), not new abstract methods."
    )


def test_force_fire_capability_detects_legacy_override():
    from cron.scheduler_provider import CronScheduler

    class Current(CronScheduler):
        @property
        def name(self):
            return "current"

        def start(self, stop_event, **kw):
            pass

    class Legacy(Current):
        def fire_due(  # type: ignore[invalid-method-override]
            self, job_id, *, adapters=None, loop=None
        ):
            return True

    class PositionalOnly(Current):
        def fire_due(  # type: ignore[invalid-method-override]
            self, job_id, force=False, /
        ):
            return True

    class KeywordSink(Current):
        def fire_due(self, job_id, **kwargs):
            return True

    assert Current().supports_force_fire is True
    assert Legacy().supports_force_fire is False
    assert PositionalOnly().supports_force_fire is False
    assert KeywordSink().supports_force_fire is True


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


def test_external_provider_falls_back_to_builtin_under_multiplex():
    from cron.scheduler_provider import (
        CronScheduler,
        InProcessCronScheduler,
        scheduler_for_profile_mode,
    )

    class External(CronScheduler):
        @property
        def name(self):
            return "external"

        def start(self, stop_event, **kwargs):
            return None

    external = External()

    assert scheduler_for_profile_mode(external, multiplex_profiles=False) is external
    assert isinstance(
        scheduler_for_profile_mode(external, multiplex_profiles=True),
        InProcessCronScheduler,
    )


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
    """The default fire_due runs the exact owner-bearing CAS snapshot."""
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.scheduler_provider import InProcessCronScheduler

    ran = []
    claims = []
    monkeypatch.setattr(
        jobs,
        "claim_job_for_fire",
        lambda jid, **kw: claims.append((jid, kw))
        or {"id": jid, "name": "t", "fire_claim": {"by": "exact-owner"}},
        raising=False,
    )
    monkeypatch.setattr(
        sched,
        "run_one_job",
        lambda job, **kw: ran.append((job["id"], job["fire_claim"]["by"])) or True,
    )

    assert InProcessCronScheduler().fire_due("j1") is True
    assert claims == [("j1", {"return_job": True})]
    assert ran == [("j1", "exact-owner")]


def test_claim_fire_persists_attempt_before_fire_claimed(monkeypatch):
    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.scheduler_provider import InProcessCronScheduler

    events = []
    monkeypatch.setattr(
        jobs,
        "claim_job_for_fire",
        lambda jid, **kwargs: events.append("claim")
        or {"id": jid, "fire_claim": {"by": "owner"}},
    )
    monkeypatch.setattr(
        executions,
        "create_execution",
        lambda jid, source: events.append("ledger") or {"id": "exec-1"},
    )
    monkeypatch.setattr(
        sched,
        "run_one_job",
        lambda job, **kwargs: events.append(("run", job["execution_id"])) or True,
    )

    provider = InProcessCronScheduler()
    claimed = provider.claim_fire("j1")

    assert events == ["ledger", "claim"]
    assert claimed is not None
    assert claimed["execution_id"] == "exec-1"
    assert provider.fire_claimed(claimed) is True
    assert events == ["ledger", "claim", ("run", "exec-1")]


def test_fire_due_forwards_manual_force_to_store_claim(monkeypatch):
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.scheduler_provider import InProcessCronScheduler

    claims = []
    monkeypatch.setattr(
        jobs,
        "claim_job_for_fire",
        lambda jid, **kw: claims.append((jid, kw))
        or {"id": jid, "name": "t", "fire_claim": {"by": "manual-owner"}},
    )
    monkeypatch.setattr(sched, "run_one_job", lambda job, **kw: True)

    assert InProcessCronScheduler().fire_due("j1", force=True) is True
    assert claims == [("j1", {"force": True, "return_job": True})]


def test_fire_due_lost_claim_does_not_run(monkeypatch):
    """If the CAS claim is lost (another machine/retry won), fire_due returns
    False and never runs the job."""
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.scheduler_provider import InProcessCronScheduler

    ran = []
    monkeypatch.setattr(
        jobs,
        "claim_job_for_fire",
        lambda jid, **kw: False,
        raising=False,
    )
    monkeypatch.setattr(sched, "run_one_job", lambda job, **kw: ran.append(job["id"]) or True)

    assert InProcessCronScheduler().fire_due("j1") is False
    assert ran == []


def test_fire_due_missing_job_does_not_run(monkeypatch):
    """If the job vanished before atomic claim, fire_due does not run it."""
    import cron.jobs as jobs
    import cron.scheduler as sched
    from cron.scheduler_provider import InProcessCronScheduler

    ran = []
    monkeypatch.setattr(
        jobs,
        "claim_job_for_fire",
        lambda jid, **kw: False,
        raising=False,
    )
    monkeypatch.setattr(sched, "run_one_job", lambda job, **kw: ran.append(job["id"]) or True)

    assert InProcessCronScheduler().fire_due("gone") is False
    assert ran == []


# ── F2a: ticker liveness — survival, heartbeat, honest status (#32612, #32895) ──


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


    def test_bare_custom_is_allowed(self):
        from cron.scheduler import _guard_job_credential_exfil

        job = {"id": "j4", "provider": "custom",
               "base_url": "https://anything.example/v1"}
        assert _guard_job_credential_exfil(job) is None

    def test_no_base_url_is_allowed(self):
        from cron.scheduler import _guard_job_credential_exfil

        assert _guard_job_credential_exfil({"id": "j5", "provider": "anthropic"}) is None
        assert _guard_job_credential_exfil({"id": "j6"}) is None

    def test_validator_exception_with_base_url_fails_closed(self, monkeypatch):
        # If the validator/import unexpectedly raises, this last-resort backstop
        # must NOT allow a base_url-bearing job through to provider resolution
        # (it cannot prove the stored pair is safe). Regression for the
        # fail-open `except Exception: err = None` path.
        import pytest
        import tools.cronjob_tools as ct
        from cron.scheduler import _guard_job_credential_exfil

        def _boom(provider, base_url):
            raise RuntimeError("validator blew up")

        monkeypatch.setattr(ct, "_validate_cron_base_url", _boom)
        job = {"id": "j7", "provider": "custom:legit",
               "base_url": "https://evil.example/v1"}
        with pytest.raises(RuntimeError) as exc:
            _guard_job_credential_exfil(job)
        assert "blocked for safety" in str(exc.value)


# ── Multiplex profiles: cron per secondary profile (issue #69377) ─────────


def test_multiplex_ticker_ticks_each_profile_once(tmp_path, monkeypatch):
    """The multiplex cron scheduler calls tick() once per profile home,
    scoped via use_cron_store, so secondary-profile jobs actually fire
    instead of languishing in an unticked store."""
    from cron.scheduler_provider import InProcessCronScheduler

    # Set up two profile directories.
    p1 = tmp_path / "default"
    p2 = tmp_path / "home-ops"
    for d in (p1, p2):
        (d / "cron").mkdir(parents=True)

    profile_homes = [("default", p1), ("home-ops", p2)]

    # Count tick() calls — should be called once per profile per iteration.
    tick_count: list[int] = []

    def _tracking_tick(*args, **kwargs):
        tick_count.append(1)
        return 0

    stop = threading.Event()
    prov = InProcessCronScheduler()

    with patch("cron.scheduler.tick", side_effect=_tracking_tick), \
         patch("cron.jobs.record_ticker_heartbeat", lambda **kw: None):
        t = threading.Thread(
            target=prov.start,
            args=(stop,),
            kwargs={"interval": 0, "profile_homes": profile_homes},
            daemon=True,
        )
        t.start()
        # Wait for at least len(profile_homes) tick calls (one full cycle).
        deadline = time.monotonic() + 10
        while len(tick_count) < len(profile_homes) and time.monotonic() < deadline:
            time.sleep(0.005)
        # Give one more cycle to ensure it keeps ticking.
        deadline = time.monotonic() + 3
        while len(tick_count) < len(profile_homes) * 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        stop.set()
        t.join(timeout=5)

    assert not t.is_alive()
    # The ticker called tick() at least once per profile per iteration.
    # With 2 profiles and multiple iterations, we should have seen at least 2 calls.
    assert len(tick_count) >= len(profile_homes), \
        f"Expected >= {len(profile_homes)} tick calls, got {len(tick_count)}"


def test_multiplex_ticker_skips_deleted_profile_from_startup_snapshot(tmp_path):
    """A stale profile_homes entry must not recreate a deleted profile."""
    import cron.jobs as jobs
    from cron.scheduler_provider import InProcessCronScheduler

    default_home = tmp_path / "default"
    (default_home / "cron").mkdir(parents=True)
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    deleted_home = profiles_dir / "deleted"
    profile_homes = [("default", default_home), ("deleted", deleted_home)]

    ticked_homes = []
    stop = threading.Event()

    def _tracking_tick(*args, **kwargs):
        ticked_homes.append(jobs._current_cron_store().cron_dir.parent)
        stop.set()
        return 0

    provider = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=_tracking_tick):
        thread = threading.Thread(
            target=provider.start,
            args=(stop,),
            kwargs={"interval": 0, "profile_homes": profile_homes},
            daemon=True,
        )
        thread.start()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert ticked_homes == [default_home.resolve()]
    assert not deleted_home.exists()


def test_existing_profile_homes_filters_deleted(tmp_path):
    """The existence filter keeps live homes and drops deleted ones, whether
    entries are (name, path) tuples or bare paths."""
    from cron.scheduler_provider import _existing_profile_homes

    live = tmp_path / "live"
    deleted = tmp_path / "deleted"
    live.mkdir(parents=True)
    # deleted intentionally not created

    as_tuples = _existing_profile_homes([("live", live), ("deleted", deleted)])
    assert [p[0] for p in as_tuples] == ["live"]

    as_paths = _existing_profile_homes([live, deleted])
    assert [p for p in as_paths] == [live]


def _run_multiplex_capture(tmp_path, *, profile_adapters, shared_adapters):
    """Run the multiplex ticker one full cycle and return the ``adapters``
    object passed to ``tick()`` for the default profile and the secondary
    profile (in ``profile_homes`` order: default first, secondary second)."""
    from cron.scheduler_provider import InProcessCronScheduler

    p_default = tmp_path / "default"
    p_sec = tmp_path / "home-ops"
    for d in (p_default, p_sec):
        (d / "cron").mkdir(parents=True)
    profile_homes = [("default", p_default), ("home-ops", p_sec)]

    captured: list = []  # adapters seen per tick call; order follows profile_homes

    def _capturing_tick(*args, **kwargs):
        captured.append(kwargs.get("adapters"))
        return 0

    stop = threading.Event()
    prov = InProcessCronScheduler()
    with patch("cron.scheduler.tick", side_effect=_capturing_tick), \
         patch("cron.jobs.record_ticker_heartbeat", lambda **kw: None):
        t = threading.Thread(
            target=prov.start,
            args=(stop,),
            kwargs={
                "interval": 0,
                "profile_homes": profile_homes,
                "adapters": shared_adapters,
                "profile_adapters": profile_adapters,
                "default_profile": "default",
            },
            daemon=True,
        )
        t.start()
        deadline = time.monotonic() + 10
        while len(captured) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        stop.set()
        t.join(timeout=5)

    assert not t.is_alive()
    assert len(captured) >= 2, f"expected >= 2 tick calls, got {len(captured)}"
    return captured[0], captured[1]  # (default, secondary) of the first cycle


def test_multiplex_default_profile_uses_shared_adapters(tmp_path):
    """The default profile's cron is delivered via the shared ``adapters`` set
    (which belongs to the default profile)."""
    shared = {"kind": "shared"}
    default_ad, _ = _run_multiplex_capture(
        tmp_path, profile_adapters={"home-ops": {"kind": "secondary"}},
        shared_adapters=shared,
    )
    assert default_ad is shared


def test_multiplex_connected_secondary_uses_its_own_adapters(tmp_path):
    """A connected secondary is delivered via ITS OWN adapters, not the shared
    default-profile set."""
    shared = {"kind": "shared"}
    sec = {"kind": "secondary"}
    default_ad, sec_ad = _run_multiplex_capture(
        tmp_path, profile_adapters={"home-ops": sec}, shared_adapters=shared,
    )
    assert default_ad is shared
    assert sec_ad is sec


def test_multiplex_empty_secondary_does_not_fall_back_to_shared(tmp_path):
    """A secondary whose adapter map is present-but-empty (its bot has not
    connected yet) must NOT fall back to the default profile's shared adapters
    — otherwise its cron output ships through the wrong bot. It receives an
    empty adapter set and simply does not deliver this tick."""
    shared = {"kind": "shared"}
    default_ad, sec_ad = _run_multiplex_capture(
        tmp_path, profile_adapters={"home-ops": {}}, shared_adapters=shared,
    )
    assert default_ad is shared
    assert sec_ad is not shared
    assert not sec_ad  # empty → no delivery, not the default bot


def test_multiplex_missing_secondary_does_not_fall_back_to_shared(tmp_path):
    """A secondary absent from profile_adapters entirely (its adapter map has
    not been created yet) also must not fall back to the shared adapters."""
    shared = {"kind": "shared"}
    default_ad, sec_ad = _run_multiplex_capture(
        tmp_path, profile_adapters={}, shared_adapters=shared,
    )
    assert default_ad is shared
    assert sec_ad is not shared
    assert not sec_ad
