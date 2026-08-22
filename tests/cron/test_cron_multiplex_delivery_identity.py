"""Regression tests for #83182 — cron delivery identity under multiplex.

Two contracts under multiplex_profiles:

1. **Scope held through delivery.** ``run_one_job`` installs the job-owning
   profile's secret scope and keeps it installed through the ENTIRE
   body — run_job, save, ``_deliver_result`` (live-adapter and standalone),
   and mark — resetting it only in the wrapper's finally. Before #83182 the
   scope was reset in a finally around ``run_job`` alone, so delivery read
   credentials from ``os.environ`` (another profile's bot token, or none).

2. **Per-profile adapter map.** The multiplex ticker ticks each profile with
   THAT profile's live adapters (``_profile_adapters[<name>]``), falling back
   to the shared map for the default profile or when no per-profile map is
   supplied. Before #83182 every profile tick delivered through the shared
   default-profile dict — wrong bot identity for every secondary profile.

The tests assert the behavior contracts (how the pieces relate), not
snapshots — they stub the pipeline and assert which credential/adapters each
stage observed.
"""
import threading

import pytest

import cron.scheduler as s
import cron.scheduler_provider as sp


def _patch_pipeline(monkeypatch, *, deliver_capture):
    """Patch the job pipeline; deliver_capture records what delivery sees."""

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        return (True, "out", "final response", None)

    def fake_save(jid, out):
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        # Capture the adapters delivery is offered AND the secret scope that
        # is active at delivery time — the two things #83182 is about.
        from agent.secret_scope import current_secret_scope

        deliver_capture.append(
            {
                "adapters": adapters,
                "scope": dict(current_secret_scope() or {}),
            }
        )
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None, **_kw):
        return True

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)


class TestScopeHeldThroughDelivery:
    """Contract 1: the profile secret scope is live at delivery time."""

    def test_scope_active_during_delivery(self, tmp_path, monkeypatch):
        from agent import secret_scope as ss

        # Build a profile home with a .env so the scope has content.
        profile_home = tmp_path / "profiles" / "alpha"
        profile_home.mkdir(parents=True)
        (profile_home / ".env").write_text("TELEGRAM_BOT_TOKEN=alpha-token\n")

        monkeypatch.setattr(
            "cron.scheduler._get_hermes_home", lambda: profile_home
        )
        # Neutralize store side effects that write into the real home.
        monkeypatch.setattr(s, "create_execution", lambda jid, **kw: {"id": "e1"})
        monkeypatch.setattr(s, "mark_execution_running", lambda eid: None)
        monkeypatch.setattr(s, "claim_dispatch", lambda jid: True)
        monkeypatch.setattr(s, "finish_execution", lambda eid, **kw: None)
        monkeypatch.setattr(s, "advance_next_runs", lambda ids: None)
        monkeypatch.setattr(
            s, "_consume_interrupted_flag", lambda jid, tok: False
        )

        capture = []
        _patch_pipeline(monkeypatch, deliver_capture=capture)

        job = {"id": "j-scope", "name": "scope-test", "deliver": "telegram:1"}
        tok = ss.set_secret_scope(None) if False else None
        prev = ss.current_secret_scope()
        assert prev is None  # hermetic: no scope leaking from the test env

        s.run_one_job(job)

        assert capture, "delivery must have run"
        assert capture[0]["scope"].get("TELEGRAM_BOT_TOKEN") == "alpha-token"

    def test_scope_reset_after_job_completes(self, tmp_path, monkeypatch):
        from agent import secret_scope as ss

        profile_home = tmp_path / "profiles" / "beta"
        profile_home.mkdir(parents=True)
        (profile_home / ".env").write_text("TELEGRAM_BOT_TOKEN=beta-token\n")

        monkeypatch.setattr(
            "cron.scheduler._get_hermes_home", lambda: profile_home
        )
        monkeypatch.setattr(s, "create_execution", lambda jid, **kw: {"id": "e1"})
        monkeypatch.setattr(s, "mark_execution_running", lambda eid: None)
        monkeypatch.setattr(s, "claim_dispatch", lambda jid: True)
        monkeypatch.setattr(s, "finish_execution", lambda eid, **kw: None)
        monkeypatch.setattr(s, "advance_next_runs", lambda ids: None)
        monkeypatch.setattr(
            s, "_consume_interrupted_flag", lambda jid, tok: False
        )

        capture = []
        _patch_pipeline(monkeypatch, deliver_capture=capture)

        job = {"id": "j-reset", "name": "reset-test", "deliver": "telegram:1"}
        assert ss.current_secret_scope() is None
        s.run_one_job(job)
        # After the wrapper returns, the scope is torn down again.
        assert ss.current_secret_scope() is None


class TestProfileTickAdapters:
    """Contract 2: the multiplex tick offers each profile its own adapters."""

    def test_secondary_profile_uses_own_adapters(self):
        shared = {"shared": object()}
        trader_map = {"trader-map": object()}
        profile_adapters = {"trader": trader_map}

        resolved = sp._profile_tick_adapters(
            "trader", shared, profile_adapters
        )
        assert resolved is trader_map

    def test_default_profile_uses_shared_adapters(self):
        shared = {"shared": object()}
        profile_adapters = {"trader": {"trader-map": object()}}

        resolved = sp._profile_tick_adapters(
            "default", shared, profile_adapters
        )
        bare = sp._profile_tick_adapters(None, shared, profile_adapters)
        assert resolved is shared
        assert bare is shared

    def test_unknown_profile_falls_back_to_shared(self):
        shared = {"shared": object()}
        profile_adapters = {"trader": {"trader-map": object()}}

        resolved = sp._profile_tick_adapters(
            "cric", shared, profile_adapters
        )
        assert resolved is shared

    def test_no_profile_adapters_supplied(self):
        shared = {"shared": object()}

        assert (
            sp._profile_tick_adapters("trader", shared, None) is shared
        )
        assert sp._profile_tick_adapters("trader", shared, {}) is shared


class TestMultiplexTickUsesProfileAdapters:
    """The ticker's per-profile loop passes the profile's own map to tick."""

    def test_tick_receives_profile_map(self, monkeypatch, tmp_path):
        calls = []
        stop = threading.Event()

        def fake_cron_tick(*, adapters=None, **kw):
            calls.append(adapters)
            # Stop after the first per-profile tick so the blocking loop exits.
            stop.set()

        # scheduler_provider reads cron.scheduler.tick lazily inside
        # _start_multiplex — patch at the source module.
        monkeypatch.setattr(
            "cron.scheduler.tick", fake_cron_tick, raising=True
        )

        # Neutralize recovery/heartbeat store side effects.
        monkeypatch.setattr(
            sp.InProcessCronScheduler, "recover_interrupted", lambda self: 0
        )
        monkeypatch.setattr(
            "cron.jobs.record_ticker_heartbeat", lambda **kw: None
        )
        monkeypatch.setattr("cron.jobs.clear_ticker_error", lambda: None)

        provider = sp.InProcessCronScheduler()
        shared = {"shared": object()}
        trader_map = {"trader-map": object()}

        homes = [("trader", tmp_path / "profiles" / "trader")]
        (tmp_path / "profiles" / "trader").mkdir(parents=True)
        provider._start_multiplex(
            stop,
            profile_homes=homes,
            adapters=shared,
            loop=None,
            interval=60,
            can_dispatch=None,
            profile_adapters={"trader": trader_map},
        )

        assert trader_map in calls
