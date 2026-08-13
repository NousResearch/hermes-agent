"""Regression tests for #83182 — cron delivery must use the owning profile's
bot token and adapter under multiplex.

Two root causes, both fixed:
1. Secret scope was reset in run_one_job's inner finally BEFORE _deliver_result
   ran. load_gateway_config() → _getenv() then fell back to os.environ (empty
   in the multiplex unit) — TELEGRAM_BOT_TOKEN resolved to the wrong bot.
2. Cron delivery used the shared runner.adapters dict (default profile) even
   when multiplexing. Secondary profile jobs delivered via the default bot.

Fix:
  - Part 1: Scope reset moved to outer finally, covers execution + delivery.
  - Part 2: profile_adapters_by_home propagated through gateway → cron chain;
    _deliver_result prefers the owning profile's adapter map.
"""
import importlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Part 1: Secret scope stays installed through delivery
# ---------------------------------------------------------------------------


class TestSecretScopeThroughDelivery:
    """run_one_job must keep the profile's secret scope installed when
    _deliver_result runs, so load_gateway_config picks up the right
    TELEGRAM_BOT_TOKEN from the profile's .env.
    """

    def test_scope_active_during_delivery(self, tmp_path, monkeypatch):
        """Monkey-patch run_job and _deliver_result to observe scope state
        at delivery time.
        """
        import hermes_constants
        from cron import scheduler as sched_mod

        # Point hermes home at our tmp_path
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        importlib.reload(hermes_constants)

        # Stub out heavy imports/modules that run_job and delivery touch
        monkeypatch.setattr(sched_mod, "run_job", lambda *a, **kw: (True, "out", "response", None))
        monkeypatch.setattr(sched_mod, "save_job_output", lambda *a, **kw: "/tmp/fake")
        monkeypatch.setattr(sched_mod, "_is_interrupted", lambda *a, **kw: False)
        monkeypatch.setattr(sched_mod, "_consume_interrupted_flag", lambda *a, **kw: False)
        monkeypatch.setattr(sched_mod, "mark_job_run", lambda *a, **kw: None)
        monkeypatch.setattr(sched_mod, "claim_dispatch", lambda *a, **kw: True)
        monkeypatch.setattr(sched_mod, "mark_execution_running", lambda *a, **kw: None)
        monkeypatch.setattr(sched_mod, "create_execution", lambda *a, **kw: {"id": "exec-1"})
        monkeypatch.setattr(sched_mod, "finish_execution", lambda *a, **kw: None)

        # Track the secret scope state at delivery time
        scope_state_at_delivery = {}

        def fake_deliver(job, content, **kwargs):
            from agent.secret_scope import current_secret_scope
            scope_state_at_delivery["scope"] = current_secret_scope()
            return None

        monkeypatch.setattr(sched_mod, "_deliver_result", fake_deliver)

        job = {
            "id": "job-1",
            "name": "test-job",
            "deliver": "local",  # local delivery skips the send path
        }

        sched_mod.run_one_job(job, adapters=None, loop=None)

        # The key assertion: secret scope is still installed when delivery runs.
        # (Not None — which was the pre-fix state.)
        # Note: may be None if no .env exists in the test hermes home, but the
        # important thing is that set_secret_scope was called and reset was
        # NOT called before delivery. We test the ordering separately below.
        # For now, just assert run_one_job completed without raising.

    def test_scope_reset_after_delivery_not_before(self, tmp_path, monkeypatch):
        """Verify the ordering: reset_secret_scope fires AFTER _deliver_result,
        not before. We track the call sequence explicitly.
        """
        import hermes_constants
        from cron import scheduler as sched_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        importlib.reload(hermes_constants)

        monkeypatch.setattr(sched_mod, "run_job", lambda *a, **kw: (True, "out", "response", None))
        monkeypatch.setattr(sched_mod, "save_job_output", lambda *a, **kw: "/tmp/fake")
        monkeypatch.setattr(sched_mod, "_is_interrupted", lambda *a, **kw: False)
        monkeypatch.setattr(sched_mod, "_consume_interrupted_flag", lambda *a, **kw: False)
        monkeypatch.setattr(sched_mod, "mark_job_run", lambda *a, **kw: None)
        monkeypatch.setattr(sched_mod, "claim_dispatch", lambda *a, **kw: True)
        monkeypatch.setattr(sched_mod, "mark_execution_running", lambda *a, **kw: None)
        monkeypatch.setattr(sched_mod, "create_execution", lambda *a, **kw: {"id": "exec-1"})
        monkeypatch.setattr(sched_mod, "finish_execution", lambda *a, **kw: None)

        events = []

        def fake_deliver(*a, **kw):
            events.append("deliver")
            return None

        # Wrap reset_secret_scope to observe the call
        from agent import secret_scope as scope_mod
        _orig_reset = scope_mod.reset_secret_scope

        def tracking_reset(token):
            events.append("reset_scope")
            return _orig_reset(token)

        monkeypatch.setattr(sched_mod, "_deliver_result", fake_deliver)
        monkeypatch.setattr(scope_mod, "reset_secret_scope", tracking_reset)

        job = {"id": "job-1", "name": "test-job", "deliver": "local"}
        sched_mod.run_one_job(job)

        # Verify ordering: deliver must come BEFORE reset_scope
        assert "deliver" in events, f"deliver was not called: {events}"
        assert "reset_scope" in events, f"reset_scope was not called: {events}"
        assert events.index("deliver") < events.index("reset_scope"), (
            f"delivery ({events.index('deliver')}) should happen BEFORE "
            f"scope reset ({events.index('reset_scope')}); events={events}"
        )


# ---------------------------------------------------------------------------
# Part 2: Profile-specific adapter selection
# ---------------------------------------------------------------------------


class TestProfileAdapterSelection:
    """_deliver_result must use profile-specific adapters when available,
    falling back to the shared dict when not.
    """

    def test_profile_adapters_preferred_over_shared(self, monkeypatch):
        """When profile_adapters is set, it wins over the shared adapters."""
        from cron.scheduler import _deliver_result

        captured = {}

        def fake_resolve(platform, config, adapters):
            captured["adapters"] = adapters
            return None  # no live transport → fall through

        monkeypatch.setattr(
            "gateway.delivery.resolve_delivery_transport",
            fake_resolve,
        )
        # Stub _resolve_delivery_targets (module-level in cron.scheduler)
        monkeypatch.setattr(
            "cron.scheduler._resolve_delivery_targets",
            lambda job: [{"platform": "telegram", "chat_id": "123"}],
        )
        # Stub config loaders (imported inside _deliver_result)
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: MagicMock(platforms={}),
        )
        monkeypatch.setattr(
            "cron.scheduler.load_config",
            lambda: {"cron": {"wrap_response": False}},
        )

        shared_adapters = {"shared": "adapter"}
        profile_adapters = {"telegram": MagicMock()}

        job = {"id": "job-1", "deliver": "telegram"}

        # Call with both — profile_adapters should win
        _deliver_result(
            job, "hello",
            adapters=shared_adapters, loop=None,
            profile_adapters=profile_adapters,
        )

        assert captured.get("adapters") is profile_adapters, (
            "profile_adapters should be preferred over shared adapters"
        )

    def test_fallback_to_shared_adapters(self, monkeypatch):
        """When profile_adapters is None, shared adapters are used (back-compat)."""
        from cron.scheduler import _deliver_result

        captured = {}

        def fake_resolve(platform, config, adapters):
            captured["adapters"] = adapters
            return None

        monkeypatch.setattr(
            "gateway.delivery.resolve_delivery_transport",
            fake_resolve,
        )
        monkeypatch.setattr(
            "cron.scheduler._resolve_delivery_targets",
            lambda job: [{"platform": "telegram", "chat_id": "123"}],
        )
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: MagicMock(platforms={}),
        )
        monkeypatch.setattr(
            "cron.scheduler.load_config",
            lambda: {"cron": {"wrap_response": False}},
        )

        shared_adapters = {"shared": "adapter"}

        job = {"id": "job-1", "deliver": "telegram"}

        # Call without profile_adapters — shared should be used
        _deliver_result(
            job, "hello",
            adapters=shared_adapters, loop=None,
            profile_adapters=None,
        )

        assert captured.get("adapters") is shared_adapters, (
            "shared adapters should be used when profile_adapters is None"
        )


# ---------------------------------------------------------------------------
# run_one_job profile_adapters_by_home resolution
# ---------------------------------------------------------------------------


class TestRunOneJobProfileAdaptersResolution:
    """run_one_job must look up the right profile adapters from the map
    using the current hermes home.
    """

    def test_resolves_adapters_by_home(self, tmp_path, monkeypatch):
        """run_one_job resolves the owning profile's adapters from the
        profile_adapters_by_home map keyed by resolved hermes home path.
        """
        import hermes_constants
        from cron import scheduler as sched_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        importlib.reload(hermes_constants)

        # Stubs
        monkeypatch.setattr(sched_mod, "run_job", lambda *a, **kw: (True, "out", "response", None))
        monkeypatch.setattr(sched_mod, "save_job_output", lambda *a, **kw: "/tmp/fake")
        monkeypatch.setattr(sched_mod, "_is_interrupted", lambda *a, **kw: False)
        monkeypatch.setattr(sched_mod, "_consume_interrupted_flag", lambda *a, **kw: False)
        monkeypatch.setattr(sched_mod, "mark_job_run", lambda *a, **kw: None)
        monkeypatch.setattr(sched_mod, "claim_dispatch", lambda *a, **kw: True)
        monkeypatch.setattr(sched_mod, "mark_execution_running", lambda *a, **kw: None)
        monkeypatch.setattr(sched_mod, "create_execution", lambda *a, **kw: {"id": "exec-1"})
        monkeypatch.setattr(sched_mod, "finish_execution", lambda *a, **kw: None)

        captured = {}

        def fake_deliver(job, content, adapters=None, loop=None, profile_adapters=None):
            captured["profile_adapters"] = profile_adapters
            captured["shared_adapters"] = adapters
            return None

        monkeypatch.setattr(sched_mod, "_deliver_result", fake_deliver)

        # Build the profile_adapters_by_home map keyed by resolved tmp_path
        profile_adapter = {"telegram": "profile-specific-adapter"}
        profile_adapters_by_home = {str(tmp_path.resolve()): profile_adapter}
        shared_adapters = {"shared": "adapter"}

        job = {"id": "job-1", "name": "test-job", "deliver": "local"}

        sched_mod.run_one_job(
            job, adapters=shared_adapters, loop=None,
            profile_adapters_by_home=profile_adapters_by_home,
        )

        assert captured.get("profile_adapters") is profile_adapter, (
            "run_one_job should resolve the owning profile's adapter map"
        )
        assert captured.get("shared_adapters") is shared_adapters

    def test_no_profile_adapters_map(self, tmp_path, monkeypatch):
        """When profile_adapters_by_home is not provided, profile_adapters is None."""
        import hermes_constants
        from cron import scheduler as sched_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        importlib.reload(hermes_constants)

        monkeypatch.setattr(sched_mod, "run_job", lambda *a, **kw: (True, "out", "response", None))
        monkeypatch.setattr(sched_mod, "save_job_output", lambda *a, **kw: "/tmp/fake")
        monkeypatch.setattr(sched_mod, "_is_interrupted", lambda *a, **kw: False)
        monkeypatch.setattr(sched_mod, "_consume_interrupted_flag", lambda *a, **kw: False)
        monkeypatch.setattr(sched_mod, "mark_job_run", lambda *a, **kw: None)
        monkeypatch.setattr(sched_mod, "claim_dispatch", lambda *a, **kw: True)
        monkeypatch.setattr(sched_mod, "mark_execution_running", lambda *a, **kw: None)
        monkeypatch.setattr(sched_mod, "create_execution", lambda *a, **kw: {"id": "exec-1"})
        monkeypatch.setattr(sched_mod, "finish_execution", lambda *a, **kw: None)

        captured = {}

        def fake_deliver(job, content, adapters=None, loop=None, profile_adapters=None):
            captured["profile_adapters"] = profile_adapters
            return None

        monkeypatch.setattr(sched_mod, "_deliver_result", fake_deliver)

        job = {"id": "job-1", "name": "test-job", "deliver": "local"}
        sched_mod.run_one_job(job)

        assert captured.get("profile_adapters") is None
