"""Selected-PostgreSQL tick gate — config-only refusal of state-mutating ticks.

A profile with ``state_store.backend=postgresql`` lacks the cron transcript
lifecycle (cron-session-transcript-lifecycle), so the built-in ticker must not
run its state-mutating steps: no bot-chat queue flush, no stale-owner reap,
no ``advance_next_runs`` (jobs.json stays byte-identical), no execution
ledger rows. The gate is a pure config read — it must never open
``state.db`` or connect to PostgreSQL; per-job refusals elsewhere
(``run_one_job``, claim paths, startup recovery) still cover manual runs,
CLI edits, and recovery, and remain unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

import cron.scheduler as scheduler_mod
from cron import scheduler_tick as scheduler_tick_mod


DSN_ENV = "HERMES_STATE_STORE_TEST_DSN"


def _force_no_skew(monkeypatch):
    """Disable the stale-code yield gate so tests reach the lock deterministically."""
    monkeypatch.setattr(scheduler_mod, "_should_yield_tick_to_fresh_gateway", lambda: None)


def _seed_due_job(home: Path) -> Path:
    """Write a jobs.json whose only job is due now; return its path."""
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = cron_dir / "jobs.json"
    job = {
        "id": "gate-job-1",
        "name": "pg gate probe",
        "prompt": "should never run under selected PG",
        "schedule": "5m",
        "next_run_at": "2020-01-01T00:00:00",
        "enabled": True,
    }
    jobs_path.write_text(json.dumps([job], indent=2), encoding="utf-8")
    return jobs_path


class TestSelectedPostgresqlTickGate:
    def test_selected_pg_skips_state_mutating_tick(self, monkeypatch, tmp_path, caplog):
        """backend=postgresql + configured DSN env → tick returns 0 without
        drain / advance / execution rows, and jobs.json is byte-identical."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv(DSN_ENV, "postgresql://fixture/gate-no-server-needed")
        _force_no_skew(monkeypatch)
        jobs_path = _seed_due_job(tmp_path)
        before_bytes = jobs_path.read_bytes()
        before_stat = jobs_path.stat()

        import hermes_cli.config as cli_config_mod
        import state_store as state_store_mod

        monkeypatch.setattr(
            cli_config_mod,
            "load_config",
            lambda: {
                "state_store": {
                    "backend": "postgresql",
                    "postgresql": {"dsn_env": DSN_ENV},
                }
            },
        )
        # Bind the secret to plain os.environ (no profile scope in the test).
        monkeypatch.setattr(
            state_store_mod, "_scoped_secret", lambda name: os.environ.get(name)
        )

        from cron.bot_chat_delivery import drain as _real_drain
        from cron.bot_chat_delivery import drain_in_background as _real_bg

        def _no_drain(*_a, **_kw):
            raise AssertionError("bot-chat queue flush must not run under selected PG")

        monkeypatch.setattr("cron.bot_chat_delivery.drain", _no_drain)
        monkeypatch.setattr("cron.bot_chat_delivery.drain_in_background", _no_drain)

        def _no_advance(*_a, **_kw):
            raise AssertionError("advance_next_runs must not run under selected PG")

        monkeypatch.setattr(scheduler_mod, "advance_next_runs", _no_advance)

        def _no_exec_rows(*_a, **_kw):
            raise AssertionError("execution ledger rows must not be created under selected PG")

        monkeypatch.setattr(scheduler_mod, "create_execution", _no_exec_rows)

        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            rc = scheduler_tick_mod.tick(verbose=False)

        assert rc == 0
        warnings = [
            r for r in caplog.records if "cron tick skipped: selected PostgreSQL" in r.getMessage()
        ]
        assert warnings, "the skip must log a warning naming the missing lifecycle"

        # The gate is config-only: never touched state.db, never connected to PG.
        assert not (tmp_path / "state.db").exists()

        # jobs.json untouched — content AND mtime (no load+save round trip).
        after_stat = jobs_path.stat()
        assert jobs_path.read_bytes() == before_bytes
        assert after_stat.st_mtime_ns == before_stat.st_mtime_ns

        # No execution rows materialized anywhere in the sandbox home.
        ledger_candidates = list(tmp_path.rglob("executions*.json*"))
        assert ledger_candidates == []

    def test_selected_pg_gate_fails_open_on_config_error(self, monkeypatch, tmp_path):
        """A malformed state_store block raises StateStoreConfigurationError;
        the gate must fail open to the existing tick behavior (no skip log,
        tick proceeds). Prove 'proceeds' by reaching get_due_jobs with no jobs."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _force_no_skew(monkeypatch)
        jobs_path = _seed_due_job(tmp_path)

        import hermes_cli.config as cli_config_mod

        monkeypatch.setattr(
            cli_config_mod,
            "load_config",
            lambda: {"state_store": {"backend": "not-a-backend"}},
        )

        reached = {"due": 0}

        def _spy_due_jobs():
            reached["due"] += 1
            return []

        monkeypatch.setattr(scheduler_mod, "get_due_jobs", _spy_due_jobs)

        from cron.bot_chat_delivery import drain as _drain_mod_drain

        monkeypatch.setattr("cron.bot_chat_delivery.drain", lambda: None)

        rc = scheduler_tick_mod.tick(verbose=False)
        assert rc == 0
        assert reached["due"] == 1, "gate must fail open: tick proceeds past the gate"

    def test_default_sqlite_tick_proceeds(self, monkeypatch, tmp_path):
        """Default sqlite backend: regression — the tick runs its normal
        sequence (drain → get_due_jobs) exactly as before the gate existed."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _force_no_skew(monkeypatch)
        _seed_due_job(tmp_path)

        import hermes_cli.config as cli_config_mod

        monkeypatch.setattr(cli_config_mod, "load_config", lambda: {})

        order: list[str] = []
        monkeypatch.setattr(
            "cron.bot_chat_delivery.drain",
            lambda: order.append("drain"),
        )
        monkeypatch.setattr(
            "cron.bot_chat_delivery.drain_in_background",
            lambda: order.append("drain_bg"),
        )
        monkeypatch.setattr(scheduler_mod, "get_due_jobs", lambda: [])
        monkeypatch.setattr(scheduler_mod, "advance_next_runs", lambda ids: order.append("advance"))
        monkeypatch.setattr(scheduler_mod, "_maybe_reap_dead_owners", lambda: order.append("reap"))
        monkeypatch.setattr(
            scheduler_mod, "_sweep_stale_inflight_for_tick", lambda due: order.append("sweep_inflight")
        )
        monkeypatch.setattr(scheduler_mod, "_sweep_mcp_orphans", lambda: order.append("mcp_sweep"))

        rc = scheduler_tick_mod.tick(verbose=False, sync=True)

        assert rc == 0
        assert order[:2] == ["drain", "reap"], "sqlite tick must keep the pre-gate step order"
        assert "drain" in order
        # Idle tick: advance is never called (no due jobs) but drain DID run —
        # the gate did not short-circuit the tick.


class TestGateIsPureConfig:
    def test_gate_never_opens_state_db_or_connects_postgresql(self, monkeypatch, tmp_path):
        """Even with a full due job present, the selected-PG tick must not
        import/connect any state-store runtime: patch the readiness/runtime
        entry modules to explode if the gate reaches for them."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv(DSN_ENV, "postgresql://fixture/gate-no-server-needed")
        _force_no_skew(monkeypatch)
        _seed_due_job(tmp_path)

        import hermes_cli.config as cli_config_mod
        import state_store as state_store_mod

        monkeypatch.setattr(
            cli_config_mod,
            "load_config",
            lambda: {"state_store": {"backend": "postgresql", "postgresql": {"dsn_env": DSN_ENV}}},
        )
        monkeypatch.setattr(state_store_mod, "_scoped_secret", lambda name: os.environ.get(name))

        import builtins

        real_import = builtins.__import__

        def _guard_import(name, *a, **kw):
            if name.startswith("psycopg") or name == "state_store_postgresql":
                raise AssertionError(f"config-only gate must not import {name}")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _guard_import)

        rc = scheduler_tick_mod.tick(verbose=False)
        assert rc == 0
        assert not (tmp_path / "state.db").exists()
