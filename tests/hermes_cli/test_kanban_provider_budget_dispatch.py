"""Dispatch-integration tests for the Kanban per-provider concurrency budget
(#123654): the gate in ``_dispatch_lane_task``, per-claim ``provider_key``
recording, host-wide counting, composition with the other caps, and the
deferral surfaces (dispatch output, describe_suppression, diagnostics).

T10–T30 per the approved plan; T23 is skipped (PR #117755 unmerged at build
time — gateway reads provider_concurrency at boot, which T27 covers).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def budget_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fresh HERMES_HOME with profiles alpha/beta/gamma (each pinned to a
    provider via config.yaml) + default, and an initialized kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    for prof, provider in (("alpha", "anthropic"), ("beta", "anthropic"),
                           ("gamma", "anthropic"), ("default", "openrouter")):
        pdir = home / "profiles" / prof
        pdir.mkdir(parents=True)
        (pdir / "config.yaml").write_text(
            f"model:\n  default: m-{prof}\n  provider: {provider}\n"
            if prof != "default" else "model:\n  default: m-default\n"
        )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_connect as kbc

    kanban_db.create_board(slug="default", name="Test")
    return home


def _fake_spawn(*args, **kwargs):
    return 12345


def _make_rows(kb, conn, rows):
    ids = []
    for title, assignee, model, provider in rows:
        ids.append(kb.create_task(
            conn, title=title, assignee=assignee,
            model_override=model, provider_override=provider,
        ))
    return ids


class TestDisabledAndRecordAlways:
    """T10 / T15c — disabled budget: identical dispatch decisions, but the key
    is still recorded on every dispatcher claim (O3 record-always)."""

    def test_disabled_budget_records_provider_key(self, budget_home):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [("a1", "alpha", None, None), ("a2", "alpha", None, None)])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency=None)
        # Both spawn (nothing gates when disabled — a budget that is off must
        # not hold anything back) and both keys are recorded (O3 record-always).
        assert len(res.spawned) == 2
        assert res.skipped_provider_budget == []
        with kbc.connect_closing() as conn:
            keys = [r["provider_key"] for r in conn.execute(
                "SELECT r.provider_key FROM tasks t JOIN task_runs r "
                "ON r.id = t.current_run_id WHERE t.status = 'running' "
                "ORDER BY t.id").fetchall()]
        assert keys == ["anthropic", "anthropic"]

    def test_empty_mapping_is_disabled(self, budget_home):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [("a1", "alpha", None, None)])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={})
        assert len(res.spawned) == 1
        assert res.skipped_provider_budget == []

    def test_resolver_error_stores_null_spawn_proceeds(self, budget_home, monkeypatch):
        """T15c: a resolution error writes NULL and never blocks the spawn."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd
        from hermes_cli import kanban_provider_budget as kpb

        def _boom(*a, **kw):
            raise RuntimeError("resolution failed")

        monkeypatch.setattr(kpb, "route_key", _boom)
        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [("a1", "alpha", None, None)])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert len(res.spawned) == 1
        with kbc.connect_closing() as conn:
            row = conn.execute(
                "SELECT r.provider_key FROM tasks t JOIN task_runs r "
                "ON r.id = t.current_run_id WHERE t.id = ?",
                (res.spawned[0][0],),
            ).fetchone()
        # resolve_requested_route succeeded, route_key raised -> the resolver
        # returns None (O3: a resolution error stores NULL, never blocks the
        # spawn). NULL on the run row — NOT the "unknown" bucket.
        assert row["provider_key"] is None


class TestUnderAndOverProtection:
    """T11 — the issue's first case: many profiles on ONE provider."""

    def test_dispatch_resolver_is_profile_scoped(self, budget_home, monkeypatch):
        """Q-F1 at the real tick: the resolver the TICK builds resolves inside
        the assignee's profile scope. A launch-profile alias with the same name
        and a launch OPENAI_BASE_URL must not leak into the claim's key."""
        from hermes_cli import config as _cfgmod

        # Launch (default) profile: alias 'fast' -> launch URL, plus a proxy env.
        (budget_home / "config.yaml").write_text(
            "model:\n  default: m-default\n"
            "model_aliases:\n  fast:\n    model: launch-model\n"
            "    provider: custom\n"
            "    base_url: https://launch-alias.example/v1\n")
        monkeypatch.setattr(_cfgmod, "_CONFIG_CACHE", {}, raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://launch-proxy.example/v1")
        # beta pins provider openai with its own .env proxy URL.
        beta = budget_home / "profiles" / "beta"
        beta.mkdir(parents=True, exist_ok=True)
        (beta / "config.yaml").write_text("model:\n  default: m-beta\n  provider: openai\n")
        (beta / ".env").write_text("OPENAI_BASE_URL=https://beta-proxy.example/v1\n")
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [("b1", "beta", None, None)])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert len(res.spawned) == 1
        with kbc.connect_closing() as conn:
            key = conn.execute(
                "SELECT r.provider_key FROM tasks t JOIN task_runs r "
                "ON r.id = t.current_run_id WHERE t.id = ?",
                (res.spawned[0][0],)).fetchone()["provider_key"]
        # beta's scoped .env proxy keyed the claim — not the launch env.
        assert key == "custom:https://beta-proxy.example/v1", key

    def test_shared_provider_budget_defers(self, budget_home):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [
                (f"{p}{i}", p, None, None)
                for i in range(2) for p in ("alpha", "beta", "gamma")
            ])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn,
                provider_concurrency={"anthropic": 2},
                max_in_progress_per_profile=5,
            )
        assert len(res.spawned) == 2
        assert len(res.skipped_provider_budget) == 4
        assert all(entry[1] == "anthropic" for entry in res.skipped_provider_budget)
        # current counts for later rows reflect in-tick consumption: 2/2
        currents = {entry[2] for entry in res.skipped_provider_budget}
        assert currents == {2}

    def test_one_provider_per_row_not_over_restricted(self, budget_home):
        """T12 — the issue's second case: distinct providers each budget 1."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [
                ("a-o", "alpha", "ma", "openrouter"),
                ("a-z", "alpha", "mz", "zai"),
                ("a-g", "alpha", "mg", "gemini"),
            ])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn,
                provider_concurrency={"openrouter": 1, "zai": 1, "gemini": 1},
                max_spawn=None, max_in_progress=10,
            )
        assert len(res.spawned) == 3
        assert res.skipped_provider_budget == []


class TestDeferSemantics:
    """T13 / T14 — defer, never kill."""

    def test_deferred_rows_stay_ready_no_side_effects(self, budget_home):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            ids = _make_rows(kb, conn, [("a1", "alpha", None, None), ("a2", "alpha", None, None)])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={"anthropic": 1})
        assert len(res.spawned) == 1
        deferred_id = res.skipped_provider_budget[0][0]
        with kbc.connect_closing() as conn:
            row = conn.execute(
                "SELECT status, claim_lock, current_run_id, consecutive_failures "
                "FROM tasks WHERE id = ?", (deferred_id,),
            ).fetchone()
            events = conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? "
                "AND kind NOT IN ('created')", (deferred_id,),
            ).fetchall()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
        assert row["current_run_id"] is None
        assert row["consecutive_failures"] == 0
        assert events == []

    def test_budget_lowered_never_kills_running(self, budget_home):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [(f"a{i}", "alpha", None, None) for i in range(3)])
        with kbc.connect_closing() as conn:
            res1 = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={"anthropic": 3})
        assert len(res1.spawned) == 3
        with kbc.connect_closing() as conn:
            res2 = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={"anthropic": 1})
        assert len(res2.spawned) == 0
        assert res2.skipped_provider_budget == []  # no ready rows left to defer
        with kbc.connect_closing() as conn:
            rows = conn.execute(
                "SELECT status, claim_lock FROM tasks WHERE status = 'running'"
            ).fetchall()
        assert len(rows) == 3  # the 3 running workers untouched


    def test_t15b_key_persisted_is_the_gated_key(self, budget_home, monkeypatch):
        """The persisted provider_key is the one that passed the gate, even
        when a later resolution would return a different value (M10)."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd
        from hermes_cli import kanban_provider_budget as kpb

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [("a1", "alpha", None, None)])

        calls = {"n": 0}
        real_resolve = kpb.RouteKeyResolver.resolve

        def _second_resolution_differs(self, assignee, model_override, provider_override):
            calls["n"] += 1
            if calls["n"] > 1:
                return "gemini"  # a mid-tick edit would resolve differently
            return real_resolve(self, assignee, model_override, provider_override)

        monkeypatch.setattr(kpb.RouteKeyResolver, "resolve", _second_resolution_differs)
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={"anthropic": 2})
        assert len(res.spawned) == 1
        with kbc.connect_closing() as conn:
            row = conn.execute(
                "SELECT r.provider_key FROM tasks t JOIN task_runs r "
                "ON r.id = t.current_run_id WHERE t.status = 'running'"
            ).fetchone()
        # The GATED key (first resolution, anthropic) is persisted — not the
        # differing second resolution.
        assert row["provider_key"] == "anthropic"


class TestPersistenceAndRestart:
    """T15 / T15b / T16 / T17 — counts survive restart; key frozen at spawn."""

    def test_restart_counts_from_run_rows(self, budget_home):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [(f"a{i}", "alpha", None, None) for i in range(3)])
        with kbc.connect_closing() as conn:
            res1 = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={"anthropic": 2})
        assert len(res1.spawned) == 2
        with kbc.connect_closing() as conn:
            for row in conn.execute(
                "SELECT r.provider_key FROM tasks t JOIN task_runs r "
                "ON r.id = t.current_run_id WHERE t.status = 'running'"
            ).fetchall():
                assert row["provider_key"] == "anthropic"
        # Simulate a restart: fresh process state (module caches of the
        # resolver are per-process; a new connect is the observable part).
        with kbc.connect_closing() as conn:
            res2 = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={"anthropic": 2})
        assert len(res2.spawned) == 0

    def test_key_frozen_at_spawn_beats_profile_edit(self, budget_home):
        """T16: editing alpha's config to openrouter after the spawn keeps the
        running rows counted under anthropic (persistence > re-derivation)."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd
        from hermes_cli import kanban_provider_budget as kpb

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [("a1", "alpha", None, None), ("a2", "alpha", None, None)])
        with kbc.connect_closing() as conn:
            res1 = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={"anthropic": 2})
        assert len(res1.spawned) == 2
        # Flip alpha's configured provider.
        cfg_path = budget_home / "profiles" / "alpha" / "config.yaml"
        cfg_path.write_text("model:\n  default: m-alpha\n  provider: openrouter\n")
        # Force re-derivation: clear every per-process memo.
        kpb._key_resolution_failures.clear()
        from hermes_cli.kanban_provider_budget import RouteKeyResolver
        with kbc.connect_closing() as conn:
            resolver = RouteKeyResolver(
                profile_exists=lambda a: True,
                profile_inputs=kbd._provider_route_inputs,
            )
            inferred: dict = {}
            counts = kpb.count_running_by_provider(conn, resolver, inferred=inferred)
        # Persisted keys win: still anthropic, and NOT re-derived.
        assert counts.get("anthropic") == 2
        assert inferred == {}

    def test_legacy_null_key_rederived_and_lanes_excluded(self, budget_home):
        """T17: NULL-key running row for a profile counts as inferred; a
        control-plane lane's running row is excluded."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd
        from hermes_cli import kanban_provider_budget as kpb
        from hermes_cli.kanban_provider_budget import RouteKeyResolver

        with kbc.connect_closing() as conn:
            tid1 = kb.create_task(conn, title="legacy", assignee="alpha")
            tid2 = kb.create_task(conn, title="lane", assignee="orion-cc")
            now = int(__import__("time").time())
            with kb.write_txn(conn):
                for tid, who in ((tid1, "alpha"), (tid2, "orion-cc")):
                    conn.execute(
                        "UPDATE tasks SET status = 'running', claim_lock = ?, "
                        "claim_expires = ?, started_at = ? WHERE id = ?",
                        (f"test-{who}", now + 600, now, tid),
                    )
        with kbc.connect_closing() as conn:
            resolver = RouteKeyResolver(
                profile_exists=lambda a: a in ("alpha", "beta", "gamma", "default"),
                profile_inputs=kbd._provider_route_inputs,
            )
            inferred: dict = {}
            counts = kpb.count_running_by_provider(
                conn, resolver,
                profile_exists=lambda a: a in ("alpha", "beta", "gamma", "default"),
                inferred=inferred)
        assert counts.get("anthropic") == 1
        assert inferred.get("anthropic") == 1
        # The control-plane lane is excluded entirely — not bucketed under its
        # assignee name, nor under "unknown" (its provider is unknowable).
        assert "orion-cc" not in counts
        assert counts.get("unknown", 0) == 0
        assert sum(counts.values()) == 1

    def test_rederivation_failure_not_counted_as_unknown(self, budget_home, monkeypatch):
        """B15b/T17: a legacy NULL row whose re-derivation FAILS (resolver
        returns None) is not counted anywhere — never bucketed 'unknown'."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd
        from hermes_cli import kanban_provider_budget as kpb

        with kbc.connect_closing() as conn:
            kb.create_task(conn, title="unresolvable", assignee="alpha")
            now = int(__import__("time").time())
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'running', claim_lock = ?, "
                    "claim_expires = ? WHERE id = (SELECT MAX(id) FROM tasks)",
                    ("z", now + 600))
        # Make every re-derivation fail loudly.
        def _boom(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(kpb, "route_key", _boom)
        from hermes_cli.kanban_provider_budget import RouteKeyResolver

        with kbc.connect_closing() as conn:
            resolver = RouteKeyResolver(
                profile_exists=lambda a: True,
                profile_inputs=kbd._provider_route_inputs,
                scope=kbd._assignee_route_scope,
            )
            inferred: dict = {}
            counts = kpb.count_running_by_provider(
                conn, resolver, profile_exists=lambda a: True, inferred=inferred)
        # The failed re-derivation produced no bucket at all — the row is
        # unknowable, not 'unknown'.
        assert counts.get("unknown", 0) == 0
        assert sum(counts.values()) == 0
        assert inferred == {}


class TestComposition:
    """T19 / T20 / T30 — gate order and composition with other caps."""

    def test_host_cap_short_circuits_provider_gate(self, budget_home):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [(f"a{i}", "alpha", None, None) for i in range(3)])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn,
                provider_concurrency={"anthropic": 1}, max_in_progress=0)
        # max_in_progress=0 -> tick returns before any gate; nothing spawned,
        # nothing recorded as provider-held.
        assert len(res.spawned) == 0
        assert res.skipped_provider_budget == []

    def test_per_profile_cap_wins_over_provider_list(self, budget_home):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [(f"a{i}", "alpha", None, None) for i in range(3)])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn,
                provider_concurrency={"anthropic": 1},
                max_in_progress_per_profile=1,
            )
        assert len(res.spawned) == 1
        # Rows at the per-profile cap land in skipped_per_profile_capped, NOT
        # the provider list.
        assert len(res.skipped_per_profile_capped) == 2
        assert res.skipped_provider_budget == []

    def test_respawn_guard_precedes_provider_gate(self, budget_home):
        """T30: a respawn-guarded row whose provider is ALSO at cap appears
        only under the guard, never in skipped_provider_budget."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            tid = kb.create_task(conn, title="a1", assignee="alpha")
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
                    ("HTTP 429 quota exceeded for provider", tid),
                )
            # anthropic at cap via a running row, so the provider gate would
            # also hold this task if it ran first.
            running = kb.create_task(conn, title="busy", assignee="beta")
            now = int(__import__("time").time())
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'running', claim_lock = ?, "
                    "claim_expires = ? WHERE id = ?", ("x", now + 600, running))
                conn.execute(
                    "INSERT INTO task_runs (task_id, profile, status, claim_lock, "
                    "claim_expires, started_at, provider_key) "
                    "VALUES (?, 'beta', 'running', 'x', ?, ?, 'anthropic')",
                    (running, now + 600, now))
                conn.execute(
                    "UPDATE tasks SET current_run_id = "
                    "(SELECT MAX(id) FROM task_runs WHERE task_id = ?) WHERE id = ?",
                    (running, running))
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={"anthropic": 1})
        assert len(res.spawned) == 0
        assert [r[0] for r in res.respawn_guarded] == [tid]
        assert res.skipped_provider_budget == []

    def test_review_mirror_releases_reservation(self, budget_home):
        """T20: a review row at provider budget + ready backlog on another
        provider -> the ready lane gets both slots."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            # Review row on anthropic (already at budget via a running task).
            rtid = kb.create_task(conn, title="rev", assignee="alpha")
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'review' WHERE id = ?", (rtid,))
            # A running anthropic row fills the budget.
            running = kb.create_task(conn, title="busy", assignee="beta")
            now = int(__import__("time").time())
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'running', claim_lock = ?, "
                    "claim_expires = ? WHERE id = ?", ("x", now + 600, running))
                conn.execute(
                    "INSERT INTO task_runs (task_id, profile, status, claim_lock, "
                    "claim_expires, started_at, provider_key) "
                    "VALUES (?, 'beta', 'running', 'x', ?, ?, 'anthropic')",
                    (running, now + 600, now))
                conn.execute(
                    "UPDATE tasks SET current_run_id = "
                    "(SELECT MAX(id) FROM task_runs WHERE task_id = ?) WHERE id = ?",
                    (running, running))
            # Ready backlog on a different provider, with slots for all of them
            # (max_spawn is a CONCURRENCY cap: running + spawns).
            _make_rows(kb, conn, [(f"o{i}", "default", None, None) for i in range(4)])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn,
                provider_concurrency={"anthropic": 1},
                max_spawn=5, max_in_progress=10,
            )
        # All four openrouter-ready rows spawned; the provider-held review row
        # did not consume the review reservation (a buggy mirror would have
        # held one ready slot back, spawning only 3).
        assert len(res.spawned) == 4
        assert res.skipped_provider_budget == []


class TestSummarySurfaces:
    """T21 / T22 / T28 — the deferral surfaces."""

    def test_describe_suppression_names_provider_budget(self, budget_home):
        from hermes_cli import kanban_db_dispatch as kbd

        res = kbd.DispatchResult()
        res.skipped_provider_budget = [("t1", "anthropic", 2, 2)]
        line = kbd.describe_suppression([res])
        assert "provider_budget[anthropic]=2/2" in line

    def test_cmd_dispatch_reports_deferrals(self, budget_home, monkeypatch, capsys):
        """T22: `hermes kanban dispatch` (dry-run) prints the deferral lines
        and --json carries skipped_provider_budget."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_ops
        import argparse

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [(f"a{i}", "alpha", None, None) for i in range(3)])
        # Point config at the budget via the home's config.yaml.
        (budget_home / "config.yaml").write_text(
            "kanban:\n  provider_concurrency:\n    anthropic: 2\n")
        # Clear the config mtime cache so the new file is seen.
        from hermes_cli import config as _cfgmod

        monkeypatch.setattr(_cfgmod, "_CONFIG_CACHE", {}, raising=False)
        args = argparse.Namespace(dry_run=True, json=False, max=None,
                                  failure_limit=None)
        rc = kanban_ops._cmd_dispatch(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "Deferred (provider budget anthropic 2/2)" in out

        # --json carries the same deferrals as skipped_provider_budget rows.
        args_json = argparse.Namespace(dry_run=True, json=True, max=None,
                                       failure_limit=None)
        rc = kanban_ops._cmd_dispatch(args_json)
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert len(payload["skipped_provider_budget"]) == 1
        entry = payload["skipped_provider_budget"][0]
        assert set(entry) == {"task_id", "provider", "current", "cap"}
        assert entry["provider"] == "anthropic"
        assert entry["current"] == 2 and entry["cap"] == 2

    def test_describe_suppression_takes_max_across_results(self, budget_home):
        """D9/R1 tests-F9: a board host-wide count is the MAX across the given
        tick results, not the min or the last."""
        from hermes_cli import kanban_db_dispatch as kbd

        r1 = kbd.DispatchResult()
        r1.skipped_provider_budget = [("t1", "anthropic", 2, 2)]
        r2 = kbd.DispatchResult()
        r2.skipped_provider_budget = [("t2", "anthropic", 5, 5), ("t3", "zai", 1, 1)]
        line = kbd.describe_suppression([r1, r2])
        assert "provider_budget[anthropic]=5/5" in line
        assert "provider_budget[zai]=1/1" in line
        assert "provider_budget[anthropic]=2/2" not in line

    def test_review_lane_gated_and_recorded_by_override_key(self, budget_home):
        """T28: a review row whose overrides name another provider is gated
        and recorded under the override's key."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            rtid = kb.create_task(
                conn, title="rev", assignee="alpha",
                model_override="m", provider_override="zai")
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (rtid,))
            # zai already at budget via a running row.
            running = kb.create_task(conn, title="busy", assignee="beta")
            now = int(__import__("time").time())
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'running', claim_lock = ?, "
                    "claim_expires = ? WHERE id = ?", ("x", now + 600, running))
                conn.execute(
                    "INSERT INTO task_runs (task_id, profile, status, claim_lock, "
                    "claim_expires, started_at, provider_key) "
                    "VALUES (?, 'beta', 'running', 'x', ?, ?, 'zai')",
                    (running, now + 600, now))
                conn.execute(
                    "UPDATE tasks SET current_run_id = "
                    "(SELECT MAX(id) FROM task_runs WHERE task_id = ?) WHERE id = ?",
                    (running, running))
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn,
                provider_concurrency={"zai": 1, "anthropic": 5},
            )
        assert len(res.spawned) == 0
        assert res.skipped_provider_budget == [(rtid, "zai", 1, 1)]
        # The review row stays review (never demoted).
        with kbc.connect_closing() as conn:
            assert conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (rtid,)
            ).fetchone()["status"] == "review"


class TestDiagnosticsSurface:
    """T24 — `hermes kanban diagnostics` prints the provider_concurrency line."""

    def test_diagnostics_off_and_on(self, budget_home, monkeypatch, capsys):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban as kanban_cli
        import argparse

        with kbc.connect_closing() as conn:
            tid = kb.create_task(conn, title="a1", assignee="alpha")
        args = argparse.Namespace(task=None, severity=None, json=False)
        rc = kanban_cli._cmd_diagnostics(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "kanban.provider_concurrency: off" in out

        # Enable: one running anthropic row, cap 1, one waiting ready row.
        (budget_home / "config.yaml").write_text(
            "kanban:\n  provider_concurrency:\n    anthropic: 1\n")
        from hermes_cli import config as _cfgmod

        monkeypatch.setattr(_cfgmod, "_CONFIG_CACHE", {}, raising=False)
        with kbc.connect_closing() as conn:
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
            kb.claim_task(conn, tid)  # running, NULL key (direct claim)
        # A second ready row on the same key -> at cap, "waiting".
        with kbc.connect_closing() as conn:
            wid = kb.create_task(conn, title="waiter", assignee="alpha")
        rc = kanban_cli._cmd_diagnostics(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "kanban.provider_concurrency:" in out
        assert "anthropic 1/1" in out
        assert "1 waiting" in out

        # --json: the trailing home-scope row carries the structured field
        # with running/inferred/cap/waiting per key (T24, R1 tests-F5a).
        args_json = argparse.Namespace(task=None, severity=None, json=True)
        rc = kanban_cli._cmd_diagnostics(args_json)
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload[-1]["task_id"] is None
        pc = payload[-1]["provider_concurrency"]
        assert pc["enabled"] is True
        assert pc["default"] is None
        assert set(pc["budgets"]) == {"anthropic"}
        anth = pc["budgets"]["anthropic"]
        assert anth["running"] == 1 and anth["cap"] == 1
        assert anth["waiting"] == 1
        assert anth["inferred"] == 1  # the NULL-key running row was re-derived

        # Addendum O1: when any counted key is 'auto', diagnostics says
        # unpinned profiles budget as 'auto' (R1 tests-F5b).
        (budget_home / "config.yaml").write_text(
            "kanban:\n  provider_concurrency:\n    anthropic: 1\n    auto: 2\n")
        monkeypatch.setattr(_cfgmod, "_CONFIG_CACHE", {}, raising=False)
        rc = kanban_cli._cmd_diagnostics(args)
        out = capsys.readouterr().out
        assert rc == 0
        assert "budget as 'auto'" in out

    def test_diagnostics_counts_sibling_boards(self, budget_home, monkeypatch, capsys):
        """T24 cross-board variant (R1 F4/Q-F4/A2): a running anthropic row on
        a sibling board counts in THIS board's diagnostics — the same
        host-wide path the dispatcher's gate uses, so the two agree."""
        import argparse
        import time as _time

        from hermes_cli import kanban as kanban_cli
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        (budget_home / "config.yaml").write_text(
            "kanban:\n  provider_concurrency:\n    anthropic: 2\n")
        from hermes_cli import config as _cfgmod

        monkeypatch.setattr(_cfgmod, "_CONFIG_CACHE", {}, raising=False)
        kb.create_board(slug="boardb", name="B")
        with kbc.connect_closing(board="boardb") as conn:
            bid = kb.create_task(conn, title="b-busy", assignee="alpha")
            now = int(_time.time())
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'running', claim_lock = ?, "
                    "claim_expires = ? WHERE id = ?", ("y", now + 600, bid))
        with kbc.connect_closing() as conn:
            tid = kb.create_task(conn, title="a1", assignee="alpha")
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        # Board A's gate holds the row (board B at 1 of cap 2? no: cap 2 -> not
        # held; use cap 1 to force the hold) — set cap 1 so the sibling row is
        # AT cap and diagnostics must show it.
        (budget_home / "config.yaml").write_text(
            "kanban:\n  provider_concurrency:\n    anthropic: 1\n")
        monkeypatch.setattr(_cfgmod, "_CONFIG_CACHE", {}, raising=False)
        args = argparse.Namespace(task=None, severity=None, json=True)
        rc = kanban_cli._cmd_diagnostics(args)
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        anth = payload[-1]["provider_concurrency"]["budgets"]["anthropic"]
        # Host-wide: the sibling's running row counts, and board A's ready row
        # is waiting at cap — the exact state the gate would hold.
        assert anth["running"] == 1
        assert anth["waiting"] == 1
        assert anth["cap"] == 1


class TestStuckWarningAndCrossBoard:
    """T18 / T29 — host-wide counting and the stuck-warning text."""

    def test_t18_cross_board_counts(self, budget_home):
        """A running anthropic worker on board B counts against board A's
        tick (host-wide); a pre-column sibling DB still counts."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd
        import time as _time

        # Board B with a running anthropic row and NO provider_key column
        # content (legacy shape: column exists post-migration, value NULL
        # -> re-derived via profile config).
        kb.create_board(slug="boardb", name="B")
        with kbc.connect_closing(board="boardb") as conn:
            bid = kb.create_task(conn, title="b-busy", assignee="alpha")
            now = int(_time.time())
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'running', claim_lock = ?, "
                    "claim_expires = ? WHERE id = ?", ("y", now + 600, bid))
        # Board A: one ready alpha row, budget 1.
        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [("a1", "alpha", None, None)])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={"anthropic": 1},
                board=None, max_spawn=5, max_in_progress=10,
            )
        # Board B's NULL-key running row was re-derived as anthropic and
        # held board A's row.
        assert len(res.spawned) == 0
        assert [(tid, key, cur, cap) for tid, key, cur, cap in res.skipped_provider_budget][0][1:] \
            == ("anthropic", 1, 1)

    def test_t18c_sibling_control_plane_lane_excluded(self, budget_home):
        """Q-F3/A3 probe P5: a running row on a sibling board whose assignee is
        NOT a Hermes profile (control-plane lane) is excluded from the sibling
        fold exactly as it would be on this board — never bucketed 'unknown'.
        The unknown-keyed READY row proves it: if the lane were counted, the
        ``unknown: 1`` cap would hold it."""
        import time as _time

        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        kb.create_board(slug="boardb", name="B")
        with kbc.connect_closing(board="boardb") as conn:
            bid = kb.create_task(conn, title="lane-busy", assignee="orion-cc")
            now = int(_time.time())
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'running', claim_lock = ?, "
                    "claim_expires = ? WHERE id = ?", ("y", now + 600, bid))
        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [
                ("a1", "alpha", None, None),
                ("u1", "alpha", "m", "no-such-provider"),
            ])
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn,
                provider_concurrency={"anthropic": 5, "unknown": 1},
                board=None, max_spawn=5, max_in_progress=10,
            )
        # The sibling control-plane lane did not consume the 'unknown' bucket:
        # the unknown-keyed ready row still spawns (exclusion, not 'unknown').
        spawned_titles = {t for t, _who, _ws in res.spawned}
        assert len(res.spawned) == 2, res.skipped_provider_budget
        assert res.skipped_provider_budget == []

    def test_t18b_sibling_corrupt_board_fail_open(self, budget_home, caplog):
        """A sibling board whose DB is unreadable logs one WARNING per board
        per process and is skipped (fail-open per board), never kills the tick."""
        import logging

        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        kb.create_board(slug="bad", name="Bad")
        kb.create_board(slug="good", name="Good")
        # Corrupt sibling: a non-SQLite file where its DB belongs.
        bad_path = kb.kanban_db_path(board="bad")
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.write_bytes(b"this is not a sqlite database at all")
        # Healthy sibling with a running anthropic row: still counted.
        import time as _time

        with kbc.connect_closing(board="good") as conn:
            gid = kb.create_task(conn, title="g-busy", assignee="beta")
            now = int(_time.time())
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'running', claim_lock = ?, "
                    "claim_expires = ? WHERE id = ?", ("z", now + 600, gid))
        # Board A: one ready alpha row, budget 1.
        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [("a1", "alpha", None, None)])
        with caplog.at_level(logging.WARNING):
            with kbc.connect_closing() as conn:
                res = kbd.dispatch_once(
                    conn, spawn_fn=_fake_spawn,
                    provider_concurrency={"anthropic": 1}, board=None,
                    max_spawn=5, max_in_progress=10)
        # Fail-open: the corrupt sibling was skipped, the healthy sibling
        # counted (anthropic at 1/1 via re-derivation), board A's row held.
        assert len(res.spawned) == 0
        assert res.skipped_provider_budget == [
            (res.skipped_provider_budget[0][0], "anthropic", 1, 1)]
        assert any("bad" in r.getMessage() for r in caplog.records)

    def test_t19_memory_elevated_limits_to_one(self, budget_home, monkeypatch):
        """Memory pressure elevated -> at most 1 spawn even with provider
        room (the host guard composes with the provider gate)."""
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd

        with kbc.connect_closing() as conn:
            _make_rows(kb, conn, [(f"a{i}", "alpha", None, None) for i in range(3)])
        monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "elevated")
        with kbc.connect_closing() as conn:
            res = kbd.dispatch_once(
                conn, spawn_fn=_fake_spawn, provider_concurrency={"anthropic": 5})
        assert len(res.spawned) == 1
        assert res.memory_pressure == "elevated"

    def test_t29_stuck_warning_names_provider_budget(self, budget_home, monkeypatch, capsys):
        """The CLI daemon health path fed repeated provider-held ticks prints
        a stuck warning whose held-back text contains provider_budget[...];
        no task changes status."""
        import argparse
        import time as _time

        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd
        from hermes_cli import kanban_ops

        with kbc.connect_closing() as conn:
            tid = kb.create_task(conn, title="a1", assignee="alpha")
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))

        captured_on_tick = {}

        def _fake_run_daemon(*, on_tick, **kwargs):
            captured_on_tick["fn"] = on_tick
            res = kbd.DispatchResult()
            res.skipped_provider_budget = [(tid, "anthropic", 1, 1)]
            for _ in range(7):
                on_tick(res)

        monkeypatch.setattr(kbd, "run_daemon", _fake_run_daemon)
        args = argparse.Namespace(
            force=True, interval=5, max=None, failure_limit=3,
            pidfile=None, verbose=False)
        monkeypatch.setattr(_time, "time", lambda: 1_000_000)
        assert kanban_ops._cmd_daemon(args) == 0
        err = capsys.readouterr().err
        assert "dispatcher stuck" in err
        assert "provider_budget[anthropic]=1/1" in err
        with kbc.connect_closing() as conn:
            assert conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (tid,)
            ).fetchone()["status"] == "ready"


class TestMigration:
    """T26 — a legacy DB gains task_runs.provider_key on connect()."""

    def test_legacy_db_gains_column(self, budget_home):
        """T26 (R1 tests-F3): a DB created BEFORE the provider_key column
        exists gains it on connect(), and its pre-upgrade rows read NULL."""
        import sqlite3

        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        db_path = kb.kanban_db_path(board=None)
        assert db_path.exists()
        # Rewind: drop the column to simulate a pre-upgrade schema, with a
        # legacy row that must keep reading NULL after the migration.
        with kbc.connect() as conn:
            with kb.write_txn(conn):
                legacy_task = kb.create_task(conn, title="legacy-row", assignee="alpha")
                conn.execute(
                    "INSERT INTO task_runs (task_id, profile, status, claim_lock, "
                    "claim_expires, started_at) VALUES (?, 'alpha', 'ended', NULL, NULL, 1)",
                    (legacy_task,))
            legacy_run = conn.execute("SELECT MAX(id) FROM task_runs").fetchone()[0]
            with kb.write_txn(conn):
                conn.execute("ALTER TABLE task_runs DROP COLUMN provider_key")
        # Re-connect: the migration re-adds the column; the legacy row is NULL.
        # The per-process _INITIALIZED_PATHS cache says this path is already
        # initialized, so simulate a FRESH process (a pre-upgrade DB being
        # opened by upgraded code for the first time) by clearing it.
        kbc._INITIALIZED_PATHS.clear()
        with kbc.connect() as conn:
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(task_runs)")}
            assert "provider_key" in cols
            row = conn.execute(
                "SELECT provider_key FROM task_runs WHERE id = ?", (legacy_run,)).fetchone()
        assert row["provider_key"] is None
        # And the counting query reads the migrated column without error.
        from hermes_cli import kanban_provider_budget as kpb
        from hermes_cli.kanban_provider_budget import RouteKeyResolver

        resolver = RouteKeyResolver(profile_exists=lambda a: False)
        with kbc.connect_closing() as conn:
            counts = kpb.count_running_by_provider(conn, resolver)
        assert isinstance(counts, dict)
        assert sqlite3.sqlite_version_info >= (3, 35, 0)  # DROP COLUMN support
