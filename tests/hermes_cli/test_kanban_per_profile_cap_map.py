"""Regression tests for ``kanban.max_in_progress_per_profile_map`` — the cap map.

The dispatcher used to read only the scalar ``kanban.max_in_progress_per_profile``; the map key
was registered in ``DEFAULT_CONFIG`` but had no reader, so a map-only config silently resolved to
no cap at all and a fan-out ran past the intended per-profile brake. The map now wins for the
profiles it names, its ``default`` entry caps the profiles it omits, and a scalar folds in as the
fallback (``setdefault``: an explicit map ``default`` wins). Scalar-only configs keep the
pre-existing contract — the scalar caps every profile.

Coverage: the shared reader ``profile_caps_setting`` (map / scalar-only / invalid entries), the
gateway settings path, and dispatch behavior for map-only, scalar-only, map+scalar,
``default``-key and no-cap configurations.
"""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
import time
import types
from dataclasses import asdict

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Fresh ``HERMES_HOME`` with alpha/beta/default profiles and a clean kanban DB."""
    test_home = tempfile.mkdtemp(prefix="kanban_profile_cap_map_test_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
        with open(os.path.join(test_home, "profiles", prof, "config.yaml"), "w") as fh:
            fh.write("{}\n")  # identity marker: a bare dir is not a profile
    monkeypatch.setenv("HERMES_HOME", test_home)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("hermes_cli")
            or mod.startswith("hermes_state")
            or mod == "hermes_constants"
        ):
            del sys.modules[mod]
    from hermes_cli import kanban_db

    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def _make_board(kb, prefix, ready_assignee, n_ready, running_assignee, n_running):
    """Disposable board: ``n_ready`` ready rows + ``n_running`` rows pinned to running.

    ``create_task`` cannot mint a ``running`` row directly, so the fixture flips the
    assignments itself with claim bookkeeping a reclaim tick respects: a future
    ``claim_expires`` (survives ``release_stale_claims``), a non-NULL foreign-host
    ``claim_lock`` (survives ``reconcile_orphaned_running`` and the local crash sweep) and a
    fresh heartbeat.
    """
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        ready = [
            kb.create_task(conn, title=f"{prefix} ready {i}", assignee=ready_assignee)
            for i in range(n_ready)
        ]
        running = [
            kb.create_task(conn, title=f"{prefix} running {i}", assignee=running_assignee)
            for i in range(n_running)
        ]
        now = int(time.time())
        for tid in running:
            conn.execute(
                "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
                "worker_pid=?, worker_started_at=?, started_at=?, last_heartbeat_at=? "
                "WHERE id=?",
                (f"foreign-host:fixture:{tid}", now + 600, 999999, now - 60, now - 60, now - 30, tid),
            )
        conn.commit()
    return ready, running


def _dispatch(conn, **kwargs):
    """One dry-run tick with deterministic budgets (nothing is spawned for real)."""
    from hermes_cli import kanban_db_dispatch as kbd

    return kbd.dispatch_once(
        conn,
        spawn_fn=_fake_spawn,
        dry_run=True,
        max_spawn=None,
        max_in_progress=16,
        stale_timeout_seconds=0,
        **kwargs,
    )


def test_config_default_registers_the_map_key():
    """The key must exist in ``DEFAULT_CONFIG`` — a registered key with no reader was the bug."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kanban"]["max_in_progress_per_profile_map"] is None


def test_profile_caps_setting_reads_the_map():
    from hermes_cli import kanban_db_dispatch as kbd

    cfg = {"max_in_progress_per_profile_map": {"alpha": 2, "default": 4}}
    assert kbd.profile_caps_setting(cfg) == {"alpha": 2, "default": 4}


def test_profile_caps_setting_scalar_only_yields_no_map():
    from hermes_cli import kanban_db_dispatch as kbd

    assert kbd.profile_caps_setting({"max_in_progress_per_profile": 2}) is None
    assert kbd.profile_caps_setting({}) is None
    assert kbd.profile_caps_setting(None) is None


def test_profile_caps_setting_drops_unusable_entries():
    """One bad value must not disable the whole map the way the pre-reader silence did."""
    from hermes_cli import kanban_db_dispatch as kbd

    assert kbd.profile_caps_setting({"max_in_progress_per_profile_map": "alpha:2"}) is None
    caps = kbd.profile_caps_setting(
        {
            "max_in_progress_per_profile_map": {
                "alpha": 2,
                "zero": 0,
                "negative": -1,
                "text": "x",
                "bool": True,
                "": 4,
            }
        }
    )
    assert caps == {"alpha": 2}


def test_gateway_settings_thread_the_map_into_dispatch(isolated_kanban_home):
    """The gateway path resolves the map and every settings field is a ``dispatch_once`` kwarg."""
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings
    from hermes_cli import kanban_db_dispatch as kbd

    settings = _resolve_dispatcher_settings(
        {"max_in_progress_per_profile_map": {"alpha": 2, "default": 4}},
        types.SimpleNamespace(DEFAULT_FAILURE_LIMIT=2),
    )
    assert settings.max_in_progress_per_profile is None
    assert settings.max_in_progress_per_profile_map == {"alpha": 2, "default": 4}

    params = inspect.signature(kbd.dispatch_once).parameters
    for key in asdict(settings):
        if key != "interval":
            assert key in params, f"dispatch_once lacks kwarg for settings field {key}"


def test_map_only_config_defers_at_cap(isolated_kanban_home):
    """The dead-config scenario: map present, scalar absent. Without a reader this board
    would spawn all three rows (cap resolved to ``None``)."""
    kb = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc

    _make_board(kb, "map_only", "alpha", 3, "alpha", 2)
    with kbc.connect_closing() as conn:
        res = _dispatch(
            conn,
            max_in_progress_per_profile=None,
            max_in_progress_per_profile_map={"alpha": 2, "default": 4},
        )
    assert res.spawned == []
    assert len(res.skipped_per_profile_capped) == 3
    for tid, who, current in res.skipped_per_profile_capped:
        assert (who, current) == ("alpha", 2)
    assert res.skipped_per_profile_capped[0][0] not in res.skipped_nonspawnable


def test_scalar_only_config_falls_back(isolated_kanban_home):
    """Scalar alone still caps every profile — the pre-map contract."""
    kb = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc

    _make_board(kb, "scalar_only", "alpha", 3, "alpha", 2)
    with kbc.connect_closing() as conn:
        res = _dispatch(conn, max_in_progress_per_profile=2, max_in_progress_per_profile_map=None)
    assert res.spawned == []
    assert len(res.skipped_per_profile_capped) == 3
    for tid, who, current in res.skipped_per_profile_capped:
        assert (who, current) == ("alpha", 2)


def test_map_wins_over_scalar(isolated_kanban_home):
    """map alpha=3 vs scalar=1 with one running alpha row: the map allows the spawn the
    scalar alone would have deferred."""
    kb = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc

    _make_board(kb, "map_wins", "alpha", 1, "alpha", 1)
    with kbc.connect_closing() as conn:
        res = _dispatch(
            conn, max_in_progress_per_profile=1, max_in_progress_per_profile_map={"alpha": 3}
        )
    assert res.skipped_per_profile_capped == []
    assert len(res.spawned) == 1


def test_map_default_key_caps_profiles_it_omits(isolated_kanban_home):
    """No scalar: the map's own ``default`` entry is the fallback."""
    kb = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc

    _make_board(kb, "map_default", "default", 1, "default", 1)
    with kbc.connect_closing() as conn:
        res = _dispatch(
            conn,
            max_in_progress_per_profile=None,
            max_in_progress_per_profile_map={"alpha": 9, "default": 1},
        )
    assert res.spawned == []
    assert [(w, c) for _tid, w, c in res.skipped_per_profile_capped] == [("default", 1)]


def test_scalar_fills_profiles_the_map_omits(isolated_kanban_home):
    """Map without a ``default`` key plus a scalar: profiles the map omits fall back to it."""
    kb = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc

    _make_board(kb, "scalar_fill", "default", 1, "default", 1)
    with kbc.connect_closing() as conn:
        res = _dispatch(
            conn, max_in_progress_per_profile=1, max_in_progress_per_profile_map={"alpha": 9}
        )
    assert res.spawned == []
    assert [(w, c) for _tid, w, c in res.skipped_per_profile_capped] == [("default", 1)]


def test_no_cap_configured_spawns(isolated_kanban_home):
    """Both absent: the historic no-cap contract stays available."""
    kb = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc

    _make_board(kb, "no_cap", "alpha", 3, "alpha", 2)
    with kbc.connect_closing() as conn:
        res = _dispatch(
            conn, max_in_progress_per_profile=None, max_in_progress_per_profile_map=None
        )
    assert res.skipped_per_profile_capped == []
    assert len(res.spawned) == 3