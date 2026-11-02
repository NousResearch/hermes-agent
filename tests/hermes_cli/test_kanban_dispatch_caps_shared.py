"""Every dispatch entry point resolves ``kanban.*`` caps through one source.

``dispatch_once`` treats ``None`` as "no cap", so each caller had to remember to
read config itself. The dashboard nudge did not (#81381): it hardcoded ``max=8``
and passed no in-progress caps, so one UI nudge could exceed
``max_in_progress`` / ``max_in_progress_per_profile``.

These assert the RELATIONSHIP between config and what reaches ``dispatch_once``,
not any particular cap value, so tuning defaults never breaks them.
"""
from __future__ import annotations

import pytest
import yaml


@pytest.fixture
def capped_home(tmp_path, monkeypatch):
    """A temp HERMES_HOME whose kanban caps differ from every default."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "kanban": {
            "max_spawn": 4,
            "max_in_progress": 6,
            "max_in_progress_per_profile": 2,
            "default_assignee": "researcher",
        }
    }))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Config loads are mtime-cached; drop any cache the process already holds.
    from hermes_cli import config as cfg
    for name in ("_CONFIG_CACHE", "_config_cache", "_READONLY_CACHE"):
        if hasattr(cfg, name):
            obj = getattr(cfg, name)
            if isinstance(obj, dict):
                obj.clear()
    return home


def _caps(capped_home):
    import hermes_cli.kanban_db_dispatch as kbd
    return kbd.resolve_dispatch_caps()


def test_resolver_reads_every_configured_cap(capped_home):
    caps = _caps(capped_home)
    assert caps["max_spawn"] == 4
    assert caps["max_in_progress"] == 6
    assert caps["max_in_progress_per_profile"] == 2
    assert caps["default_assignee"] == "researcher"


def test_cli_dispatch_passes_the_configured_caps(capped_home, monkeypatch):
    """`hermes kanban dispatch` must apply the same caps as every other caller."""
    import argparse

    import hermes_cli.kanban_db_dispatch as kbd
    import hermes_cli.kanban_ops as ops

    seen: dict = {}

    def _spy(conn, **kwargs):
        seen.update(kwargs)
        class _Result:
            spawned: list = []
            skipped: list = []
            timed_out: list = []
            stale: list = []
            auto_blocked: list = []
            promoted: list = []
            skipped_locked = False
        return _Result()

    monkeypatch.setattr(kbd, "dispatch_once", _spy)
    monkeypatch.setattr(ops.kbd, "dispatch_once", _spy)

    args = argparse.Namespace(dry_run=True, max=None, failure_limit=3, board=None)
    try:
        ops._cmd_dispatch(args)
    except Exception:
        # Rendering the result is not what this test pins; the kwargs are.
        pass

    caps = _caps(capped_home)
    assert seen["max_spawn"] == caps["max_spawn"]
    assert seen["max_in_progress"] == caps["max_in_progress"]
    assert seen["max_in_progress_per_profile"] == caps["max_in_progress_per_profile"]


def test_explicit_max_overrides_spawn_only(capped_home):
    """--max / ?max= is a spawn-rate signal; it must not widen the host caps."""
    import hermes_cli.kanban_db_dispatch as kbd
    base = _caps(capped_home)
    overridden = kbd.resolve_dispatch_caps(99)
    assert overridden["max_spawn"] == 99
    assert overridden["max_in_progress"] == base["max_in_progress"]
    assert overridden["max_in_progress_per_profile"] == base["max_in_progress_per_profile"]


def test_dashboard_nudge_passes_the_configured_caps(capped_home, monkeypatch):
    """The regression that started this: the nudge must not dispatch uncapped."""
    import hermes_cli.kanban_db_dispatch as kbd
    import plugins.kanban.dashboard.plugin_api as api

    seen: dict = {}

    def _spy(conn, **kwargs):
        seen.update(kwargs)
        class _Result:
            spawned: list = []
            skipped: list = []
        return _Result()

    monkeypatch.setattr(kbd, "dispatch_once", _spy)
    monkeypatch.setattr(api.kbd, "dispatch_once", _spy)
    api.dispatch(dry_run=True, max_n=None, board=None)

    caps = _caps(capped_home)
    assert seen["max_spawn"] == caps["max_spawn"]
    assert seen["max_in_progress"] == caps["max_in_progress"]
    assert seen["max_in_progress_per_profile"] == caps["max_in_progress_per_profile"]


def test_dashboard_nudge_has_no_hardcoded_spawn_default(capped_home, monkeypatch):
    """With no ?max=, the nudge takes config — never a literal baked into the route."""
    import hermes_cli.kanban_db_dispatch as kbd
    import plugins.kanban.dashboard.plugin_api as api

    seen: dict = {}

    def _spy(conn, **kwargs):
        seen.update(kwargs)
        class _Result:
            spawned: list = []
            skipped: list = []
        return _Result()

    monkeypatch.setattr(kbd, "dispatch_once", _spy)
    monkeypatch.setattr(api.kbd, "dispatch_once", _spy)
    api.dispatch(dry_run=True, max_n=None, board=None)

    assert seen["max_spawn"] == _caps(capped_home)["max_spawn"]


def test_in_progress_caps_are_never_none_when_configured(capped_home):
    """A None cap means unlimited downstream — the exact over-spawn failure."""
    caps = _caps(capped_home)
    assert caps["max_in_progress"] is not None
    assert caps["max_in_progress_per_profile"] is not None


def test_invalid_caps_fall_through_instead_of_crashing(tmp_path, monkeypatch):
    """Garbage config must not take the dispatcher down; it falls back to defaults."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "kanban": {"max_spawn": "eight", "max_in_progress_per_profile": 0}
    }))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.kanban_db_dispatch as kbd
    caps = kbd.resolve_dispatch_caps()
    # Non-integer and below-1 values are ignored, not propagated as caps.
    assert caps["max_spawn"] is None
    assert caps["max_in_progress_per_profile"] is None
