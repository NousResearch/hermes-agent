"""Plugin-registered auxiliary tasks reach the dashboard/Desktop Models settings.

``PluginContext.register_auxiliary_task`` already puts a plugin task in the ``hermes model``
picker (``main_provider_setup._all_aux_tasks``), but the REST surface behind the Desktop
Settings → Models page enumerated only the built-in ``_AUX_TASK_SLOTS``: ``GET
/api/model/auxiliary`` never listed the task, ``POST /api/model/set`` rejected it with
``unknown auxiliary task``, ``__reset__`` skipped it and the stale-pin nudge ignored it.

These tests load a real plugin from a temp ``HERMES_HOME`` through ``PluginManager`` discovery
(no fake registry) and cover the profile seam the reviewer of #40922 flagged: two profiles with
different plugins must each see their own tasks from one ``hermes serve`` process.
Regression for #40880.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginManager
from hermes_cli.web_server_config import (
    _AUX_TASK_SLOTS, _apply_aux_assignment_sync, _aux_task_slots, _stale_aux_pins,
)
from hermes_cli.web_routers.models import get_auxiliary_models


def _write_aux_plugin(home: Path, name: str, task_key: str, *, display: str) -> None:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "0.1.0", "description": f"{name} probe"})
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    ctx.register_auxiliary_task({task_key!r}, display_name={display!r},\n"
        f"                                description='side model for {name}', defaults={{'timeout': 7}})\n"
    )
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [name]}}))


@pytest.fixture
def two_profile_homes(tmp_path, monkeypatch):
    """Process home ``a`` with plugin task ``alpha_task``; named profile ``b`` with ``beta_task``."""
    root = tmp_path / ".hermes"
    home_a = root
    home_b = root / "profiles" / "b"
    _write_aux_plugin(home_a, "alpha_plugin", "alpha_task", display="Alpha side model")
    _write_aux_plugin(home_b, "beta_plugin", "beta_task", display="Beta side model")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "empty-bundled")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    plugins_mod._reset_plugin_managers_for_tests()
    yield home_a, home_b
    plugins_mod._reset_plugin_managers_for_tests()


def test_slots_and_listing_include_the_active_profiles_plugin_task(two_profile_homes):
    slots = _aux_task_slots()
    assert slots[: len(_AUX_TASK_SLOTS)] == _AUX_TASK_SLOTS
    assert "alpha_task" in slots and "beta_task" not in slots

    listing = get_auxiliary_models(profile=None)
    by_task = {row["task"]: row for row in listing["tasks"]}
    assert set(_AUX_TASK_SLOTS) <= set(by_task)
    alpha = by_task["alpha_task"]
    assert alpha["label"] == "Alpha side model"
    assert alpha["hint"] == "side model for alpha_plugin"
    assert alpha["plugin"] == "alpha_plugin"
    assert alpha["provider"] == "auto" and alpha["model"] == ""
    # Built-in rows keep their pre-existing shape: no plugin fields.
    assert "label" not in by_task["vision"] and "plugin" not in by_task["vision"]
    # Plugin rows come after every built-in so the UI order stays stable.
    assert [row["task"] for row in listing["tasks"]][: len(_AUX_TASK_SLOTS)] == list(_AUX_TASK_SLOTS)


def test_listing_scopes_plugin_tasks_to_the_requested_profile(two_profile_homes):
    """One serve process, two profiles: each request enumerates ITS profile's plugins."""
    tasks_a = {row["task"] for row in get_auxiliary_models(profile=None)["tasks"]}
    tasks_b = {row["task"] for row in get_auxiliary_models(profile="b")["tasks"]}
    assert "alpha_task" in tasks_a and "beta_task" not in tasks_a
    assert "beta_task" in tasks_b and "alpha_task" not in tasks_b


def test_assignment_reset_and_stale_pins_cover_plugin_tasks(two_profile_homes, monkeypatch):
    saved: dict = {}
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.update(cfg))
    cfg: dict = {"auxiliary": {}}

    out = _apply_aux_assignment_sync(cfg, "openrouter", "fast-model", "alpha_task", "", "")
    assert out["tasks"] == ["alpha_task"]
    assert cfg["auxiliary"]["alpha_task"] == {"provider": "openrouter", "model": "fast-model"}
    assert saved["auxiliary"]["alpha_task"]["model"] == "fast-model"

    # A pin on a plugin task is a stale pin like any other when main moves elsewhere.
    assert {"task": "alpha_task", "provider": "openrouter", "model": "fast-model"} in _stale_aux_pins(cfg, "nous")

    # Empty task = broadcast; it must reach the plugin slot too.
    _apply_aux_assignment_sync(cfg, "nous", "hermes-4", "", "", "")
    assert cfg["auxiliary"]["alpha_task"]["provider"] == "nous"

    _apply_aux_assignment_sync(cfg, "", "", "__reset__", "", "")
    assert cfg["auxiliary"]["alpha_task"] == {"provider": "auto", "model": ""}

    # Another profile's task is still unknown here — the validation is per-profile, not global.
    with pytest.raises(HTTPException) as exc:
        _apply_aux_assignment_sync(cfg, "openrouter", "m", "beta_task", "", "")
    assert exc.value.status_code == 400 and "unknown auxiliary task" in exc.value.detail


def test_plugin_discovery_failure_leaves_builtins_working(two_profile_homes, monkeypatch):
    def _boom():
        raise RuntimeError("plugin scan exploded")

    # Patch where production reads: the resolver imports the name from hermes_cli.plugins at call time.
    monkeypatch.setattr(plugins_mod, "get_plugin_auxiliary_tasks", _boom)
    assert _aux_task_slots() == _AUX_TASK_SLOTS
    tasks = [row["task"] for row in get_auxiliary_models(profile=None)["tasks"]]
    assert tasks == list(_AUX_TASK_SLOTS)
