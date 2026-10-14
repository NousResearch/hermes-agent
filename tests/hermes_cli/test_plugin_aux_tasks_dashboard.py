"""Plugin-registered auxiliary tasks reach the dashboard/Desktop Models settings.

``PluginContext.register_auxiliary_task`` already puts a plugin task in the ``hermes model``
picker (``main_provider_setup._all_aux_tasks``), but the REST surface behind the Desktop
Settings → Models page enumerated only the built-in ``_AUX_TASK_SLOTS``: ``GET
/api/model/auxiliary`` never listed the task, ``POST /api/model/set`` rejected it with
``unknown auxiliary task``, ``__reset__`` skipped it and the stale-pin nudge ignored it.

These tests load real plugins from temp homes through ``PluginManager`` discovery (no fake
registry) and cover the profile seam: one ``hermes serve`` process, two profiles with their own
plugins and their own config. Regression for #40880 / #129189.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

import hermes_cli.plugins as plugins_mod
from agent.auxiliary_client import _get_auxiliary_task_config
from hermes_cli.plugins import PluginManager
from hermes_cli.web_server_config import (
    _AUX_TASK_SLOTS, _apply_aux_assignment_sync, _apply_model_assignment_sync, _aux_task_slots,
    _stale_aux_pins,
)
from hermes_cli.web_server_profiles import _profile_scope
from hermes_cli.web_routers.models import get_auxiliary_models


def _write_plugin(home: Path, name: str, register_call: str) -> None:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    # JSON is valid YAML; keeps the test free of a yaml dependency.
    (plugin_dir / "plugin.yaml").write_text(
        json.dumps({"name": name, "version": "0.1.0", "description": f"{name} probe"}))
    (plugin_dir / "__init__.py").write_text(f"def register(ctx):\n    ctx.{register_call}\n")


def _write_home(home: Path, plugins: dict, compression_model: str) -> None:
    for name, call in plugins.items():
        _write_plugin(home, name, call)
    (home / "config.yaml").write_text(json.dumps({
        "plugins": {"enabled": list(plugins)},
        "auxiliary": {"compression": {"provider": "openrouter", "model": compression_model}},
    }))


_SIDE = ("register_auxiliary_task('side_task', display_name='Side model', "
         "description='side model for side', inherit_from='compression')")
_ALPHA = ("register_auxiliary_task('alpha_task', display_name='Alpha side model', "
          "description='side model for alpha_plugin', defaults={'timeout': 7})")


@pytest.fixture
def two_profile_homes(tmp_path, monkeypatch):
    """Process home ``a`` (plugins side + alpha) and named profile ``b`` (plugin side only), each
    with its own ``auxiliary.compression`` model."""
    home_a = tmp_path / ".hermes"
    home_b = home_a / "profiles" / "b"
    _write_home(home_a, {"side": _SIDE, "alpha_plugin": _ALPHA}, "vendor/model-a")
    _write_home(home_b, {"side": _SIDE}, "vendor/model-b")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "empty-bundled")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    plugins_mod._reset_plugin_managers_for_tests()
    yield home_a, home_b
    plugins_mod._reset_plugin_managers_for_tests()


def _rows(profile):
    return {row["task"]: row for row in get_auxiliary_models(profile=profile)["tasks"]}


def _resolved_model(profile):
    with _profile_scope(profile):
        return _get_auxiliary_task_config("side_task").get("model")


def test_listing_is_per_profile_and_carries_inheritance(two_profile_homes):
    listing = get_auxiliary_models(profile=None)["tasks"]
    # Built-ins first, in their fixed order and pre-existing shape; plugin rows appended.
    assert [row["task"] for row in listing][: len(_AUX_TASK_SLOTS)] == list(_AUX_TASK_SLOTS)
    assert "label" not in listing[0] and "plugin" not in listing[0]

    rows_a, rows_b = _rows(None), _rows("b")
    assert "alpha_task" in rows_a and "alpha_task" not in rows_b
    side_a, side_b = rows_a["side_task"], rows_b["side_task"]
    assert (side_a["label"], side_a["hint"], side_a["plugin"]) == ("Side model", "side model for side", "side")
    assert side_a["inherit_from"] == "compression" and rows_a["alpha_task"]["inherit_from"] is None
    # Unpinned: own provider stays "auto"; ``effective`` is the base slot of THAT profile.
    assert side_a["provider"] == "auto"
    assert side_a["effective"] == {"provider": "openrouter", "model": "vendor/model-a", "base_url": ""}
    assert side_b["effective"]["model"] == "vendor/model-b"


def test_pin_persists_per_profile_and_is_honored_by_resolution(two_profile_homes):
    home_a, _home_b = two_profile_homes
    with _profile_scope("b"):
        _apply_model_assignment_sync("auxiliary", "nous", "hermes-4", "side_task", "", "")
    assert _rows("b")["side_task"]["provider"] == "nous"
    assert "side_task" not in (home_a / "config.yaml").read_text()
    assert (_resolved_model("b"), _resolved_model(None)) == ("hermes-4", "vendor/model-a")

    # "auto" on an inheriting slot (Desktop's "Follow <base>") goes back to the base.
    with _profile_scope("b"):
        _apply_model_assignment_sync("auxiliary", "auto", "", "side_task", "", "")
    assert _resolved_model("b") == "vendor/model-b"

    # A base-slot change in B reaches B's plugin slot, never A's.
    with _profile_scope("b"):
        _apply_model_assignment_sync("auxiliary", "openrouter", "vendor/model-b2", "compression", "", "")
    assert (_resolved_model("b"), _resolved_model(None)) == ("vendor/model-b2", "vendor/model-a")


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
    with _profile_scope("b"):
        _apply_aux_assignment_sync({"auxiliary": {}}, "openrouter", "m", "side_task", "", "")
        with pytest.raises(HTTPException) as exc:
            _apply_aux_assignment_sync({"auxiliary": {}}, "openrouter", "m", "alpha_task", "", "")
    assert exc.value.status_code == 400 and "unknown auxiliary task" in exc.value.detail


def test_plugin_discovery_failure_leaves_builtins_working(two_profile_homes, monkeypatch):
    def _boom():
        raise RuntimeError("plugin scan exploded")

    # Patch where production reads: the resolver imports the name from hermes_cli.plugins at call time.
    monkeypatch.setattr(plugins_mod, "get_plugin_auxiliary_tasks", _boom)
    assert _aux_task_slots() == _AUX_TASK_SLOTS
    tasks = [row["task"] for row in get_auxiliary_models(profile=None)["tasks"]]
    assert tasks == list(_AUX_TASK_SLOTS)
