"""The bundled ``prompt-refinement`` plugin gives Desktop's refinement call a selectable model slot.

Loads the real in-tree plugin through ``PluginManager`` discovery against temp homes (no fake
registry): enabled, the ``prompt_refinement`` slot is listed on the Settings > Model surface and a
pin there is what ``call_llm(task="prompt_refinement")`` resolves; not enabled (the default), the
slot does not exist and the same task falls back to the automatic route instead of failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hermes_cli.plugins as plugins_mod
from agent.auxiliary_client import _get_auxiliary_task_config
from hermes_cli.plugins import PluginManager
from hermes_cli.web_server_config import _apply_model_assignment_sync, _aux_task_slots
from hermes_cli.web_server_profiles import _profile_scope
from hermes_cli.web_routers.models import get_auxiliary_models

TASK = "prompt_refinement"


def _write_home(home: Path, *, enabled: bool) -> None:
    home.mkdir(parents=True)
    # JSON is valid YAML; keeps the test free of a yaml dependency.
    (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["prompt-refinement"] if enabled else []}}))


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Process home ``a`` enables the plugin; named profile ``b`` does not."""
    home_a = tmp_path / ".hermes"
    home_b = home_a / "profiles" / "b"
    _write_home(home_a, enabled=True)
    _write_home(home_b, enabled=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.delenv("HERMES_BUNDLED_PLUGINS", raising=False)
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    plugins_mod._reset_plugin_managers_for_tests()
    yield home_a, home_b
    plugins_mod._reset_plugin_managers_for_tests()


def _row(profile):
    return {row["task"]: row for row in get_auxiliary_models(profile=profile)["tasks"]}.get(TASK)


def test_enabled_plugin_lists_the_slot_with_its_own_copy(homes):
    row = _row(None)
    assert row is not None
    assert (row["label"], row["plugin"]) == ("Prompt refinement", "prompt-refinement")
    assert row["provider"] == "auto"
    with _profile_scope(None):
        assert TASK in _aux_task_slots()


def test_slot_is_absent_where_the_plugin_is_not_enabled(homes):
    assert _row("b") is None
    with _profile_scope("b"):
        assert TASK not in _aux_task_slots()
        # The Desktop call still resolves: an unknown task has no pin, so the auto route applies.
        assert _get_auxiliary_task_config(TASK) == {}


def test_pin_is_per_profile_and_is_what_the_refinement_call_resolves(homes):
    home_a, home_b = homes
    with _profile_scope(None):
        _apply_model_assignment_sync("auxiliary", "openrouter", "vendor/refiner", TASK, "", "")
        assert _get_auxiliary_task_config(TASK).get("model") == "vendor/refiner"
    assert TASK in (home_a / "config.yaml").read_text()
    assert TASK not in (home_b / "config.yaml").read_text()
