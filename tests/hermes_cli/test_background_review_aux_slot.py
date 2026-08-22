"""#84411 — background_review must be a first-class auxiliary management slot.

Runtime + DEFAULT_CONFIG honor ``auxiliary.background_review``, but the
management allow-lists (REST ``_AUX_TASK_SLOTS``, CLI ``_AUX_TASKS``) used to
omit it. That hid the pin from Desktop/Dashboard/CLI and made bulk assign
and Reset-all leave an expensive model pin active.

These tests exercise production helpers against a real temp HERMES_HOME
(no MagicMock config managers).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _seed_hidden_pro_pin(home: Path) -> None:
    cfg = {
        "model": {"provider": "deepseek", "default": "deepseek-v4-flash"},
        "auxiliary": {
            "vision": {"provider": "deepseek", "model": "deepseek-v4-flash"},
            "background_review": {
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "timeout": 120,
            },
        },
    }
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_background_review_in_default_config_schema():
    from hermes_cli.config import DEFAULT_CONFIG

    assert "background_review" in DEFAULT_CONFIG["auxiliary"]
    slot = DEFAULT_CONFIG["auxiliary"]["background_review"]
    assert slot["provider"] == "auto"
    assert slot["model"] == ""
    assert int(slot.get("timeout") or 0) > 0


def test_background_review_in_management_registries():
    """Every management surface that gates assign/reset must know the slot."""
    from hermes_cli.main import _AUX_TASKS
    from hermes_cli.web_server import _AUX_TASK_SLOTS

    assert "background_review" in _AUX_TASK_SLOTS, (
        "background_review missing from _AUX_TASK_SLOTS "
        "(GET/set/reset/stale_aux allow-list)"
    )
    cli_keys = {k for k, _name, _desc in _AUX_TASKS}
    assert "background_review" in cli_keys, (
        "background_review missing from CLI _AUX_TASKS"
    )


def test_get_auxiliary_models_exposes_background_review(hermes_home):
    _seed_hidden_pro_pin(hermes_home)
    from hermes_cli.web_server import get_auxiliary_models

    api = get_auxiliary_models()
    tasks = {row["task"]: row for row in api["tasks"]}
    assert "background_review" in tasks
    row = tasks["background_review"]
    assert row["provider"] == "deepseek"
    assert row["model"] == "deepseek-v4-pro"


def test_direct_assignment_accepts_background_review(hermes_home):
    _seed_hidden_pro_pin(hermes_home)
    from hermes_cli.config import load_config
    from hermes_cli.web_server import _apply_model_assignment_sync

    result = _apply_model_assignment_sync(
        "auxiliary", "deepseek", "deepseek-v4-flash", "background_review", "", ""
    )
    assert result.get("ok") is True
    pin = load_config()["auxiliary"]["background_review"]
    assert pin["provider"] == "deepseek"
    assert pin["model"] == "deepseek-v4-flash"
    # Non-routing fields preserved
    assert int(pin.get("timeout") or 0) == 120


def test_bulk_auxiliary_assignment_includes_background_review(hermes_home):
    _seed_hidden_pro_pin(hermes_home)
    from hermes_cli.config import load_config
    from hermes_cli.web_server import _apply_model_assignment_sync

    # Empty task → bulk over every _AUX_TASK_SLOTS entry
    result = _apply_model_assignment_sync(
        "auxiliary", "deepseek", "deepseek-v4-flash", "", "", ""
    )
    assert result.get("ok") is True
    pin = load_config()["auxiliary"]["background_review"]
    assert pin["provider"] == "deepseek"
    assert pin["model"] == "deepseek-v4-flash"
    assert int(pin.get("timeout") or 0) == 120


def test_reset_all_clears_background_review_pin_preserves_timeout(hermes_home):
    _seed_hidden_pro_pin(hermes_home)
    from hermes_cli.config import load_config
    from hermes_cli.web_server import _apply_model_assignment_sync

    result = _apply_model_assignment_sync(
        "auxiliary", "deepseek", "deepseek-v4-flash", "__reset__", "", ""
    )
    assert result.get("ok") is True
    assert result.get("reset") is True
    pin = load_config()["auxiliary"]["background_review"]
    assert pin["provider"] == "auto"
    assert pin["model"] == ""
    assert int(pin.get("timeout") or 0) == 120


def test_stale_aux_reports_background_review_when_main_provider_changes(hermes_home):
    """Main-model set should surface mismatched background_review pins."""
    _seed_hidden_pro_pin(hermes_home)
    from hermes_cli.web_server import _apply_model_assignment_sync

    result = _apply_model_assignment_sync(
        "main", "openrouter", "anthropic/claude-sonnet-4", "", "", ""
    )
    assert result.get("ok") is True
    stale_tasks = {row["task"] for row in result.get("stale_aux") or []}
    assert "background_review" in stale_tasks


def test_cli_reset_aux_to_auto_clears_background_review(hermes_home):
    """CLI 'Reset all to auto' must clear the pin (same class as REST __reset__)."""
    _seed_hidden_pro_pin(hermes_home)
    from hermes_cli.config import load_config
    from hermes_cli.main import _reset_aux_to_auto

    n = _reset_aux_to_auto()
    assert n >= 1
    pin = load_config()["auxiliary"]["background_review"]
    assert pin["provider"] == "auto"
    assert pin["model"] == ""
    assert int(pin.get("timeout") or 0) == 120
