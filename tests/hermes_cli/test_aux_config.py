"""Tests for the auxiliary-model configuration UI in ``hermes model``.

Covers the helper functions:
  - ``_save_aux_choice`` writes to config.yaml without touching main model config
  - ``_reset_aux_to_auto`` clears routing fields but preserves timeouts
  - ``_format_aux_current`` renders current task config for the menu
  - ``_AUX_TASKS`` stays in sync with ``DEFAULT_CONFIG["auxiliary"]``

These are pure-function tests — the interactive menu loops are not covered
here (they're stdin-driven curses prompts).
"""

from __future__ import annotations

import pytest

from hermes_cli.config import DEFAULT_CONFIG, load_config
from hermes_cli.main import (
    _AUX_TASKS,
    _format_aux_current,
    _reset_aux_to_auto,
    _save_aux_choice,
)


# ── Default config ──────────────────────────────────────────────────────────


def test_title_generation_present_in_default_config():
    """`title_generation` task must be defined in DEFAULT_CONFIG.

    Regression for an existing gap: title_generator.py calls
    ``call_llm(task="title_generation", ...)`` but the task was missing
    from DEFAULT_CONFIG["auxiliary"], so the config-backed timeout/provider
    overrides never worked for that task.
    """
    assert "title_generation" in DEFAULT_CONFIG["auxiliary"]
    tg = DEFAULT_CONFIG["auxiliary"]["title_generation"]
    assert tg["provider"] == "auto"
    assert tg["model"] == ""
    assert tg["timeout"] > 0
    assert tg["extra_body"] == {}


def test_session_search_defaults_include_extra_body_and_concurrency():
    ss = DEFAULT_CONFIG["auxiliary"]["session_search"]
    assert ss["provider"] == "auto"
    assert ss["model"] == ""
    assert ss["extra_body"] == {}
    assert ss["max_concurrency"] == 3


def test_aux_tasks_keys_all_exist_in_default_config():
    """Every task the menu offers must be defined in DEFAULT_CONFIG."""
    aux_keys = {k for k, _name, _desc in _AUX_TASKS}
    default_keys = set(DEFAULT_CONFIG["auxiliary"].keys())
    missing = aux_keys - default_keys
    assert not missing, (
        f"_AUX_TASKS references tasks not in DEFAULT_CONFIG.auxiliary: {missing}"
    )


# ── _format_aux_current ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "task_cfg,expected",
    [
        ({}, "auto"),
        ({"provider": "", "model": ""}, "auto"),
        ({"provider": "auto", "model": ""}, "auto"),
        ({"provider": "auto", "model": "gpt-4o"}, "auto · gpt-4o"),
        ({"provider": "openrouter", "model": ""}, "openrouter"),
        (
            {"provider": "openrouter", "model": "google/gemini-2.5-flash"},
            "openrouter · google/gemini-2.5-flash",
        ),
        ({"provider": "nous", "model": "gemini-3-flash"}, "nous · gemini-3-flash"),
        (
            {"provider": "custom", "base_url": "http://localhost:11434/v1", "model": ""},
            "custom (localhost:11434/v1)",
        ),
        (
            {
                "provider": "custom",
                "base_url": "http://localhost:11434/v1/",
                "model": "qwen2.5:32b",
            },
            "custom (localhost:11434/v1) · qwen2.5:32b",
        ),
    ],
)
def test_format_aux_current(task_cfg, expected):
    assert _format_aux_current(task_cfg) == expected


def test_format_aux_current_handles_non_dict():
    assert _format_aux_current(None) == "auto"
    assert _format_aux_current("string") == "auto"


# ── _save_aux_choice ────────────────────────────────────────────────────────


def test_save_aux_choice_persists_to_config_yaml(tmp_path, monkeypatch):
    """Saving a task writes provider/model/base_url/api_key to auxiliary.<task>."""
    from pathlib import Path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    _save_aux_choice(
        "vision", provider="openrouter", model="google/gemini-2.5-flash",
    )
    cfg = load_config()
    v = cfg["auxiliary"]["vision"]
    assert v["provider"] == "openrouter"
    assert v["model"] == "google/gemini-2.5-flash"
    assert v["base_url"] == ""
    assert v["api_key"] == ""


def test_save_aux_choice_preserves_timeout(tmp_path, monkeypatch):
    """Saving must NOT clobber user-tuned timeout values."""
    from pathlib import Path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    # Default vision timeout is 120
    cfg_before = load_config()
    default_timeout = cfg_before["auxiliary"]["vision"]["timeout"]
    assert default_timeout == 120

    _save_aux_choice("vision", provider="nous", model="gemini-3-flash")
    cfg_after = load_config()
    assert cfg_after["auxiliary"]["vision"]["timeout"] == default_timeout
    # download_timeout also preserved for vision
    assert cfg_after["auxiliary"]["vision"].get("download_timeout") == 30


def test_save_aux_choice_does_not_touch_main_model(tmp_path, monkeypatch):
    """Aux config must never mutate model.default / model.provider / model.base_url."""
    from pathlib import Path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    # Simulate a configured main model
    from hermes_cli.config import save_config

    cfg = load_config()
    cfg["model"] = {
        "default": "claude-sonnet-4.6",
        "provider": "anthropic",
        "base_url": "",
    }
    save_config(cfg)

    _save_aux_choice(
        "compression", provider="custom",
        base_url="http://localhost:11434/v1", model="qwen2.5:32b",
    )

    cfg = load_config()
    # Main model untouched
    assert cfg["model"]["default"] == "claude-sonnet-4.6"
    assert cfg["model"]["provider"] == "anthropic"
    # Aux saved correctly
    c = cfg["auxiliary"]["compression"]
    assert c["provider"] == "custom"
    assert c["model"] == "qwen2.5:32b"
    assert c["base_url"] == "http://localhost:11434/v1"


def test_save_aux_choice_creates_missing_task_entry(tmp_path, monkeypatch):
    """Saving a task that was wiped from config.yaml should recreate it."""
    from pathlib import Path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    # Remove vision from config entirely
    from hermes_cli.config import save_config

    cfg = load_config()
    cfg.setdefault("auxiliary", {}).pop("vision", None)
    save_config(cfg)

    _save_aux_choice("vision", provider="nous", model="gemini-3-flash")
    cfg = load_config()
    assert cfg["auxiliary"]["vision"]["provider"] == "nous"
    assert cfg["auxiliary"]["vision"]["model"] == "gemini-3-flash"


# ── _reset_aux_to_auto ──────────────────────────────────────────────────────


def test_reset_aux_to_auto_clears_routing_preserves_timeouts(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    # Configure two tasks non-auto, and bump a timeout
    _save_aux_choice("vision", provider="openrouter", model="gpt-4o")
    _save_aux_choice("compression", provider="nous", model="gemini-3-flash")
    from hermes_cli.config import save_config

    cfg = load_config()
    cfg["auxiliary"]["vision"]["timeout"] = 300  # user-tuned
    save_config(cfg)

    n = _reset_aux_to_auto()
    assert n == 2  # both changed

    cfg = load_config()
    for task in ("vision", "compression"):
        v = cfg["auxiliary"][task]
        assert v["provider"] == "auto"
        assert v["model"] == ""
        assert v["base_url"] == ""
        assert v["api_key"] == ""
    # User-tuned timeout survives reset
    assert cfg["auxiliary"]["vision"]["timeout"] == 300
    # Default compression timeout preserved
    assert cfg["auxiliary"]["compression"]["timeout"] == 120


def test_reset_aux_to_auto_idempotent(tmp_path, monkeypatch):
    """Second reset on already-auto config returns 0 without errors."""
    from pathlib import Path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    assert _reset_aux_to_auto() == 0
    _save_aux_choice("vision", provider="nous", model="gemini-3-flash")
    assert _reset_aux_to_auto() == 1
    assert _reset_aux_to_auto() == 0


# ── Menu dispatch ───────────────────────────────────────────────────────────


def test_select_provider_and_model_dispatches_to_aux_menu(tmp_path, monkeypatch):
    """Picking 'Configure auxiliary models...' in the provider list calls _aux_config_menu."""
    from pathlib import Path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    from hermes_cli import main as main_mod

    called = {"aux": 0, "flow": 0}

    def fake_prompt(choices, *, default=0):
        # Find the aux-config entry by its label text and return its index
        for i, label in enumerate(choices):
            if "Configure auxiliary models" in label:
                return i
        raise AssertionError("aux entry not in provider list")

    monkeypatch.setattr(main_mod, "_prompt_provider_choice", fake_prompt)
    monkeypatch.setattr(main_mod, "_aux_config_menu", lambda: called.__setitem__("aux", called["aux"] + 1))
    # Guard against any main flow accidentally running
    monkeypatch.setattr(main_mod, "_model_flow_openrouter",
                        lambda *a, **kw: called.__setitem__("flow", called["flow"] + 1))

    main_mod.select_provider_and_model()

    assert called["aux"] == 1, "aux menu not invoked"
    assert called["flow"] == 0, "main provider flow should not run"


def test_leave_unchanged_replaces_cancel_label(tmp_path, monkeypatch):
    """The bottom cancel entry now reads 'Leave unchanged' (UX polish)."""
    from pathlib import Path
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    from hermes_cli import main as main_mod

    captured: list[list[str]] = []

    def fake_prompt(choices, *, default=0):
        captured.append(list(choices))
        # Pick 'Leave unchanged' (last item) to exit cleanly
        for i, label in enumerate(choices):
            if label == "Leave unchanged":
                return i
        raise AssertionError("Leave unchanged not in provider list")

    monkeypatch.setattr(main_mod, "_prompt_provider_choice", fake_prompt)

    main_mod.select_provider_and_model()

    assert captured, "provider menu never rendered"
    labels = captured[0]
    assert "Leave unchanged" in labels
    assert "Cancel" not in labels, "Cancel label should be replaced"
    assert any("Configure auxiliary models" in label for label in labels)
