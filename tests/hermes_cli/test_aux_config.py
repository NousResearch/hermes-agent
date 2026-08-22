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
    _DELEGATION_TASK_KEY,
    _all_aux_tasks,
    _apply_aux_choice_to_all,
    _delegation_cfg_as_task,
    _format_aux_all_current,
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
    assert tg["enabled"] is True
    assert tg["provider"] == "auto"
    assert tg["model"] == ""
    assert tg["prefer_fast_model"] is False
    assert tg["timeout"] > 0
    assert tg["extra_body"] == {}






# ── _format_aux_current ─────────────────────────────────────────────────────




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




# ── _reset_aux_to_auto ──────────────────────────────────────────────────────






# ── Menu dispatch ───────────────────────────────────────────────────────────




# ── Delegation entry (top-level `delegation.*`, not `auxiliary.*`) ──────────


def _isolate_home(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)


def test_save_delegation_writes_top_level_section(tmp_path, monkeypatch):
    """Delegation picks write to delegation.*, never auxiliary.delegation."""
    _isolate_home(tmp_path, monkeypatch)

    _save_aux_choice(
        _DELEGATION_TASK_KEY, provider="openrouter", model="google/gemini-3-flash",
    )
    cfg = load_config()
    d = cfg["delegation"]
    assert d["provider"] == "openrouter"
    assert d["model"] == "google/gemini-3-flash"
    assert d["base_url"] == ""
    assert d["api_key"] == ""
    aux = cfg.get("auxiliary", {})
    entry = aux.get("delegation", {}) if isinstance(aux, dict) else {}
    assert not (isinstance(entry, dict) and entry.get("provider")), (
        "delegation routing leaked into auxiliary.delegation"
    )


def test_save_delegation_auto_stores_empty_provider(tmp_path, monkeypatch):
    """'auto' (inherit parent) persists as empty strings — never the literal
    'auto', which delegate_tool would resolve as a provider name."""
    _isolate_home(tmp_path, monkeypatch)

    _save_aux_choice(_DELEGATION_TASK_KEY, provider="openrouter", model="m")
    _save_aux_choice(_DELEGATION_TASK_KEY, provider="auto", model="", base_url="", api_key="")
    cfg = load_config()
    d = cfg["delegation"]
    assert d["provider"] == ""
    assert d["model"] == ""
    assert d["base_url"] == ""
    assert d["api_key"] == ""


def test_reset_aux_clears_delegation_routing_preserves_settings(tmp_path, monkeypatch):
    """Reset-all clears delegation provider/model/base_url/api_key but leaves
    non-routing delegation settings (max_concurrent_children, etc.) alone."""
    from hermes_cli.config import load_config as _lc, save_config

    _isolate_home(tmp_path, monkeypatch)

    cfg = _lc()
    cfg.setdefault("delegation", {})
    cfg["delegation"].update(
        {"provider": "openrouter", "model": "x", "max_concurrent_children": 7}
    )
    save_config(cfg)

    n = _reset_aux_to_auto()
    assert n >= 1

    cfg = _lc()
    d = cfg["delegation"]
    assert d["provider"] == ""
    assert d["model"] == ""
    assert d["max_concurrent_children"] == 7


def test_delegation_cfg_as_task_projection():
    """Projection renders empty provider as auto via _format_aux_current."""
    assert _format_aux_current(_delegation_cfg_as_task({})) == "auto"
    shaped = _delegation_cfg_as_task(
        {"delegation": {"provider": "nous", "model": "Hermes-4.5"}}
    )
    assert _format_aux_current(shaped) == "nous · Hermes-4.5"
    # Non-dict delegation section must not crash
    assert _format_aux_current(_delegation_cfg_as_task({"delegation": "bogus"})) == "auto"


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


# ── Set one provider/model for ALL auxiliary tasks ──────────────────────────


def test_apply_aux_choice_to_all_writes_every_task(tmp_path, monkeypatch):
    """Apply-all writes provider/model to every aux task AND the delegation section."""
    _isolate_home(tmp_path, monkeypatch)

    n = _apply_aux_choice_to_all(provider="openrouter", model="google/gemini-3-flash")
    assert n == len(_all_aux_tasks()) + 1  # built-ins + delegation

    cfg = load_config()
    aux = cfg["auxiliary"]
    for task, _name, _desc in _all_aux_tasks():
        entry = aux.get(task, {})
        assert entry.get("provider") == "openrouter", task
        assert entry.get("model") == "google/gemini-3-flash", task
    d = cfg["delegation"]
    assert d["provider"] == "openrouter"
    assert d["model"] == "google/gemini-3-flash"


def test_apply_aux_choice_to_all_preserves_timeouts(tmp_path, monkeypatch):
    """Apply-all overwrites only routing fields; per-task timeouts survive."""
    from hermes_cli.config import load_config as _lc, save_config

    _isolate_home(tmp_path, monkeypatch)
    cfg = _lc()
    cfg.setdefault("auxiliary", {}).setdefault("compression", {})["timeout"] = 999
    cfg["auxiliary"]["compression"]["extra_body"] = {"plugins": [{"id": "x"}]}
    save_config(cfg)

    _apply_aux_choice_to_all(provider="openrouter", model="m")

    cfg = _lc()
    comp = cfg["auxiliary"]["compression"]
    assert comp["provider"] == "openrouter"
    assert comp["model"] == "m"
    assert comp["timeout"] == 999
    assert comp["extra_body"] == {"plugins": [{"id": "x"}]}


def test_apply_aux_choice_to_all_auto_resets(tmp_path, monkeypatch):
    """provider='auto' resets every aux task to auto and delegation to empty."""
    _isolate_home(tmp_path, monkeypatch)

    _apply_aux_choice_to_all(provider="openrouter", model="m")
    _apply_aux_choice_to_all(provider="auto", model="", base_url="", api_key="")

    cfg = load_config()
    aux = cfg["auxiliary"]
    for task, _name, _desc in _all_aux_tasks():
        entry = aux.get(task, {})
        assert entry.get("provider") in {None, "", "auto"}, task
        assert not entry.get("model"), task
    d = cfg["delegation"]
    assert d["provider"] == ""
    assert d["model"] == ""


def test_apply_aux_choice_to_all_custom_endpoint(tmp_path, monkeypatch):
    """Apply-all custom endpoint writes base_url/api_key into every slot."""
    _isolate_home(tmp_path, monkeypatch)

    n = _apply_aux_choice_to_all(
        provider="custom",
        model="qwen2.5-coder",
        base_url="http://localhost:11434/v1",
        api_key="",
    )
    assert n == len(_all_aux_tasks()) + 1

    cfg = load_config()
    vision = cfg["auxiliary"]["vision"]
    assert vision["provider"] == "custom"
    assert vision["model"] == "qwen2.5-coder"
    assert vision["base_url"] == "http://localhost:11434/v1"
    assert cfg["delegation"]["base_url"] == "http://localhost:11434/v1"


def test_format_aux_all_current():
    """Summary shows 'auto', a shared value, or 'mixed' across all tasks."""
    assert _format_aux_all_current({}) == "auto"

    cfg: dict = {"auxiliary": {}}
    for task, _name, _desc in _all_aux_tasks():
        cfg["auxiliary"][task] = {"provider": "nous", "model": "Hermes-4.5"}
    cfg["delegation"] = {"provider": "nous", "model": "Hermes-4.5"}
    assert _format_aux_all_current(cfg) == "nous · Hermes-4.5"

    cfg["auxiliary"]["vision"]["provider"] = "openrouter"
    assert _format_aux_all_current(cfg) == "mixed (configured per task)"


def test_aux_config_menu_shows_set_all_entry(tmp_path, monkeypatch):
    """The aux menu surfaces 'Set one provider/model for all auxiliary tasks...'."""
    _isolate_home(tmp_path, monkeypatch)

    from hermes_cli import main as main_mod

    captured: list[list[str]] = []

    def fake_prompt(choices, *, default=0, title="Select provider:"):
        captured.append(list(choices))
        for i, label in enumerate(choices):
            if label == "Back":
                return i
        raise AssertionError("Back not in aux menu")

    monkeypatch.setattr(main_mod, "_prompt_provider_choice", fake_prompt)
    main_mod._aux_config_menu()

    assert captured, "aux menu never rendered"
    labels = captured[0]
    assert any(
        "Set one provider/model for all auxiliary tasks" in label for label in labels
    )
