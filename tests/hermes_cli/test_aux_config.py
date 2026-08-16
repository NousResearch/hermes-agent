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
