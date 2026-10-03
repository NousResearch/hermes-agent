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

from hermes_cli.config import load_config
from hermes_cli.main_provider_setup import (
    _DELEGATION_TASK_KEY,
    _aux_select_for_task,
    _delegation_cfg_as_task,
    _format_aux_current,
    _reset_aux_to_auto,
    _save_aux_choice,
)

# ── Default config ──────────────────────────────────────────────────────────

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


def test_micro_compaction_picker_auto_removes_route_overrides_preserves_tuning(tmp_path, monkeypatch):
    from hermes_cli import inventory
    from hermes_cli import main_provider_setup as setup
    from hermes_cli.config import save_config

    _isolate_home(tmp_path, monkeypatch)
    cfg = load_config()
    cfg["auxiliary"]["micro_compaction"] = {
        "provider": "custom",
        "model": "micro-model",
        "base_url": "https://micro.example/v1",
        "api_key": "micro-key",
        "reasoning_effort": "none",
        "timeout": 17,
    }
    save_config(cfg)
    monkeypatch.setattr(inventory, "build_aux_picker_rows", lambda **_kwargs: [])
    monkeypatch.setattr(inventory, "format_aux_picker_entries", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(setup, "_prompt_provider_choice", lambda _choices, default=0: 0)

    _aux_select_for_task("micro_compaction")

    micro = load_config()["auxiliary"]["micro_compaction"]
    assert not ({"provider", "model", "base_url", "api_key", "reasoning_effort"} & micro.keys())
    assert micro["timeout"] == 17


def test_micro_compaction_explicit_route_persists_reasoning_override(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)

    _save_aux_choice(
        "micro_compaction", provider="openrouter", model="fast-summary",
        reasoning_effort="high",
    )

    micro = load_config()["auxiliary"]["micro_compaction"]
    assert micro["provider"] == "openrouter"
    assert micro["model"] == "fast-summary"
    assert micro["reasoning_effort"] == "high"

# ── _reset_aux_to_auto ──────────────────────────────────────────────────────


def test_reset_aux_removes_micro_route_overrides_preserves_tuning(tmp_path, monkeypatch):
    from hermes_cli.config import save_config

    _isolate_home(tmp_path, monkeypatch)
    cfg = load_config()
    cfg["auxiliary"]["micro_compaction"] = {
        "provider": "openrouter",
        "model": "micro-model",
        "base_url": "https://micro.example/v1",
        "api_key": "micro-key",
        "reasoning_effort": "high",
        "timeout": 23,
    }
    save_config(cfg)

    assert _reset_aux_to_auto() >= 1

    micro = load_config()["auxiliary"]["micro_compaction"]
    assert not ({"provider", "model", "base_url", "api_key", "reasoning_effort"} & micro.keys())
    assert micro["timeout"] == 23

# ── Menu dispatch ───────────────────────────────────────────────────────────


def test_micro_compaction_auto_label_communicates_compression_inheritance(tmp_path, monkeypatch):
    from hermes_cli import inventory
    from hermes_cli import main_provider_setup as setup

    _isolate_home(tmp_path, monkeypatch)
    shown = []
    monkeypatch.setattr(inventory, "build_aux_picker_rows", lambda **_kwargs: [])
    monkeypatch.setattr(inventory, "format_aux_picker_entries", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(setup, "_prompt_provider_choice", lambda choices, default=0: shown.extend(choices))

    _aux_select_for_task("micro_compaction")

    assert shown[0].startswith("auto (inherit compression)")

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
