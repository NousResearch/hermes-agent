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
    _AUX_TASKS, _DELEGATION_TASK_KEY, _delegation_cfg_as_task, _embedding_picker_rows,
    _format_aux_current, _format_embedding_picker_entries, _reset_aux_to_auto,
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


def test_embedding_route_has_capability_specific_default_and_picker(tmp_path, monkeypatch):
    """Embedding auto stays independent of chat and the picker only offers supported auth routes."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    _isolate_home(tmp_path, monkeypatch)
    assert "embedding" in {task for task, _name, _description in _AUX_TASKS}
    default = DEFAULT_CONFIG["auxiliary"]["embedding"]
    assert (default["provider"], default["model"], default["dimensions"]) == ("auto", "", None)

    rows = [
        {"slug": "openai-api", "name": "OpenAI API", "models": ["gpt-4.1"]},
        {"slug": "openrouter", "name": "OpenRouter", "models": ["openai/text-embedding-3-small"]},
        {"slug": "openai-codex", "name": "Codex OAuth", "models": ["gpt-5"]},
        {"slug": "anthropic", "name": "Anthropic", "models": ["claude-sonnet"]},
        {"slug": "custom:vectorbox", "name": "Vectorbox", "models": ["embed-v2"]},
        {"slug": "custom:claudegateway", "name": "Claude gateway", "models": ["embed-v2"]},
    ]
    config = {"providers": {
        "vectorbox": {"name": "Vectorbox", "api": "https://vectors.example/v1"},
        "claudegateway": {
            "name": "Claude gateway", "api": "https://gateway.example/anthropic",
        },
    }}
    selectable = _embedding_picker_rows(rows, config)
    assert [row["slug"] for row in selectable] == ["openai-api", "openrouter", "custom:vectorbox"]
    options = _format_embedding_picker_entries(selectable, current_provider="openrouter")
    assert all(models == [] for _, _, models in options)
    assert "enter an embedding model" in options[1][1]


def test_embedding_route_save_and_reset_preserve_capability_defaults(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    _save_aux_choice("embedding", provider="openrouter", model="openai/text-embedding-3-small")
    cfg = load_config()
    assert cfg["auxiliary"]["embedding"]["provider"] == "openrouter"
    assert cfg["auxiliary"]["embedding"]["model"] == "openai/text-embedding-3-small"
    assert cfg["auxiliary"]["embedding"]["api_key"] == ""
    assert cfg["model"] == DEFAULT_CONFIG["model"]

    cfg["auxiliary"]["embedding"]["dimensions"] = 512
    from hermes_cli.config import save_config
    save_config(cfg)
    assert _reset_aux_to_auto() >= 1
    reset = load_config()["auxiliary"]["embedding"]
    assert (reset["provider"], reset["model"], reset["dimensions"]) == ("auto", "", None)

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
