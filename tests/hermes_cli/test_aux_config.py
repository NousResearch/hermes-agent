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

from hermes_cli.config import DEFAULT_CONFIG, get_env_value, load_config
from hermes_cli.main import (
    _AUX_TASKS,
    _DELEGATION_TASK_KEY,
    _aux_flow_custom_endpoint,
    _delegation_cfg_as_task,
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
    assert not v.get("api_key")
    assert not v.get("key_env")


def test_aux_flow_custom_endpoint_persists_key_env_not_secret(tmp_path, monkeypatch):
    """CLI auxiliary custom setup must not write the API key into config.yaml."""
    from pathlib import Path

    from hermes_cli.config import get_env_value, load_config
    from hermes_cli.main import _aux_flow_custom_endpoint

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    answers = iter(["http://127.0.0.1:11434/v1", "gemma"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(answers))
    monkeypatch.setattr(
        "hermes_cli.secret_prompt.masked_secret_prompt",
        lambda *_a, **_k: "super-secret-aux-key",
    )

    _aux_flow_custom_endpoint("vision", {})

    cfg = load_config()
    slot = cfg["auxiliary"]["vision"]
    dumped = (home / "config.yaml").read_text(encoding="utf-8")
    assert "super-secret-aux-key" not in dumped
    assert not slot.get("api_key")
    assert not slot.get("api")
    assert str(slot.get("key_env") or "").startswith("HERMES_CUSTOM_")
    assert get_env_value(slot["key_env"]) == "super-secret-aux-key"
    assert slot["provider"] == "custom"
    assert slot["base_url"] == "http://127.0.0.1:11434/v1"
    assert slot["model"] == "gemma"


def test_aux_flow_custom_rotation_without_key_clears_key_env(tmp_path, monkeypatch):
    """A→B without a replacement key must not keep the previous key_env."""
    from pathlib import Path

    from hermes_cli.main import _aux_flow_custom_endpoint, _save_aux_choice

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    _save_aux_choice(
        "vision",
        provider="custom",
        model="gemma",
        base_url="http://127.0.0.1:11434/v1",
        key_env="HERMES_CUSTOM_127_0_0_1_11434_API_KEY",
    )

    answers = iter(["http://evil.example:8080/v1", "gemma"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(answers))
    monkeypatch.setattr(
        "hermes_cli.secret_prompt.masked_secret_prompt",
        lambda *_a, **_k: "",
    )

    _aux_flow_custom_endpoint(
        "vision",
        {
            "provider": "custom",
            "model": "gemma",
            "base_url": "http://127.0.0.1:11434/v1",
            "key_env": "HERMES_CUSTOM_127_0_0_1_11434_API_KEY",
        },
    )

    slot = load_config()["auxiliary"]["vision"]
    assert slot["base_url"] == "http://evil.example:8080/v1"
    assert not slot.get("key_env")
    assert not slot.get("api_key")


def test_save_delegation_custom_endpoint_persists_key_env_not_secret(
    tmp_path, monkeypatch
):
    """The special top-level delegation slot follows aux secret hygiene."""
    from pathlib import Path

    from hermes_cli.config import load_config

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    answers = iter(["https://delegation.example/v1", "delegation-model"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(answers))
    monkeypatch.setattr(
        "hermes_cli.secret_prompt.masked_secret_prompt",
        lambda *_a, **_k: "delegation-secret",
    )

    _aux_flow_custom_endpoint(_DELEGATION_TASK_KEY, {})

    cfg = load_config()
    delegation = cfg["delegation"]
    dumped = (home / "config.yaml").read_text(encoding="utf-8")
    assert str(delegation.get("key_env") or "").startswith("HERMES_CUSTOM_")
    assert not delegation.get("api_key")
    assert "api_key_env" not in delegation
    assert "delegation-model" in dumped
    assert "delegation-secret" not in dumped
    assert get_env_value(delegation["key_env"]) == "delegation-secret"




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
