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

from copy import deepcopy

import pytest

from hermes_constants import VALID_REASONING_EFFORTS
from hermes_cli.config import DEFAULT_CONFIG, load_config, save_config
import hermes_cli.main_provider_setup as provider_setup
from hermes_cli.main_provider_setup import _AUX_TASKS, _DELEGATION_TASK_KEY, _delegation_cfg_as_task, _format_aux_current, _reset_aux_to_auto, _save_aux_choice


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
    import hermes_cli.main_provider_setup as hermes_cli_main_provider_setup

    captured: list[list[str]] = []

    def fake_prompt(choices, *, default=0):
        captured.append(list(choices))
        # Pick 'Leave unchanged' (last item) to exit cleanly
        for i, label in enumerate(choices):
            if label == "Leave unchanged":
                return i
        raise AssertionError("Leave unchanged not in provider list")

    monkeypatch.setattr(main_mod, "_prompt_provider_choice", fake_prompt)
    monkeypatch.setattr(hermes_cli_main_provider_setup, "_prompt_provider_choice", fake_prompt)

    main_mod.select_provider_and_model()

    assert captured, "provider menu never rendered"
    labels = captured[0]
    assert "Leave unchanged" in labels
    assert "Cancel" not in labels, "Cancel label should be replaced"
    assert any("Configure auxiliary models" in label for label in labels)


@pytest.mark.parametrize("route", ["auto", "provider", "custom"])
@pytest.mark.parametrize(
    ("effort_choice", "expected_effort", "cancelled"),
    [
        ("high", "high", False),
        ("none", "none", False),
        ("", "", False),
        (None, "low", False),
        (provider_setup._CANCELLED, "low", True),
    ],
)
def test_delegation_route_and_reasoning_are_saved_atomically(
    tmp_path, monkeypatch, route, effort_choice, expected_effort, cancelled
):
    """Every delegation route shares select/inherit/skip/cancel semantics."""
    _isolate_home(tmp_path, monkeypatch)
    cfg = load_config()
    cfg["model"] = {"default": "parent-model", "provider": "parent-provider"}
    cfg["auth"] = {"active_provider": "parent-provider"}
    cfg["delegation"].update(
        {
            "provider": "old-provider",
            "model": "old-model",
            "base_url": "https://old.example/v1",
            "api_key": "old-key",
            "reasoning_effort": "low",
            "max_concurrent_children": 7,
        }
    )
    save_config(cfg)
    original = deepcopy(load_config())

    route_label = {
        "auto": "auto (inherit main agent)",
        "provider": "OpenRouter",
        "custom": "Custom endpoint (direct URL)",
    }[route]

    def choose_route(choices, **_kwargs):
        return next(i for i, label in enumerate(choices) if route_label in label)

    monkeypatch.setattr(provider_setup, "_prompt_provider_choice", choose_route)
    monkeypatch.setattr("hermes_cli.inventory.build_aux_picker_rows", lambda **_kwargs: [object()])
    monkeypatch.setattr(
        "hermes_cli.inventory.format_aux_picker_entries",
        lambda *_args, **_kwargs: [("openrouter", "OpenRouter", ["child-model"])],
    )
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *_args, **_kwargs: "child-model")
    monkeypatch.setattr("hermes_cli.models_pricing.get_pricing_for_provider", lambda *_args: {})
    custom_answers = iter(["https://new.example/v1", "custom-model", "new-key"])
    monkeypatch.setattr(provider_setup, "_ask", lambda *_args, **_kwargs: next(custom_answers))

    effort_prompts = []

    def choose_effort(efforts, current_effort="", **kwargs):
        effort_prompts.append((tuple(efforts), current_effort, kwargs))
        return effort_choice

    monkeypatch.setattr(provider_setup, "_prompt_reasoning_effort_selection", choose_effort)

    provider_setup._aux_select_for_task(_DELEGATION_TASK_KEY)

    persisted = load_config()
    assert effort_prompts == [
        (
            tuple(VALID_REASONING_EFFORTS),
            "low",
            {"allow_inherit": True, "distinguish_cancel": True},
        )
    ]
    assert persisted["model"] == original["model"]
    assert persisted["auth"] == original["auth"]
    if cancelled:
        assert persisted["delegation"] == original["delegation"]
        return

    expected_routes = {
        "auto": {"provider": "", "model": "", "base_url": "", "api_key": ""},
        "provider": {
            "provider": "openrouter",
            "model": "child-model",
            "base_url": "",
            "api_key": "",
        },
        "custom": {
            "provider": "custom",
            "model": "custom-model",
            "base_url": "https://new.example/v1",
            "api_key": "new-key",
        },
    }
    delegation = persisted["delegation"]
    assert {key: delegation[key] for key in expected_routes[route]} == expected_routes[route]
    assert delegation["reasoning_effort"] == expected_effort
    assert delegation["max_concurrent_children"] == 7


@pytest.mark.parametrize(
    ("selected_label", "expected"),
    [
        (None, provider_setup._CANCELLED),
        ("Disable reasoning", "none"),
        ("Inherit parent", ""),
        ("Skip (keep current)", None),
    ],
)
def test_delegation_reasoning_prompt_distinguishes_cancel_inherit_and_skip(
    monkeypatch, selected_label, expected
):
    """The shared reasoning picker can expose delegation-only atomic choices."""
    def choose_reasoning(_title, choices, _default):
        if selected_label is None:
            return -1
        return next(i for i, label in enumerate(choices) if selected_label in label)

    monkeypatch.setattr(provider_setup, "_radiolist", choose_reasoning)

    result = provider_setup._prompt_reasoning_effort_selection(
        VALID_REASONING_EFFORTS,
        current_effort="low",
        allow_inherit=True,
        distinguish_cancel=True,
    )

    assert result is expected or result == expected
