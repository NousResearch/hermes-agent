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
from hermes_cli.main_provider_setup import _DELEGATION_TASK_KEY, _delegation_cfg_as_task, _format_aux_current, _reset_aux_to_auto, _save_aux_choice

# ── Default config ──────────────────────────────────────────────────────────

def test_aux_task_registries_cover_default_config():
    """Both aux-slot registries must cover every per-task block in the defaults.

    ``_AUX_TASKS`` drives the ``hermes model`` auxiliary picker;
    ``_AUX_TASK_SLOTS`` gates the dashboard REST assignment path and feeds
    ``_stale_aux_pins``. Both are hand-maintained and drifted before #125413:
    five slots were configurable only via config.yaml. Non-task keys
    (``transient_retries``, ``free_only``, ``openrouter_model``,
    ``stream_only_base_urls``) are excluded by shape — a task block is a dict
    carrying a ``provider`` key.
    """
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.main_provider_setup import _AUX_TASKS
    from hermes_cli.web_server_config import _AUX_TASK_SLOTS

    expected = {
        key for key, val in DEFAULT_CONFIG["auxiliary"].items()
        if isinstance(val, dict) and val.get("provider") is not None
    }
    cli_keys = {key for key, _name, _desc in _AUX_TASKS}
    rest_keys = set(_AUX_TASK_SLOTS)

    assert cli_keys == expected, (
        "hermes model picker out of sync with DEFAULT_CONFIG['auxiliary']: "
        f"missing={sorted(expected - cli_keys)} extra={sorted(cli_keys - expected)}"
    )
    assert rest_keys == expected, (
        "dashboard REST allowlist out of sync with DEFAULT_CONFIG['auxiliary']: "
        f"missing={sorted(expected - rest_keys)} extra={sorted(rest_keys - expected)}"
    )


def test_bulk_exclusion_registries_cannot_drift():
    """The CLI and REST bulk-exclusion sets must name exactly the MoA slots (#125435 review).

    Both are independent literals (the modules stay import-light); this pins them
    to each other AND to the MoA slots' presence in ``_AUX_TASK_SLOTS``, so adding
    a slot to one exclusion without the other fails here instead of silently
    resetting MoA pins on one path but not the other.
    """
    from hermes_cli.main_provider_setup import _BULK_EXCLUDED_AUX_TASKS
    from hermes_cli.web_server_config import _AUX_TASK_SLOTS, _BULK_EXCLUDED_AUX_SLOTS

    assert _BULK_EXCLUDED_AUX_TASKS == _BULK_EXCLUDED_AUX_SLOTS, (
        "CLI and REST bulk-exclusion sets drifted apart: "
        f"cli-only={sorted(_BULK_EXCLUDED_AUX_TASKS - _BULK_EXCLUDED_AUX_SLOTS)} "
        f"rest-only={sorted(_BULK_EXCLUDED_AUX_SLOTS - _BULK_EXCLUDED_AUX_TASKS)}"
    )
    assert _BULK_EXCLUDED_AUX_SLOTS <= set(_AUX_TASK_SLOTS), (
        "bulk-exclusion names a slot outside _AUX_TASK_SLOTS: "
        f"{sorted(_BULK_EXCLUDED_AUX_SLOTS - set(_AUX_TASK_SLOTS))}"
    )

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

def test_reset_aux_preserves_moa_pins(tmp_path, monkeypatch):
    """CLI reset-all must not collapse the MoA topology (#125435 review).

    The widened ``_AUX_TASKS`` added the MoA slots to the sweep: "Reset all to
    auto" would rewrite ``auxiliary.moa_reference``.``provider``/``model``/``base_url``
    to auto/empty — MoA reference/aggregator pins are topology, not task routing.
    """
    from hermes_cli.config import load_config as _lc, save_config

    _isolate_home(tmp_path, monkeypatch)

    cfg = _lc()
    aux = cfg.setdefault("auxiliary", {})
    aux["vision"] = {"provider": "openrouter", "model": "m1"}
    aux["moa_reference"] = {
        "provider": "ollama", "model": "ref-model",
        "base_url": "http://box:11434", "api_key": "sk-local", "timeout": 900,
    }
    aux["moa_aggregator"] = {"provider": "openrouter", "model": "agg-model", "timeout": 900}
    save_config(cfg)

    n = _reset_aux_to_auto()
    assert n >= 1  # vision was reset

    cfg = _lc()
    assert cfg["auxiliary"]["vision"]["provider"] == "auto"
    # MoA pins survive untouched, including the local endpoint + key + non-routing fields.
    # (load_config normalization may add inert keys like extra_body — check the routed fields.)
    ref = cfg["auxiliary"]["moa_reference"]
    assert (ref["provider"], ref["model"], ref["base_url"], ref["api_key"], ref["timeout"]) == (
        "ollama", "ref-model", "http://box:11434", "sk-local", 900)
    agg = cfg["auxiliary"]["moa_aggregator"]
    assert (agg["provider"], agg["model"], agg["timeout"]) == ("openrouter", "agg-model", 900)

def test_delegation_cfg_as_task_projection():
    """Projection renders empty provider as auto via _format_aux_current."""
    assert _format_aux_current(_delegation_cfg_as_task({})) == "auto"
    shaped = _delegation_cfg_as_task(
        {"delegation": {"provider": "nous", "model": "Hermes-4.5"}}
    )
    assert _format_aux_current(shaped) == "nous · Hermes-4.5"
    # Non-dict delegation section must not crash
    assert _format_aux_current(_delegation_cfg_as_task({"delegation": "bogus"})) == "auto"
