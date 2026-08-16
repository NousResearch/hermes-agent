"""Tests for ``model.picker`` hide + order preferences.

The interactive ``/model`` picker honors two optional, purely-cosmetic config
knobs::

    model:
      picker:
        hide:  [openai-api, "claude-apx-*"]
        order: [anthropic, openai-codex]

These tests pin the behavior contract rather than any current provider list:

- hide drops matching rows (exact slug *or* glob), except the active provider
- order front-anchors listed slugs and leaves everything else stably behind
- hide and order are independent (an unlisted row is not hidden by `order`)
- malformed / missing config is always a no-op, never an exception
- every picker surface applies the same prefs (CLI/Discord and payload pickers
  share one helper, so they cannot drift)
"""

import pytest

from hermes_cli import inventory
from hermes_cli import model_switch
from hermes_cli.model_switch import _apply_picker_preferences


def _row(slug, models=("m1",)):
    return {"slug": slug, "models": list(models), "total_models": len(models)}


def _slugs(rows):
    return [r["slug"] for r in rows]


@pytest.fixture
def cfg(monkeypatch):
    """Install a config dict that `_apply_picker_preferences` will read."""

    def _install(picker):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"model": {"picker": picker}},
        )

    return _install


# ─── hide: exact slugs ──────────────────────────────────────────────────


def test_hide_drops_exact_slug(cfg):
    cfg({"hide": ["openai-api"]})
    rows = [_row("anthropic"), _row("openai-api"), _row("openrouter")]
    assert _slugs(_apply_picker_preferences(rows)) == ["anthropic", "openrouter"]


def test_hide_is_case_insensitive(cfg):
    cfg({"hide": ["OpenAI-API"]})
    rows = [_row("anthropic"), _row("openai-api")]
    assert _slugs(_apply_picker_preferences(rows)) == ["anthropic"]


def test_hide_never_drops_the_current_provider(cfg):
    """You must always be able to see and switch off what you're on."""
    cfg({"hide": ["openai-api"]})
    rows = [_row("anthropic"), _row("openai-api")]
    out = _apply_picker_preferences(rows, current_provider="openai-api")
    assert _slugs(out) == ["anthropic", "openai-api"]


def test_hide_ignores_unknown_slugs(cfg):
    cfg({"hide": ["not-a-real-provider"]})
    rows = [_row("anthropic"), _row("openrouter")]
    assert _slugs(_apply_picker_preferences(rows)) == ["anthropic", "openrouter"]


# ─── hide: glob patterns ────────────────────────────────────────────────


def test_hide_glob_collapses_a_provider_family(cfg):
    """One glob line replaces listing every numbered failover lane."""
    cfg({"hide": ["claude-apx-*"]})
    rows = [
        _row("openrouter"),
        _row("claude-apx-0"),
        _row("claude-apx-1"),
        _row("claude-apx-10"),
        _row("claude-apr"),  # not a *-N lane, must be kept
    ]
    assert _slugs(_apply_picker_preferences(rows)) == ["openrouter", "claude-apr"]


def test_hide_glob_and_exact_entries_coexist(cfg):
    cfg({"hide": ["claude-apx-*", "openai-api"]})
    rows = [_row("claude-apx-0"), _row("openai-api"), _row("anthropic")]
    assert _slugs(_apply_picker_preferences(rows)) == ["anthropic"]


def test_hide_glob_still_spares_the_current_provider(cfg):
    cfg({"hide": ["claude-apx-*"]})
    rows = [_row("claude-apx-0"), _row("claude-apx-1"), _row("anthropic")]
    out = _apply_picker_preferences(rows, current_provider="claude-apx-1")
    assert _slugs(out) == ["claude-apx-1", "anthropic"]


def test_hide_suffix_glob(cfg):
    cfg({"hide": ["*-preview"]})
    rows = [_row("gemini-preview"), _row("gemini"), _row("anthropic")]
    assert _slugs(_apply_picker_preferences(rows)) == ["gemini", "anthropic"]


# ─── order ──────────────────────────────────────────────────────────────


def test_order_front_anchors_listed_slugs(cfg):
    cfg({"order": ["openrouter", "anthropic"]})
    rows = [_row("anthropic"), _row("openai-api"), _row("openrouter")]
    assert _slugs(_apply_picker_preferences(rows)) == [
        "openrouter",
        "anthropic",
        "openai-api",
    ]


def test_order_keeps_unlisted_rows_in_original_relative_order(cfg):
    cfg({"order": ["zzz"]})
    rows = [_row("a"), _row("b"), _row("c")]
    assert _slugs(_apply_picker_preferences(rows)) == ["a", "b", "c"]


def test_order_does_not_hide_unlisted_rows(cfg):
    """Ordering and hiding are independent knobs."""
    cfg({"order": ["anthropic"]})
    rows = [_row("openai-api"), _row("anthropic")]
    assert set(_slugs(_apply_picker_preferences(rows))) == {"openai-api", "anthropic"}


def test_order_ignores_blank_entries_without_collision(cfg):
    """A blank entry must not collide with a real rank and reshuffle rows."""
    cfg({"order": ["", "anthropic", "  "]})
    rows = [_row("openai-api"), _row("anthropic"), _row("openrouter")]
    assert _slugs(_apply_picker_preferences(rows))[0] == "anthropic"


def test_hide_and_order_compose(cfg):
    cfg({"hide": ["openai-api"], "order": ["openrouter"]})
    rows = [_row("anthropic"), _row("openai-api"), _row("openrouter")]
    assert _slugs(_apply_picker_preferences(rows)) == ["openrouter", "anthropic"]


# ─── robustness: never raise into the picker path ───────────────────────


@pytest.mark.parametrize(
    "picker",
    [
        {},
        {"hide": None},
        {"hide": "not-a-list"},
        {"order": "not-a-list"},
        {"hide": [None, ""]},
        "not-a-dict",
    ],
)
def test_malformed_config_is_a_noop(cfg, picker):
    cfg(picker)
    rows = [_row("a"), _row("b")]
    assert _slugs(_apply_picker_preferences(rows)) == ["a", "b"]


def test_config_load_failure_is_a_noop(monkeypatch):
    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    rows = [_row("a"), _row("b")]
    assert _slugs(_apply_picker_preferences(rows)) == ["a", "b"]


# ─── surface parity: payload pickers apply the same prefs ───────────────


def test_payload_helper_delegates_to_the_shared_picker_helper(cfg):
    """`inventory` must not grow a second, drift-prone implementation."""
    cfg({"hide": ["openai-api"], "order": ["openrouter"]})
    rows = [_row("anthropic"), _row("openai-api"), _row("openrouter")]
    assert inventory._apply_picker_prefs_rows(rows) == _apply_picker_preferences(rows)


def test_payload_helper_survives_a_broken_shared_helper(monkeypatch):
    """Best-effort: a failure in the shared helper must not break the picker."""

    def _boom(*a, **kw):
        raise RuntimeError("nope")

    monkeypatch.setattr(model_switch, "_apply_picker_preferences", _boom)
    rows = [_row("a"), _row("b")]
    assert _slugs(inventory._apply_picker_prefs_rows(rows)) == ["a", "b"]


def test_build_models_payload_exposes_the_pref_switch():
    """The knob exists and defaults OFF for non-picker consumers."""
    import inspect

    sig = inspect.signature(inventory.build_models_payload)
    assert "apply_picker_prefs" in sig.parameters
    assert sig.parameters["apply_picker_prefs"].default is False
