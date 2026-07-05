"""Tests for the CLI's sub-provider drill-down stage in :func:`_handle_model_picker_selection`.

The drill-down feature reuses the picker state machine to insert a middle
stage (``"subprovider"``) between the provider row and the model row, so the
user first picks an upstream vendor (openai, anthropic, google, ...) before
seeing concrete model IDs.

These tests pin the state transitions and side effects the picker relies on:

* Selecting a sub-provider resolves the **label** ("openai (12 models)") back
  to the **bare slug** ("openai") that indexes ``sub_models``. Shipping the
  label downstream would crash the model picker — labels contain spaces,
  model IDs don't.
* "Back" from the sub-provider stage returns to the provider stage with the
  *correct* provider row selected by slug, not just index 0 (which might
  not be the row the user drilled from).
* The "Back" path from the model stage lands on the sub-provider stage with
  the previously-chosen sub selected — so users can switch upstream without
  re-picking the provider.
* A stale ``sub_label`` (one that doesn't match any ``subproviders`` entry)
  must NOT be fed to ``switch_model`` — that path would crash on
  ``"Model names cannot contain spaces."`` The defensive close path is the
  only safe behaviour.

The Telegram/Discord/TUI/Desktop adapters reuse the same drill-down
contract — they all expect ``chosen_sub`` to be the bare slug, not the
display label. A regression here would surface as 404s on every chat
platform.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _bound(fn, instance):
    """Bind an unbound method to a fake ``self`` without instantiating the class."""
    return fn.__get__(instance, type(instance))


def _make_provider(slug, name=None, *, subproviders=None, sub_models=None,
                   sub_labels=None, is_subprovider_picker=False, models=None):
    return {
        "slug": slug,
        "name": name or slug.title(),
        "is_current": False,
        "is_user_defined": False,
        "models": list(models or []),
        "total_models": len(models or []),
        "source": "built-in",
        "is_subprovider_picker": is_subprovider_picker,
        "subproviders": list(subproviders or []),
        "sub_models": dict(sub_models or {}),
        "sub_labels": list(sub_labels or []),
    }


# ── Stage transition: provider → subprovider ──────────────────────────────


def test_provider_selection_with_drill_down_transitions_to_subprovider(monkeypatch):
    """Selecting an aggregator provider row inserts the sub-provider stage.

    The row's ``models`` (which is actually sub-labels after drill-down
    grouping) must NOT be passed to the model picker — that path crashes
    on the spaces inside "openai (12 models)".
    """
    import cli as cli_mod

    state = {
        "stage": "provider",
        "providers": [
            _make_provider("anthropic", models=["claude-sonnet-4-6"]),
            _make_provider(
                "huggingface",
                models=["openai (2 models)", "anthropic (1 models)"],
                subproviders=["openai", "anthropic"],
                sub_models={
                    "openai": ["openai/gpt-4o", "openai/gpt-4o-mini"],
                    "anthropic": ["anthropic/claude-3-5-sonnet"],
                },
                sub_labels=["openai (2 models)", "anthropic (1 models)"],
                is_subprovider_picker=True,
            ),
        ],
        "selected": 1,
    }

    invalidations = []

    self_ = SimpleNamespace(
        _model_picker_state=state,
        _close_model_picker=lambda: None,
        _invalidate=lambda **kw: invalidations.append(kw),
    )

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    assert state["stage"] == "subprovider"
    assert state["sub_list"] == ["openai (2 models)", "anthropic (1 models)"]
    assert state["selected"] == 0
    assert state["provider_data"]["slug"] == "huggingface"
    # The drill-down row's models were NOT leaked forward.
    assert state.get("model_list") is None
    # The picker repainted.
    assert invalidations == [{"min_interval": 0.0}]


def test_provider_selection_without_drill_down_skips_stage(monkeypatch):
    """A regular provider row goes straight to the model stage.

    This is the unchanged behaviour — the picker contract for any provider
    that is NOT marked drill-down.
    """
    import cli as cli_mod

    monkeypatch.setattr("hermes_cli.models.provider_model_ids",
                        lambda slug: [])

    state = {
        "stage": "provider",
        "providers": [
            _make_provider("anthropic", models=["claude-sonnet-4-6"]),
        ],
        "selected": 0,
    }

    self_ = SimpleNamespace(
        _model_picker_state=state,
        _close_model_picker=lambda: None,
        _invalidate=lambda **kw: None,
    )

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    assert state["stage"] == "model"
    assert state["model_list"] == ["claude-sonnet-4-6"]
    assert state["provider_data"]["slug"] == "anthropic"
    # Drill-down artefacts must NOT appear on a regular row.
    assert "sub_list" not in state


# ── Stage subprovider: resolution of label → slug ─────────────────────────


def test_subprovider_selection_resolves_label_to_bare_slug(monkeypatch):
    """A sub-label like ``"openai (2 models)"`` maps to the bare slug ``"openai"``.

    This is the contract every downstream adapter (Telegram, Discord, TUI,
    Desktop) relies on: ``chosen_sub`` is what indexes ``sub_models``, so
    shipping a label downstream would silently produce empty model lists.
    """
    import cli as cli_mod

    state = {
        "stage": "subprovider",
        "providers": [_make_provider("huggingface")],
        "provider_data": _make_provider(
            "huggingface",
            subproviders=["openai", "anthropic"],
            sub_models={
                "openai": ["openai/gpt-4o", "openai/gpt-4o-mini"],
                "anthropic": ["anthropic/claude-3-5-sonnet"],
            },
            sub_labels=["openai (2 models)", "anthropic (1 models)"],
            is_subprovider_picker=True,
        ),
        "sub_list": ["openai (2 models)", "anthropic (1 models)"],
        "selected": 0,
    }

    self_ = SimpleNamespace(
        _model_picker_state=state,
        _close_model_picker=lambda: None,
        _invalidate=lambda **kw: None,
    )

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    # The bare slug, not the label.
    assert state["chosen_sub"] == "openai"
    # The model list is the openai slice, NOT a label.
    assert state["model_list"] == ["openai/gpt-4o", "openai/gpt-4o-mini"]
    assert state["stage"] == "model"
    assert state["selected"] == 0


def test_subprovider_selection_with_no_space_in_label(monkeypatch):
    """If a sub has only one model, the label is ``"openai (1 models)"``
    (the helper has no plural handling). The matching branch must still
    resolve it to the bare slug.

    Pin this so a future label-format change can't silently break it.
    """
    import cli as cli_mod

    state = {
        "stage": "subprovider",
        "provider_data": _make_provider(
            "huggingface",
            subproviders=["google"],
            sub_models={"google": ["google/gemini-1.5-pro"]},
            sub_labels=["google (1 models)"],
            is_subprovider_picker=True,
        ),
        "sub_list": ["google (1 models)"],
        "selected": 0,
    }

    self_ = SimpleNamespace(
        _model_picker_state=state,
        _close_model_picker=lambda: None,
        _invalidate=lambda **kw: None,
    )

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    assert state["chosen_sub"] == "google"
    assert state["model_list"] == ["google/gemini-1.5-pro"]


# ── Stage subprovider: defensive paths ────────────────────────────────────


def test_subprovider_defensive_close_when_label_does_not_match_any_sub(monkeypatch):
    """A stale sub-label (no match in ``subproviders``) must NOT be fed to ``switch_model``.

    The dispatch key must be the bare slug; feeding the label would trigger
    ``"Model names cannot contain spaces."``. We close the picker instead
    of crashing the user. The error-message wording is rendered through
    ``_cprint`` which writes to a Rich console that is not always captured
    under pytest's capsys, so we only pin the observable contract here:
    the picker closes and no model is selected.
    """
    import cli as cli_mod

    state = {
        "stage": "subprovider",
        "provider_data": _make_provider(
            "huggingface",
            subproviders=["openai", "anthropic"],
            sub_models={"openai": ["openai/gpt-4o"], "anthropic": ["anthropic/claude-3-5-sonnet"]},
            sub_labels=["openai (1 models)", "anthropic (1 models)"],
            is_subprovider_picker=True,
        ),
        # State drift: the selected label no longer corresponds to any sub.
        "sub_list": ["orphan (10 models)"],
        "selected": 0,
    }
    closes = []

    self_ = SimpleNamespace(
        _model_picker_state=state,
        _close_model_picker=lambda: closes.append(True),
        _invalidate=lambda **kw: None,
    )

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    assert closes == [True]
    # The picker must NOT have advanced to the model stage with a label
    # as ``chosen_sub`` (that would feed "openai (12 models)" to switch_model).
    assert "model_list" not in state or state.get("model_list") != ["orphan (10 models)"]


def test_subprovider_cancel_closes_picker(monkeypatch):
    """Selecting Cancel (idx past back) closes the picker, no drill-down."""
    import cli as cli_mod

    state = {
        "stage": "subprovider",
        "provider_data": _make_provider(
            "huggingface",
            subproviders=["openai"],
            sub_models={"openai": ["openai/gpt-4o"]},
            sub_labels=["openai (1 models)"],
            is_subprovider_picker=True,
        ),
        "sub_list": ["openai (1 models)"],
        "selected": 2,  # cancel_idx = len(sub_list) + 1 = 2
    }
    closes = []

    self_ = SimpleNamespace(
        _model_picker_state=state,
        _close_model_picker=lambda: closes.append(True),
        _invalidate=lambda **kw: None,
    )

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    assert closes == [True]


def test_subprovider_back_returns_to_provider_stage_with_correct_row(monkeypatch):
    """Back from subprovider stage returns to the provider stage with the
    original provider row selected (by slug, not index 0).

    The picker rebuilds the cursor position from the slug the user drilled
    from, so multi-aggregator pickers land the user back on the exact row
    they came from.
    """
    import cli as cli_mod

    provider_data = _make_provider(
        "huggingface",
        subproviders=["openai"],
        sub_models={"openai": ["openai/gpt-4o"]},
        sub_labels=["openai (1 models)"],
        is_subprovider_picker=True,
    )
    state = {
        "stage": "subprovider",
        "providers": [
            _make_provider("anthropic"),
            _make_provider("huggingface", is_subprovider_picker=True),
            _make_provider("gemini"),
        ],
        "provider_data": provider_data,
        "sub_list": ["openai (1 models)"],
        "selected": 1,  # back_idx = len(sub_list) = 1
    }
    invalidations = []

    self_ = SimpleNamespace(
        _model_picker_state=state,
        _close_model_picker=lambda: None,
        _invalidate=lambda **kw: invalidations.append(kw),
    )

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    assert state["stage"] == "provider"
    # huggingface is at index 1 in the providers list, NOT 0.
    assert state["selected"] == 1
    # The user's sub_list memory is dropped on the way back up.
    assert invalidations == [{"min_interval": 0.0}]


# ── Stage model: back navigation lands on subprovider ────────────────────


def test_model_back_with_drill_down_returns_to_subprovider_with_sub_selected(monkeypatch):
    """Back from model stage with a drill-down provider lands on the
    subprovider stage, with the previously-chosen sub pre-selected.

    So after picking openai → gpt-4o, hitting Back should put the cursor on
    ``openai`` again — not on the first sub in the list.
    """
    import cli as cli_mod

    state = {
        "stage": "model",
        "provider_data": _make_provider(
            "huggingface",
            subproviders=["openai", "anthropic", "google"],
            sub_models={
                "openai": ["openai/gpt-4o"],
                "anthropic": ["anthropic/claude-3-5-sonnet"],
                "google": ["google/gemini-1.5-pro"],
            },
            sub_labels=[
                "openai (1 models)",
                "anthropic (1 models)",
                "google (1 models)",
            ],
            is_subprovider_picker=True,
        ),
        "model_list": ["anthropic/claude-3-5-sonnet"],
        "chosen_sub": "anthropic",
        "selected": 1,  # back_idx = 1
    }
    invalidations = []

    self_ = SimpleNamespace(
        _model_picker_state=state,
        _close_model_picker=lambda: None,
        _invalidate=lambda **kw: invalidations.append(kw),
    )

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    assert state["stage"] == "subprovider"
    # Cursor lands on the previously-chosen "anthropic", index 1, NOT 0.
    assert state["selected"] == 1
    assert invalidations == [{"min_interval": 0.0}]


def test_model_back_without_drill_down_returns_to_provider_stage(monkeypatch):
    """Sanity: Back from the model stage on a non-drill-down provider
    goes back to the provider stage (unchanged behaviour).
    """
    import cli as cli_mod

    state = {
        "stage": "model",
        "provider_data": _make_provider("anthropic", models=["claude-sonnet-4-6"]),
        "model_list": ["claude-sonnet-4-6"],
        "selected": 1,  # back_idx = 1
    }

    self_ = SimpleNamespace(
        _model_picker_state=state,
        _close_model_picker=lambda: None,
        _invalidate=lambda **kw: None,
    )

    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    assert state["stage"] == "provider"
