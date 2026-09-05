"""In-chat (/model) model picker must show $/Mtok pricing + Nous sale chrome (#102990).

The standalone ``hermes model`` picker passes ``pricing=get_pricing_for_provider(slug)`` into
``_prompt_model_selection``; the prompt_toolkit in-chat picker previously rendered bare model ids.
These tests prove the chat path now builds the same ``_ModelPickerRows`` decoration when the
provider catalog is priced, while selection/resolution keep operating on bare model ids.
"""

from types import SimpleNamespace

import cli as cli_mod
import pytest

_MODELS = ["m-a", "m-b"]
_PRICING = {
    "m-a": {"prompt": "0.000006", "completion": "0.000018"},
    "m-b": {
        "prompt": "0.000012", "completion": "0.000036",
        "original": {"prompt": "0.000060", "completion": "0.000120"},
    },
}


def _bound(fn, instance):
    return fn.__get__(instance, type(instance))


def _chat_picker_state(stage="provider", **extra):
    state = {
        "stage": stage,
        "providers": [{"slug": "nous", "name": "Nous", "models": list(_MODELS)}],
        "selected": 0,
        "current_model": "m-a",
        "current_provider": "nous",
        "user_provs": None,
        "custom_provs": None,
        "filter": "",
    }
    state.update(extra)
    return state


def _select_provider(monkeypatch, fake_pricing, slug="nous"):
    """Drive the picker's Enter handler on the provider stage; returns the cli stub."""
    monkeypatch.setattr(
        "hermes_cli.models_pricing.get_pricing_for_provider",
        lambda provider, **kwargs: fake_pricing(provider) if callable(fake_pricing) else fake_pricing,
    )
    self_ = SimpleNamespace(
        _model_picker_state=_chat_picker_state(providers=[
            {"slug": slug, "name": "P", "models": list(_MODELS)}]),
        _invalidate=lambda **kwargs: None,
    )
    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()
    return self_


@pytest.mark.parametrize("slug,expected_extra", [("nous", True), ("openrouter", False)])
def test_provider_select_builds_pricing_rows_with_sale_chrome_only_for_nous(
        monkeypatch, slug, expected_extra):
    self_ = _select_provider(monkeypatch, dict(_PRICING), slug=slug)
    state = self_._model_picker_state
    assert state["stage"] == "model"
    rows = state["_model_price_rows"]
    assert rows is not None and rows.has_pricing
    sale_row = rows.label("m-b")
    assert "$12.00" in sale_row and "$36.00" in sale_row
    if expected_extra:
        assert "★ " in sale_row and "-80%" in sale_row
        assert "was $60.00/$120.00" in sale_row
    else:
        assert "★" not in sale_row and "-80%" not in sale_row
        assert "was " not in sale_row
    # Non-sale row keeps the price columns but no chrome.
    plain_row = rows.label("m-a")
    assert "$6.00" in plain_row and "$18.00" in plain_row
    assert "★" not in plain_row and "was " not in plain_row


def test_provider_select_without_pricing_catalog_keeps_plain_rows(monkeypatch):
    self_ = _select_provider(monkeypatch, lambda provider: {})
    state = self_._model_picker_state
    assert state["stage"] == "model"
    assert state["_model_price_rows"] is None


def test_pricing_fetch_failure_degrades_to_plain_rows(monkeypatch):
    def _boom(provider, **kwargs):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(
        "hermes_cli.models_pricing.get_pricing_for_provider", _boom)
    self_ = SimpleNamespace(
        _model_picker_state=_chat_picker_state(),
        _invalidate=lambda **kwargs: None,
    )
    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()
    assert self_._model_picker_state["_model_price_rows"] is None


def test_model_stage_fragments_show_pricing_but_resolution_stays_on_bare_ids(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.models_pricing.get_pricing_for_provider",
        lambda provider, **kwargs: dict(_PRICING))
    self_ = SimpleNamespace(
        _model_picker_state=_chat_picker_state(),
        _invalidate=lambda **kwargs: None,
    )
    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    captured = {}

    def _fake_render(state, title, hint, labels, **kwargs):
        captured["hint"] = hint
        captured["labels"] = list(labels)
        return []

    self_._render_scroll_list_panel = _fake_render
    self_._filter_model_picker_entries = cli_mod.HermesCLI._filter_model_picker_entries
    _bound(cli_mod.HermesCLI._get_model_picker_display_fragments, self_)()

    state = self_._model_picker_state
    # Decoration reached the panel rows (sale chrome for the Nous provider).
    assert any("$12.00" in label and "★ " in label and "-80%" in label
               for label in captured["labels"])
    assert captured["hint"].endswith("★ = on sale")
    assert "$/Mtok" in captured["hint"]
    # Back/Cancel appended after the priced rows.
    assert captured["labels"][-2:] == ["← Back", "Cancel"]
    # Resolution data is untouched: pairs carry bare ids and the Enter handler would map
    # the visible row back to exactly the original model id.
    assert state["_filtered_pairs"] == [(0, "m-a"), (1, "m-b")]
    assert [e for _i, e in state["_filtered_pairs"]] == ["m-a", "m-b"]

    # Filtering still matches bare ids and yields decorated labels for the survivors.
    state["filter"] = "m-b"
    _bound(cli_mod.HermesCLI._get_model_picker_display_fragments, self_)()
    assert state["_filtered_pairs"] == [(1, "m-b")]
    assert len(captured["labels"]) == 3  # 1 model + Back + Cancel
    assert "$36.00" in captured["labels"][0]


def test_model_stage_fragments_unpriced_catalog_shows_bare_ids(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.models_pricing.get_pricing_for_provider",
        lambda provider, **kwargs: {})
    self_ = SimpleNamespace(
        _model_picker_state=_chat_picker_state(),
        _invalidate=lambda **kwargs: None,
    )
    _bound(cli_mod.HermesCLI._handle_model_picker_selection, self_)()

    captured = {}
    self_._render_scroll_list_panel = (
        lambda state, title, hint, labels, **kwargs: captured.update(
            hint=hint, labels=list(labels)) or [])
    self_._filter_model_picker_entries = cli_mod.HermesCLI._filter_model_picker_entries
    _bound(cli_mod.HermesCLI._get_model_picker_display_fragments, self_)()
    assert captured["labels"] == ["m-a", "m-b", "← Back", "Cancel"]
    assert "$" not in captured["hint"]
