"""Picker search aliases for brand-less wire model ids."""

from hermes_cli.curses_ui import _filter_indices
from hermes_cli.model_search import model_search_text


def test_model_search_text_keeps_ordinary_ids():
    assert model_search_text("kimi-k2.6") == "kimi-k2.6"
    assert model_search_text("glm-5.2") == "glm-5.2"


def test_model_search_text_adds_kimi_aliases_for_k3():
    assert model_search_text("k3") == "k3 kimi-k3 kimi"
    assert model_search_text("K3") == "K3 kimi-k3 kimi"


def test_filter_indices_surfaces_k3_for_kimi_query():
    models = ["kimi-k2.6", "kimi-k2.5", "k3", "kimi-for-coding"]
    haystacks = [model_search_text(m) for m in models]
    ranked = [models[i] for i in _filter_indices(haystacks, "kimi")]
    assert "k3" in ranked


def test_filter_indices_still_finds_k3_by_wire_id():
    models = ["kimi-k2.6", "k3", "kimi-for-coding"]
    haystacks = [model_search_text(m) for m in models]
    ranked = [models[i] for i in _filter_indices(haystacks, "k3")]
    assert ranked == ["k3"]
