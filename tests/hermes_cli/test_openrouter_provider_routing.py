"""Provider-routing step of the OpenRouter model picker.

Contracts covered:
- a pin is written to ``provider_routing.models.<model>`` and nothing else (other models and the
  flat keys survive, unmanaged keys of the same entry survive);
- routing written there reaches the request body of the next turn
  (``chat_completion_helpers._provider_preferences_for_agent``);
- the picker step is skipped — and writes nothing — for a single-endpoint model, for a non-OpenRouter
  provider, and when the user confirms with nothing checked.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import hermes_cli.provider_routing_picker as prp
from hermes_cli.cli_model_switch_mixin import _picker_offers_routing
from hermes_cli.provider_routing_picker import (
    _clean_entry,
    configure_after_selection,
    existing_model_routing,
    prompt_provider_routing,
    provider_rows,
    row_label,
    save_model_routing,
)


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write_config(home, """
        model:
          default: deepseek/deepseek-v4.1-flash
          provider: openrouter
    """)
    return home


def _write_config(home: Path, body: str) -> None:
    (home / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def _read_config(home: Path) -> dict:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


# --------------------------------------------------------------------------- write contract


def test_pin_lands_only_on_the_picked_model(config_home):
    _write_config(config_home, """
        model:
          default: deepseek/deepseek-v4.1-flash
        provider_routing:
          sort: price
          models:
            qwen/qwen3.7-flash:
              order: [alibaba]
    """)

    save_model_routing("deepseek/deepseek-v4.1-flash", {"order": ["DeepSeek", "alibaba"], "sort": "throughput"})

    pr = _read_config(config_home)["provider_routing"]
    assert pr["sort"] == "price", "flat keys are not this screen's to touch"
    assert pr["models"]["qwen/qwen3.7-flash"] == {"order": ["alibaba"]}
    assert pr["models"]["deepseek/deepseek-v4.1-flash"] == {
        "order": ["deepseek", "alibaba"], "sort": "throughput"}


def test_unmanaged_keys_of_the_entry_survive(config_home):
    _write_config(config_home, """
        provider_routing:
          models:
            deepseek/deepseek-v4.1-flash:
              only: [deepseek]
              require_parameters: true
    """)

    save_model_routing("deepseek/deepseek-v4.1-flash", {"order": ["deepseek"], "sort": "price"})

    assert _read_config(config_home)["provider_routing"]["models"]["deepseek/deepseek-v4.1-flash"] == {
        "only": ["deepseek"], "require_parameters": True, "order": ["deepseek"], "sort": "price"}


def test_clearing_removes_the_entry_and_empty_sections(config_home):
    _write_config(config_home, """
        provider_routing:
          sort: price
          models:
            deepseek/deepseek-v4.1-flash:
              order: [deepseek]
    """)

    assert save_model_routing("deepseek/deepseek-v4.1-flash", {}) == {}
    assert _read_config(config_home)["provider_routing"] == {"sort": "price"}

    # Last model gone -> its `models` map goes too, while the flat keys stay.
    save_model_routing("deepseek/deepseek-v4.1-flash", {"order": ["alibaba"]})
    save_model_routing("deepseek/deepseek-v4.1-flash", None)
    assert _read_config(config_home)["provider_routing"] == {"sort": "price"}


def test_last_section_removal_drops_the_empty_root(config_home):
    save_model_routing("deepseek/deepseek-v4.1-flash", {"order": ["alibaba"]})
    save_model_routing("deepseek/deepseek-v4.1-flash", {})

    assert "provider_routing" not in _read_config(config_home)


def test_clearing_matches_a_spelling_variant_of_the_same_model(config_home):
    _write_config(config_home, """
        provider_routing:
          models:
            deepseek-4-1-flash:
              order: [deepseek]
    """)

    save_model_routing("deepseek/deepseek-4.1-flash", {})

    assert _read_config(config_home).get("provider_routing", {}) == {}


@pytest.mark.parametrize("raw,expected", [
    ({"order": ["A", "a", "", "  "], "sort": "PRICE"}, {"order": ["a"], "sort": "price"}),
    ({"sort": "bogus"}, {}),
    ({"only": "deepseek"}, {"only": ["deepseek"]}),
    ({"data_collection": None, "require_parameters": False}, {}),
])
def test_entry_normalization(raw, expected):
    assert _clean_entry(raw) == expected


def test_pin_reaches_the_request_body(config_home):
    """The whole point: what the screen writes must show up in ``extra_body['provider']``."""
    from agent.chat_completion_helpers import _provider_preferences_for_agent

    _write_config(config_home, """
        model:
          default: deepseek/deepseek-v4.1-flash
        provider_routing:
          sort: price
    """)
    save_model_routing("deepseek/deepseek-v4.1-flash", {"order": ["deepseek"], "sort": "latency"})

    agent = SimpleNamespace(
        model="deepseek/deepseek-v4.1-flash", providers_allowed=None, providers_ignored=None,
        providers_order=None, provider_sort=None, provider_require_parameters=False,
        provider_data_collection=None)

    prefs = _provider_preferences_for_agent(agent)
    assert prefs["order"] == ["deepseek"]
    assert prefs["sort"] == "latency", "per-model sort must win over the flat 'price'"


# --------------------------------------------------------------------------- endpoint rows


def _endpoint(tag, prompt, completion="0.000001", context=200_000, tools=True, throughput=None):
    ep = {
        "tag": tag, "provider_name": tag, "context_length": context, "status": 0,
        "quantization": "unknown", "uptime_last_30m": 99.0,
        "pricing": {"prompt": prompt, "completion": completion, "input_cache_read": "0.0000001"},
        "supported_parameters": (["tools"] if tools else ["temperature"]),
    }
    if throughput is not None:
        ep["throughput_last_30m"] = {"p50": throughput}
    return ep


def test_rows_dedupe_by_tag_and_sort_cheapest_first(monkeypatch):
    endpoints = [
        _endpoint("expensive", "0.000003"),
        _endpoint("cheap", "0.0000005"),
        _endpoint("cheap", "0.000001"),  # same provider, pricier variant -> dropped
        _endpoint("mid", "0.0000015"),
    ]
    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: endpoints)

    rows = provider_rows("some/model")

    assert [r["tag"] for r in rows] == ["cheap", "mid", "expensive"]
    assert [r["prompt"] for r in rows] == [0.5, 1.5, 3.0], "USD per Mtok"


def test_rows_flag_providers_that_would_break_the_agent_loop(monkeypatch):
    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: [
        _endpoint("good", "0.000002", tools=True),
        _endpoint("no-tools", "0.000001", tools=False, throughput=50),
    ])

    rows = provider_rows("some/model")

    assert [(r["tag"], r["tool_capable"]) for r in rows] == [("no-tools", False), ("good", True)]
    assert "⚠" in row_label(rows[0]), "a pin that kills tool calls must be visible in the row"
    assert "⚠" not in row_label(rows[1])


def test_row_labels_share_one_tag_width(monkeypatch):
    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: [
        _endpoint("short", "0.000001"),
        _endpoint("a-much-longer-provider-tag", "0.000002"),
    ])

    labels = prp.row_labels(provider_rows("some/model"))

    assert len({len(label) for label in labels}) == 1, "columns must not drift per row"


def test_endpoints_cache_marks_perf_less_responses(monkeypatch, config_home):
    """Keyless responses carry no throughput: they must be cached briefly, not for hours."""
    calls = []

    def fake_get_json(url, **kwargs):
        calls.append(kwargs.get("api_key", ""))
        return {"data": {"endpoints": [_endpoint("p", "0.000001")]}}

    monkeypatch.setattr(prp, "_get_json", fake_get_json)
    monkeypatch.setattr(prp, "_api_key", lambda: "")

    prp.fetch_model_endpoints("some/model")
    assert len(calls) == 1
    prp.fetch_model_endpoints("some/model")
    assert len(calls) == 1, "second call is served from cache"
    assert prp._read_cache()["some/model"]["perf"] is False


# --------------------------------------------------------------------------- wizard screen


def test_screen_skipped_for_single_endpoint_model(monkeypatch, config_home):
    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: [_endpoint("only", "0.000001")])

    assert prompt_provider_routing("solo/model") is None
    assert configure_after_selection("solo/model") is None
    assert "provider_routing" not in _read_config(config_home)


def test_screen_skipped_for_other_providers(monkeypatch, config_home):
    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: [
        _endpoint("a", "0.000001"), _endpoint("b", "0.000002")])

    assert prompt_provider_routing("m/x", provider="nous") is None
    assert "provider_routing" not in _read_config(config_home)


def test_screen_empty_selection_leaves_an_existing_pin_untouched(monkeypatch, config_home):
    _write_config(config_home, """
        provider_routing:
          models:
            some/model:
              order: [kept]
              sort: price
    """)
    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: [
        _endpoint("a", "0.000001"), _endpoint("b", "0.000002")])
    monkeypatch.setattr(prp.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(prp, "curses_checklist", lambda *a, **k: set())

    assert configure_after_selection("some/model") is None
    assert existing_model_routing("some/model")[1] == {"order": ["kept"], "sort": "price"}


def test_screen_selection_writes_order_and_sort(monkeypatch, config_home):
    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: [
        _endpoint("b", "0.000002"), _endpoint("a", "0.000001")])
    monkeypatch.setattr(prp.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(prp, "curses_checklist", lambda *a, **k: {0, 1})
    monkeypatch.setattr(prp, "curses_radiolist", lambda *a, **k: 1)  # price

    stored = configure_after_selection("some/model")

    assert stored == {"order": ["a", "b"], "sort": "price"}
    assert _read_config(config_home)["provider_routing"]["models"]["some/model"] == stored


def test_screen_clear_row_removes_the_pin(monkeypatch, config_home):
    _write_config(config_home, """
        provider_routing:
          models:
            some/model:
              order: [a]
    """)
    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: [
        _endpoint("a", "0.000001"), _endpoint("b", "0.000002")])
    monkeypatch.setattr(prp.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(prp, "curses_checklist", lambda *a, **k: {2})  # trailing clear row

    assert configure_after_selection("some/model") == {}
    assert existing_model_routing("some/model") == ("", {})


# --------------------------------------------------------------------------- picker stages


def _picker_self(state, model="some/model"):
    """Minimal HermesCLI stand-in: the stage handlers read the picker state and write config."""
    import cli as cli_mod
    from hermes_cli.model_switch import ModelSwitchResult

    state["switch_result"] = ModelSwitchResult(
        success=True, new_model=model, target_provider="openrouter")
    self_ = SimpleNamespace(
        _model_picker_state=state,
        _invalidate=lambda **_kwargs: None,
        _close_model_picker=lambda: state.clear(),
        _pending_commit={})
    bound = lambda name: getattr(cli_mod.HermesCLI, name).__get__(self_, type(self_))
    self_._advance_after_model_pick = bound("_advance_after_model_pick")
    self_._handle_picker_routing_stage = bound("_handle_picker_routing_stage")
    self_._handle_picker_routing_sort_stage = bound("_handle_picker_routing_sort_stage")
    self_._tui_model_picker_toggle = bound("_tui_model_picker_toggle")
    self_._commit_picker_result = (
        lambda result, persist_global, reasoning_effort="": self_._pending_commit.update(
            result=result, persist_global=persist_global, effort=reasoning_effort))
    return self_


def _routing_state(model, monkeypatch, endpoints):
    from hermes_cli.cli_model_switch_mixin import _picker_routing_state
    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: endpoints)
    return _picker_routing_state(model)


def test_space_toggle_is_exclusive_with_the_clear_row():
    state = {"stage": "routing", "selected": 0, "routing_rows": [{"tag": "a"}, {"tag": "b"}],
             "routing_checked": set(), "routing_clear_index": 2}
    self_ = _picker_self(state)
    event = SimpleNamespace(app=SimpleNamespace(invalidate=lambda: None))

    self_._tui_model_picker_toggle(event)
    assert state["routing_checked"] == {0}
    state["selected"] = 2  # the clear row
    self_._tui_model_picker_toggle(event)
    assert state["routing_checked"] == {2}, "marking clear drops the provider checks"


def test_stage_gating_skips_models_with_a_single_endpoint(monkeypatch):
    endpoints = [_endpoint("only", "0.000001")]
    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: endpoints)
    assert _picker_offers_routing({"slug": "openrouter"}, "solo/model") is False
    assert _picker_offers_routing({"slug": "nous"}, "some/model") is False
    assert _picker_offers_routing({"slug": "openrouter"}, "") is False

    monkeypatch.setattr(prp, "fetch_model_endpoints", lambda *a, **k: [
        _endpoint("a", "0.000001"), _endpoint("b", "0.000002")])
    assert _picker_offers_routing({"slug": "openrouter"}, "some/model") is True


def test_state_preselects_the_current_pin(monkeypatch, config_home):
    _write_config(config_home, """
        provider_routing:
          models:
            some/model:
              order: [b]
              sort: latency
    """)

    state = _routing_state("some/model", monkeypatch, [
        _endpoint("a", "0.000001"), _endpoint("b", "0.000002")])

    assert state["routing_checked"] == {1}, "the pinned provider is pre-checked"
    assert state["routing_sort"] == "latency"
    assert state["routing_clear_index"] == 2, "clear row sits after the provider rows"
    assert len(state["routing_labels"]) == 3


def test_empty_selection_leaves_an_existing_pin_untouched(monkeypatch, config_home):
    _write_config(config_home, """
        provider_routing:
          models:
            some/model:
              order: [kept]
              sort: price
    """)
    state = _routing_state("some/model", monkeypatch, [
        _endpoint("a", "0.000001"), _endpoint("b", "0.000002")])
    state.update(stage="routing", routing_checked=set())
    self_ = _picker_self(state)

    self_._handle_picker_routing_stage(state, 0, persist_global=False)

    assert existing_model_routing("some/model")[1] == {"order": ["kept"], "sort": "price"}
    assert state["stage"] == "reasoning", "the pick continues down the stage machine"


def test_checked_providers_advance_to_the_sort_step(monkeypatch, config_home):
    state = _routing_state("some/model", monkeypatch, [
        _endpoint("a", "0.000001"), _endpoint("b", "0.000002")])
    state.update(stage="routing", routing_checked={0, 1})
    self_ = _picker_self(state)

    self_._handle_picker_routing_stage(state, 0, persist_global=False)

    assert state["stage"] == "routing_sort"
    assert state["routing_order"] == ["a", "b"], "checklist order becomes the priority order"
    assert "provider_routing" not in _read_config(config_home), "nothing is written before sort"


def test_sort_step_persists_order_and_sort(monkeypatch, config_home):
    state = _routing_state("some/model", monkeypatch, [
        _endpoint("a", "0.000001"), _endpoint("b", "0.000002")])
    state.update(stage="routing_sort", routing_checked={0, 1}, routing_order=["a", "b"])
    self_ = _picker_self(state)

    self_._handle_picker_routing_sort_stage(state, 1, persist_global=False)  # 1 = price

    assert _read_config(config_home)["provider_routing"]["models"]["some/model"] == {
        "order": ["a", "b"], "sort": "price"}
    assert state["stage"] == "reasoning", "sort applies the pick instead of stopping the flow"


def test_clear_row_removes_the_pin(monkeypatch, config_home):
    _write_config(config_home, """
        provider_routing:
          models:
            some/model:
              order: [a]
    """)
    state = _routing_state("some/model", monkeypatch, [
        _endpoint("a", "0.000001"), _endpoint("b", "0.000002")])
    state.update(stage="routing", routing_checked={state["routing_clear_index"]})
    self_ = _picker_self(state)

    self_._handle_picker_routing_stage(state, state["routing_clear_index"], persist_global=False)

    assert existing_model_routing("some/model") == ("", {})


def test_wizard_flow_offers_the_screen_after_a_model_is_picked(monkeypatch, config_home):
    """``hermes model`` / ``hermes setup`` must run the same step the CLI picker does."""
    import hermes_cli.auth as auth_mod
    import hermes_cli.model_setup_flows as flows
    import hermes_cli.models as models_mod
    import hermes_cli.models_pricing as pricing_mod

    seen = []
    monkeypatch.setattr(flows, "_ensure_flow_api_key", lambda *a, **k: ("k", "k", False))
    monkeypatch.setattr(flows, "_finish_model", lambda *a, **k: None)
    monkeypatch.setattr(models_mod, "model_ids", lambda **k: ["some/model"])
    monkeypatch.setattr(pricing_mod, "get_pricing_for_provider", lambda *a, **k: {})
    monkeypatch.setattr(auth_mod, "_prompt_model_selection", lambda *a, **k: "some/model")
    monkeypatch.setattr(prp, "configure_after_selection",
                        lambda model, **k: seen.append(model) or {})

    flows._model_flow_openrouter({}, "")

    assert seen == ["some/model"]

    # A cancelled model pick must not open the provider step at all.
    seen.clear()
    monkeypatch.setattr(auth_mod, "_prompt_model_selection", lambda *a, **k: None)
    flows._model_flow_openrouter({}, "")
    assert seen == []
