"""``providers.<id>.label``: a display-only provider name for pickers and status (#158907)."""

import pytest

from hermes_cli import config_providers
from hermes_cli import providers as providers_mod
from hermes_cli.model_switch_providers import _finalize_picker_rows

PLUGIN = "claude-subscription-directsdk-experimental"


def _helper(name):
    fn = getattr(config_providers, name, None)
    assert callable(fn), f"hermes_cli.config_providers.{name} is missing"
    return fn


def _labels(providers):
    return _helper("user_provider_labels")(providers)


def test_label_maps_canonical_provider_id():
    assert _labels({"nous": {"label": "My Nous"}}) == {"nous": "My Nous"}


def test_label_key_is_normalized_like_provider_ids():
    # Case and aliases resolve to the canonical id the pickers use as the row slug.
    assert _labels({"NOUS": {"label": "Portal"}}) == {"nous": "Portal"}
    # Aliases resolve through Hermes' own provider-id table (the picker slugs), not models.dev ids.
    assert _labels({"kimi-cn": {"label": "Aliased"}}) == {"kimi-coding-cn": "Aliased"}
    assert _labels({"vercel": {"label": "Aliased"}}) == {"ai-gateway": "Aliased"}


def _listed_slugs():
    from hermes_cli.models_catalog_static import listed_canonical_providers

    return [p.slug for p in listed_canonical_providers()]


def test_every_listed_provider_labels_only_its_own_row():
    # Slugs that models.dev folds together (kimi-coding / kimi-coding-cn, kilocode, ...) stay distinct.
    slugs = _listed_slugs()
    assert {"kimi-coding", "kimi-coding-cn", "kilocode", "opencode-zen", "ai-gateway"} <= set(slugs)
    for slug in [s for s in slugs if s != "custom"]:  # the custom route is never labelled
        rows = [_row(s, f"Default {s}") for s in slugs]
        _helper("apply_user_provider_labels")(rows, {slug: {"label": "Mine"}})
        assert [r["slug"] for r in rows if r["name"] == "Mine"] == [slug], slug


def test_sibling_providers_keep_separate_labels():
    labels = _labels({"kimi-coding": {"label": "Kimi"}, "kimi-coding-cn": {"label": "Kimi CN"}})
    assert labels == {"kimi-coding": "Kimi", "kimi-coding-cn": "Kimi CN"}


@pytest.mark.parametrize(("key", "row_slug"), [
    ("llamacpp", "llamacpp"), ("llama.cpp", "llamacpp"), ("ollama", "ollama"), ("vllm", "vllm"), ("moa", "moa")])
def test_local_runtime_label_keys_on_its_own_slug(key, row_slug):
    # normalize_provider folds local runtimes into the generic "custom" route; a label must not.
    assert _labels({key: {"label": "Short"}}) == {row_slug: "Short"}


def test_custom_route_cannot_be_labelled():
    assert _labels({"custom": {"label": "Mine"}}) == {}


def test_local_runtime_label_does_not_rename_custom_endpoints(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda: {"providers": {"llamacpp": {"label": "On-box"}}})
    assert providers_mod.get_label("llamacpp") == "On-box"
    assert providers_mod.get_label("custom") != "On-box"


def test_label_is_stripped_of_surrounding_spaces():
    assert _labels({"nous": {"label": "  Portal  "}}) == {"nous": "Portal"}


@pytest.mark.parametrize("bad", [
    "", "   ", 7, None, ["x"], {"a": 1},
    "two\nlines", "\u2028Portal", "Portal\u2029", "Por\x85tal", "\tPortal", "Portal\x00", "Port\x1bal",
])
def test_invalid_label_is_ignored(bad):
    assert _labels({"nous": {"label": bad}}) == {}


def test_disabled_entry_label_is_ignored():
    assert _labels({"nous": {"label": "Portal", "enabled": False}}) == {}
    assert _labels({"nous": {"label": "Portal", "enabled": "false"}}) == {}


@pytest.mark.parametrize("providers", [None, [], "x", {"nous": "Portal"}, {"nous": None}, {7: {"label": "x"}}])
def test_malformed_providers_section_yields_no_labels(providers):
    assert _labels(providers) == {}


def test_get_label_prefers_user_label(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"providers": {"nous": {"label": "Portal"}}})
    assert providers_mod.get_label("nous") == "Portal"
    assert providers_mod.get_label("NOUS") == "Portal"


def test_get_label_without_user_label_is_unchanged(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"providers": {}})
    assert providers_mod.get_label("nous") == "Nous Portal"


def test_get_label_ignores_malformed_providers_section(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"providers": ["not", "a", "map"]})
    assert providers_mod.get_label("nous") == "Nous Portal"


def _row(slug, name, *, user=False, models=1):
    return {"slug": slug, "name": name, "is_current": False, "is_user_defined": user,
            "models": ["m"] * models, "total_models": models, "source": "hermes"}


def test_picker_rows_use_label_for_builtin_rows_only():
    rows = [_row(PLUGIN, "Claude Subscription DirectSDK (Experimental)"),
            _row("custom:mine", "Mine", user=True), _row("nous", "Nous Portal")]
    out = _finalize_picker_rows(rows, {PLUGIN: {"label": "Claude Sub"},
                                       "mine": {"label": "Ignored", "api": "http://x/v1"}}, "")
    names = {r["slug"]: r["name"] for r in out}
    assert names == {PLUGIN: "Claude Sub", "custom:mine": "Mine", "nous": "Nous Portal"}
    assert {r["slug"] for r in out} == {PLUGIN, "custom:mine", "nous"}  # no row added or dropped


def test_label_only_entry_does_not_route_as_custom_endpoint():
    from hermes_cli.runtime_provider_custom import _get_named_custom_provider

    cfg = {"providers": {PLUGIN: {"label": "Claude Sub"}}}
    assert _get_named_custom_provider(PLUGIN, config=cfg, resolve_credentials=False) is None
    assert _get_named_custom_provider("Claude Sub", config=cfg, resolve_credentials=False) is None


def test_label_is_a_known_provider_key():
    assert "label" in config_providers._KNOWN_PROVIDER_KEYS


def test_models_payload_labels_skeleton_rows(monkeypatch):
    from hermes_cli import inventory

    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", lambda **_k: [])
    monkeypatch.setattr(inventory, "_moa_provider_row", lambda *_a, **_k: None)
    monkeypatch.setattr(inventory, "inject_local_runtime_row", lambda rows, ctx: (rows, None))
    monkeypatch.setattr("hermes_cli.models_validate.drop_unofferable_model_ids", lambda rows: None)
    ctx = inventory.ConfigContext(current_provider="", current_model="", current_base_url="",
                                  user_providers={"nous": {"label": "Portal"}}, custom_providers=[],
                                  excluded_providers=[])
    payload = inventory.build_models_payload(ctx, include_unconfigured=True)
    nous = [r for r in payload["providers"] if r["slug"] == "nous"]
    assert nous and nous[0]["name"] == "Portal"
    others = [r for r in payload["providers"] if r["slug"] != "nous"]
    assert others and all(r["name"] != "Portal" for r in others)


def test_hermes_model_label_map_applies_user_labels():
    from hermes_cli import models

    fn = getattr(config_providers, "provider_labels_for_config", None)
    assert callable(fn), "hermes_cli.config_providers.provider_labels_for_config is missing"
    labels = fn({"providers": {"nous": {"label": "Portal"}}})
    assert labels["nous"] == "Portal"
    assert labels["custom"] == models._PROVIDER_LABELS["custom"]
    assert fn({})["nous"] == models._PROVIDER_LABELS["nous"]
    assert fn(None) == dict(models._PROVIDER_LABELS)


@pytest.mark.parametrize("slug", ["nous", "kilocode", "ai-gateway"])
def test_hermes_model_rows_use_label_for_listed_slug(slug):
    from hermes_cli.main_provider_setup import _build_provider_picker_rows

    rows, _ = _build_provider_picker_rows({"providers": {slug: {"label": "Mine"}}}, "", {}, {})
    assert [key for key, label, _m in rows if label == "Mine"] == [slug]


@pytest.mark.parametrize(("group", "slug"), [("kimi", "kimi-coding-cn"), ("opencode", "opencode-zen")])
def test_hermes_model_group_member_uses_label(monkeypatch, group, slug):
    # Grouped providers appear in the group's sub-picker: open the group and read the member rows.
    from hermes_cli import main

    cfg = {"providers": {slug: {"label": "Mine"}}}
    prompts = []

    def fake_prompt(labels, **_kw):
        prompts.append(list(labels))
        if len(prompts) == 1:
            return next(i for i, (key, _l, _m) in enumerate(rows) if key == f"group:{group}")
        return None

    from hermes_cli.main_provider_setup import _build_provider_picker_rows

    labels = _helper("provider_labels_for_config")(cfg)
    rows, _ = _build_provider_picker_rows(cfg, "", labels, {})
    monkeypatch.setattr(main, "_prompt_provider_choice", fake_prompt)
    main._pick_provider(cfg, "", labels, {})
    members = prompts[1]
    assert members.count("Mine") == 1
    assert len(members) >= 2


@pytest.mark.parametrize("slug", ["nous", "kilocode", "opencode-zen", "ai-gateway", "kimi-coding-cn"])
def test_hermes_model_active_line_uses_label(monkeypatch, capsys, slug):
    # The active provider resolves to a models.dev id for some slugs (kilocode -> kilo); the label still applies.
    from hermes_cli import main

    cfg = {"model": {"default": "m", "provider": slug}, "providers": {slug: {"label": "Mine"}}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(main, "_prompt_provider_choice", lambda labels, **_kw: None)
    main.select_provider_and_model()
    assert "Active provider:  Mine" in capsys.readouterr().out


def test_hermes_model_picker_rows_show_user_label(monkeypatch, capsys):
    # Drive the real ``hermes model`` flow through the real row builder; only the prompt is stubbed.
    from hermes_cli import main

    cfg = {"model": {"default": "m", "provider": "nous"}, "providers": {"nous": {"label": "Portal"}}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    shown = []

    def fake_prompt(labels, **_kw):
        shown.extend(labels)

    monkeypatch.setattr(main, "_prompt_provider_choice", fake_prompt)
    main.select_provider_and_model()
    assert any(label.startswith("Portal") for label in shown), shown
    assert not any(label.startswith("Nous Portal") for label in shown), shown
    assert "Active provider:  Portal" in capsys.readouterr().out


@pytest.fixture
def _offline_picker(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    for name in ("fetch_api_models", "cached_provider_model_ids", "provider_model_ids"):
        monkeypatch.setattr(f"hermes_cli.models.{name}", lambda *_a, **_kw: None if name == "fetch_api_models" else [])
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("key", ["openai", "moa", "llamacpp", "llama.cpp", "ollama", "local", "vllm"])
@pytest.mark.parametrize("block", [{"label": "Short"}, {"request_timeout_seconds": 30}])
def test_endpointless_block_for_known_provider_adds_no_row(_offline_picker, key, block):
    from hermes_cli.model_switch import list_authenticated_providers

    def rows(user_providers):
        return sorted((r["slug"], r["is_user_defined"]) for r in list_authenticated_providers(user_providers=user_providers))

    assert rows({key: block}) == rows({})


def test_endpoint_entry_keyed_like_builtin_still_gets_a_row(_offline_picker):
    from hermes_cli.model_switch import list_authenticated_providers

    rows = list_authenticated_providers(
        user_providers={"openai": {"base_url": "http://127.0.0.1:9/v1", "models": ["m1"]}},
        probe_custom_providers=False)
    assert [r for r in rows if r["slug"] == "openai" and r["is_user_defined"]]


def test_picker_late_rows_get_user_labels(monkeypatch):
    from hermes_cli import model_switch_providers as msp

    base = [_row("nous", "Nous Portal")]
    monkeypatch.setattr("hermes_cli.model_switch.list_authenticated_providers", lambda **_k: list(base))
    monkeypatch.setattr(msp, "_with_local_runtime_row",
                        lambda rows, *_a, **_k: rows + [_row("llamacpp", "Local")])
    monkeypatch.setattr(msp, "_prepend_moa_picker_provider",
                        lambda rows, **_k: [_row("moa", "Mixture of Agents")] + rows)
    monkeypatch.setattr("hermes_cli.models_validate.drop_unofferable_model_ids", lambda rows: None)
    monkeypatch.setattr(msp, "_attach_window_model_notes", lambda rows: None)
    out = msp.list_picker_providers(
        user_providers={"moa": {"label": "Ensemble"}, "llamacpp": {"label": "On-box"}, "nous": {"label": "Portal"}},
        include_moa=True)
    assert {r["slug"]: r["name"] for r in out} == {"moa": "Ensemble", "llamacpp": "On-box", "nous": "Portal"}
