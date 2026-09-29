"""Picker-only catalog curation, using real config loading and row discovery."""

import pytest

from hermes_cli import model_switch_providers as listing
from hermes_cli.config import atomic_config_write
from hermes_cli.inventory import build_models_payload, load_picker_context

_build_curated_lists = listing._build_curated_lists
_live_or_curated_ids = listing._live_or_curated_ids


@pytest.fixture
def catalogs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(listing, "_build_curated_lists", lambda *a, **kw: {})
    monkeypatch.setattr(listing, "_overlay_has_creds",
                        lambda b, pid, slug, overlay: slug in {"anthropic", "openai-codex"})
    discovered = {
        "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-4-6"],
        "openai-codex": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.5"],
    }
    monkeypatch.setattr(listing, "_live_or_curated_ids",
                        lambda slug, *a, **kw: list(discovered.get(slug, [])))
    return tmp_path, discovered


def write_config(home, filters):
    atomic_config_write(home / "config.yaml", {
        "model": {"provider": "anthropic", "default": "claude-opus-4-6"},
        "model_catalog": {"model_filters": filters},
        "nous": {"guest": False},
    })


@pytest.mark.parametrize("surface", ["inventory", "gateway"])
def test_config_filters_after_discovery(catalogs, surface):
    home, discovered = catalogs
    write_config(home, {
        "anthropic": {"allow": ["claude-*-5-5"]},
        "openai-codex": {"deny": ["gpt-5.4*"]},
    })
    if surface == "inventory":
        payload = build_models_payload(load_picker_context())
        rows = payload["providers"]
        # A hidden active model remains the session's current model.
        assert payload["model"] == "claude-opus-4-6"
    else:
        rows = listing.list_picker_providers(
            current_provider="anthropic", current_model="claude-opus-4-6")
    by_slug = {row["slug"]: row for row in rows}
    assert by_slug["anthropic"]["models"] == discovered["anthropic"][:2]
    assert by_slug["anthropic"]["total_models"] == 2
    assert by_slug["openai-codex"]["models"] == ["gpt-5.5"]


@pytest.mark.parametrize("surface", ["inventory", "gateway"])
@pytest.mark.parametrize("rule", [{"allow": []}, {"allow": ["missing*"]}, {"deny": ["*"]}])
def test_empty_result_is_not_repopulated(catalogs, surface, rule):
    home, _ = catalogs
    write_config(home, {"anthropic": rule})
    if surface == "inventory":
        rows = build_models_payload(load_picker_context(), explicit_only=True)["providers"]
    else:
        rows = listing.list_picker_providers(current_provider="anthropic", current_model="claude-opus-4-6")
    assert all(row["slug"] != "anthropic" for row in rows)


@pytest.mark.parametrize("surface", ["inventory", "gateway"])
def test_filter_before_limit_and_count_full_matches(catalogs, surface):
    home, discovered = catalogs
    # The desired IDs occur past the first item: truncating discovery first loses them.
    allowed = discovered["anthropic"][1:]
    write_config(home, {"anthropic": {"allow": allowed}})
    if surface == "inventory":
        rows = build_models_payload(load_picker_context(), max_models=1)["providers"]
    else:
        rows = listing.list_picker_providers(max_models=1)
    row = next(row for row in rows if row["slug"] == "anthropic")
    assert row["models"] == allowed[:1]
    assert row["total_models"] == len(allowed)


def test_profile_rules_do_not_leak(catalogs):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home, _ = catalogs
    other = home / "other"
    other.mkdir()
    write_config(home, {"anthropic": {"allow": ["claude-opus-5-5"]}})
    write_config(other, {"anthropic": {"allow": ["claude-sonnet-5-5"]}})
    for target, expected in [(home, "claude-opus-5-5"), (other, "claude-sonnet-5-5"), (home, "claude-opus-5-5")]:
        token = set_hermes_home_override(target)
        try:
            for rows in (build_models_payload(load_picker_context())["providers"], listing.list_picker_providers()):
                assert next(row["models"] for row in rows if row["slug"] == "anthropic") == [expected]
        finally:
            reset_hermes_home_override(token)


@pytest.mark.parametrize("rule,expected", [
    ({}, ["Model-A", "model-b", "other"]),
    ({"allow": ["Model-?", "model-*"], "deny": ["*-b"]}, ["Model-A"]),
    ({"deny": []}, ["Model-A", "model-b", "other"]),
    ({"allow": ["model-*"]}, ["model-b"]),
    ({"allow": "model-*"}, ["Model-A", "model-b", "other"]),
    ({"deny": [None]}, ["Model-A", "model-b", "other"]),
])
def test_rule_semantics(rule, expected):
    from hermes_cli.model_catalog import filter_picker_model_ids, get_picker_model_filters

    filters = get_picker_model_filters({"model_catalog": {"model_filters": {" CHATGPT ": rule}}})
    assert filter_picker_model_ids("openai-codex", ["Model-A", "model-b", "other"], filters) == expected


def test_filters_do_not_mutate_discovery_rows():
    from hermes_cli.model_catalog import filter_picker_rows

    row = {"slug": "custom:local", "models": ["keep", "hide"], "total_models": 2,
           "is_user_defined": True, "api_url": "http://127.0.0.1:1234/v1"}
    filtered = filter_picker_rows([row], {"custom:local": {"deny": ["hide"]}})
    assert filtered[0]["models"] == ["keep"]
    assert row["models"] == ["keep", "hide"]
    assert row["total_models"] == 2
    assert filter_picker_rows([row], {"custom:local": {"allow": []}}) == []


def test_oauth_setup_picker_and_manual_entry(catalogs, monkeypatch):
    from hermes_cli.auth_model_picker import _prompt_model_selection

    home, discovered = catalogs
    write_config(home, {"anthropic": {"allow": ["claude-opus-5-5"]}})
    menus = []

    def choose_custom(title, choices, **kwargs):
        menus.append(choices)
        return len(choices) - 2  # Enter custom model name

    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", choose_custom)
    monkeypatch.setattr("hermes_cli.cli_output.line_input", lambda prompt: "claude-opus-4-6")
    monkeypatch.setattr("hermes_cli.auth_model_picker._confirm_selection_guards", lambda *a, **kw: True)
    result = _prompt_model_selection(discovered["anthropic"], current_model="claude-opus-4-6",
                                     confirm_provider="anthropic", unavailable_models=["claude-sonnet-4-6"])
    assert len(menus[0]) == 3  # one model, manual entry, skip
    assert result == "claude-opus-4-6"


def test_explicit_model_switch_ignores_display_filter(catalogs, monkeypatch):
    from hermes_cli.model_switch import switch_model

    home, _ = catalogs
    write_config(home, {"openai-api": {"deny": ["*"]}})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model",
                        lambda *a, **kw: {"accepted": True, "recognized": True, "persist": True, "message": None})
    result = switch_model("gpt-5.5", "openai-api", "gpt-5.4", explicit_provider="openai-api")
    assert result.success
    assert result.new_model == "gpt-5.5"


def test_unset_preserves_models(catalogs):
    home, discovered = catalogs
    write_config(home, {})
    for rows in (build_models_payload(load_picker_context())["providers"], listing.list_picker_providers()):
        for slug, ids in discovered.items():
            assert next(row["models"] for row in rows if row["slug"] == slug) == ids


@pytest.mark.parametrize("surface", ["inventory", "gateway"])
def test_other_provider_filter_preserves_custom_catalog(catalogs, surface):
    from hermes_cli.config import load_config

    home, _ = catalogs
    local_models = ["local-a", "local-b", "local-c"]
    results = []
    for filters in ({}, {"anthropic": {"allow": ["claude-opus-5-5"]}}):
        write_config(home, filters)
        cfg = load_config()
        cfg["providers"] = {
            "local": {"base_url": "http://127.0.0.1:1234/v1", "models": local_models,
                      "discover_models": False},
        }
        atomic_config_write(home / "config.yaml", cfg)
        ctx = load_picker_context()
        if surface == "inventory":
            rows = build_models_payload(ctx, max_models=1)["providers"]
        else:
            rows = listing.list_picker_providers(user_providers=ctx.user_providers, max_models=1)
        results.append(next(row["models"] for row in rows if row["slug"] == "local"))
    assert results[0] == local_models
    assert results[1] == results[0]


@pytest.mark.parametrize("surface", ["inventory", "gateway"])
@pytest.mark.parametrize("filter_key", ["local", "custom:local"])
def test_custom_provider_filter_uses_stable_identity(catalogs, surface, filter_key):
    from hermes_cli.config import load_config

    home, _ = catalogs
    write_config(home, {filter_key: {"deny": ["local-b"]}})
    cfg = load_config()
    cfg["providers"] = {
        "local": {"name": "Renamed Local Server", "base_url": "http://127.0.0.1:1234/v1",
                  "models": ["local-a", "local-b", "local-c"], "discover_models": False},
    }
    atomic_config_write(home / "config.yaml", cfg)
    ctx = load_picker_context()
    if surface == "inventory":
        rows = build_models_payload(ctx, max_models=1)["providers"]
    else:
        rows = listing.list_picker_providers(user_providers=ctx.user_providers, max_models=1)
    row = next(row for row in rows if row["slug"] == "local")
    # Named custom endpoints retain their uncapped display policy.
    assert row["models"] == ["local-a", "local-c"]
    assert row["total_models"] == 2


@pytest.mark.parametrize("source", ["live", "fallback"])
def test_discovery_and_offline_fallback_are_filtered(catalogs, monkeypatch, source):
    home, discovered = catalogs
    atomic_config_write(home / "config.yaml", {
        "model_catalog": {"enabled": False, "model_filters": {"anthropic": {"allow": []}}},
        "nous": {"guest": False},
    })
    monkeypatch.setattr(listing, "_live_or_curated_ids", _live_or_curated_ids)
    monkeypatch.setattr(listing, "_build_curated_lists", lambda *a, **kw: discovered)
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids",
                        lambda slug, **kw: discovered.get(slug, []) if source == "live" else [])
    monkeypatch.setattr("hermes_cli.models._merge_with_models_dev", lambda slug, ids: ids)
    for rows in (build_models_payload(load_picker_context())["providers"], listing.list_picker_providers()):
        assert all(row["slug"] != "anthropic" for row in rows)
        assert next(row["models"] for row in rows if row["slug"] == "openai-codex") == discovered["openai-codex"]


def test_lmstudio_direct_catalog_is_filtered(catalogs, monkeypatch):
    home, _ = catalogs
    atomic_config_write(home / "config.yaml", {
        "model": {"provider": "lmstudio", "default": "qwen-old"},
        "model_catalog": {"model_filters": {"lmstudio": {"deny": ["*-old"]}}},
        "nous": {"guest": False},
    })
    monkeypatch.setattr(listing, "_build_curated_lists", _build_curated_lists)
    monkeypatch.setattr("hermes_cli.models.get_curated_nous_model_ids", lambda: [])
    monkeypatch.setattr("hermes_cli.models.fetch_ollama_cloud_models", lambda **kw: [])
    monkeypatch.setattr("hermes_cli.models_local.fetch_lmstudio_models", lambda **kw: ["qwen-local", "qwen-old"])
    for rows in (build_models_payload(load_picker_context())["providers"],
                 listing.list_picker_providers(current_provider="lmstudio", current_model="qwen-old")):
        row = next(row for row in rows if row["slug"] == "lmstudio")
        assert row["models"] == ["qwen-local"]


def test_openrouter_replacement_cannot_undo_filter(catalogs, monkeypatch):
    home, _ = catalogs
    write_config(home, {"openrouter": {"deny": ["anthropic/*"]}})
    monkeypatch.setattr(listing, "_overlay_has_creds", lambda b, pid, slug, overlay: slug == "openrouter")
    monkeypatch.setattr("hermes_cli.models.fetch_openrouter_models",
                        lambda **kw: [("anthropic/claude-opus-5-5", "Claude"), ("openai/gpt-5.5", "GPT")])
    row = next(row for row in listing.list_picker_providers(max_models=1) if row["slug"] == "openrouter")
    assert row["models"] == ["openai/gpt-5.5"]
    assert row["total_models"] == 1


@pytest.mark.asyncio
async def test_gateway_text_picker_honors_rules(catalogs):
    from gateway.slash_commands_model import GatewayModelCommandsMixin, _ModelSwitchContext

    home, _ = catalogs
    write_config(home, {"anthropic": {"allow": ["claude-opus-5-5"]}})

    class TextGateway(GatewayModelCommandsMixin):
        def _delivery_adapter_for(self, source):
            return None

    ctx = _ModelSwitchContext("test", None, home / "config.yaml", False,
                              current_provider="anthropic", current_model="claude-opus-5-5")
    reply = await TextGateway()._model_listing_reply(None, ctx, None)
    assert "claude-opus-5-5" in reply
    assert "claude-sonnet-5-5" not in reply
    assert "claude-opus-4-6" not in reply


def test_moa_setup_picker_honors_rules(monkeypatch):
    from hermes_cli import model_setup_flows as flows

    preset = {"aggregator": {"provider": "openrouter", "model": "openai/gpt-5.5"},
              "reference_models": [{"provider": "anthropic", "model": "claude-sonnet-5-5"}]}
    config = {"moa": {"presets": {"visible": preset, "hidden": preset}},
              "model_catalog": {"model_filters": {"moa": {"deny": ["hidden"]}}}}
    menus = []

    def cancel(title, rows, default):
        menus.append(rows)
        return -1

    monkeypatch.setattr(flows, "_curses_choice", cancel)
    flows._model_flow_moa(config)
    assert len(menus[0]) == 1
    assert menus[0][0].startswith("visible ")
    config["model_catalog"]["model_filters"]["moa"] = {"allow": []}
    flows._model_flow_moa(config)
    assert len(menus) == 1
