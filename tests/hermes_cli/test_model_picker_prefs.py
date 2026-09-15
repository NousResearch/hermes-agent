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
from hermes_cli.model_switch_providers import _apply_picker_preferences


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


def test_config_load_failure_is_a_noop(monkeypatch, caplog):
    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    rows = [_row("a"), _row("b")]
    with caplog.at_level("DEBUG", logger="hermes_cli.model_switch"):
        assert _slugs(_apply_picker_preferences(rows)) == ["a", "b"]
    assert "RuntimeError" in caplog.text


# ─── surface parity: payload pickers apply the same prefs ───────────────


def test_build_models_payload_exposes_the_pref_switch():
    """The knob exists and defaults OFF for non-picker consumers."""
    import inspect

    sig = inspect.signature(inventory.build_models_payload)
    assert "apply_picker_prefs" in sig.parameters
    assert sig.parameters["apply_picker_prefs"].default is False


@pytest.fixture
def entry_env(tmp_path, monkeypatch):
    """Real disk config and policy; replace only discovery/network and terminal UI."""
    import copy
    import yaml
    home = tmp_path / "picker-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    rows = [dict(_row(s), name=s, is_current=False, is_user_defined=False)
            for s in ("anthropic", "openai-api", "openai-codex", "deepseek")]
    def discover(**kw):
        out = copy.deepcopy(rows)
        for row in out:
            row["is_current"] = row["slug"] == kw.get("current_provider")
        return out
    monkeypatch.setattr(model_switch, "list_authenticated_providers", discover)
    for name in ("_local_runtime_row", "_moa_provider_row"):
        monkeypatch.setattr(inventory, name, lambda *a: None)
    monkeypatch.setattr(inventory, "_prewarm_pricing_async", lambda *a, **kw: None)
    monkeypatch.setattr(inventory, "_apply_pricing", lambda *a, **kw: None)
    monkeypatch.setattr(inventory, "_apply_capabilities", lambda *a: None)
    monkeypatch.setattr(inventory, "_apply_featured", lambda *a: None)
    def write(picker, current="anthropic", excluded=()):
        config = {"model": {"provider": current, "default": "m1", "picker": picker},
                  "model_catalog": {"excluded_providers": list(excluded)},
                  "auxiliary": {"vision": {"provider": "openai-codex", "model": "m1"}}}
        (home / "config.yaml").write_text(yaml.safe_dump(config))
        return config
    return write


def test_entry_classic(entry_env):
    from types import SimpleNamespace
    from hermes_cli.cli_model_switch_mixin import _show_model_picker
    entry_env({"hide": ["openai-*"], "order": ["deepseek"]}, "openai-codex")
    captured = []
    cli = SimpleNamespace(model="m1", provider="openai-codex",
                          _open_model_picker=lambda rows, *a, **kw: captured.extend(rows))
    _show_model_picker(cli, inventory.load_picker_context(), False)
    assert _slugs(captured) == ["deepseek", "anthropic", "openai-codex"]


def test_entry_options(entry_env):
    entry_env({"hide": ["openai-*"], "order": ["deepseek"]}, "openai-codex")
    ctx = inventory.load_picker_context()
    assert _slugs(inventory.build_model_options_payload(ctx)["providers"]) == [
        "deepseek", "anthropic", "openai-codex"]
    assert _slugs(inventory.build_models_payload(ctx)["providers"]) == [
        "anthropic", "openai-api", "openai-codex", "deepseek"]


def test_entry_auxiliary(entry_env, monkeypatch):
    from hermes_cli import main_provider_setup as setup
    entry_env({"hide": ["openai-*", "anthropic"], "order": ["deepseek"]})
    captured = []
    def cancel(labels, **kw):
        captured.extend(labels)
        return None
    monkeypatch.setattr(setup, "_prompt_provider_choice", cancel)
    setup._aux_select_for_task("vision")
    text = "\n".join(captured)
    assert "openai-api" not in text and "anthropic" not in text
    assert "openai-codex" in text and "deepseek" in text
    assert text.index("deepseek") < text.index("openai-codex")


def test_entry_discovery(entry_env):
    from hermes_cli.model_switch_providers import list_picker_providers
    entry_env({"hide": ["openai-*"], "order": ["deepseek"]})
    assert _slugs(list_picker_providers(current_provider="openai-codex")) == [
        "deepseek", "anthropic", "openai-codex"]


def test_entry_hermes_model(entry_env, monkeypatch):
    from hermes_cli import main
    entry_env({"hide": ["openrouter", "openai-*"], "order": ["deepseek"]}, "openai-codex")
    captured = []
    def cancel(labels, **kw):
        captured.extend(labels)
        return None
    monkeypatch.setattr(main, "_prompt_provider_choice", cancel)
    main.select_provider_and_model()
    text = "\n".join(captured)
    assert "OpenRouter" not in text
    assert "OpenAI API" not in text
    assert "Codex" in text
    assert "Leave unchanged" in text


@pytest.mark.parametrize("current", ["openai-api", "moa", "llamacpp"])
def test_entry_excluded_current(entry_env, monkeypatch, current):
    entry_env({"hide": ["*"], "order": [current]}, current, [current])
    if current != "openai-api":
        monkeypatch.setattr(inventory, "_moa_provider_row" if current == "moa" else "_local_runtime_row",
                            lambda *a: dict(_row(current), is_current=True))
    ctx = inventory.load_picker_context()
    assert current not in _slugs(inventory.build_model_options_payload(ctx, include_unconfigured=True)["providers"])


def test_entry_rest(entry_env, monkeypatch):
    import asyncio
    from hermes_cli.web_routers import models
    entry_env({"hide": ["openai-*"], "order": ["deepseek"]}, "openai-codex")
    monkeypatch.setattr(models, "_dashboard_code_skew_guard", lambda: None)
    result = asyncio.run(models.get_model_options())
    assert _slugs(result["providers"]) == ["deepseek", "anthropic", "openai-codex"]


def test_entry_rpc_live_current(entry_env, monkeypatch):
    from types import SimpleNamespace
    import tui_gateway.server as server
    entry_env({"hide": ["openai-*", "anthropic"], "order": ["deepseek"]})
    monkeypatch.setitem(server._sessions, "picker-test", {
        "agent": SimpleNamespace(provider="openai-codex", model="m1", base_url="")})
    response = server._methods["model.options"](1, {"session_id": "picker-test"})
    assert _slugs(response["result"]["providers"]) == ["deepseek", "openai-codex"]


def test_entry_tools_vision(entry_env, monkeypatch):
    from hermes_cli import tools_config, tools_config_providers
    config = entry_env({"hide": ["openai-*", "anthropic"], "order": ["deepseek"]})
    captured = []
    def cancel(title, labels, default):
        captured.extend(labels)
        return len(labels) - 1
    monkeypatch.setattr(tools_config, "_prompt_choice", cancel)
    tools_config_providers._configure_vision_provider_model(config, config["auxiliary"]["vision"])
    assert "anthropic" not in "\n".join(captured)
    assert "openai-api" not in "\n".join(captured)
    assert captured[0].startswith("deepseek")
    assert captured[1].startswith("openai-codex")
    assert captured[-1] == "Cancel"


def test_entry_group_custom_actions(entry_env):
    from hermes_cli.main_provider_setup import _build_provider_picker_rows
    from hermes_cli.models import _PROVIDER_LABELS
    custom = {"custom:local": {"name": "Local Test", "base_url": "http://localhost:1/v1"}}
    config = entry_env({}, "openai-codex")
    baseline, _ = _build_provider_picker_rows(config, "openai-codex", _PROVIDER_LABELS, custom)
    config["model"]["picker"] = {"order": ["custom:local", "openai-api", "deepseek"]}
    ordered, selected = _build_provider_picker_rows(config, "openai-codex", _PROVIDER_LABELS, custom)
    assert next(r[2] for r in ordered if r[0] == "group:openai") == ["openai-api", "openai-codex"]
    assert ordered[selected][0] == "group:openai"
    assert ordered[-6:] == baseline[-6:]  # saved custom plus trailing action group
    config["model"]["picker"] = {"hide": ["openai-*", "custom:*"]}
    hidden, selected = _build_provider_picker_rows(config, "openai-codex", _PROVIDER_LABELS, custom)
    assert hidden[selected][0] == "openai-codex"
    assert "custom:local" not in [r[0] for r in hidden]
    config["model_catalog"]["excluded_providers"] = ["openai-codex"]
    excluded, _ = _build_provider_picker_rows(config, "openai-codex", _PROVIDER_LABELS, custom)
    assert "openai-codex" not in [r[0] for r in excluded]


def test_entry_default_unchanged(entry_env):
    entry_env({})
    ctx = inventory.load_picker_context()
    assert inventory.build_models_payload(ctx) == inventory.build_models_payload(ctx, apply_picker_prefs=True)


def test_custom_identity_and_order_groups(cfg):
    cfg({"hide": ["custom:*"], "order": ["custom:b", "z", "custom:a"]})
    rows = [_row("a"), dict(_row("custom:a"), is_user_defined=True, is_current=True), _row("z")]
    out = _apply_picker_preferences(rows, "custom:alias")
    assert _slugs(out) == ["z", "custom:a", "a"]
    assert _slugs(rows) == ["a", "custom:a", "z"]


def test_excluded_alias_wins_over_current(cfg):
    cfg({"hide": ["*"], "order": ["anthropic"]})
    assert _apply_picker_preferences([dict(_row("anthropic"), is_current=True)],
                                     "anthropic", excluded_providers=["claude"]) == []
