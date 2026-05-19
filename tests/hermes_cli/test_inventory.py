"""Behavior tests for hermes_cli.inventory.

Locks the invariants the three migrated consumers (web_server.py
/api/model/options, tui_gateway model.options, tui_gateway model.save_key)
depend on:

- load_picker_context() reproduces the inline 17-LOC config-slice exactly.
- with_overrides() is truthy-only (empty agent attrs must not clobber).
- build_models_payload() returns a stable {providers, model, provider}
  shape and delegates curation to list_authenticated_providers (does not
  call provider_model_ids per row).
- canonical_order keys on slug membership, not is_user_defined — section
  3 of list_authenticated_providers sets is_user_defined=True for
  canonical slugs in the providers: dict, and that flag must NOT demote
  them to the tail.
- picker_hints adds authenticated/auth_type/key_env/warning per row,
  matching the TUI ModelPickerDialog shape.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.inventory import (
    ConfigContext,
    build_models_payload,
    load_picker_context,
)


# ─── load_picker_context ───────────────────────────────────────────────


def _cfg(model=None, providers=None, custom_providers=None) -> dict:
    return {
        "model": model if model is not None else {},
        "providers": providers if providers is not None else {},
        "custom_providers": custom_providers if custom_providers is not None else [],
    }


def test_load_picker_context_full_dict():
    cfg = _cfg(
        model={
            "default": "anthropic/claude-sonnet-4.6",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
        },
        providers={"openrouter": {}},
        custom_providers=[{"name": "Ollama", "base_url": "http://localhost:11434/v1"}],
    )
    with patch("hermes_cli.config.load_config", return_value=cfg):
        ctx = load_picker_context()
    assert ctx.current_model == "anthropic/claude-sonnet-4.6"
    assert ctx.current_provider == "openrouter"
    assert ctx.current_base_url == "https://openrouter.ai/api/v1"
    assert "openrouter" in ctx.user_providers
    # custom_providers comes from get_compatible_custom_providers, which
    # merges legacy list + v12+ keyed providers — both present here means
    # at least one row.
    assert isinstance(ctx.custom_providers, list)


def test_load_picker_context_falls_back_to_name_when_default_missing():
    cfg = _cfg(model={"name": "gpt-5.4", "provider": "openai"})
    with patch("hermes_cli.config.load_config", return_value=cfg):
        ctx = load_picker_context()
    assert ctx.current_model == "gpt-5.4"
    assert ctx.current_provider == "openai"


def test_load_picker_context_string_model_legacy_shape():
    """config.model can be a bare string in older configs."""
    cfg = {"model": "some-model", "providers": {}, "custom_providers": []}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        ctx = load_picker_context()
    assert ctx.current_model == "some-model"
    assert ctx.current_provider == ""
    assert ctx.current_base_url == ""


def test_load_picker_context_empty_config():
    cfg = _cfg()
    with patch("hermes_cli.config.load_config", return_value=cfg):
        ctx = load_picker_context()
    assert ctx.current_provider == ""
    assert ctx.current_model == ""
    assert ctx.current_base_url == ""
    assert ctx.user_providers == {}
    assert ctx.custom_providers == []


# ─── with_overrides ────────────────────────────────────────────────────


def _empty_ctx(provider="orig", model="orig-model", base_url="orig-url"):
    return ConfigContext(
        current_provider=provider,
        current_model=model,
        current_base_url=base_url,
        user_providers={},
        custom_providers=[],
    )


def test_with_overrides_truthy_only_strings():
    """Empty strings must NOT clobber disk config — TUI calls this with
    empty getattr(agent, 'provider', '') when no agent is spawned yet."""
    ctx = _empty_ctx()
    overlaid = ctx.with_overrides(
        current_provider="",
        current_model="",
        current_base_url="",
    )
    assert overlaid.current_provider == "orig"
    assert overlaid.current_model == "orig-model"
    assert overlaid.current_base_url == "orig-url"


def test_with_overrides_truthy_value_replaces():
    ctx = _empty_ctx()
    overlaid = ctx.with_overrides(current_provider="anthropic")
    assert overlaid.current_provider == "anthropic"
    assert overlaid.current_model == "orig-model"  # untouched


def test_with_overrides_no_args_returns_self_or_equivalent():
    ctx = _empty_ctx()
    assert ctx.with_overrides() == ctx


# ─── build_models_payload ──────────────────────────────────────────────


def _list_auth_returning(rows: list[dict]):
    """Patch list_authenticated_providers to return a fixed row list."""
    return patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=rows,
    )


def test_build_models_payload_returns_expected_shape():
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["m1"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "built-in"},
    ]
    ctx = _empty_ctx(provider="openrouter", model="m1", base_url="")
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)
    assert set(payload.keys()) == {"providers", "model", "provider"}
    assert payload["model"] == "m1"
    assert payload["provider"] == "openrouter"
    assert payload["providers"] == rows


def test_build_models_payload_does_not_call_provider_model_ids():
    """Curated lists must come from list_authenticated_providers, not
    provider_model_ids — that would pull TTS/embeddings/etc.
    """
    rows = [{"slug": "nous", "name": "Nous", "models": ["hermes-4-405b"],
             "total_models": 1, "is_current": False, "is_user_defined": False,
             "source": "built-in"}]
    ctx = _empty_ctx()
    with _list_auth_returning(rows), \
         patch("hermes_cli.models.provider_model_ids") as mock_pm:
        build_models_payload(ctx)
    mock_pm.assert_not_called()


def test_include_unconfigured_appends_canonical_skeletons():
    """include_unconfigured=True adds CANONICAL_PROVIDERS rows that
    list_authenticated_providers didn't emit. Skeleton rows have empty
    models and source='canonical'."""
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["m1"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "built-in"},
    ]
    ctx = _empty_ctx(provider="openrouter")
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx, include_unconfigured=True)
    # All canonical providers other than openrouter should appear as
    # skeleton rows.
    from hermes_cli.models import CANONICAL_PROVIDERS

    seen_slugs = {r["slug"] for r in payload["providers"]}
    for entry in CANONICAL_PROVIDERS:
        assert entry.slug in seen_slugs, f"missing {entry.slug}"
    # Skeletons have empty models and source='canonical'.
    skeletons = [r for r in payload["providers"]
                 if r.get("source") == "canonical"]
    assert all(r["models"] == [] for r in skeletons)
    assert all(r["total_models"] == 0 for r in skeletons)


def test_include_unconfigured_skips_already_present_slugs():
    """If list_authenticated_providers already returned a row for a
    canonical slug, include_unconfigured must NOT duplicate it."""
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["m1"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "built-in"},
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx, include_unconfigured=True)
    or_rows = [r for r in payload["providers"] if r["slug"] == "openrouter"]
    assert len(or_rows) == 1
    assert or_rows[0]["models"] == ["m1"]  # the authenticated row, not skeleton


# ─── picker_hints ──────────────────────────────────────────────────────


def test_picker_hints_marks_authed_rows_authenticated():
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["m1"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "built-in"},
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx, picker_hints=True)
    assert payload["providers"][0]["authenticated"] is True


def test_picker_hints_adds_warning_to_skeleton_rows():
    """Skeleton rows (unconfigured canonical providers) must carry the
    setup hint the picker UI displays."""
    rows = []
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(
            ctx, include_unconfigured=True, picker_hints=True,
        )
    skeleton_rows = [r for r in payload["providers"]
                     if r.get("source") == "canonical"]
    assert skeleton_rows, "test setup: expected at least one skeleton row"
    for row in skeleton_rows:
        assert row["authenticated"] is False
        assert "auth_type" in row
        assert "warning" in row
        # api_key providers get "paste X to activate" / others get the
        # hermes model fallback.
        assert (
            row["warning"].startswith("paste ")
            or row["warning"].startswith("run `hermes model`")
        )


def test_picker_hints_api_key_warning_format():
    """For api_key providers with a defined env var, the warning must
    point to that env var."""
    rows = []
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(
            ctx, include_unconfigured=True, picker_hints=True,
        )
    # anthropic uses api_key + ANTHROPIC_API_KEY.
    anthropic = next(
        r for r in payload["providers"] if r["slug"] == "anthropic"
    )
    assert "ANTHROPIC_API_KEY" in anthropic["warning"]
    assert anthropic["warning"].startswith("paste ")


# ─── canonical_order ───────────────────────────────────────────────────


def test_canonical_order_uses_slug_not_is_user_defined_flag():
    """Section 3 of list_authenticated_providers sets is_user_defined=True
    for canonical slugs that appear in the providers: config dict.
    canonical_order MUST key on slug membership, not the flag — otherwise
    canonical providers configured via the keyed schema get demoted to
    the tail.
    """
    from hermes_cli.models import CANONICAL_PROVIDERS

    canonical_slug = CANONICAL_PROVIDERS[2].slug  # any canonical
    rows = [
        # A truly-custom row (correct: is_user_defined=True)
        {"slug": "custom:Ollama", "name": "Ollama", "models": [],
         "total_models": 0, "is_current": False, "is_user_defined": True,
         "source": "user-config"},
        # A canonical row that the substrate flagged as user-defined
        # because the user configured it via providers: dict.
        {"slug": canonical_slug, "name": "x", "models": ["m1"],
         "total_models": 1, "is_current": False, "is_user_defined": True,
         "source": "built-in"},
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx, canonical_order=True)
    slugs = [r["slug"] for r in payload["providers"]]
    # Canonical-slug row must come BEFORE truly-custom rows, regardless
    # of is_user_defined.
    canonical_idx = slugs.index(canonical_slug)
    custom_idx = slugs.index("custom:Ollama")
    assert canonical_idx < custom_idx, (
        f"canonical {canonical_slug} demoted to tail "
        f"(canonical_idx={canonical_idx} > custom_idx={custom_idx})"
    )


def test_canonical_order_with_unconfigured_preserves_full_universe():
    """Combined picker call: include_unconfigured + picker_hints +
    canonical_order is the production TUI shape. Verify the result
    has CANONICAL_PROVIDERS in declaration order, hints applied,
    custom rows trailing.
    """
    from hermes_cli.models import CANONICAL_PROVIDERS

    rows = [
        {"slug": "custom:Ollama", "name": "Ollama", "models": [],
         "total_models": 0, "is_current": False, "is_user_defined": True,
         "source": "user-config"},
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(
            ctx,
            include_unconfigured=True,
            picker_hints=True,
            canonical_order=True,
        )
    slugs = [r["slug"] for r in payload["providers"]]
    # First row: first canonical provider in declaration order.
    assert slugs[0] == CANONICAL_PROVIDERS[0].slug
    # Custom row trails canonical universe.
    assert slugs.index("custom:Ollama") >= len(CANONICAL_PROVIDERS)


# ─── Integration: end-to-end through real load_picker_context ──────────


def test_end_to_end_with_real_context_no_credentials_leak(monkeypatch):
    """Full pipeline: real load_picker_context + real
    list_authenticated_providers. Verify no credential string ever
    appears in the returned payload, even with picker_hints=True."""
    canary = "sk-canary-XYZ-must-not-appear"
    monkeypatch.setenv("OPENROUTER_API_KEY", canary)
    monkeypatch.setenv("ANTHROPIC_API_KEY", canary)
    cfg = _cfg(model={"provider": "openrouter"})
    with patch("hermes_cli.config.load_config", return_value=cfg):
        ctx = load_picker_context()
    payload = build_models_payload(
        ctx, include_unconfigured=True, picker_hints=True,
    )
    import json as _json

    assert canary not in _json.dumps(payload)


def test_payload_shape_compatible_with_modelpickerdialog_frontend():
    """Frontend (web/src/components/ModelPickerDialog.tsx) reads:
    name, slug, models, total_models, is_current, warning, authenticated.
    Verify every authenticated/skeleton row exposes those keys.
    """
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["m1"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "built-in"},
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(
            ctx, include_unconfigured=True, picker_hints=True,
        )
    required_keys = {"name", "slug", "models", "total_models", "is_current",
                     "authenticated"}
    for row in payload["providers"]:
        missing = required_keys - row.keys()
        assert not missing, f"row {row['slug']} missing keys: {missing}"
