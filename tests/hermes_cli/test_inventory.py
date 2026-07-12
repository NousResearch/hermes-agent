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


def test_load_picker_context_normalizes_list_of_dict_models():
    cfg = _cfg(
        providers={
            "static-gateway": {
                "name": "Static Gateway",
                "api": "https://router.example.com/v1",
                "default_model": "claude-3-7-sonnet",
                "models": [
                    {"id": "claude-3-7-sonnet"},
                    {"id": "claude-sonnet-4", "context_length": 200000},
                ],
                "discover_models": False,
            }
        },
    )
    with patch("hermes_cli.config.load_config", return_value=cfg):
        ctx = load_picker_context()

    assert len(ctx.custom_providers) == 1
    assert ctx.custom_providers[0]["models"] == {
        "claude-3-7-sonnet": {},
        "claude-sonnet-4": {"context_length": 200000},
    }
    assert ctx.custom_providers[0]["discover_models"] is False


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


def _nous_row(model: str = "openai/gpt-5.5") -> dict:
    return {
        "slug": "nous",
        "name": "Nous",
        "models": [model],
        "total_models": 1,
        "is_current": True,
        "is_user_defined": False,
        "source": "built-in",
    }


def test_build_models_payload_returns_expected_shape():
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["m1"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "built-in"},
    ]
    ctx = _empty_ctx(provider="openrouter", model="m1", base_url="")
    raw_config = {
        "moa": {
            "presets": {
                "default": {"enabled": True},
            },
        },
    }
    with (
        _list_auth_returning(rows),
        patch("hermes_cli.config.read_raw_config", return_value=raw_config),
        patch("hermes_cli.config.load_config", return_value=raw_config),
    ):
        payload = build_models_payload(ctx)
    assert set(payload.keys()) == {"providers", "model", "provider"}
    assert payload["model"] == "m1"
    assert payload["provider"] == "openrouter"
    assert payload["providers"][0]["slug"] == "moa"
    assert payload["providers"][0]["models"] == ["default"]
    assert payload["providers"][1:] == rows


def test_build_models_payload_does_not_call_provider_model_ids():
    """``build_models_payload`` is a thin shape adapter — it delegates the
    actual curation to ``list_authenticated_providers`` (which DOES call
    ``cached_provider_model_ids`` internally for live discovery, with disk
    caching). ``build_models_payload`` itself must not call the live fetcher
    directly; the test pins that boundary.
    """
    rows = [{"slug": "nous", "name": "Nous", "models": ["hermes-4-405b"],
             "total_models": 1, "is_current": False, "is_user_defined": False,
             "source": "built-in"}]
    ctx = _empty_ctx()
    with _list_auth_returning(rows), \
         patch("hermes_cli.models.provider_model_ids") as mock_pm:
        build_models_payload(ctx)
    mock_pm.assert_not_called()


def test_build_models_payload_uses_cached_nous_tier_by_default():
    """Picker payloads should not force fresh Nous account checks.

    Desktop/status picker opens are request/response UI paths. They can hit
    the short free-tier cache; explicit model/auth flows can still opt into a
    fresh account check when needed.
    """
    ctx = _empty_ctx(provider="nous", model="openai/gpt-5.5")
    rows = [_nous_row()]
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=rows,
    ) as mock_list:
        build_models_payload(ctx)

    mock_list.assert_called_once()
    assert mock_list.call_args.kwargs["force_fresh_nous_tier"] is False


def test_build_models_payload_can_force_fresh_nous_tier():
    ctx = _empty_ctx(provider="nous", model="openai/gpt-5.5")
    rows = [_nous_row()]
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=rows,
    ) as mock_list:
        build_models_payload(ctx, force_fresh_nous_tier=True)

    mock_list.assert_called_once()
    assert mock_list.call_args.kwargs["force_fresh_nous_tier"] is True


def test_build_models_payload_can_skip_custom_provider_probes():
    ctx = _empty_ctx()
    rows = []
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=rows,
    ) as mock_list:
        build_models_payload(ctx, probe_custom_providers=False)

    mock_list.assert_called_once()
    assert mock_list.call_args.kwargs["probe_custom_providers"] is False


def test_build_models_payload_can_probe_only_current_custom_provider():
    ctx = _empty_ctx()
    rows = []
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=rows,
    ) as mock_list:
        build_models_payload(
            ctx,
            probe_custom_providers=False,
            probe_current_custom_provider=True,
        )

    mock_list.assert_called_once()
    assert mock_list.call_args.kwargs["probe_custom_providers"] is False
    assert mock_list.call_args.kwargs["probe_current_custom_provider"] is True


def test_list_authenticated_providers_force_fresh_is_keyword_only():
    """``force_fresh_nous_tier`` must be keyword-only on the public listing API.

    It was inserted between ``custom_providers`` and ``max_models``; making it
    keyword-only ensures no positional caller passing ``max_models`` as the 5th
    arg silently mis-binds it to the tier-refresh flag. Pin the contract so a
    future signature edit that drops the ``*`` separator is caught.
    """
    import inspect

    from hermes_cli.model_switch import list_authenticated_providers

    sig = inspect.signature(list_authenticated_providers)
    param = sig.parameters["force_fresh_nous_tier"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False


def test_pricing_uses_cached_nous_tier_by_default():
    rows = [_nous_row()]
    ctx = _empty_ctx(provider="nous", model="openai/gpt-5.5")
    with (
        _list_auth_returning(rows),
        patch(
            "hermes_cli.models.get_pricing_for_provider",
            return_value={
                "openai/gpt-5.5": {
                    "prompt": "0.000001",
                    "completion": "0.000002",
                },
            },
        ),
        patch("hermes_cli.models.check_nous_free_tier", return_value=False) as mock_free,
    ):
        build_models_payload(ctx, pricing=True)

    mock_free.assert_called_once_with(force_fresh=False)


def test_pricing_can_force_fresh_nous_tier():
    rows = [_nous_row()]
    ctx = _empty_ctx(provider="nous", model="openai/gpt-5.5")
    with (
        _list_auth_returning(rows),
        patch(
            "hermes_cli.models.get_pricing_for_provider",
            return_value={
                "openai/gpt-5.5": {
                    "prompt": "0.000001",
                    "completion": "0.000002",
                },
            },
        ),
        patch("hermes_cli.models.check_nous_free_tier", return_value=False) as mock_free,
    ):
        build_models_payload(ctx, pricing=True, force_fresh_nous_tier=True)

    mock_free.assert_called_once_with(force_fresh=True)


def test_include_unconfigured_appends_canonical_skeletons():
    """include_unconfigured=True adds CANONICAL_PROVIDERS rows that
    list_authenticated_providers didn't emit. Skeleton rows have empty
    models and source='canonical'. MoA is a virtual routing mode, not a
    canonical provider, so it is excluded — see issue #63353."""
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["m1"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "built-in"},
    ]
    ctx = _empty_ctx(provider="openrouter")
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx, include_unconfigured=True)
    # All canonical providers other than openrouter should appear as
    # skeleton rows. MoA is virtual/opt-in and excluded from unconfigured.
    from hermes_cli.models import CANONICAL_PROVIDERS

    seen_slugs = {r["slug"] for r in payload["providers"]}
    for entry in CANONICAL_PROVIDERS:
        if entry.slug == "moa":
            continue  # virtual; only shown when explicitly configured
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


def test_explicit_only_filters_ambient_credentials_but_keeps_current_and_custom_rows():
    rows = [
        {"slug": "openai-codex", "name": "OpenAI Codex", "models": ["gpt-5.4"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "hermes"},
        {"slug": "gemini", "name": "Gemini", "models": ["gemini-2.5-pro"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "built-in"},
        {"slug": "copilot", "name": "Copilot", "models": ["gpt-5.4"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "hermes"},
        {"slug": "nous", "name": "Nous", "models": ["anthropic/claude-sonnet-5"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "hermes"},
        {"slug": "custom:lab", "name": "Lab", "models": ["lab-1"],
         "total_models": 1, "is_current": False, "is_user_defined": True,
         "source": "user-config"},
        {"slug": "moa", "name": "MoA", "models": ["default"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "virtual"},
    ]
    ctx = _empty_ctx(provider="openai-codex", model="gpt-5.4")
    with (
        _list_auth_returning(rows),
        patch("hermes_cli.config.read_raw_config", return_value={}),
        patch(
            "hermes_cli.auth.is_provider_explicitly_configured",
            side_effect=lambda slug: slug == "gemini",
        ),
    ):
        payload = build_models_payload(ctx, explicit_only=True)

    assert [row["slug"] for row in payload["providers"]] == [
        "openai-codex",
        "gemini",
        "custom:lab",
    ]


def test_explicit_only_keeps_unauthenticated_current_provider_visible():
    """Desktop's configured-only picker must retain its saved provider row."""
    ctx = _empty_ctx(provider="deepseek", model="deepseek-v4-pro")
    with _list_auth_returning([]):
        payload = build_models_payload(
            ctx,
            explicit_only=True,
            picker_hints=True,
        )

    assert [row["slug"] for row in payload["providers"]] == ["deepseek"]
    row = payload["providers"][0]
    assert row["source"] == "configured-current"
    assert row["authenticated"] is False
    assert row["models"] == ["deepseek-v4-pro"]
    assert "DEEPSEEK_API_KEY" in row["warning"]
def test_include_unconfigured_keeps_current_provider_visible_without_credentials():
    """If the saved provider is currently unauthenticated, keep a visible row
    with the saved model so GUI pickers don't silently jump to another
    authenticated provider."""
    ctx = _empty_ctx(provider="deepseek", model="deepseek-v4-pro")
    with _list_auth_returning([]):
        payload = build_models_payload(
            ctx, include_unconfigured=True, picker_hints=True,
        )

    deepseek = next(r for r in payload["providers"] if r["slug"] == "deepseek")
    assert deepseek["source"] == "configured-current"
    assert deepseek["is_current"] is True
    assert deepseek["authenticated"] is False
    assert deepseek["models"] == ["deepseek-v4-pro"]
    assert deepseek["total_models"] == 1
    assert deepseek["auth_type"] == "api_key"
    assert "DEEPSEEK_API_KEY" in deepseek["warning"]
    assert "saved model only" in deepseek["warning"]


def test_include_unconfigured_does_not_duplicate_configured_current_row():
    ctx = _empty_ctx(provider="deepseek", model="deepseek-v4-pro")
    with _list_auth_returning([]):
        payload = build_models_payload(
            ctx,
            explicit_only=True,
            include_unconfigured=True,
            picker_hints=True,
        )

    assert sum(row["slug"] == "deepseek" for row in payload["providers"]) == 1

def test_explicit_only_keeps_moa_when_raw_config_has_enabled_preset():
    rows = [
        {"slug": "moa", "name": "MoA", "models": ["review"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "virtual"},
    ]
    ctx = _empty_ctx(provider="openrouter", model="anthropic/claude-opus-4.8")
    raw_config = {
        "moa": {
            "active_preset": "review",
            "presets": {
                "review": {
                    "enabled": True,
                    "reference_models": [
                        {"provider": "openai-codex", "model": "gpt-5.5"},
                    ],
                    "aggregator": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-opus-4.8",
                    },
                },
            },
        },
    }

    with (
        _list_auth_returning(rows),
        patch("hermes_cli.config.load_config", return_value=raw_config),
        patch("hermes_cli.config.read_raw_config", return_value=raw_config),
        patch("hermes_cli.auth.is_provider_explicitly_configured", return_value=False),
    ):
        payload = build_models_payload(ctx, explicit_only=True)

    assert [row["slug"] for row in payload["providers"]] == ["moa", "openrouter"]
    assert payload["providers"][0]["models"] == ["review"]
    assert payload["providers"][1]["source"] == "configured-current"
    assert payload["providers"][1]["authenticated"] is False
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
    custom rows trailing. MoA is excluded from unconfigured skeletons
    (virtual, opt-in only — issue #63353).
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
    # Custom row trails all visible canonical rows. MoA is virtual/opt-in
    # so it is excluded from the unconfigured skeleton set.
    visible_canonical_count = sum(
        1 for e in CANONICAL_PROVIDERS if e.slug != "moa"
    )
    assert slugs.index("custom:Ollama") >= visible_canonical_count


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


# ─── Aggregator dedup (issue #45954) ───────────────────────────────────


def _user_provider_row(slug: str, models: list[str]) -> dict:
    return {
        "slug": slug,
        "name": slug.title(),
        "models": models,
        "total_models": len(models),
        "is_current": False,
        "is_user_defined": True,
        "source": "user-config",
    }


def _aggregator_row(slug: str, models: list[str]) -> dict:
    return {
        "slug": slug,
        "name": slug.title(),
        "models": models,
        "total_models": len(models),
        "is_current": False,
        "is_user_defined": False,
        "source": "built-in",
    }


def test_aggregator_dedup_removes_overlapping_models():
    """Models served by a user-defined provider are removed from
    aggregator rows so the picker doesn't show them under the wrong
    provider.  (#45954)"""
    rows = [
        _user_provider_row("litellm-proxy", [
            "nvidia/nim/minimax-m3",
            "nvidia/nim/kimi-k2.6",
        ]),
        _aggregator_row("openrouter", [
            "minimax/minimax-m3",
            "nvidia/nim/minimax-m3",  # overlaps with litellm-proxy
            "anthropic/claude-sonnet-4.6",
        ]),
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    or_row = next(r for r in payload["providers"] if r["slug"] == "openrouter")
    proxy_row = next(r for r in payload["providers"] if r["slug"] == "litellm-proxy")

    # User-defined provider keeps all its models
    assert proxy_row["models"] == ["nvidia/nim/minimax-m3", "nvidia/nim/kimi-k2.6"]

    # Aggregator lost the overlapping model but kept the rest
    assert "nvidia/nim/minimax-m3" not in or_row["models"]
    assert "minimax/minimax-m3" in or_row["models"]
    assert "anthropic/claude-sonnet-4.6" in or_row["models"]
    assert or_row["total_models"] == 2


def test_aggregator_dedup_case_insensitive():
    """Dedup uses case-insensitive matching.  (#45954)"""
    rows = [
        _user_provider_row("my-proxy", ["NVIDIA/NIM/MiniMax-M3"]),
        _aggregator_row("openrouter", ["nvidia/nim/minimax-m3", "other/model"]),
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    or_row = next(r for r in payload["providers"] if r["slug"] == "openrouter")
    assert "nvidia/nim/minimax-m3" not in or_row["models"]
    assert or_row["total_models"] == 1


def test_aggregator_dedup_no_overlap_unchanged():
    """When there's no overlap, aggregator models are untouched.  (#45954)"""
    rows = [
        _user_provider_row("litellm-proxy", ["custom/model-a"]),
        _aggregator_row("openrouter", ["anthropic/claude-sonnet-4.6"]),
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    or_row = next(r for r in payload["providers"] if r["slug"] == "openrouter")
    assert or_row["models"] == ["anthropic/claude-sonnet-4.6"]
    assert or_row["total_models"] == 1


def test_aggregator_dedup_no_user_providers_unchanged():
    """When there are no user-defined providers, nothing is filtered.
    (#45954)"""
    rows = [
        _aggregator_row("openrouter", [
            "nvidia/nim/minimax-m3",
            "anthropic/claude-sonnet-4.6",
        ]),
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    or_row = next(r for r in payload["providers"] if r["slug"] == "openrouter")
    assert len(or_row["models"]) == 2


def test_aggregator_dedup_multiple_user_providers():
    """Models from all user-defined providers are excluded from aggregators.
    (#45954)"""
    rows = [
        _user_provider_row("proxy-a", ["model-x"]),
        _user_provider_row("proxy-b", ["model-y"]),
        _aggregator_row("openrouter", ["model-x", "model-y", "model-z"]),
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    or_row = next(r for r in payload["providers"] if r["slug"] == "openrouter")
    assert or_row["models"] == ["model-z"]
    assert or_row["total_models"] == 1


def test_aggregator_dedup_does_not_empty_user_defined_custom_provider():
    """A named custom provider has slug ``custom:<name>``, which makes it
    *both* ``is_user_defined=True`` *and* ``is_aggregator()==True``
    (is_aggregator reports True for every ``custom:*`` slug).  The dedup
    must skip user-defined rows: their models populate ``user_models``, so
    filtering them against that set would strip the row's entire catalog and
    hide the provider from the picker.  Regression for the #45954 dedup
    emptying ``custom:*`` providers (e.g. a local llama.cpp endpoint or an
    Anthropic-compatible proxy)."""
    rows = [
        _user_provider_row("custom:my-proxy", ["my-model-a", "my-model-b"]),
        _aggregator_row("openrouter", ["my-model-a", "other/model"]),
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    proxy_row = next(
        r for r in payload["providers"] if r["slug"] == "custom:my-proxy"
    )
    or_row = next(r for r in payload["providers"] if r["slug"] == "openrouter")

    # The user's own custom provider keeps all of its models.
    assert proxy_row["models"] == ["my-model-a", "my-model-b"]
    assert proxy_row["total_models"] == 2

    # A genuine aggregator is still deduped against the user's models.
    assert "my-model-a" not in or_row["models"]
    assert "other/model" in or_row["models"]
    assert or_row["total_models"] == 1


def test_flat_namespace_reseller_keeps_first_party_models_overlapping_user_proxy():
    """opencode-go / opencode-zen are flagged ``is_aggregator=True`` (their
    flat ``/v1/models`` returns bare IDs the model-switch resolver searches),
    but they are NOT routing aggregators — every model they list is a
    first-party model under the user's subscription. When a user also runs a
    custom proxy that happens to serve a same-named model, the picker dedup
    must NOT strip the reseller's own catalog. Regression for #47077, where
    opencode-go showed only 13 of 19 models because minimax-m3/m2.7/m2.5,
    glm-5/5.1, and deepseek-v4-flash were deduped against an overlapping
    custom provider.
    """
    rows = [
        _user_provider_row("custom:my-proxy", [
            "minimax-m3", "minimax-m2.7", "glm-5", "deepseek-v4-flash",
        ]),
        _aggregator_row("opencode-go", [
            "kimi-k2.6", "minimax-m3", "minimax-m2.7", "glm-5",
            "deepseek-v4-flash", "qwen3.7-max",
        ]),
        _aggregator_row("openrouter", ["minimax-m3", "anthropic/claude-sonnet-4.6"]),
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    go_row = next(r for r in payload["providers"] if r["slug"] == "opencode-go")
    or_row = next(r for r in payload["providers"] if r["slug"] == "openrouter")

    # The reseller keeps ALL of its first-party models — nothing stripped.
    assert go_row["models"] == [
        "kimi-k2.6", "minimax-m3", "minimax-m2.7", "glm-5",
        "deepseek-v4-flash", "qwen3.7-max",
    ]
    assert go_row["total_models"] == 6

    # A TRUE routing aggregator is still deduped against the user's models.
    assert "minimax-m3" not in or_row["models"]
    assert "anthropic/claude-sonnet-4.6" in or_row["models"]


def test_two_custom_providers_with_overlap_both_survive():
    """Two user-defined custom endpoints that happen to expose an
    overlapping model must each keep their full catalog. Neither is the
    aggregator the dedup exists to trim, so cross-filtering between two
    user-defined rows must not happen.
    """
    rows = [
        _user_provider_row("custom:proxy-a", ["shared/model", "a/only"]),
        _user_provider_row("custom:proxy-b", ["shared/model", "b/only"]),
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    a_row = next(r for r in payload["providers"] if r["slug"] == "custom:proxy-a")
    b_row = next(r for r in payload["providers"] if r["slug"] == "custom:proxy-b")
    assert a_row["models"] == ["shared/model", "a/only"]
    assert b_row["models"] == ["shared/model", "b/only"]
    assert a_row["total_models"] == 2
    assert b_row["total_models"] == 2


def test_build_models_payload_keeps_static_provider_models_from_providers_dict():
    """The inventory payload must keep configured static models from a
    ``providers:`` entry even when the same endpoint also appears via the
    compatibility ``custom_providers`` view and live discovery would fail."""
    cfg = _cfg(
        model={
            "provider": "static-gateway",
            "default": "claude-3-7-sonnet",
        },
        providers={
            "static-gateway": {
                "name": "Static Gateway",
                "api": "https://router.example.com/v1",
                "api_key": "sk-test",
                "default_model": "claude-3-7-sonnet",
                "models": [
                    {"id": "claude-3-7-sonnet"},
                    {"id": "claude-sonnet-4"},
                ],
                "discover_models": False,
            }
        },
    )
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("agent.models_dev.fetch_models_dev", return_value={}),
        patch("hermes_cli.providers.HERMES_OVERLAYS", {}),
        patch(
            "hermes_cli.models.fetch_api_models",
            side_effect=AssertionError("fetch_api_models must not be called"),
        ),
    ):
        ctx = load_picker_context()
        payload = build_models_payload(ctx)

    rows = [
        row
        for row in payload["providers"]
        if row.get("api_url") == "https://router.example.com/v1"
    ]
    assert len(rows) == 1
    assert rows[0]["slug"] == "static-gateway"
    assert rows[0]["models"] == ["claude-3-7-sonnet", "claude-sonnet-4"]
    assert rows[0]["total_models"] == 2


def test_build_models_payload_no_max_models_returns_full_list():
    """When max_models is not passed (None), build_models_payload must
    return the full model list — not truncate to the old default of 50.
    Regression for #48279: Kilo Gateway picker was capped at 50 of 336
    models, making most models undiscoverable via search."""
    full_models = [f"model-{i}" for i in range(100)]
    rows = [
        {
            "slug": "kilocode",
            "name": "Kilo Code",
            "models": full_models,
            "total_models": len(full_models),
            "is_current": False,
            "is_user_defined": False,
            "source": "built-in",
        },
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        # No max_models argument — should return all 100 models
        payload = build_models_payload(ctx)

    kilo_row = next(r for r in payload["providers"] if r["slug"] == "kilocode")
    assert kilo_row["models"] == full_models
    assert kilo_row["total_models"] == 100
    assert len(kilo_row["models"]) == 100


# ─── refresh flag (cache-bust) ─────────────────────────────────────────


def test_build_models_payload_forwards_refresh_flag():
    """build_models_payload must forward refresh= to list_authenticated_providers.

    The desktop picker's "Refresh Models" control passes refresh=True; the
    flag has to reach list_authenticated_providers so the per-provider
    model-id cache gets busted. Default opens pass refresh=False.
    """
    captured: dict = {}

    def _capture(*args, **kwargs):
        captured["refresh"] = kwargs.get("refresh")
        return []

    with patch("hermes_cli.model_switch.list_authenticated_providers", side_effect=_capture):
        build_models_payload(_empty_ctx())
    assert captured["refresh"] is False

    with patch("hermes_cli.model_switch.list_authenticated_providers", side_effect=_capture):
        build_models_payload(_empty_ctx(), refresh=True)
    assert captured["refresh"] is True


def test_list_authenticated_providers_refresh_busts_cache():
    """refresh=True clears the provider-model disk cache exactly once;
    refresh=False leaves it untouched (so normal picker opens stay snappy)."""
    from hermes_cli import model_switch

    with patch("hermes_cli.models.clear_provider_models_cache") as clear:
        model_switch.list_authenticated_providers(refresh=False)
        assert clear.call_count == 0
        model_switch.list_authenticated_providers(refresh=True)
        assert clear.call_count == 1
