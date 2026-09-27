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


def test_load_picker_context_coerces_numeric_yaml_provider():
    """YAML parses unquoted `provider: 2070` as int; picker context must be str.

    Desktop GET /api/model/options crashed when a custom endpoint was named
    after a GPU: current_provider.strip() and providers dict keys .lower().
    """
    cfg = _cfg(
        model={
            "provider": 2070,
            "default": "Qwen3.5-9B-Q4_K_M.gguf",
            "base_url": "http://192.168.1.10:8082/v1",
        },
        providers={
            2070: {
                "name": 2070,
                "base_url": "http://192.168.1.10:8082/v1",
                "model": "Qwen3.5-9B-Q4_K_M.gguf",
            }
        },
    )
    with patch("hermes_cli.config.load_config", return_value=cfg):
        ctx = load_picker_context()
    assert ctx.current_provider == "2070"
    assert isinstance(ctx.current_provider, str)
    assert list(ctx.user_providers) == ["2070"]
    assert all(isinstance(k, str) for k in ctx.user_providers)
    assert ctx.current_model == "Qwen3.5-9B-Q4_K_M.gguf"
    assert ctx.current_base_url == "http://192.168.1.10:8082/v1"


# ─── with_overrides ────────────────────────────────────────────────────


def _empty_ctx(provider="orig", model="orig-model", base_url="orig-url"):
    return ConfigContext(
        current_provider=provider,
        current_model=model,
        current_base_url=base_url,
        user_providers={},
        custom_providers=[],
    )






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


def test_explicit_only_keeps_authenticated_row_for_aliased_current():
    """The explicit-only filter must resolve current identity from context.

    Do not pre-seed ``is_current`` here: the row flag is intentionally false so
    this test proves that the filter maps ``google`` to the canonical ``gemini``
    row instead of dropping it and replacing it with a missing-credential
    skeleton. The alias hop goes through ``_picker_provider_identity``, not the
    general normalizer (which would collapse unrelated provider families).
    """
    row = {
        "slug": "gemini",
        "name": "Google AI Studio",
        "models": ["gemini-3.8-flash"],
        "total_models": 1,
        "is_current": False,
        "is_user_defined": False,
        "source": "built-in",
    }
    ctx = _empty_ctx(provider="google", model="gemini-3.8-flash")
    with (
        _list_auth_returning([row]),
        patch("hermes_cli.config.read_raw_config", return_value={}),
        patch("hermes_cli.auth.is_provider_explicitly_configured", return_value=False),
        patch("hermes_cli.inventory._anthropic_oauth_credentials_present", return_value=False),
        patch("hermes_cli.inventory._external_process_signed_in", return_value=False),
    ):
        payload = build_models_payload(ctx, explicit_only=True)

    assert payload["providers"] == [row]
    assert payload["providers"][0]["source"] == "built-in"


def test_explicit_only_keeps_aliased_current_skeleton_when_credentials_missing():
    """Accepted runtime aliases must restore the saved current-provider row.

    ``list_authenticated_providers`` omits a provider when its credential has
    disappeared. The explicit-only picker then appends a configured-current
    skeleton; accepted spellings such as ``google`` and ``hf`` must resolve to
    the canonical slug instead of leaving the current selection invisible.
    """
    ctx = _empty_ctx(provider="google", model="gemini-3.8-flash")
    with _list_auth_returning([]):
        payload = build_models_payload(ctx, explicit_only=True)

    row = next(r for r in payload["providers"] if r["slug"] == "gemini")
    assert row["is_current"] is True
    assert row["source"] == "configured-current"
    assert row["authenticated"] is False
    assert row["models"] == ["gemini-3.8-flash"]
    assert [r["slug"] for r in payload["providers"]] == ["gemini"]


def test_explicit_only_keeps_aliased_current_skeleton_for_regional_provider():
    """``kimi-cn`` is an accepted alias, not the exact canonical ID.

    It must restore the China-specific row rather than disappearing or
    collapsing onto the non-China Kimi provider.
    """
    ctx = _empty_ctx(provider="kimi-cn", model="kimi-k2.6")
    with _list_auth_returning([]):
        payload = build_models_payload(ctx, explicit_only=True)

    assert [r["slug"] for r in payload["providers"]] == ["kimi-coding-cn"]
    assert payload["providers"][0]["models"] == ["kimi-k2.6"]


def test_explicit_only_skeleton_does_not_fall_back_from_raw_openai_to_openrouter():
    """An explicit ``providers.openai`` entry is a distinct raw identity.

    The generic provider normalizer maps ``openai`` to OpenRouter, but that
    lossy fallback must never make the missing-credential skeleton show
    OpenRouter for a configured direct-OpenAI endpoint.
    """
    ctx = ConfigContext(
        current_provider="openai",
        current_model="gpt-5.6-sol",
        current_base_url="https://api.openai.com/v1",
        user_providers={
            "openai": {
                "name": "OpenAI Direct",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-5.6-sol",
            }
        },
        custom_providers=[],
    )
    with _list_auth_returning([]):
        payload = build_models_payload(ctx, explicit_only=True)

    assert "openrouter" not in [r["slug"] for r in payload["providers"]]
    assert "openai" not in [r["slug"] for r in payload["providers"]]
    assert payload["providers"] == []


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


def test_explicit_only_keeps_anthropic_row_with_oauth_credentials():
    """Anthropic OAuth logins are deliberate sign-ins, not ambient credentials.

    Claude Code (~/.claude/.credentials.json) and Hermes' own device flow
    leave no trace in active_provider / model.provider / API-key env vars,
    so is_provider_explicitly_configured() returns False even though
    list_authenticated_providers just accepted those same credentials when
    building the row. The desktop explicit-only filter must keep it.
    """
    rows = [
        {"slug": "anthropic", "name": "Anthropic", "models": ["claude-sonnet-5"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "hermes"},
        {"slug": "copilot", "name": "Copilot", "models": ["gpt-5.4"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "hermes"},
    ]
    ctx = _empty_ctx(provider="opencode-go", model="glm-5.3")
    with (
        _list_auth_returning(rows),
        patch("hermes_cli.config.read_raw_config", return_value={}),
        patch(
            "hermes_cli.auth.is_provider_explicitly_configured",
            return_value=False,
        ),
        patch(
            "hermes_cli.inventory._anthropic_oauth_credentials_present",
            return_value=True,
        ),
    ):
        payload = build_models_payload(ctx, explicit_only=True)

    slugs = [row["slug"] for row in payload["providers"]]
    assert "anthropic" in slugs, (
        "Anthropic OAuth login must survive the explicit-only filter"
    )
    assert "copilot" not in slugs, (
        "ambient credential discovery must stay filtered"
    )


def test_explicit_only_drops_anthropic_row_without_oauth_credentials():
    """No OAuth token and no explicit config -> Anthropic stays hidden."""
    rows = [
        {"slug": "anthropic", "name": "Anthropic", "models": ["claude-sonnet-5"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "hermes"},
    ]
    ctx = _empty_ctx(provider="opencode-go", model="glm-5.3")
    with (
        _list_auth_returning(rows),
        patch("hermes_cli.config.read_raw_config", return_value={}),
        patch(
            "hermes_cli.auth.is_provider_explicitly_configured",
            return_value=False,
        ),
        patch(
            "hermes_cli.inventory._anthropic_oauth_credentials_present",
            return_value=False,
        ),
    ):
        payload = build_models_payload(ctx, explicit_only=True)

    assert "anthropic" not in [row["slug"] for row in payload["providers"]]


def test_anthropic_oauth_presence_accepts_pool_only_oauth_entry():
    """A pool-only OAuth entry (auth.json credential_pool.anthropic) counts.

    Wired/device-flow tokens land in the credential pool, not in
    .anthropic_oauth.json or ~/.claude/.credentials.json. The presence
    check must accept them or the row is built and then silently dropped.
    """
    from hermes_cli.inventory import _anthropic_oauth_credentials_present

    with (
        patch(
            "agent.anthropic_credentials.read_hermes_oauth_credentials",
            return_value=None,
        ),
        patch(
            "agent.anthropic_credentials.read_claude_code_credentials",
            return_value=None,
        ),
        patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[
                {"auth_type": "oauth", "access_token": "sk-ant-oat01-pool"}
            ],
        ),
    ):
        assert _anthropic_oauth_credentials_present() is True

    # api_key pool entries are NOT OAuth logins — presence must stay False
    # (they are handled by the explicit-config gate / env var paths).
    with (
        patch(
            "agent.anthropic_credentials.read_hermes_oauth_credentials",
            return_value=None,
        ),
        patch(
            "agent.anthropic_credentials.read_claude_code_credentials",
            return_value=None,
        ),
        patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[
                {"auth_type": "api_key", "access_token": "sk-ant-api03-key"}
            ],
        ),
    ):
        assert _anthropic_oauth_credentials_present() is False



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


def test_user_defined_rows_carry_alias_set_for_gui_current_match():
    """Custom provider rows must expose `aliases` so the desktop picker can
    match a session's canonical `custom:<key>` identity against the row's
    bare-key slug (#87035). Built-in rows carry no aliases.
    """
    rows = [
        {
            "slug": "myep",
            "name": "My Endpoint",
            "models": ["my-model"],
            "total_models": 1,
            "is_current": True,
            "is_user_defined": True,
            "source": "user-config",
            "api_url": "http://localhost:8000/v1",
        },
        _nous_row() | {"is_current": False},
    ]
    ctx = _empty_ctx(provider="custom:myep", model="my-model")

    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    by_slug = {r["slug"]: r for r in payload["providers"]}
    aliases = by_slug["myep"]["aliases"]
    # The canonical session identity must be matchable via the alias set.
    assert "custom:myep" in aliases
    assert "myep" in aliases
    assert "custom:my-endpoint" in aliases
    assert "aliases" not in by_slug["nous"]


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




def test_current_provider_keeps_selected_overlap_in_picker():
    """The dedup stage preserves an overlap on a row already marked current.

    Identity resolution is covered by the authenticated-row builder tests;
    this unit fixture deliberately isolates the second-stage dedup contract.
    """
    target = "stealth/space-bunny-alpha"
    rows = [
        _user_provider_row("custom:freebuff", [target]),
        _aggregator_row("openrouter", [target, "anthropic/claude-sonnet-4.6"]),
    ]
    for row in rows:
        if row["slug"] == "openrouter":
            row["is_current"] = True
    ctx = _empty_ctx(provider="openrouter", model=target)
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    openrouter = next(r for r in payload["providers"] if r["slug"] == "openrouter")
    assert target in openrouter["models"]
    assert openrouter["total_models"] == 2


def test_inactive_noncurrent_row_still_loses_overlap():
    """Identity equality alone never preserves an overlap.

    This isolates dedup's row-local ``is_current`` guard. The requested
    provider is OpenRouter while the only candidate row is inactive Kilo;
    even if those names shared an identity, the overlap would still have to
    be removed from the non-current row.
    """
    target = "stealth/space-bunny-alpha"
    rows = [
        _user_provider_row("custom:freebuff", [target]),
        _aggregator_row("kilocode", [target, "anthropic/claude-sonnet-4.6"]),
    ]
    for row in rows:
        if row["slug"] == "kilocode":
            row["is_current"] = False
    ctx = _empty_ctx(provider="OpenRouter", model=target)
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    kilo = next(r for r in payload["providers"] if r["slug"] == "kilocode")
    assert kilo["is_current"] is False
    assert target not in kilo["models"]
    assert kilo["total_models"] == 1


def test_configured_raw_openai_current_does_not_wedge_the_dedup():
    """A configured ``providers.openai`` resolves to identity ``openai``, which is NOT a canonical
    slug — so no *canonical* row is ever ``is_current`` and the current-model escape hatch in
    ``_strip_aggregator_overlaps`` cannot fire.

    That is still correct: the configured row is ``is_user_defined``, and the dedup only ever mutates
    non-user-defined routing-aggregator rows, so the user's own row is untouched and the built-in
    OpenRouter row is stripped as intended. Pinned here because the behaviour depends on three
    separate guards agreeing; if any one changes, this fails loudly instead of silently
    under-preserving.
    """
    target = "stealth/space-bunny-alpha"
    user_row = _user_provider_row("openai", [target])
    rows = [user_row, _aggregator_row("openrouter", [target, "anthropic/claude-sonnet-4.6"])]
    # ``with_overrides`` only covers provider/model/base_url, so build the context directly:
    # ``user_providers`` is the field that makes ``openai`` a CONFIGURED raw key.
    ctx = ConfigContext(
        current_provider="openai",
        current_model=target,
        current_base_url="",
        user_providers={"openai": {"base_url": "https://api.openai.com/v1"}},
        custom_providers=[],
    )
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    # The user's own row keeps the model it advertises - the dedup never touches user-defined rows.
    mine = next(r for r in payload["providers"] if r["slug"] == "openai")
    assert target in mine["models"]
    # The built-in aggregator loses the overlap, because it is not the selected route.
    builtin = next(r for r in payload["providers"] if r["slug"] == "openrouter")
    assert target not in builtin["models"]
    assert builtin["total_models"] == 1


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




def test_list_authenticated_providers_refresh_busts_cache():
    """refresh=True clears the provider-model disk cache exactly once;
    refresh=False leaves it untouched (so normal picker opens stay snappy)."""
    from hermes_cli import model_switch

    with patch("hermes_cli.models.clear_provider_models_cache") as clear:
        model_switch.list_authenticated_providers(refresh=False)
        assert clear.call_count == 0
        model_switch.list_authenticated_providers(refresh=True)
        assert clear.call_count == 1


def test_picker_metadata_uses_one_config_read_for_real_models_dev_lookups(tmp_path, monkeypatch):
    """Custom-provider metadata stays constant-read as its model count grows (#119048).

    ``get_model_capabilities`` and ``get_model_info`` deliberately stay real:
    each lookup resolves ``providers.lab.catalog_provider`` before consulting
    the seeded models.dev catalog.  Removing snapshot threading from that
    path makes the larger payload re-open config.yaml once per lookup.
    """
    from agent import models_dev
    from hermes_cli import config as config_module

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "providers:\n"
        "  lab:\n"
        "    catalog_provider: openrouter\n"
        "model_overrides: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()

    models = [
        "openai/model-a", "anthropic/model-b", "openai/model-c",
        "anthropic/model-d", "openai/model-e", "anthropic/model-f",
        "openai/model-g", "anthropic/model-h",
    ]
    registry = {
        "openrouter": {
            "models": {
                model: {
                    "id": model,
                    "tool_call": True,
                    "reasoning": model.startswith("openai/"),
                    "release_date": f"2026-01-{index:02d}",
                    "limit": {"context": 200000, "output": 8192},
                }
                for index, model in enumerate(models, start=1)
            },
        },
    }
    monkeypatch.setattr(models_dev, "_models_dev_cache", registry)
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", float("inf"))

    snapshot = config_module.load_config_readonly()
    baseline = [
        (
            models_dev.get_model_capabilities("custom:lab", model),
            models_dev.get_model_info("custom:lab", model),
        )
        for model in models
    ]
    snapshot_result = [
        (
            models_dev.get_model_capabilities("custom:lab", model, config=snapshot),
            models_dev.get_model_info("custom:lab", model, config=snapshot),
        )
        for model in models
    ]
    assert snapshot_result == baseline

    def _rows(model_ids):
        return [{
            "slug": "custom:lab",
            "name": "Lab",
            "models": model_ids,
            "total_models": len(model_ids),
            "is_current": False,
            "is_user_defined": True,
            "source": "user-config",
        }]

    def _build_and_count(model_ids):
        cfg_get_calls = 0
        real_cfg_get = models_dev._cfg_get

        def counted_cfg_get(*keys, **kwargs):
            nonlocal cfg_get_calls
            cfg_get_calls += 1
            return real_cfg_get(*keys, **kwargs)

        with (
            _list_auth_returning(_rows(model_ids)),
            patch("hermes_cli.inventory._local_runtime_row", return_value=None),
            patch("hermes_cli.inventory._moa_provider_row", return_value=None),
            patch("hermes_cli.models.model_supports_fast_mode", return_value=False),
            patch("hermes_cli.inventory._reasoning_catalog_reader", return_value=None),
            patch.object(models_dev, "_cfg_get", side_effect=counted_cfg_get),
            patch.object(
                config_module, "load_config_readonly",
                wraps=config_module.load_config_readonly,
            ) as readonly_load,
            patch.object(
                config_module, "_load_config_impl", wraps=config_module._load_config_impl,
            ) as config_impl,
        ):
            payload = build_models_payload(
                _empty_ctx(), capabilities=True, featured=True,
            )
        return payload, cfg_get_calls, readonly_load.call_count, config_impl.call_count

    small_payload, small_cfg_get, small_reads, small_impls = _build_and_count(models[:3])
    large_payload, large_cfg_get, large_reads, large_impls = _build_and_count(models)

    # Snapshot threading leaves _cfg_get's cheap dict traversals proportional
    # to model count, while eliminating config/signature reads from the hot path.
    assert small_cfg_get < large_cfg_get
    assert (small_reads, large_reads) == (1, 1)
    assert (small_impls, large_impls) == (1, 1)

    small_row = small_payload["providers"][0]
    large_row = large_payload["providers"][0]
    assert small_row["capabilities"] == {
        model: {"fast": False, "reasoning": model.startswith("openai/")}
        for model in models[:3]
    }
    assert large_row["featured_models"] == models
