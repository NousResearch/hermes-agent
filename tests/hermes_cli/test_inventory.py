"""Tests for ``hermes_cli.inventory`` — the issue #23359 consolidation.

Locks invariants only:
  - Concept A: ``offline=True`` does not call ``fetch_models_dev``.
  - Concept B: ``offline=True`` does not call any provider live HTTP.
  - Concept C: status payload is HTTP-free (no ``provider_model_ids``).
  - H2: probe-skip predicate covers oauth_external + aws_sdk + api_key
        without env var.
  - H3: openrouter offline uses the local OPENROUTER_MODELS constant.
  - H4: ``--all`` populates models from curated/static for unconfigured.
  - H5: provider-arg resolution accepts canonical, custom:NAME, profile
        aliases, and bare custom names.
  - H6: env-only auth derivation (offline path) doesn't read auth.json /
        credential pool / external auth stores.
  - W1: canonical-slug rows from ``providers:`` config dict slot into
        canonical position (not user-defined tail).
  - Schema: payload always has ``schema_version``/``current``/``providers``.
  - Mutex: unknown provider raises ``ValueError``; CLI exits 2.
  - Credential leak: env-var values never appear in the JSON payload.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    h = tmp_path / "hermes_home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    for var in [
        "ARCEEAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "NOUS_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_PROFILE",
        "GH_TOKEN",
        "COPILOT_GITHUB_TOKEN",
        "DEEPSEEK_API_KEY",
        "GLM_API_KEY",
        "ZAI_API_KEY",
        "STEPFUN_API_KEY",
        "GMI_API_KEY",
        "LM_BASE_URL",
        "LM_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "QWEN_API_KEY",
        "MOONSHOT_API_KEY",
        "MINIMAX_API_KEY",
        "BEDROCK_API_KEY",
        "CEREBRAS_API_KEY",
        "VERCEL_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)
    return h


@pytest.fixture
def fail_on_network(monkeypatch):
    """Concept A+B canary: monkeypatch every live-fetch helper to fail."""
    sentinels: list[str] = []

    def _fail(name: str):
        def _f(*a: Any, **k: Any) -> Any:
            sentinels.append(name)
            raise AssertionError(f"network call to {name} during offline path")

        return _f

    from agent import models_dev as md
    from hermes_cli import auth as ha
    from hermes_cli import models as hm
    from hermes_cli import model_switch as ms

    for fn in [
        "fetch_anthropic_models",
        "fetch_ai_gateway_models",
        "fetch_ollama_cloud_models",
        "fetch_lmstudio_models",
        "fetch_openrouter_models",
        "_fetch_github_models",
        "fetch_api_models",
        "fetch_nous_models",
    ]:
        if hasattr(hm, fn):
            monkeypatch.setattr(hm, fn, _fail(f"hermes_cli.models.{fn}"))
        if hasattr(ha, fn):
            monkeypatch.setattr(ha, fn, _fail(f"hermes_cli.auth.{fn}"))
    for fn in [
        "fetch_models_dev",
        "fetch_ollama_cloud_models",
        "fetch_lmstudio_models",
        "fetch_openrouter_models",
    ]:
        if hasattr(md, fn):
            monkeypatch.setattr(md, fn, _fail(f"agent.models_dev.{fn}"))
    if hasattr(hm, "model_ids"):
        monkeypatch.setattr(hm, "model_ids", _fail("hermes_cli.models.model_ids"))
    if hasattr(hm, "get_curated_nous_model_ids"):
        monkeypatch.setattr(hm, "get_curated_nous_model_ids", _fail("get_curated_nous_model_ids"))
    try:
        from agent import bedrock_adapter as ba
    except ImportError:
        ba = None
    if ba is not None and hasattr(ba, "bedrock_model_ids_or_none"):
        monkeypatch.setattr(ba, "bedrock_model_ids_or_none", _fail("bedrock_model_ids_or_none"))
    monkeypatch.setattr(ms, "list_authenticated_providers", _fail("list_authenticated_providers"))
    return sentinels


# ─── Concept canaries ───────────────────────────────────────────────────


def test_concept_a_offline_skips_models_dev(isolated_home, fail_on_network):
    from hermes_cli import inventory

    inventory.build_payload("providers", include_unconfigured=True, offline=True)
    inventory.build_payload("models", live=False, include_unconfigured=True, offline=True)
    inventory.build_payload("status", offline=True)

    leaked = [s for s in fail_on_network if "models_dev" in s]
    assert leaked == [], f"models.dev fetched when offline: {leaked}"


def test_concept_b_offline_skips_provider_live(isolated_home, fail_on_network):
    from hermes_cli import inventory

    inventory.build_payload("providers", include_unconfigured=True, offline=True)
    inventory.build_payload("models", live=False, include_unconfigured=True, offline=True)
    inventory.build_payload("status", offline=True)

    assert fail_on_network == [], (
        f"provider HTTP / remote-catalog leaked when offline: {fail_on_network}"
    )


def test_concept_c_status_does_not_call_provider_model_ids(isolated_home, monkeypatch):
    """``build_payload('status')`` is an auth snapshot — no probe."""
    from hermes_cli import inventory
    from hermes_cli import models

    seen: list[str] = []
    real = models.provider_model_ids

    def _spy(slug: str, *a: Any, **k: Any) -> Any:
        seen.append(slug)
        return real(slug, *a, **k)

    monkeypatch.setattr(models, "provider_model_ids", _spy)
    inventory.build_payload("status", offline=True)
    assert seen == [], f"provider_model_ids called {seen} from build_payload('status')"


def test_status_payload_omits_models(isolated_home):
    """status rows must drop ``models``/``total_models``/``model_source``."""
    from hermes_cli import inventory

    payload = inventory.build_payload("status", offline=True)
    assert payload["providers"]
    for r in payload["providers"]:
        assert "models" not in r, f"status row {r['slug']!r} has models"
        assert "total_models" not in r
        assert "model_source" not in r


# ─── H2: probe-skip predicate ───────────────────────────────────────────


@pytest.mark.parametrize(
    "slug,expected_skip",
    [
        ("openai-codex", False),  # in allowlist
        ("google-gemini-cli", True),  # oauth_external; no /models REST
        ("qwen-oauth", True),  # oauth_external; no /models REST
        ("bedrock", True),  # aws_sdk
        ("anthropic", True),  # api_key, no env var set in isolated_home
    ],
)
def test_h2_skip_live_for(isolated_home, slug, expected_skip):
    from hermes_cli.inventory import _skip_live_for

    assert _skip_live_for(slug) is expected_skip, (
        f"_skip_live_for({slug!r}) != {expected_skip}"
    )


# ─── H3: openrouter offline gotcha ──────────────────────────────────────


def test_h3_openrouter_offline_uses_local_constant(isolated_home, fail_on_network):
    """``_PROVIDER_MODELS['openrouter']`` is None — must use OPENROUTER_MODELS."""
    from hermes_cli import inventory
    from hermes_cli.models import OPENROUTER_MODELS

    payload = inventory.build_payload("models", provider="openrouter", offline=True)
    row = payload["providers"][0]
    assert row["total_models"] == len(OPENROUTER_MODELS), (
        f"openrouter offline got {row['total_models']} models; "
        f"expected len(OPENROUTER_MODELS)={len(OPENROUTER_MODELS)}"
    )
    # And no network calls fired.
    assert fail_on_network == []


# ─── H4: --all populates models from curated/static for unconfigured ────


def test_h4_all_offline_populates_models(isolated_home, fail_on_network):
    from hermes_cli import inventory

    payload = inventory.build_payload(
        "models", include_unconfigured=True, offline=True
    )
    rows = {r["slug"]: r for r in payload["providers"]}
    assert "anthropic" in rows
    # Anthropic is unconfigured in isolated_home but should still get curated.
    assert rows["anthropic"]["total_models"] > 0
    assert rows["anthropic"]["auth_state"] == "unconfigured"
    assert rows["anthropic"]["model_source"] in ("curated", "fallback")
    assert fail_on_network == []


# ─── H5: provider-arg resolution ────────────────────────────────────────


def test_h5_canonical_slug_resolves(isolated_home):
    from hermes_cli.inventory import _resolve_provider, load_picker_context

    ctx = load_picker_context()
    r = _resolve_provider("anthropic", ctx)
    assert r is not None and r["slug"] == "anthropic" and r["kind"] == "canonical"


def test_h5_alias_resolves(isolated_home):
    """``google`` should resolve via profile alias to ``gemini``."""
    from hermes_cli.inventory import _resolve_provider, load_picker_context

    ctx = load_picker_context()
    r = _resolve_provider("google", ctx)
    # Some installs alias google → gemini-cli, others google → gemini.
    assert r is not None and r["kind"] == "canonical"
    assert r["slug"] in {"gemini", "gemini-cli", "google", "google-gemini-cli"}


def test_h5_unknown_slug_returns_none(isolated_home):
    from hermes_cli.inventory import _resolve_provider, load_picker_context

    assert _resolve_provider("does-not-exist-xyz", load_picker_context()) is None


# ─── H6: offline auth-state is env-only ─────────────────────────────────


def test_h6_offline_auth_state_env_only(isolated_home, monkeypatch):
    """``--offline`` must derive ``auth_state`` from env vars only."""
    from hermes_cli import inventory

    # Set ARCEEAI_API_KEY (arcee's only env var, no auth_store seeding).
    monkeypatch.setenv("ARCEEAI_API_KEY", "test-canary")

    payload = inventory.build_payload(
        "providers", include_unconfigured=True, offline=True
    )
    rows = {r["slug"]: r for r in payload["providers"]}
    assert "arcee" in rows
    assert rows["arcee"]["auth_state"] == "configured", (
        f"arcee should be configured via ARCEEAI_API_KEY env var; got {rows['arcee']['auth_state']!r}"
    )


def test_h6_openrouter_overlay_bridge(isolated_home, monkeypatch):
    """OpenRouter accepts ``OPENAI_API_KEY`` via HERMES_OVERLAYS."""
    from hermes_cli import inventory

    monkeypatch.setenv("OPENAI_API_KEY", "test-canary")
    payload = inventory.build_payload(
        "providers", include_unconfigured=True, offline=True
    )
    rows = {r["slug"]: r for r in payload["providers"]}
    assert rows["openrouter"]["auth_state"] == "configured"


# ─── W1: canonical-slug rows from providers: config dict ────────────────


def test_w1_canonical_slug_in_user_providers_keeps_canonical_position(
    isolated_home, monkeypatch
):
    """``list_authenticated_providers`` section 3 emits rows with
    ``is_user_defined=True`` even for canonical slugs (when the user has
    ``providers: { openrouter: {...} }`` in config). They must slot into
    canonical position under ``canonical_order=True``, NOT the tail.
    """
    from hermes_cli import inventory
    from hermes_cli import model_switch

    fixture = [
        {
            "slug": "my-custom",
            "name": "My Custom",
            "is_current": False,
            "is_user_defined": True,
            "models": ["foo"],
            "total_models": 1,
            "source": "user-config",
        },
        {
            "slug": "openrouter",
            "name": "OpenRouter",
            "is_current": False,
            "is_user_defined": True,  # section 3 sets True even for canonical slug
            "models": ["anthropic/claude-opus-4.7"],
            "total_models": 1,
            "source": "user-config",
        },
        {
            "slug": "anthropic",
            "name": "Anthropic",
            "is_current": False,
            "is_user_defined": False,
            "models": ["claude-opus-4-7"],
            "total_models": 1,
            "source": "built-in",
        },
    ]
    monkeypatch.setattr(
        model_switch, "list_authenticated_providers", lambda **_: [dict(r) for r in fixture],
    )
    payload = inventory.build_payload(
        "models", include_unconfigured=False,
        picker_hints=True, canonical_order=True,
    )
    slugs = [r["slug"] for r in payload["providers"]]
    # openrouter and anthropic are both canonical; must precede my-custom.
    assert slugs.index("openrouter") < slugs.index("my-custom")
    assert slugs.index("anthropic") < slugs.index("my-custom")


# ─── Picker hints ────────────────────────────────────────────────────────


def test_picker_hints_emits_authenticated_quartet(isolated_home):
    """``picker_hints=True`` adds ``authenticated``/``key_env``/``warning``
    on every row. Configured rows: authenticated=True, no warning.
    Unconfigured api-key rows: authenticated=False + warning.
    """
    from hermes_cli import inventory

    payload = inventory.build_payload(
        "models", include_unconfigured=True, picker_hints=True, offline=True
    )
    rows = {r["slug"]: r for r in payload["providers"]}
    assert "anthropic" in rows
    assert rows["anthropic"]["authenticated"] is False
    assert "key_env" in rows["anthropic"]
    assert "warning" in rows["anthropic"]
    assert "ANTHROPIC" in rows["anthropic"]["key_env"]


def test_without_picker_hints_omits_authenticated(isolated_home):
    from hermes_cli import inventory

    payload = inventory.build_payload(
        "models", include_unconfigured=True, offline=True
    )
    for r in payload["providers"]:
        assert "authenticated" not in r, f"{r['slug']} leaked authenticated field"


# ─── ConfigContext.with_overrides ───────────────────────────────────────


def test_with_overrides_truthy_only(isolated_home):
    from hermes_cli.inventory import ConfigContext

    base = ConfigContext("a", "m", "u", {}, [])
    # Empty values preserve original.
    assert base.with_overrides(current_provider="").current_provider == "a"
    # Truthy overrides apply.
    assert base.with_overrides(current_provider="b").current_provider == "b"


# ─── ctx kwarg bypasses load_picker_context ─────────────────────────────


def test_ctx_kwarg_bypasses_load(isolated_home, monkeypatch):
    from hermes_cli import inventory

    sentinel: list[str] = []

    def _spy():
        sentinel.append("called")
        return inventory.ConfigContext("", "", "", {}, [])

    monkeypatch.setattr(inventory, "load_picker_context", _spy)
    custom = inventory.ConfigContext("openrouter", "x/y", "", {}, [])
    inventory.build_payload("models", ctx=custom, include_unconfigured=True, offline=True)
    assert sentinel == [], f"load_picker_context called despite ctx kwarg"


# ─── Schema invariants ──────────────────────────────────────────────────


def test_payload_has_schema_version(isolated_home):
    from hermes_cli import inventory

    for kind in ("providers", "models", "status"):
        p = inventory.build_payload(kind, include_unconfigured=True, offline=True)
        assert p["schema_version"] == 1
        assert "current" in p
        assert "providers" in p and isinstance(p["providers"], list)


def test_unknown_kind_raises(isolated_home):
    from hermes_cli import inventory

    with pytest.raises(ValueError, match="unknown kind"):
        inventory.build_payload("nonsense")


def test_unknown_provider_raises(isolated_home):
    from hermes_cli import inventory

    with pytest.raises(ValueError, match="unknown provider"):
        inventory.build_payload("models", provider="does-not-exist")


# ─── Credential leak negative test ──────────────────────────────────────


def test_credential_values_never_appear_in_payload(isolated_home, monkeypatch):
    """env_vars contains NAMES only; env_vars_present is BOOLEANS only."""
    from hermes_cli import inventory

    canary = "sk-LEAK-CANARY-DO-NOT-RECORD-89f0a1b2c3d4e5"
    for var in [
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "ARCEEAI_API_KEY",
        "OPENAI_API_KEY",
        "NOUS_API_KEY",
        "GROQ_API_KEY",
    ]:
        monkeypatch.setenv(var, canary)

    for kind in ("providers", "models", "status"):
        payload = inventory.build_payload(kind, include_unconfigured=True, offline=True)
        s = inventory.dump_json(payload)
        assert canary not in s, f"credential leaked into {kind!r} payload"


# ─── Universe: must iterate CANONICAL_PROVIDERS ─────────────────────────


def test_universe_iterates_canonical_providers(isolated_home):
    """``providers list --all --offline`` returns >= every CANONICAL slug."""
    from hermes_cli import inventory
    from hermes_cli.models import CANONICAL_PROVIDERS

    payload = inventory.build_payload(
        "providers", include_unconfigured=True, offline=True
    )
    canonical_slugs = {p.slug for p in CANONICAL_PROVIDERS}
    payload_slugs = {r["slug"] for r in payload["providers"] if not r.get("is_user_defined")}
    missing = canonical_slugs - payload_slugs
    assert not missing, f"missing canonical providers: {missing}"
    # The two legacy registry-only providers must appear.
    assert "lmstudio" in payload_slugs
    assert "tencent-tokenhub" in payload_slugs


# ─── Renderers smoke ────────────────────────────────────────────────────


def test_render_text_smoke(isolated_home):
    from hermes_cli import inventory

    p1 = inventory.build_payload("providers", include_unconfigured=True, offline=True)
    p2 = inventory.build_payload("models", provider="openrouter", offline=True)
    p3 = inventory.build_payload("status", offline=True)
    assert "SLUG" in inventory.render_text(p1, "providers")
    assert "PROVIDER" in inventory.render_text(p2, "models")
    assert "PROVIDER" in inventory.render_text(p3, "status")


def test_dump_json_round_trips(isolated_home):
    import json
    from hermes_cli import inventory

    p = inventory.build_payload("providers", include_unconfigured=True, offline=True)
    parsed = json.loads(inventory.dump_json(p))
    assert parsed["schema_version"] == 1
    assert len(parsed["providers"]) == len(p["providers"])


# ─── Web-server / TUI consumer parity (the consolidation contract) ──────


def test_web_server_payload_preserves_list_authenticated_providers_shape(
    isolated_home, monkeypatch
):
    """``build_payload('models', ctx=ctx, max_models=50)`` (web_server consumer)
    returns the same row set + order as the OLD direct
    ``list_authenticated_providers`` call. Required dashboard contract
    fields are preserved with original values.
    """
    from hermes_cli import inventory, model_switch

    fixture = [
        {
            "slug": "openrouter", "name": "OpenRouter",
            "is_current": True, "is_user_defined": False,
            "models": ["anthropic/claude-opus-4.7"], "total_models": 1,
            "source": "models.dev",
        },
        {
            "slug": "anthropic", "name": "Anthropic",
            "is_current": False, "is_user_defined": False,
            "models": ["claude-opus-4-7"], "total_models": 1,
            "source": "built-in",
        },
    ]
    monkeypatch.setattr(
        model_switch, "list_authenticated_providers",
        lambda **_: [dict(r) for r in fixture],
    )
    payload = inventory.build_payload("models", max_models=50)
    assert [r["slug"] for r in payload["providers"]] == [r["slug"] for r in fixture]
    for got, want in zip(payload["providers"], fixture):
        for f in ("slug", "name", "is_current", "is_user_defined",
                  "models", "total_models", "source"):
            assert got[f] == want[f], f"{got['slug']}: {f} drifted"


def test_tui_consumer_payload_has_picker_hints_on_every_row(
    isolated_home, monkeypatch
):
    """TUI ``model.options`` consumer pattern:
    ``build_payload('models', include_unconfigured=True, picker_hints=True,
    canonical_order=True)``. EVERY row must carry ``authenticated``.
    """
    from hermes_cli import inventory, model_switch

    fixture = [
        {
            "slug": "openrouter", "name": "OpenRouter",
            "is_current": False, "is_user_defined": False,
            "models": ["x"], "total_models": 1, "source": "built-in",
        },
    ]
    monkeypatch.setattr(
        model_switch, "list_authenticated_providers",
        lambda **_: [dict(r) for r in fixture],
    )
    payload = inventory.build_payload(
        "models", include_unconfigured=True,
        picker_hints=True, canonical_order=True, max_models=50,
    )
    for r in payload["providers"]:
        assert "authenticated" in r, f"{r['slug']} missing authenticated"


# ─── TUI live-path contract canary ──────────────────────────────────────


def test_live_path_does_not_call_models_for(isolated_home, monkeypatch):
    """TUI/dashboard contract: ``model.options`` must NOT call ``_models_for``
    per row in the live path.

    ``list_authenticated_providers`` already populates each row's
    ``models`` with the picker-curated list. Calling ``_models_for`` again
    would either (a) double the live HTTP/OAuth work per invocation, or
    (b) for some providers (Nous), pull non-agentic models — TTS,
    embeddings, rerankers, image/video generators (see longstanding
    warning in tui_gateway/server.py).

    Strategy: spy on ``_models_for`` (the post-processing hook that COULD
    refill rows) and assert it fires zero times on the live ``kind="models"``
    path with ``include_unconfigured=True`` (the TUI's exact call shape).
    """
    from hermes_cli import inventory

    spy_calls: list[str] = []
    real = inventory._models_for

    def _spy(slug, *, live):
        spy_calls.append(slug)
        return real(slug, live=live)

    monkeypatch.setattr(inventory, "_models_for", _spy)

    inventory.build_payload(
        "models",
        include_unconfigured=True,
        picker_hints=True,
        canonical_order=True,
        max_models=50,
    )

    assert spy_calls == [], (
        f"_models_for called {spy_calls} from build_payload live path. "
        "list_authenticated_providers should have populated models already; "
        "refilling would pull non-agentic models (TTS/embeddings) on Nous "
        "and double live HTTP work elsewhere. See tui_gateway/server.py "
        "longstanding warning."
    )


# ─── model.save_key migration coverage ──────────────────────────────────


def test_save_key_migration_returns_populated_row(isolated_home, monkeypatch):
    """``model.save_key`` (in tui_gateway) calls ``build_payload("models",
    ctx=…, picker_hints=True)`` after persisting a key, and looks up the
    saved slug in the result. This test verifies the build_payload
    contract that backs that handler:

      1. A slug present in ``list_authenticated_providers``'s output IS
         present in the payload's providers list (no off-by-name lookup).
      2. The matched row carries ``models`` populated from the
         list_authenticated_providers response (not stripped or replaced).
      3. ``picker_hints=True`` adds ``authenticated`` per row, including
         the freshly-saved one.
    """
    from hermes_cli import inventory, model_switch

    fake_authed = [
        {
            "slug": "anthropic",
            "name": "Anthropic",
            "is_current": False,
            "is_user_defined": False,
            "models": ["claude-opus-4-7", "claude-sonnet-4-6"],
            "total_models": 2,
            "source": "built-in",
        }
    ]
    monkeypatch.setattr(
        model_switch, "list_authenticated_providers", lambda **kw: fake_authed
    )

    payload = inventory.build_payload(
        "models", include_unconfigured=True, picker_hints=True, max_models=50
    )

    # save_key handler does: next((p for p in providers if p["slug"] == slug), None)
    # against this exact payload shape. Verify the saved-slug lookup works.
    matched = next(
        (p for p in payload["providers"] if p["slug"] == "anthropic"), None
    )
    assert matched is not None, (
        "save_key's next(... slug == saved_slug ...) lookup would return "
        "None for a slug that IS in list_authenticated_providers's output"
    )
    assert matched["models"] == ["claude-opus-4-7", "claude-sonnet-4-6"], (
        "build_payload stripped models from the live row; save_key would "
        "return an empty model list to the TUI even when the key is valid"
    )
    assert matched["total_models"] == 2
    assert matched["authenticated"] is True, (
        "picker_hints=True must set authenticated for every configured row"
    )
