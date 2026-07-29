"""Test that kimi-coding and kimi-coding-cn both appear in the /model picker.

Both providers share the same models.dev ID (kimi-for-coding) but are distinct
profiles with different API keys, base URLs, and endpoints.  The /model picker
must show both so users can pick the right endpoint for their key type.

Regression: the original ``seen_mdev_ids`` dedup by mdev_id alone would skip
kimi-coding-cn after kimi-coding was emitted because both map to
``kimi-for-coding`` (#10526).  The fix deduplicates by
``(mdev_id, canonical_profile_name)`` instead, allowing distinct profiles
through.
"""

import os
from unittest.mock import patch

from hermes_cli.model_switch import (
    list_authenticated_providers,
    parse_model_flags,
    switch_model,
)
from hermes_cli.providers import resolve_provider_full


# -- Only KIMI_CN_API_KEY set ------------------------------------------------


@patch.dict(os.environ, {"KIMI_CN_API_KEY": "sk-cn-fake"}, clear=False)
def test_kimi_cn_appears_when_only_cn_key_set():
    """kimi-coding-cn should appear when only KIMI_CN_API_KEY is set."""
    providers = list_authenticated_providers(current_provider="kimi-coding-cn")

    # kimi-coding-cn must be listed (it has credentials)
    cn = next((p for p in providers if p["slug"] == "kimi-coding-cn"), None)
    assert cn is not None, (
        "kimi-coding-cn should appear when KIMI_CN_API_KEY is set"
    )
    assert cn["is_current"] is True
    assert cn["total_models"] > 0

    # kimi-coding must NOT appear (no KIMI_API_KEY)
    intl = next((p for p in providers if p["slug"] == "kimi-coding"), None)
    assert intl is None, (
        "kimi-coding should NOT appear when only KIMI_CN_API_KEY is set"
    )


# -- Only KIMI_API_KEY set ---------------------------------------------------


@patch.dict(os.environ, {"KIMI_API_KEY": "sk-intl-fake"}, clear=False)
def test_kimi_intl_appears_when_only_intl_key_set():
    """kimi-coding (international) should appear when only KIMI_API_KEY is set."""
    providers = list_authenticated_providers(current_provider="kimi-coding")

    intl = next((p for p in providers if p["slug"] == "kimi-coding"), None)
    assert intl is not None, (
        "kimi-coding should appear when KIMI_API_KEY is set"
    )
    assert intl["is_current"] is True

    # kimi-coding-cn must NOT appear (no KIMI_CN_API_KEY)
    cn = next((p for p in providers if p["slug"] == "kimi-coding-cn"), None)
    assert cn is None, (
        "kimi-coding-cn should NOT appear when only KIMI_API_KEY is set"
    )


# -- Both keys set -----------------------------------------------------------

@patch.dict(os.environ, {
    "KIMI_API_KEY": "sk-intl-fake",
    "KIMI_CN_API_KEY": "sk-cn-fake",
}, clear=False)
def test_both_kimi_providers_appear_when_both_keys_set():
    """Both kimi-coding and kimi-coding-cn should appear when both keys set.

    They are distinct profiles with different env vars and endpoints.  The
    existing aliases (kimi, moonshot → kimi-coding; kimi-cn, moonshot-cn →
    kimi-coding-cn) must NOT create additional rows.
    """
    providers = list_authenticated_providers(current_provider="kimi-coding")

    # Both profile slugs must appear
    intl = next((p for p in providers if p["slug"] == "kimi-coding"), None)
    assert intl is not None, "kimi-coding should appear when KIMI_API_KEY is set"
    assert intl["is_current"] is True

    cn = next((p for p in providers if p["slug"] == "kimi-coding-cn"), None)
    assert cn is not None, (
        "kimi-coding-cn should appear when KIMI_CN_API_KEY is set"
    )
    assert cn["is_current"] is False  # `current_provider` is kimi-coding

    # Exactly 2 Kimi entries — no duplicates for aliases (kimi, moonshot,
    # moonshot-cn, kimi-cn)
    kimi_slugs = [p["slug"] for p in providers if "kimi" in p["slug"] or "moonshot" in p["slug"]]
    assert len(kimi_slugs) == 2, (
        f"Expected exactly 2 Kimi entries (kimi-coding, kimi-coding-cn), "
        f"got {kimi_slugs}"
    )


# -- Both aliases deduped correctly ------------------------------------------

@patch.dict(os.environ, {
    "KIMI_API_KEY": "sk-intl-fake",
    "KIMI_CN_API_KEY": "sk-cn-fake",
}, clear=False)
def test_kimi_aliases_not_listed_separately():
    """Alias hermes_ids (kimi, moonshot) must NOT create phantom picker rows.

    They resolve to the same canonical profile (kimi-coding) and should be
    deduped.  Only the canonical slug (kimi-coding) should appear.
    """
    providers = list_authenticated_providers(current_provider="kimi-coding-cn")

    slugs = {p["slug"] for p in providers}
    # These alias slugs must NOT appear
    for bad_slug in ("kimi", "moonshot", "moonshot-cn", "kimi-cn"):
        assert bad_slug not in slugs, (
            f"Alias slug '{bad_slug}' must not appear in picker (resolved to "
            f"canonical profile)"
        )


@patch.dict(os.environ, {
    "KIMI_API_KEY": "sk-intl-fake",
    "KIMI_CN_API_KEY": "sk-cn-fake",
}, clear=False)
def test_resolve_provider_full_preserves_kimi_cn_provider_identity():
    """Explicit kimi-coding-cn must not collapse to shared models.dev alias.

    Regression: resolve_provider_full('kimi-coding-cn') used normalize_provider(),
    which mapped both kimi-coding and kimi-coding-cn to the models.dev alias
    'kimi-for-coding'. That silently rewired CN users to the international
    endpoint and KIMI_API_KEY.
    """
    pdef = resolve_provider_full("kimi-coding-cn", None, None)
    assert pdef is not None
    assert pdef.id == "kimi-coding-cn"
    assert pdef.base_url == "https://api.moonshot.cn/v1"
    assert pdef.api_key_env_vars == ("KIMI_CN_API_KEY",)


@patch.dict(os.environ, {
    "KIMI_API_KEY": "sk-intl-fake",
    "KIMI_CN_API_KEY": "sk-cn-fake",
}, clear=False)
def test_switch_model_with_explicit_kimi_cn_provider_stays_on_cn_endpoint():
    """/model ... --provider kimi-coding-cn must stay on moonshot.cn.

    This hits the real switch path used by gateway /model: parse flags first,
    then call switch_model() with explicit_provider. The result must not rewrite
    the target provider/base_url back to the international Kimi endpoint.
    """
    model_input, explicit_provider, *_ = parse_model_flags(
        "kimi-k2.6 —provider kimi-coding-cn"
    )
    result = switch_model(
        raw_input=model_input,
        current_provider="deepseek",
        current_model="deepseek-v4-flash",
        current_base_url="https://api.deepseek.com/v1",
        current_api_key="***",
        is_global=False,
        explicit_provider=explicit_provider,
        user_providers={},
        custom_providers=None,
    )

    assert result.success is True
    assert result.target_provider == "kimi-coding-cn"
    assert result.new_model == "kimi-k2.6"
    assert result.base_url == "https://api.moonshot.cn/v1"
    assert result.api_key == "sk-cn-fake"
