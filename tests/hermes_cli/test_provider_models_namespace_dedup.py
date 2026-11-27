"""provider_model_ids: a curated id namespaced with the provider's OWN slug and the bare id the
live /models endpoint returns are the same model, and must merge to one picker row.

Plugin provider profiles commonly ship ``fallback_models`` spelled ``"<slug>/<model>"`` while
the endpoint's live ``/v1/models`` returns bare ``"<model>"``. The live+curated merge compared
the two spellings case-insensitively but exactly, so every model appeared twice per provider in
the ``/model`` pickers (the adapters display the tail, so both rows render identically).

Only the provider's own ``<slug>/`` prefix is canonical: vendor-namespaced aggregator ids that
merely share a tail (``openai/gpt-5`` vs ``azure/gpt-5``) must both survive.

These drive the real ``provider_model_ids()`` generic profile path; only the provider profile
and the credential lookup are stubbed.
"""

from unittest.mock import MagicMock, patch

from hermes_cli.models import _provider_catalog_dedup_key, provider_model_ids

_SLUG = "test-ns-provider"


def _profile(live, fallback_models=None):
    p = MagicMock()
    p.auth_type = "api_key"
    p.base_url = "http://127.0.0.1:59999/v1"
    p.fetch_models.return_value = list(live)
    p.fallback_models = fallback_models
    return p


def _run(live, *, curated=None, fallback_models=None, live_first=False):
    patches = [
        patch("providers.get_provider_profile", return_value=_profile(live, fallback_models)),
        patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "k", "base_url": ""},
        ),
    ]
    if curated is not None:
        patches.append(patch.dict("hermes_cli.models._PROVIDER_MODELS", {_SLUG: list(curated)}))
    if live_first:
        patches.append(patch("hermes_cli.models._LIVE_FIRST_PICKER_PROVIDERS", frozenset({_SLUG})))
    for p in patches:
        p.start()
    try:
        return provider_model_ids(_SLUG)
    finally:
        for p in reversed(patches):
            p.stop()


class TestNamespacedCuratedVsBareLive:
    def test_plugin_fallback_models_namespaced_live_bare(self):
        """The shape seen in the wild: a plugin profile with no static catalog row whose
        fallback_models carry its own slug, against a live endpoint returning bare ids."""
        fallback = (f"{_SLUG}/model-a", f"{_SLUG}/model-b", f"{_SLUG}/model-c")
        merged = _run(["model-a", "model-b", "model-c"], fallback_models=fallback)
        tails = [m.split("/")[-1].lower() for m in merged]
        assert len(tails) == len(set(tails)), f"duplicate picker rows: {merged}"
        assert merged == list(fallback)

    def test_static_curated_namespaced_live_bare(self):
        curated = [f"{_SLUG}/model-a", f"{_SLUG}/model-b"]
        merged = _run(["model-a", "model-b"], curated=curated)
        assert merged == curated

    def test_live_only_models_still_appended(self):
        merged = _run(["model-a", "model-new"], curated=[f"{_SLUG}/model-a"])
        assert merged == [f"{_SLUG}/model-a", "model-new"]

    def test_other_vendor_namespaces_not_collapsed(self):
        merged = _run(["azure/gpt-5", "model-a"], curated=["openai/gpt-5", f"{_SLUG}/model-a"])
        assert "openai/gpt-5" in merged
        assert "azure/gpt-5" in merged
        assert merged.count(f"{_SLUG}/model-a") == 1
        assert "model-a" not in merged

    def test_live_first_provider_keeps_live_spelling(self):
        merged = _run(
            ["model-a", "model-new"],
            curated=[f"{_SLUG}/model-a", f"{_SLUG}/model-stale"],
            live_first=True,
        )
        assert merged == ["model-a", "model-new", f"{_SLUG}/model-stale"]


class TestProviderCatalogDedupKey:
    def test_own_prefix_and_bare_id_share_a_key(self):
        assert _provider_catalog_dedup_key("prov/model-a", "prov") == _provider_catalog_dedup_key("model-a", "prov")

    def test_case_insensitive(self):
        assert _provider_catalog_dedup_key("Prov/Model-A", "prov") == _provider_catalog_dedup_key("model-a", "prov")

    def test_other_namespace_kept(self):
        assert _provider_catalog_dedup_key("other/model-a", "prov") != _provider_catalog_dedup_key("model-a", "prov")

    def test_prefix_matches_on_slash_boundary_only(self):
        assert _provider_catalog_dedup_key("provider/model-a", "prov") != _provider_catalog_dedup_key("model-a", "prov")
