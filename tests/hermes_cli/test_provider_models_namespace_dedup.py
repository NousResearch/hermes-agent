"""Regression tests: provider_model_ids merge must dedup namespaced-vs-bare ids.

Bug class: identifier-representation dedup (same logical id in two spellings).
Plugin provider catalogs namespace curated entries with the provider's own slug
("my-proxy/claude-x") while the provider's live /models endpoint returns bare
ids ("claude-x"). The curated+live merge in provider_model_ids() compared the
two spellings with a naive exact (lowercased) match, so every model appeared
TWICE in /model pickers — the desktop/TUI/gateway adapters display the bare
tail, so both rows render identically ("Fable 5" twice per provider).

These tests drive the REAL provider_model_ids() with a stubbed provider
profile + credential resolution (not a mock of the function under test), per
the repo's E2E-over-mocks rule.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.models import (
    _PROVIDER_MODELS,
    _canonical_model_key,
    provider_model_ids,
)

_SLUG = "test-ns-provider"


class _StubProfile:
    """Minimal stand-in for providers.base.ProviderProfile."""

    auth_type = "api_key"
    base_url = "http://127.0.0.1:59999/v1"
    fallback_models: tuple = ()

    def __init__(self, live_models):
        self._live = list(live_models)

    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
        return list(self._live)


def _run_merge(curated, live, slug=_SLUG):
    """Invoke the real provider_model_ids() generic api-key-provider path."""
    profile = _StubProfile(live)
    with (
        patch.dict(_PROVIDER_MODELS, {slug: list(curated)}),
        patch("providers.get_provider_profile", return_value=profile),
        patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "test-key", "base_url": profile.base_url},
        ),
    ):
        return provider_model_ids(slug)


class TestNamespacedCuratedVsBareLiveDedup:
    def test_bare_live_ids_do_not_duplicate_namespaced_curated(self):
        """The reported bug: 4 curated namespaced + 4 live bare -> 8 rows."""
        curated = [
            f"{_SLUG}/claude-fable-5",
            f"{_SLUG}/claude-opus-4-8",
            f"{_SLUG}/claude-sonnet-5",
            f"{_SLUG}/claude-haiku-4-5",
        ]
        live = [
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-haiku-4-5",
        ]
        merged = _run_merge(curated, live)

        # Every model must appear exactly once by display tail — the picker
        # adapters render model.split("/")[-1], so a bare+namespaced pair is
        # a visible duplicate row.
        tails = [m.split("/")[-1].lower() for m in merged]
        assert len(tails) == len(set(tails)), f"duplicate rows in merge: {merged}"
        # Curated order/spelling wins (curated-first provider).
        assert merged == curated

    def test_live_only_models_still_appended(self):
        """Dedup must not eat genuinely-new live models (discovery intact)."""
        curated = [f"{_SLUG}/claude-opus-4-8"]
        live = ["claude-opus-4-8", "claude-brand-new-6"]
        merged = _run_merge(curated, live)
        assert merged == [f"{_SLUG}/claude-opus-4-8", "claude-brand-new-6"]

    def test_vendor_namespaces_not_collapsed(self):
        """Over-collapse guard: distinct vendor namespaces sharing a tail must
        BOTH survive — only the provider's OWN slug prefix is canonical."""
        curated = ["openai/gpt-5", f"{_SLUG}/model-a"]
        live = ["azure/gpt-5", "model-a"]
        merged = _run_merge(curated, live)
        assert "openai/gpt-5" in merged
        assert "azure/gpt-5" in merged  # NOT deduped against openai/gpt-5
        assert merged.count(f"{_SLUG}/model-a") == 1
        assert "model-a" not in merged  # bare spelling of the SAME model deduped

    def test_live_first_provider_order_prefers_live_spelling(self):
        """Same dedup on the live-first branch (opencode-style providers)."""
        curated = [f"{_SLUG}/model-a", f"{_SLUG}/model-stale"]
        live = ["model-a", "model-new"]
        with patch(
            "hermes_cli.models._LIVE_FIRST_PICKER_PROVIDERS", frozenset({_SLUG})
        ):
            merged = _run_merge(curated, live)
        assert merged == ["model-a", "model-new", f"{_SLUG}/model-stale"]


class TestCanonicalModelKey:
    def test_strips_own_prefix_only(self):
        assert _canonical_model_key("prov/model-a", "prov") == "model-a"
        assert _canonical_model_key("other/model-a", "prov") == "other/model-a"

    def test_case_insensitive(self):
        assert _canonical_model_key("Prov/Model-A", "prov") == "model-a"

    def test_bare_id_passthrough(self):
        assert _canonical_model_key("model-a", "prov") == "model-a"

    def test_no_substring_false_match(self):
        # "prov" prefix must match on the "/" boundary, not as a substring.
        assert _canonical_model_key("provider/model-a", "prov") == "provider/model-a"
