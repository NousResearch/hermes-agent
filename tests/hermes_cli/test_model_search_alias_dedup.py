"""Picker dedup must fold live bare wire-ids into their curated public slug.

Kimi Coding Plan live-discovers its flagship as the bare id ``k3`` while the
curated catalog carries ``kimi-k3``. The curated-first picker merge must not
render both as separate rows for the same model.
"""

from unittest.mock import patch

from hermes_cli.model_search import model_alias_canonical
from hermes_cli.models import provider_model_ids


class TestModelAliasCanonical:
    def test_bare_k3_folds_to_public_slug(self):
        assert model_alias_canonical("k3") == "kimi-k3"
        assert model_alias_canonical("K3") == "kimi-k3"

    def test_kimi_for_coding_folds_to_k2_8_preview(self):
        assert model_alias_canonical("kimi-for-coding") == "kimi-k2.8-preview"


class TestPickerMergeAliasDedup:
    def test_live_bare_k3_not_duplicated_against_curated_kimi_k3(self):
        """Coding Plan key: live returns bare ``k3``; curated has ``kimi-k3``.
        Exactly one k3-family row must survive (the curated slug leads)."""
        with (
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={
                    "api_key": "sk-kimi-x",
                    "base_url": "https://api.kimi.com/coding",
                },
            ),
            patch(
                "providers.base.ProviderProfile.fetch_models",
                return_value=["k3", "kimi-for-coding"],
            ),
        ):
            out = provider_model_ids("kimi-coding")

        k3_rows = [m for m in out if model_alias_canonical(m) == "kimi-k3"]
        assert k3_rows == ["kimi-k3"], out
        # Live wire-id kimi-for-coding folds into curated kimi-k2.8-preview.
        k28_rows = [m for m in out if model_alias_canonical(m) == "kimi-k2.8-preview"]
        assert k28_rows == ["kimi-k2.8-preview"], out

