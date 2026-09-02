"""Picker merge must not double-list Kimi Coding live bare ids.

Kimi Coding Plan live-discovers its flagship as the bare wire id ``k3``.
The curated catalog now leads with the same live id (retired public slug
``kimi-k3`` is alias-only). Merge must keep a single k3-family row.
"""

from unittest.mock import patch

from hermes_cli.model_search import model_alias_canonical
from hermes_cli.models import provider_model_ids


class TestModelAliasCanonical:
    def test_bare_k3_folds_to_public_slug(self):
        # Alias table still maps bare wire → legacy public slug for search
        # / config compatibility; picker catalog itself leads with bare k3.
        assert model_alias_canonical("k3") == "kimi-k3"
        assert model_alias_canonical("K3") == "kimi-k3"


class TestPickerMergeAliasDedup:
    def test_live_bare_k3_not_duplicated_against_curated_kimi_k3(self):
        """Coding Plan key: live returns bare ``k3``; curated also leads with
        ``k3``. Exactly one k3-family row must survive (live bare id)."""
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
        assert k3_rows == ["k3"], out
        # Live-only entries with no curated twin still surface.
        assert "kimi-for-coding" in out
        # Retired public slug must not reappear as a second picker row.
        assert "kimi-k3" not in out

