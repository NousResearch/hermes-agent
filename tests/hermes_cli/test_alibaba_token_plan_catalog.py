"""Alibaba Token Plan curated-floor sync, 2026-09-14.

Alibaba rotated the Token Plan (``sk-sp-...`` tier) catalog: kimi-k2.5/-k2.6/-k2.7-code,
glm-5/glm-5.1, qwen3.6-plus and deepseek-v4-flash/v3.2 now 403 on chat probes,
qwen3.8-max-0902 404s, while qwen3.8-max/flash, deepseek-v4.1-flash and
deepseek-v4-flash-0731 joined the live ``/compatible-mode/v1/models`` listing. The
curated-first merge in ``provider_model_ids`` kept the dead ids at the TOP of the
picker, above the live-only replacements — same failure class as the delisted
ox-alpha-free on opencode-go (see ``test_provider_live_curated_merge.py``).

These tests pin the sync so a revert (stale floor resurrected) fails loudly,
without freezing the list against future legitimate updates: they assert the
KNOWN-DEAD ids stay out and the live replacements are present, never an exact
snapshot of the floor.
"""

from unittest.mock import MagicMock, patch

from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, _longest_key_match
from hermes_cli.models import provider_model_ids
from hermes_cli.models_catalog_static import _ALIBABA_TOKEN_PLAN_MODELS

# Chat-probed 2026-09-14 against token-plan.ap-southeast-1.maas.aliyuncs.com
# (HTTP 403 "Access to model denied", or 404 for the -0902 snapshot id).
_KNOWN_DEAD = (
    "kimi-k2.5", "kimi-k2.6", "kimi-k2.7-code",
    "glm-5", "glm-5.1",
    "qwen3.6-plus", "qwen3.8-max-0902",
    "deepseek-v4-flash", "deepseek-v3.2",
)
# HTTP 200 on the same probes.
_LIVE_REPLACEMENTS = (
    "qwen3.8-max", "qwen3.8-flash", "qwen3.7-max", "qwen3.7-plus",
    "qwen3.6-flash", "deepseek-v4-pro", "deepseek-v4.1-flash",
    "deepseek-v4-flash-0731", "glm-5.2",
)


def _mock_profile(models):
    p = MagicMock()
    p.auth_type = "api_key"
    p.base_url = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    p.fetch_models.return_value = models
    p.fallback_models = None
    return p


class TestTokenPlanCuratedFloorSync:
    """The curated floor must not offer models the tier no longer serves."""

    def test_dead_models_not_in_curated_floor(self):
        """Chat-probe-verified dead ids stay out of the offline fallback (REVERT-PROOF:
        restoring the stale floor re-adds them and this fails)."""
        floor = set(_ALIBABA_TOKEN_PLAN_MODELS)
        assert not (floor & set(_KNOWN_DEAD)), sorted(floor & set(_KNOWN_DEAD))

    def test_live_replacements_present_in_curated_floor(self):
        """Every 200-OK model from the live listing is curated, so the offline picker
        offers the full current tier even when the live fetch fails."""
        floor = set(_ALIBABA_TOKEN_PLAN_MODELS)
        assert set(_LIVE_REPLACEMENTS) <= floor

    def test_merge_leads_with_live_models(self):
        """End-to-end through provider_model_ids with the REAL floor and a live listing
        that omits the dead ids (as the real API does): dead curated entries must not
        lead the picker over the live-only replacements."""
        live = list(_LIVE_REPLACEMENTS)
        with (
            patch("providers.get_provider_profile", return_value=_mock_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
        ):
            result = provider_model_ids("alibaba-token-plan")
        assert not (set(result) & set(_KNOWN_DEAD))
        assert set(_LIVE_REPLACEMENTS) <= set(result)


class TestTokenPlanContextWindows:
    """qwen3.7-max / qwen3.6-flash must resolve to their real 1M window (Model Studio
    docs), not the 131,072 "qwen" catch-all — the #69881 premature-compaction class."""

    @staticmethod
    def _resolve(model):
        hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
        return hit[1] if hit else None

    def test_qwen37_max_resolves_to_1m(self):
        assert self._resolve("qwen3.7-max") == 1_000_000

    def test_qwen36_flash_resolves_to_1m(self):
        assert self._resolve("qwen3.6-flash") == 1_000_000
