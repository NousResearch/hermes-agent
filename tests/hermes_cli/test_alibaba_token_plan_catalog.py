"""Alibaba Token Plan picker sync, 2026-09-19.

Two independent layers keep the ``alibaba-token-plan`` picker wrong: options the tier no
longer serves, and missing options it does serve.

1. The curated floor in ``models_catalog_static`` went stale after Alibaba rotated the
   Personal tier: kimi-k2.5/-k2.6/-k2.7-code, glm-5/glm-5.1, qwen3.6-plus, deepseek-v4-flash
   and deepseek-v3.2 all answer 403, qwen3.8-max-0902 404s (chat-probed 2026-09-19), and the
   curated-first merge in ``provider_model_ids`` kept the dead ids at the TOP of the picker
   — same failure class as the delisted ox-alpha-free on opencode-go (see
   ``test_provider_live_curated_merge.py``).
2. The live ``/compatible-mode/v1/models`` listing is not a complete source: it omits
   ``deepseek-v4-pro-0813`` — a documented, 200-OK Personal-tier chat model — so the curated
   floor is the only layer that can carry it; and it mixes in image/audio SKUs
   (wan2.7-image*, qwen-audio-3.0-*) that 400 on ``/chat/completions``, which the chat
   picker must never offer.

These tests pin the contracts so a revert (stale floor resurrected) fails loudly, without
freezing the list against future legitimate updates: known-dead ids stay out, the
documented chat set stays in, the e2e merge leads with it, non-chat live entries are
filtered, and the 1M context windows resolve.
"""

from unittest.mock import MagicMock, patch

from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, _longest_key_match
from hermes_cli.models import provider_model_ids
from hermes_cli.models_catalog_static import _ALIBABA_TOKEN_PLAN_MODELS

# Chat-probed 2026-09-19 against token-plan.ap-southeast-1.maas.aliyuncs.com
# (HTTP 403 "Access to model denied", or 404 for the -0902 snapshot id).
_KNOWN_DEAD = (
    "kimi-k2.5", "kimi-k2.6", "kimi-k2.7-code",
    "glm-5", "glm-5.1",
    "qwen3.6-plus", "qwen3.8-max-0902",
    "deepseek-v4-flash", "deepseek-v3.2",
)
# Every chat-capable id on the official Token Plan (Personal Edition) supported-models list
# (help.aliyun.com/en/model-studio/token-plan-personal-overview); HTTP 200 on the same probes.
_VERIFIED_CHAT = (
    "qwen3.8-max", "qwen3.8-flash", "qwen3.7-max", "qwen3.7-plus", "qwen3.6-flash",
    "deepseek-v4.1-flash", "deepseek-v4-pro", "deepseek-v4-pro-0813", "deepseek-v4-flash-0731",
    "glm-5.3", "glm-5.2",
)
# Non-chat SKUs the live listing actually returns today (400 on /chat/completions).
_NON_CHAT_LIVE = (
    "wan2.7-image", "wan2.7-image-pro",
    "qwen-audio-3.0-tts-plus", "qwen-audio-3.0-realtime-plus",
    "wanx2.1-t2v-plus", "wanx2.1-i2v-plus",
    "happyhorse-video", "qwen-image-edit",
)
# Also covers the -cn slug: same listing, same non-chat SKUs.
_NON_CHAT_LIVE_CN = _NON_CHAT_LIVE


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

    def test_verified_chat_models_present_in_curated_floor(self):
        """Every documented 200-OK chat model from the Personal-tier list is curated, so the
        offline picker offers the full current tier even when the live fetch fails —
        including deepseek-v4-pro-0813, which the live /models listing does not return."""
        floor = set(_ALIBABA_TOKEN_PLAN_MODELS)
        assert set(_VERIFIED_CHAT) <= floor


class TestTokenPlanMerge:
    """End-to-end through provider_model_ids with the REAL floor and the live listing the
    tier actually returns today (chat ids that overlap curated plus non-chat SKUs)."""

    @staticmethod
    def _merge(live):
        with (
            patch("providers.get_provider_profile", return_value=_mock_profile(live)),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
        ):
            return provider_model_ids("alibaba-token-plan")

    def test_merge_offers_verified_chat_and_no_dead_ids(self):
        """Dead curated ids must not lead the picker over working models, and floor-only
        models (deepseek-v4-pro-0813) must survive the live fetch's omission."""
        live = [
            "qwen3.7-max", "qwen3.7-plus", "qwen3.6-flash", "glm-5.2", "deepseek-v4-pro",
            "deepseek-v4-flash-0731", "qwen3.8-max", "qwen3.8-flash", "deepseek-v4.1-flash",
            "glm-5.3",
        ]
        result = self._merge(live)
        assert not (set(result) & set(_KNOWN_DEAD))
        assert set(_VERIFIED_CHAT) <= set(result)

    def test_merge_filters_non_chat_live_skus(self):
        """wan2.7-image* / qwen-audio-3.0-* are plan models but 400 on /chat/completions —
        the chat picker must not offer them even though the live listing returns them."""
        live = list(_NON_CHAT_LIVE) + ["qwen3.8-max"]
        result = self._merge(live)
        assert not (set(result) & set(_NON_CHAT_LIVE))
        assert "qwen3.8-max" in result

    def test_merge_filters_non_chat_live_skus_cn_slug(self):
        """-cn slug uses the same non-chat filter."""
        with (
            patch("providers.get_provider_profile", return_value=_mock_profile(list(_NON_CHAT_LIVE_CN) + ["qwen3.8-max"])),
            patch(
                "hermes_cli.auth.resolve_api_key_provider_credentials",
                return_value={"api_key": "k", "base_url": ""},
            ),
        ):
            result = provider_model_ids("alibaba-token-plan-cn")
        assert not (set(result) & set(_NON_CHAT_LIVE_CN))
        assert "qwen3.8-max" in result


class TestTokenPlanContextWindows:
    """The 1M-window ids must resolve to their real window (Model Studio docs), not the
    131,072 "qwen" / 128K "deepseek" catch-alls — the #69881 premature-compaction class."""

    @staticmethod
    def _resolve(model):
        hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
        return hit[1] if hit else None

    def test_qwen37_max_resolves_to_1m(self):
        assert self._resolve("qwen3.7-max") == 1_000_000

    def test_qwen36_flash_resolves_to_1m(self):
        assert self._resolve("qwen3.6-flash") == 1_000_000

    def test_deepseek_v4_pro_0813_resolves_to_1m(self):
        assert self._resolve("deepseek-v4-pro-0813") == 1_000_000

    def test_qwen38_max_resolves_to_1m(self):
        assert self._resolve("qwen3.8-max") == 1_000_000

    def test_qwen38_flash_resolves_to_1m(self):
        assert self._resolve("qwen3.8-flash") == 1_000_000

    def test_qwen37_plus_resolves_to_1m(self):
        assert self._resolve("qwen3.7-plus") == 1_048_576

    def test_deepseek_v4_pro_resolves_to_1m(self):
        assert self._resolve("deepseek-v4-pro") == 1_000_000

    def test_deepseek_v4_1_flash_resolves_to_1m(self):
        assert self._resolve("deepseek-v4.1-flash") == 1_000_000

    def test_deepseek_v4_flash_0731_resolves_to_1m(self):
        assert self._resolve("deepseek-v4-flash-0731") == 1_000_000

    def test_glm_5_3_resolves_to_1m3(self):
        assert self._resolve("glm-5.3") == 1_310_720

    def test_glm_5_2_resolves_to_1m(self):
        assert self._resolve("glm-5.2") == 1_048_576
