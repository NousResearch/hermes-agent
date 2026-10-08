"""Token Plan picker: stale Personal ids stay out, served-but-unlisted ids stay in.

The live ``/compatible-mode/v1/models`` listing omits ``deepseek-v4-pro-0813`` and mixes
image/audio SKUs plus the ``auto`` router alias. A curated-first merge used to lead with
rotated-out ids (403/404 on Personal chat). These contracts fail if either comes back.
"""

from unittest.mock import MagicMock, patch

from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, _longest_key_match
from hermes_cli.models import provider_model_ids
from hermes_cli.models_catalog_static import _ALIBABA_TOKEN_PLAN_MODELS

_KNOWN_DEAD = (
    "kimi-k2.5", "kimi-k2.6", "kimi-k2.7-code",
    "glm-5", "glm-5.1",
    "qwen3.6-plus", "qwen3.8-max-0902",
    "deepseek-v4-flash", "deepseek-v3.2",
    "auto",
)
_FLOOR_ONLY = "deepseek-v4-pro-0813"
_NON_CHAT = (
    "wan2.7-image", "wan2.7-image-pro",
    "qwen-audio-3.0-tts-plus", "qwen-audio-3.0-realtime-plus",
)


def _mock_profile(models):
    profile = MagicMock()
    profile.auth_type = "api_key"
    profile.base_url = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    profile.fetch_models.return_value = models
    profile.fallback_models = ()
    return profile


def _merge(slug, live):
    with (
        patch("providers.get_provider_profile", return_value=_mock_profile(live)),
        patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "k", "base_url": ""},
        ),
    ):
        return provider_model_ids(slug)


class TestTokenPlanFloor:
    def test_dead_ids_stay_out_of_the_floor(self):
        assert not (set(_ALIBABA_TOKEN_PLAN_MODELS) & set(_KNOWN_DEAD))

    def test_floor_carries_the_id_live_omits(self):
        assert _FLOOR_ONLY in _ALIBABA_TOKEN_PLAN_MODELS


class TestTokenPlanMerge:
    def test_merge_drops_dead_and_non_chat_and_keeps_omitted_chat(self):
        live = [
            "qwen3.7-max", "wan2.7-image", "qwen-audio-3.0-tts-plus", "auto",
            "qwen3.8-max-0902",
        ]
        result = _merge("alibaba-token-plan", live)
        assert not (set(result) & set(_KNOWN_DEAD))
        assert not (set(result) & set(_NON_CHAT))
        assert _FLOOR_ONLY in result
        assert "qwen3.7-max" in result

    def test_cn_slug_uses_the_same_filter(self):
        result = _merge("alibaba-token-plan-cn", ["wan2.7-image-pro", "glm-5.2"])
        assert "wan2.7-image-pro" not in result
        assert "glm-5.2" in result


class TestTokenPlanContext:
    def test_qwen37_max_and_qwen36_flash_are_1m(self):
        for model in ("qwen3.7-max", "qwen3.6-flash", "deepseek-v4-pro-0813"):
            hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model)
            assert hit and hit[1] >= 1_000_000, model
