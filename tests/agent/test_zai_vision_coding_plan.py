"""ZAI vision endpoint/model defaults cover the GLM Coding Plan (issue: vision 1113/1311).

Coding Plan keys have no balance on the general pay-as-you-go endpoints (error 1113
"Insufficient balance") and no glm-5v-turbo access (error 1311 "not included in your
subscription plan"). The vision client must try /coding/ surfaces first and default to
glm-5.3-flash, the Coding Plan's multimodal model.
"""
import pytest

from agent import auxiliary_client


def test_zai_vision_urls_try_coding_plan_first():
    urls = auxiliary_client._ZAI_OPENAI_VISION_URLS
    assert urls[0] == "https://api.z.ai/api/coding/paas/v4"
    assert any("/coding/" in u for u in urls), "coding surfaces must be present"
    # General endpoints remain as last-resort fallbacks (pay-as-you-go keys).
    assert "https://api.z.ai/api/paas/v4" in urls
    assert "https://open.bigmodel.cn/api/paas/v4" in urls
    # Coding surfaces must come BEFORE the general ones.
    first_general = min(urls.index(u) for u in urls if "/coding/" not in u)
    first_coding = min(urls.index(u) for u in urls if "/coding/" in u)
    assert first_coding < first_general


def test_zai_vision_default_model_is_coding_plan_multimodal():
    # glm-5v-turbo is NOT available on the Coding Plan (1311); glm-5.3-flash is.
    assert auxiliary_client._PROVIDER_VISION_MODELS["zai"] == "glm-5.3-flash"
    assert (
        auxiliary_client._resolve_provider_vision_default("zai") == "glm-5.3-flash"
    )
