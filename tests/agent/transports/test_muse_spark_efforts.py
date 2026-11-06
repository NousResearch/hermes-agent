"""Regression for #140229: Muse Spark 1.3 accepts reasoning effort ``max``.

``max`` and ``xhigh`` are distinct levels on Spark 1.3. A request for ``max``
must stay ``max`` on the direct Meta route and on the OpenCode Zen/Go
Responses route. Spark 1.2 (and the unversioned slug) still tops out at
``xhigh``.
"""

import pytest

from agent.reasoning_effort import clamp_effort, codex_supported_efforts
from agent.transports.codex import _resolve_reasoning
from providers import get_provider_profile

# Family slugs the picker and OpenCode catalog actually send, including the
# vendor prefix and the contributor / free tiers.
_SPARK_13 = (
    "muse-spark-1.3",
    "muse-spark-1.3-contributor",
    "meta/muse-spark-1.3-contributor",
    "muse-spark-1.3-contributor-free",
)
_CAPPED = (
    "muse-spark-1.2",
    "muse-spark-1.2-contributor",
    "meta/muse-spark-1.2-2026-04-01",
    "muse-spark",
)

# Direct Meta is a Responses route (api.meta.ai). OpenCode Zen and Go serve
# Muse Spark on their Responses endpoints too.
_RESPONSES_ROUTES = (
    ("meta-ai", "https://api.meta.ai/v1"),
    ("opencode-go", "https://opencode.ai/zen/go/v1"),
    ("opencode-zen", "https://opencode.ai/zen/v1"),
)


def _wire_effort(provider, base_url, model, effort="max"):
    resolved, enabled = _resolve_reasoning(
        model,
        {
            "provider": provider,
            "base_url": base_url,
            "reasoning_config": {"enabled": True, "effort": effort},
        },
    )
    assert enabled is True
    return resolved


def _meta_effort(model, effort="max"):
    profile = get_provider_profile("meta-ai")
    assert profile is not None
    _extra, top = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": effort},
        model=model,
    )
    return top["reasoning_effort"]


class TestMuseSpark13MaxEffort:
    @pytest.mark.parametrize("model", _SPARK_13)
    @pytest.mark.parametrize("provider,base_url", _RESPONSES_ROUTES)
    def test_max_stays_max_on_spark_13(self, model, provider, base_url):
        assert clamp_effort("max", codex_supported_efforts(model)) == "max"
        assert _wire_effort(provider, base_url, model) == "max"

    @pytest.mark.parametrize("model", _CAPPED)
    @pytest.mark.parametrize("provider,base_url", _RESPONSES_ROUTES)
    def test_max_stays_xhigh_below_spark_13(self, model, provider, base_url):
        assert clamp_effort("max", codex_supported_efforts(model)) == "xhigh"
        assert _wire_effort(provider, base_url, model) == "xhigh"

    @pytest.mark.parametrize("model", _SPARK_13)
    def test_direct_meta_profile_keeps_max_on_spark_13(self, model):
        assert _meta_effort(model) == "max"

    @pytest.mark.parametrize("model", _CAPPED)
    def test_direct_meta_profile_caps_older_spark_at_xhigh(self, model):
        assert _meta_effort(model) == "xhigh"

    def test_spark_13_keeps_xhigh_distinct_from_max(self):
        """xhigh must not round up to max; ultra (internal) clamps down to max."""
        supported = codex_supported_efforts("muse-spark-1.3")
        assert clamp_effort("xhigh", supported) == "xhigh"
        assert clamp_effort("ultra", supported) == "max"
        assert _meta_effort("muse-spark-1.3", "xhigh") == "xhigh"
