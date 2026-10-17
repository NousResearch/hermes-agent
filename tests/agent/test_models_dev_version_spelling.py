"""models.dev keys ONE version spelling per provider (alibaba ``qwen2-5-vl-72b-instruct``, zai
``glm-5.3-flash``); the vendor API accepts the other, so every catalog consumer must find the entry
under either spelling, with the listed spelling winning and sizes/dates never rewritten."""

from unittest.mock import patch

from agent.models_dev import get_model_capabilities, get_model_info, lookup_models_dev_context

REGISTRY = {
    "alibaba": {"id": "alibaba", "models": {
        "qwen2-5-vl-72b-instruct": {"id": "qwen2-5-vl-72b-instruct", "limit": {"context": 131072, "output": 8192},
                                    "modalities": {"input": ["text", "image"], "output": ["text"]}},
        # Both spellings listed with different data: the exact spelling must win.
        "qwen9.1-max": {"id": "qwen9.1-max", "limit": {"context": 1000000}},
        "qwen9-1-max": {"id": "qwen9-1-max", "limit": {"context": 32768}},
        "qwen3-235b-a22b": {"id": "qwen3-235b-a22b", "limit": {"context": 131072}},
    }},
    "zai": {"id": "zai", "models": {"glm-5.3-flash": {"id": "glm-5.3-flash", "limit": {"context": 1000000}}}},
}


def test_either_version_spelling_resolves_across_consumers():
    with patch("agent.models_dev.fetch_models_dev", return_value=REGISTRY):
        assert lookup_models_dev_context("alibaba", "qwen2.5-vl-72b-instruct") == 131072
        assert get_model_capabilities("alibaba", "qwen2.5-vl-72b-instruct").supports_vision is True
        assert get_model_info("alibaba", "qwen2.5-vl-72b-instruct").context_window == 131072
        assert lookup_models_dev_context("zai", "glm-5-3-flash") == 1000000


def test_listed_spelling_wins_and_sizes_are_not_versions():
    with patch("agent.models_dev.fetch_models_dev", return_value=REGISTRY):
        assert lookup_models_dev_context("alibaba", "qwen9.1-max") == 1000000
        assert lookup_models_dev_context("alibaba", "qwen9-1-max") == 32768
        # "3-235b" is a size, not a minor version: no "qwen3.235b-a22b" alias exists to collide.
        assert lookup_models_dev_context("alibaba", "qwen3.235b-a22b") is None
        assert lookup_models_dev_context("alibaba", "qwen2.6-vl-72b-instruct") is None
