"""End-to-end wire check: the Zen relay cap reaches ``api_kwargs``."""

from agent.transports.chat_completions import _apply_max_tokens
from providers import get_provider_profile


def _mt_fn(v):
    return {"max_tokens": v}


class TestOpenCodeRelayWireCap:
    def test_plain_model_cap_lands_in_api_kwargs(self):
        p = get_provider_profile("opencode-go")
        assert p is not None
        kwargs: dict = {}
        _apply_max_tokens(
            kwargs, "deepseek-v4.1-flash", None,
            {"max_tokens_param_fn": _mt_fn}, p.get_max_tokens("deepseek-v4.1-flash"),
        )
        assert kwargs == {"max_tokens": 65536}

    def test_model_specific_cap_lands_in_api_kwargs(self):
        p = get_provider_profile("opencode-go")
        assert p is not None
        kwargs: dict = {}
        _apply_max_tokens(
            kwargs, "mimo-v2.5-pro", None,
            {"max_tokens_param_fn": _mt_fn}, p.get_max_tokens("mimo-v2.5-pro"),
        )
        assert kwargs == {"max_tokens": 131072}
