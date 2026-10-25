"""A plugin profile's ``build_api_kwargs_extras`` written against the older keyword set keeps working.

Split from ``test_chat_completions.py`` (size cap): this pins the transport's hook-call contract,
not response handling.
"""

import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.chat_completions
    return get_transport("chat_completions")


class TestProfileHookKeywordCompat:
    """``lmstudio_reasoning_options`` is LM Studio-only: a plugin hook that names the older keywords
    without ``**context`` must not start raising TypeError on every request."""

    def test_hook_without_var_kwargs_still_called(self, transport):
        from providers.base import ProviderProfile

        class OldSignature(ProviderProfile):
            def build_api_kwargs_extras(self, *, reasoning_config=None, supports_reasoning=False,
                                        qwen_session_metadata=None, model=None, base_url=None,
                                        ollama_num_ctx=None, session_id=None, cache_scope_id=None):
                return {}, {"reasoning_effort": "low"}

        kw = transport.build_kwargs(model="m", messages=[{"role": "user", "content": "Hi"}],
                                    provider_profile=OldSignature(name="old-signature"))
        assert kw["reasoning_effort"] == "low"
