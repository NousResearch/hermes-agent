"""Truncation-retry boost base and ceiling (issue #110126 layer 3).

The retry ladder must seed from the cap the failed request actually carried
(user config, or the provider's declared request default), not the raw 4096 —
a 4096-seeded ladder can never exceed 64K in four steps even when the server
accepts 384K. The ceiling resolves requested cap → provider hard output limit →
context window, and keeps the pre-declaration behaviour when a provider
declares no hard limit.
"""
from unittest.mock import MagicMock

from agent.turn_truncation import _retry_truncated_tool_call, _truncation_boost_ceiling


class _StubTrunc:
    def __init__(self, agent, retries, is_stub=False):
        self.agent = agent
        self.truncated_tool_call_retries = retries
        self._is_stub = is_stub
        self.action = None
        self.result = None

    @property
    def is_stub(self):
        return self._is_stub

    def done(self, action, result=None):
        self.action, self.result = action, result
        return self


def _mock_agent(**kw):
    a = MagicMock()
    a.max_tokens = None
    a.provider = "deepseek"
    a.model = "deepseek-v4-pro"
    a._config_context_length = None
    a.context_compressor = None
    a._requested_output_cap_from_api_kwargs.return_value = None
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# --- ceiling helper -------------------------------------------------------

def test_ceiling_authorized_by_provider_hard_limit():
    from providers import get_provider_profile

    profile = get_provider_profile("deepseek")
    assert profile.max_output_tokens == 384000
    agent = _mock_agent()
    assert _truncation_boost_ceiling(agent, 65536) == 384000


def test_ceiling_shrinks_to_context_window():
    agent = _mock_agent(_config_context_length=200000)
    agent.context_compressor = MagicMock(last_prompt_tokens=150000)
    assert _truncation_boost_ceiling(agent, 65536) == 50000  # 200K - 150K


def test_ceiling_never_below_32768():
    agent = _mock_agent(_config_context_length=40000)
    agent.context_compressor = MagicMock(last_prompt_tokens=39000)
    assert _truncation_boost_ceiling(agent, None) == 32768


def test_ceiling_keeps_old_behaviour_without_declaration():
    agent = _mock_agent(provider="openrouter", model="some-model")
    assert _truncation_boost_ceiling(agent, 65536) == 65536
    assert _truncation_boost_ceiling(agent, None) == 32768


# --- retry base -----------------------------------------------------------

def test_retry_base_uses_requested_cap_when_unset():
    """With no user max_tokens, the ladder seeds from the cap the request carried
    (the 65536 declared default), so retry 1 escalates to 128K."""
    agent = _mock_agent()
    agent._requested_output_cap_from_api_kwargs.return_value = 65536
    st = _StubTrunc(agent, retries=0, is_stub=False)
    _retry_truncated_tool_call(st, {})
    assert agent._ephemeral_max_output_tokens == 131072  # 65536 * 2
    assert st.action == "continue"


def test_retry_base_falls_back_to_4096_without_any_cap():
    agent = _mock_agent(provider="openrouter")  # no declared request default
    agent._requested_output_cap_from_api_kwargs.return_value = None
    st = _StubTrunc(agent, retries=0, is_stub=False)
    _retry_truncated_tool_call(st, {})
    assert agent._ephemeral_max_output_tokens == 8192  # 4096 * 2, old ladder


def test_retry_honours_user_max_tokens_as_base():
    agent = _mock_agent()
    agent.max_tokens = 32768
    agent._requested_output_cap_from_api_kwargs.return_value = 32768
    st = _StubTrunc(agent, retries=0, is_stub=False)
    _retry_truncated_tool_call(st, {})
    assert agent._ephemeral_max_output_tokens == 65536  # 32768 * 2
