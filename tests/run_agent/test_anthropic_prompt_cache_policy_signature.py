import unittest
from unittest.mock import MagicMock

from agent.agent_runtime_helpers import anthropic_prompt_cache_policy


class AnthropicPromptCachePolicySignatureTest(unittest.TestCase):
    """Regression guard: anthropic_prompt_cache_policy MUST accept `agent`.

    Call sites that pass `agent` positionally:
      - agent/moa_loop.py
      - agent/agent_runtime_helpers.py (recursive)
      - run_agent.py::_anthropic_prompt_cache_policy (forwarder)

    If the positional `agent` parameter is ever dropped again, this test
    fails loudly instead of crashing at runtime.
    """

    def setUp(self):
        self.agent = MagicMock()
        self.agent.provider = None
        self.agent.base_url = None
        self.agent.api_mode = None
        self.agent.model = None

    def test_positional_agent_accepted(self):
        try:
            anthropic_prompt_cache_policy(
                self.agent,
                provider=None,
                base_url=None,
                api_mode=None,
                model=None,
            )
        except TypeError:
            self.fail("anthropic_prompt_cache_policy rejected positional `agent`")

    def test_return_value_shape(self):
        result = anthropic_prompt_cache_policy(
            self.agent,
            provider=None,
            base_url=None,
            api_mode=None,
            model=None,
        )
        self.assertEqual(len(result), 2)
        for item in result:
            self.assertIsInstance(item, bool)


if __name__ == "__main__":
    unittest.main()
