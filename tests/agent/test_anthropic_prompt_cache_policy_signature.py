"""Contract guard: ``anthropic_prompt_cache_policy`` keeps its positional ``agent``.

The policy is called positionally from every shipped call shape — the module-level
function itself (``agent/moa_loop.py`` and recursively inside
``agent/agent_runtime_helpers.py``) and the ``AIAgent._anthropic_prompt_cache_policy``
alias that ``run_agent.py`` binds to it via ``_forward``. Dropping the leading
``agent`` parameter, or turning it keyword-only, breaks those call sites with a
runtime ``TypeError``; this file pins the signature and exercises the forwarded
method for real instead of waiting for that crash.

Decision coverage for the policy itself (which provider/model caches, and in which
layout) lives in ``tests/agent/test_anthropic_prompt_cache_policy.py``.
"""
import inspect
import unittest
from types import SimpleNamespace

from agent.agent_runtime_helpers import anthropic_prompt_cache_policy

_STUB_ATTRS = {
    "provider": None,
    "base_url": None,
    "api_mode": None,
    "model": None,
    "_cache_disabled": False,
}


def _stub() -> SimpleNamespace:
    """Only the documented attributes exist: any other attribute the policy reads
    raises AttributeError instead of silently returning a truthy MagicMock."""
    return SimpleNamespace(**_STUB_ATTRS)


class AnthropicPromptCachePolicySignatureTest(unittest.TestCase):
    def test_positional_agent_accepted(self):
        try:
            anthropic_prompt_cache_policy(
                _stub(), provider=None, base_url=None, api_mode=None, model=None,
            )
        except TypeError:
            self.fail("anthropic_prompt_cache_policy rejected positional `agent`")

    def test_agent_parameter_is_positional(self):
        params = list(inspect.signature(anthropic_prompt_cache_policy).parameters.values())
        self.assertTrue(params, "policy declares no parameters at all")
        first = params[0]
        self.assertEqual(first.name, "agent", f"first parameter is {first.name!r}, not 'agent'")
        self.assertIn(
            first.kind,
            (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD),
            "`agent` must stay passable positionally",
        )

    def test_forwarded_agent_method_takes_no_arguments(self):
        """``run_agent.py`` binds the policy onto AIAgent as a forwarded alias, so
        the shipped call shape is ``agent._anthropic_prompt_cache_policy()`` with
        only ``self``. A keyword-only ``agent`` breaks exactly this call."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        for key, value in _STUB_ATTRS.items():
            setattr(agent, key, value)
        # getattr: the alias is bound dynamically via `_forward`, so static tools
        # cannot see through it (Pyright reads the target's own parameter list).
        method = getattr(agent, "_anthropic_prompt_cache_policy")
        result = method()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2, f"expected (should_cache, native_layout), got {result!r}")
        self.assertTrue(all(isinstance(v, bool) for v in result), f"non-bool members: {result!r}")


if __name__ == "__main__":
    unittest.main()
