import unittest
from types import SimpleNamespace

from agent.agent_runtime_helpers import anthropic_prompt_cache_policy


class AnthropicPromptCachePolicySignatureTest(unittest.TestCase):
    """Regression guard: anthropic_prompt_cache_policy MUST accept `agent`.

    Call sites that pass `agent` positionally:
      - agent/moa_loop.py
      - agent/agent_runtime_helpers.py (recursive)
      - run_agent.py::_anthropic_prompt_cache_policy (forwarder)

    If the positional `agent` parameter is ever dropped again, this test
    fails loudly instead of crashing at runtime.

    SCOPE NOTE: this is a signature/contract guard, not a behavioural test
    of the policy. It does not assert the caching *decision* for real
    provider/model inputs — only that (a) the function keeps its positional
    `agent` contract, (b) the deterministic all-None input returns the
    exact canonical tuple, and (c) the documented call sites still pass
    `agent` positionally. Behavioural coverage of the decision logic lives
    in test_anthropic_prompt_cache_policy.py.
    """

    def setUp(self):
        # Explicit stub: only the documented attributes exist. Any attribute
        # the policy reads that is NOT set here raises AttributeError instead
        # of silently returning a truthy MagicMock, so an unexpected access
        # surfaces immediately.
        self.agent = SimpleNamespace(
            provider=None,
            base_url=None,
            api_mode=None,
            model=None,
            _cache_disabled=False,
        )

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

    def test_return_value_exact_contract(self):
        # The all-None input is deterministic: with no provider/model/api_mode
        # and no _cache_disabled flag, the policy resolves to (False, False).
        # Asserting the exact tuple catches a semantic regression (e.g. the
        # policy flipping its caching decision) that a pure shape check misses.
        result = anthropic_prompt_cache_policy(
            self.agent,
            provider=None,
            base_url=None,
            api_mode=None,
            model=None,
        )
        self.assertEqual(result, (False, False))

    def test_documented_call_sites_pass_agent_positionally(self):
        # If any of the listed call sites is refactored to pass `agent`
        # differently (or a new positional site is added), this assertion
        # forces a conscious update of the guard instead of a silent drift.
        expected_sources = [
            "agent/moa_loop.py",
            "agent/agent_runtime_helpers.py",
            "run_agent.py",
        ]
        import subprocess

        found = set()
        for rel in expected_sources:
            out = subprocess.run(
                ["grep", "-rn", "anthropic_prompt_cache_policy(", rel],
                capture_output=True,
                text=True,
            ).stdout
            # The call must appear as a positional call: identifier immediately
            # followed by '(' then a newline OR 'agent'/stub/self on the next line.
            for line in out.splitlines():
                if "def anthropic_prompt_cache_policy" in line:
                    continue  # the definition itself
                if "anthropic_prompt_cache_policy(" in line and (
                    "stub" in line
                    or "agent" in line
                    or "self" in line
                    or line.rstrip().endswith("(")
                ):
                    found.add(rel)
                    break
        self.assertEqual(
            found,
            set(expected_sources),
            msg="A documented call site no longer passes `agent` positionally: "
            + ", ".join(sorted(set(expected_sources) - found)),
        )


if __name__ == "__main__":
    unittest.main()
