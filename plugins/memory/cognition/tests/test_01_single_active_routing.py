"""Acceptance: single-active provider with internal multi-repo routing.

Exactly one provider registers under the name 'cognition' (one slot, never a
provider per repo), while facts route internally to several repos by
origin_repo. Also pins the user-facing copy contract: push is fixed to session
end, Pending means unpushed, Dream is read-only.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load  # noqa: E402

provider_mod = load("provider")
merge_mod = load("merge")
provenance_mod = load("provenance")

CognitionMemoryProvider = provider_mod.CognitionMemoryProvider


class FakeCtx:
    """Minimal ctx surface: single-slot provider activation."""

    def __init__(self):
        self.providers = []

    def register_memory_provider(self, provider):
        self.providers.append(provider)


def make_provider(home, **config):
    full = dict(provider_mod._DEFAULTS)
    full.update(config)
    provider = CognitionMemoryProvider(config=full)
    provider.initialize("sess-1", hermes_home=home, platform="cli",
                        agent_context="primary")
    return provider


class SingleActiveRoutingTest(unittest.TestCase):
    def test_register_emits_exactly_one_provider_named_cognition(self):
        ctx = FakeCtx()
        provider_mod.register(ctx)
        self.assertEqual(len(ctx.providers), 1)
        self.assertEqual(ctx.providers[0].name, "cognition")

    def test_one_provider_routes_many_repos_internally(self):
        home = tempfile.mkdtemp(prefix="cog-home-")
        provider = make_provider(home, repos=["alpha", "beta", "gamma"])
        log = provider._log
        base_time = 1_700_000_000.0
        for i, repo in enumerate(["alpha", "beta", "gamma", "alpha"]):
            ref = "cog-sess-1-%d" % (i + 1)
            entry = {
                "ref": ref,
                "content": "fact %d" % i,
                "salience": 0.5,
                "provenance": provenance_mod.make_provenance(
                    source_session="sess-1", author="Ada", origin_repo=repo,
                    trace_id="t%d" % i, created_at=base_time + i),
            }
            status, _ = merge_mod.fetch_first_push(log, entry, None, now=base_time)
            self.assertEqual(status, "applied")
        # One provider instance serves all three repos — no provider per repo.
        self.assertEqual(len(log.entries_for_repo("alpha")), 2)
        self.assertEqual(len(log.entries_for_repo("beta")), 1)
        self.assertEqual(len(log.entries_for_repo("gamma")), 1)
        self.assertEqual(provider.identity_signature()["repos"],
                         ["alpha", "beta", "gamma"])
        provider.shutdown()

    def test_system_prompt_states_push_pending_and_dream_copy(self):
        home = tempfile.mkdtemp(prefix="cog-home-")
        provider = make_provider(home)
        block = provider.system_prompt_block()
        self.assertIn("session end", block)
        self.assertIn("Pending", block)
        self.assertIn("unpushed", block)
        self.assertIn("read-only", block)
        provider.shutdown()


if __name__ == "__main__":
    unittest.main()
