"""Custom-endpoint catalog last resort: a sibling model id must not inherit a table window.

Repro: a self-hosted ``qwen3.8-flash-next`` (262,144-token window) behind a relay whose
``/v1/models`` publishes no context metadata. Every probe fails; the step-3b substring match
then hits the ``qwen3.8-flash`` entry and inherits 1,000,000 — a 4x inflation. Compression
thresholds are computed from the inflated window, so compaction is deferred until the endpoint
itself rejects the oversized prompt mid-turn. This is a distinct failure class from "probe finds
nothing" (#142334, #133606, #63161): here a wrong value IS found — a *different model's* — and it
outranks the conservative fallback.
"""
import unittest
from unittest.mock import patch

import agent.model_metadata as mm
from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, DEFAULT_FALLBACK_CONTEXT


def _custom_endpoint_ctx(model: str, table: dict) -> int:
    """Drive ``_resolve_custom_endpoint_context_length`` with every endpoint probe failing."""
    with patch.object(mm, "DEFAULT_CONTEXT_LENGTHS", table), \
         patch.object(mm, "_resolve_endpoint_context_length", return_value=None), \
         patch.object(mm, "is_local_endpoint", return_value=False), \
         patch.object(mm, "_probe_local_context_length", return_value=None), \
         patch.object(mm, "_query_ollama_api_show", return_value=None), \
         patch.object(mm, "_save_unless_skipped"):
        return mm._resolve_custom_endpoint_context_length(model, "https://relay.example/v1", "key", "custom")


class SiblingIdentityGuard(unittest.TestCase):
    def test_sibling_generation_does_not_inherit_the_sibling_window(self):
        """``qwen3.8-flash-next`` must not inherit ``qwen3.8-flash``'s 1M (the reported 4x inflation)."""
        self.assertEqual(
            _custom_endpoint_ctx("qwen3.8-flash-next", {"qwen3.8-flash": 1_000_000}),
            DEFAULT_FALLBACK_CONTEXT)

    def test_exact_model_id_still_takes_the_catalog_window(self):
        self.assertEqual(_custom_endpoint_ctx("qwen3.8-flash", {"qwen3.8-flash": 1_000_000}), 1_000_000)

    def test_decorative_residuals_still_inherit(self):
        for model in ("qwen3.8-flash:batch", "qwen3.8-flash:us", "qwen3.8-flash-latest",
                      "qwen3.8-flash-preview", "org/qwen3.8-flash:global", "qwen3.8-flash-1017-free"):
            self.assertEqual(_custom_endpoint_ctx(model, {"qwen3.8-flash": 1_000_000}), 1_000_000, model)

    def test_family_catch_all_still_answers_unknown_lineup_members(self):
        """Rejecting the sibling entry must fall through to the family catch-all, not just give up."""
        table = {"qwen3.8-flash": 1_000_000, "qwen": 131_072}
        self.assertEqual(_custom_endpoint_ctx("qwen3.8-flash-next", table), 131_072)

    def test_catch_all_answers_sibling_when_specific_entry_is_rejected(self):
        """``glm-4.5-air`` falls to the ``glm`` family entry instead of inheriting ``glm-4.5``."""
        table = {"glm-4.5": 200_000, "glm": 100_000}
        self.assertEqual(_custom_endpoint_ctx("glm-4.5-air", table), 100_000)

    def test_relay_slug_normalization_survives_the_guard(self):
        """#97398 must not regress: the hyphenated slug still hits the specific entry."""
        table = {"glm-5.3": 200_000, "glm": 100_000}
        self.assertEqual(_custom_endpoint_ctx("z-ai-glm-5-3", table), 200_000)

    def test_live_table_case(self):
        """The reported model against the real table: 1M must no longer be claimed."""
        self.assertNotEqual(_custom_endpoint_ctx("qwen3.8-flash-next", DEFAULT_CONTEXT_LENGTHS), 1_000_000)

    def test_family_head_answers_lineup_continuation(self):
        """``minimax`` (bare family head, no version digits) must keep answering ``MiniMax-M2.7``."""
        self.assertEqual(_custom_endpoint_ctx("minimax-m2.7", {"minimax": 204_800}), 204_800)

    def test_hosted_paths_keep_legacy_semantics(self):
        """Without the kwarg the matcher is byte-for-byte the old behaviour (hosted resolvers,
        Codex OAuth table and context pins rely on plain substring inheritance)."""
        hit = mm._longest_key_match(DEFAULT_CONTEXT_LENGTHS, "qwen3.8-flash-next")
        assert hit is not None
        self.assertEqual(hit[1], DEFAULT_CONTEXT_LENGTHS["qwen3.8-flash"])


if __name__ == "__main__":
    unittest.main()
