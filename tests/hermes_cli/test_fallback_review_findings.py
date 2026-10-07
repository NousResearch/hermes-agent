"""Tests verifying resolution of all 11 review findings (F1–F11) on PR #133676."""

import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from hermes_cli.fallback_heuristics import (
    FallbackCriteria,
    ModelCandidateMetadata,
    enrich_model_metadata,
    filter_and_rank_candidates,
    get_candidate_models,
    clear_candidate_cache,
    resolve_fallback_entry,
)
from hermes_cli.fallback_config import (
    get_fallback_chain,
    get_stored_fallback_rules,
    _iter_fallback_entries,
)
from agent.fallback_reasoning import (
    resolve_fallback_entry_reasoning_config,
    reresolve_fallback_reasoning_config,
)
from hermes_constants import (
    parse_reasoning_effort,
    resolve_reasoning_config,
    set_hermes_home_override,
    reset_hermes_home_override,
)


class TestReviewFindings(unittest.TestCase):
    def setUp(self):
        clear_candidate_cache()

    def tearDown(self):
        clear_candidate_cache()

    # ─── F1 · P1: Keep free-only and other hard filters hard ───────────────────

    def test_f1_free_only_hard_filter_no_match_returns_empty(self):
        """When catalog contains only paid models, free_only=True returns empty list, not paid or hardcoded fallback."""
        paid_candidate = ModelCandidateMetadata(
            id="lab/paid-405b", provider="nous", is_free=False, supports_tools=True
        )
        with patch("hermes_cli.fallback_heuristics.get_candidate_models", return_value=[paid_candidate]):
            entry = {"provider": "nous", "criteria": {"free_only": True}}
            res = resolve_fallback_entry(entry)
            self.assertEqual(res, [], "Free-only rule must not relax to paid models or inject hardcoded fallbacks")

    def test_f1_vendor_filter_no_match_returns_empty(self):
        """When candidate vendor does not match criteria, returns empty list."""
        other_vendor = ModelCandidateMetadata(
            id="lab/model-a", provider="nous", vendor="lab", is_free=True, supports_tools=True
        )
        with patch("hermes_cli.fallback_heuristics.get_candidate_models", return_value=[other_vendor]):
            entry = {"provider": "nous", "criteria": {"vendor": "google", "free_only": True}}
            res = resolve_fallback_entry(entry)
            self.assertEqual(res, [], "Vendor mismatch must return no match")

    def test_f1_require_tools_no_match_returns_empty(self):
        """When candidates lack tool support, returns empty list."""
        no_tools = ModelCandidateMetadata(
            id="lab/no-tools", provider="nous", is_free=True, supports_tools=False
        )
        with patch("hermes_cli.fallback_heuristics.get_candidate_models", return_value=[no_tools]):
            entry = {"provider": "nous", "criteria": {"require_tools": True, "free_only": True}}
            res = resolve_fallback_entry(entry)
            self.assertEqual(res, [], "Tools requirement must not be bypassed")

    def test_f1_matching_free_candidate_succeeds(self):
        """A valid candidate matching all criteria is returned."""
        matching = ModelCandidateMetadata(
            id="lab/free-match", provider="nous", is_free=True, supports_tools=True
        )
        with patch("hermes_cli.fallback_heuristics.get_candidate_models", return_value=[matching]):
            entry = {"provider": "nous", "criteria": {"free_only": True}}
            res = resolve_fallback_entry(entry)
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]["model"], "lab/free-match")

    # ─── F2 · P1: Do not persist effective chain over stored rules ────────────

    def test_f2_cmd_fallback_remove_preserves_stored_inactive_heuristics(self):
        """Removing a static route via fallback_cmd must preserve stored inactive heuristic rules."""
        from hermes_cli.fallback_cmd import cmd_fallback_remove

        initial_config = {
            "fallback_heuristics": False,
            "fallback_providers": [
                {"provider": "nous", "criteria": {"free_only": True}},
                {"provider": "openai", "model": "gpt-4o-mini"},
            ],
        }

        saved_configs = []

        def mock_load():
            return dict(initial_config)

        def mock_save(cfg):
            saved_configs.append(cfg)

        with patch("hermes_cli.config.load_config", side_effect=mock_load), \
             patch("hermes_cli.config.save_config", side_effect=mock_save), \
             patch("hermes_cli.setup._curses_prompt_choice", return_value=1):  # Select second entry (gpt-4o-mini)
            cmd_fallback_remove(MagicMock())

        self.assertTrue(len(saved_configs) > 0)
        saved = saved_configs[-1]
        stored = saved.get("fallback_providers", [])
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0], {"provider": "nous", "criteria": {"free_only": True}})

    # ─── F3 · P2: Make normalization of materialized entries idempotent ───────

    def test_f3_normalization_idempotent_materialized_entries(self):
        """Calling normalization on materialized entries must preserve cardinality and identities."""
        candidates = [
            ModelCandidateMetadata(id=f"lab/model-{i}", provider="nous", is_free=True, supports_tools=True)
            for i in range(1, 4)
        ]
        with patch("hermes_cli.fallback_heuristics.get_candidate_models", return_value=candidates):
            cfg = {
                "fallback_heuristics": True,
                "fallback_providers": [{"heuristic": "free", "max_candidates": 3}],
            }
            pass1 = get_fallback_chain(cfg)
            self.assertEqual(len(pass1), 3)

            # Second pass through _iter_fallback_entries
            pass2 = _iter_fallback_entries(pass1, config=cfg)
            self.assertEqual(len(pass2), 3)
            self.assertEqual([e["model"] for e in pass1], [e["model"] for e in pass2])

            # Third pass
            pass3 = _iter_fallback_entries(pass2, config=cfg)
            self.assertEqual(len(pass3), 3)

    # ─── F4 · P2: Preserve absent effort instead of inventing override ────────

    def test_f4_absent_effort_preserves_per_model_override(self):
        """Heuristic route with no effort specified must respect config reasoning_overrides."""
        cfg = {
            "agent": {
                "reasoning_overrides": {
                    "lab/model-30b:free": "none",
                }
            }
        }
        entry = {
            "model": "lab/model-30b:free",
            "provider": "nous",
            "_is_heuristic": True,
            # No reasoning_effort in entry or criteria
        }
        res = resolve_fallback_entry_reasoning_config(cfg, "lab/model-30b:free", entry)
        self.assertEqual(res, {"enabled": False}, "Absent effort in heuristic route must respect per-model override")

    # ─── F5 · P2: Preserve YAML false through effort extraction ───────────────

    def test_f5_preserve_yaml_false(self):
        """reasoning_effort: False must resolve to disabled, not None or default."""
        agent = SimpleNamespace(model="stepfun/step-3.7-flash:free", reasoning_config=None)
        reresolve_fallback_reasoning_config(agent, {"reasoning_effort": False})
        self.assertEqual(agent.reasoning_config, {"enabled": False})

        agent2 = SimpleNamespace(model="stepfun/step-3.7-flash:free", reasoning_config=None)
        reresolve_fallback_reasoning_config(agent2, {"criteria": {"reasoning_effort": False}})
        self.assertEqual(agent2.reasoning_config, {"enabled": False})

    # ─── F6 · P2: Carry explicit native intent through provider wire hook ──────

    def test_f6_nous_wire_hook_omits_reasoning_on_native_intent(self):
        """NousProfile.build_api_kwargs_extras omits reasoning on explicit native intent, but fills medium on None."""
        from plugins.model_providers.nous import nous

        # Explicit native default: omits reasoning parameter
        extra, top = nous.build_api_kwargs_extras(reasoning_config={"native": True}, supports_reasoning=True)
        self.assertEqual((extra, top), ({}, {}), "Explicit native must omit reasoning kwargs")

        # Unset reasoning (None): applies profile default (medium)
        extra_unset, _ = nous.build_api_kwargs_extras(reasoning_config=None, supports_reasoning=True)
        self.assertEqual(extra_unset, {"reasoning": {"enabled": True, "effort": "medium"}})

    def test_f6_openrouter_wire_hook_omits_reasoning_on_native_intent(self):
        """OpenRouterProfile.build_api_kwargs_extras omits reasoning on explicit native intent."""
        from plugins.model_providers.openrouter import openrouter

        extra, _ = openrouter.build_api_kwargs_extras(
            reasoning_config={"native": True}, supports_reasoning=True, model="openai/gpt-5.6"
        )
        self.assertNotIn("reasoning", extra)

    # ─── F7 · P2: Apply fallback effort before AIAgent exists ─────────────────

    def test_f7_pre_agent_fallback_effort_applied(self):
        """resolve_fallback_entry_reasoning_config resolves effort from selected fallback entry."""
        cfg = {"agent": {"reasoning_effort": "high"}}
        entry = {"model": "lab/fallback", "reasoning_effort": "none"}
        rc = resolve_fallback_entry_reasoning_config(cfg, "lab/fallback", entry)
        self.assertEqual(rc, {"enabled": False})

        # Explicit level
        entry_low = {"model": "lab/fallback", "reasoning_effort": "low"}
        rc_low = resolve_fallback_entry_reasoning_config(cfg, "lab/fallback", entry_low)
        self.assertEqual(rc_low, {"enabled": True, "effort": "low"})

    # ─── F8 · P1: Candidate cache is profile-scoped ────────────────────────────

    def test_f8_candidate_cache_profile_scoped(self, tmp_path=None):
        """Candidate cache must be isolated per active profile (hermes_home_key)."""
        import tempfile
        from pathlib import Path

        dir_a = Path(tempfile.mkdtemp(prefix="profile_a_"))
        dir_b = Path(tempfile.mkdtemp(prefix="profile_b_"))

        cand_a = ModelCandidateMetadata(id="model-a-free", provider="nous", is_free=True)
        cand_b = ModelCandidateMetadata(id="model-b-free", provider="nous", is_free=True)

        tok_a = set_hermes_home_override(dir_a)
        try:
            with patch("hermes_cli.fallback_heuristics._gather_nous_candidates", return_value={"model-a-free": cand_a}):
                res_a = get_candidate_models(provider="nous", force_refresh=False)
                self.assertEqual([c.id for c in res_a], ["model-a-free"])

            # Switch to Profile B in same process
            reset_hermes_home_override(tok_a)
            tok_b = set_hermes_home_override(dir_b)
            with patch("hermes_cli.fallback_heuristics._gather_nous_candidates", return_value={"model-b-free": cand_b}):
                res_b = get_candidate_models(provider="nous", force_refresh=False)
                self.assertEqual([c.id for c in res_b], ["model-b-free"])

            # Switch back to Profile A: gets profile A's cached candidates
            reset_hermes_home_override(tok_b)
            tok_a2 = set_hermes_home_override(dir_a)
            res_a2 = get_candidate_models(provider="nous", force_refresh=False)
            self.assertEqual([c.id for c in res_a2], ["model-a-free"])
            reset_hermes_home_override(tok_a2)
        except Exception:
            # Clean up on failure
            try:
                reset_hermes_home_override(tok_a)
            except Exception:
                pass
            raise

    # ─── F9 · P2: Per-model native/default/auto stops before global effort ─────

    def test_f9_per_model_native_stops_before_global_effort(self):
        """per-model override 'native'/'default'/'auto' must resolve to native dict and stop before global effort."""
        cfg = {
            "agent": {
                "reasoning_effort": "high",
                "reasoning_overrides": {
                    "provider/test-native": "native",
                    "provider/test-default": "default",
                    "provider/test-auto": "auto",
                },
            }
        }
        for model in ("provider/test-native", "provider/test-default", "provider/test-auto"):
            rc = resolve_reasoning_config(cfg, model)
            self.assertEqual(rc, {"native": True}, f"{model} must resolve to native dict, not global high")

    # ─── F10 · P1: Nous heuristic metadata uses established resolver ───────────

    def test_f10_nous_metadata_laguna_context_window(self):
        """poolside/laguna-s-2.1:free must resolve to verified 262,144 context, not synthetic 8,192."""
        meta = enrich_model_metadata("poolside/laguna-s-2.1:free", provider="nous")
        self.assertEqual(meta.context_window, 262144)

    # ─── F11 · P2: Static fallback reasoning semantics preserved ──────────────

    def test_f11_static_fallback_inherits_global_effort(self):
        """Static fallback route with no entry effort must inherit global agent.reasoning_effort."""
        cfg = {"agent": {"reasoning_effort": "high"}}
        static_entry = {"provider": "openrouter", "model": "google/gemini-2.5-flash"}
        rc = resolve_fallback_entry_reasoning_config(cfg, "google/gemini-2.5-flash", static_entry)
        self.assertEqual(rc, {"enabled": True, "effort": "high"}, "Static fallback must inherit global effort")

        heuristic_entry = {"provider": "nous", "model": "stepfun/step-3.7-flash:free", "_is_heuristic": True}
        rc_heur = resolve_fallback_entry_reasoning_config(cfg, "stepfun/step-3.7-flash:free", heuristic_entry)
        self.assertEqual(rc_heur, {"native": True}, "Heuristic route must default to native default")


if __name__ == "__main__":
    unittest.main()
