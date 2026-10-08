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

    @staticmethod
    def _get_provider_fixture(name: str):
        """Retrieve provider profile via get_provider_profile, falling back to direct repo-relative module spec."""
        try:
            from providers import get_provider_profile

            profile = get_provider_profile(name)
            if profile is not None:
                return profile
        except Exception:
            pass
        import importlib.util
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        plugin_init = repo_root / "plugins" / "model-providers" / name / "__init__.py"
        if plugin_init.exists():
            spec = importlib.util.spec_from_file_location(
                f"_hermes_test_{name.replace('-', '_')}",
                plugin_init,
                submodule_search_locations=[str(plugin_init.parent)],
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return getattr(mod, name.replace("-", "_"), None)
        return None

    def test_f6_nous_wire_hook_omits_reasoning_on_native_intent(self):
        """NousProfile.build_api_kwargs_extras omits reasoning on explicit native intent, but fills medium on None."""
        nous = self._get_provider_fixture("nous")
        self.assertIsNotNone(nous, "nous provider profile must load")

        # Explicit native default: omits reasoning parameter
        extra, top = nous.build_api_kwargs_extras(reasoning_config={"native": True}, supports_reasoning=True)
        self.assertEqual((extra, top), ({}, {}), "Explicit native must omit reasoning kwargs")

        # Unset reasoning (None): applies profile default (medium)
        extra_unset, _ = nous.build_api_kwargs_extras(reasoning_config=None, supports_reasoning=True)
        self.assertEqual(extra_unset, {"reasoning": {"enabled": True, "effort": "medium"}})

    def test_f6_openrouter_wire_hook_omits_reasoning_on_native_intent(self):
        """OpenRouterProfile.build_api_kwargs_extras omits reasoning on explicit native intent."""
        openrouter = self._get_provider_fixture("openrouter")
        self.assertIsNotNone(openrouter, "openrouter provider profile must load")

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

    def test_f7_oneshot_pre_agent_fallback_captures_reasoning_config(self):
        """Composed oneshot _run_agent captures AIAgent's reasoning_config for a pre-agent fallback."""
        from hermes_cli.oneshot import _run_agent

        cfg = {"agent": {"reasoning_effort": "high"}}
        fallback_entry = {"provider": "openrouter", "model": "lab/fallback:free", "reasoning_effort": "none"}
        runtime = {"provider": "openrouter", "api_key": "dummy"}

        with patch("hermes_cli.config.load_config", return_value=cfg), \
             patch("hermes_cli.runtime_provider.resolve_runtime_with_fallback", return_value=(runtime, fallback_entry)), \
             patch("hermes_cli.oneshot._create_session_db_for_oneshot"), \
             patch("hermes_cli.oneshot._load_resume_target", return_value=(None, [], None)), \
             patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build"), \
             patch("run_agent.AIAgent") as mock_agent:
            mock_inst = MagicMock()
            mock_inst.chat.return_value = "response"
            mock_agent.return_value = mock_inst

            _run_agent("test query")
            self.assertTrue(mock_agent.called)
            kwargs = mock_agent.call_args.kwargs
            self.assertEqual(kwargs.get("model"), "lab/fallback:free")
            self.assertEqual(kwargs.get("reasoning_config"), {"enabled": False})

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
        laguna_catalog = {
            "poolside/laguna-s-2.1:free": {
                "context_length": 262144,
                "name": "Poolside: Laguna S 2.1 (free)",
                "pricing": {"prompt": "0", "completion": "0"},
            }
        }
        with patch("agent.model_metadata.fetch_model_metadata", return_value=laguna_catalog), \
             patch("agent.models_dev.lookup_models_dev_context", return_value=None):
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

    # ─── P3: get_candidate_models tolerates base_url=None ─────────────────────

    def test_p3_get_candidate_models_tolerates_none_base_url(self):
        """get_candidate_models must not crash with AttributeError when base_url=None."""
        cand = ModelCandidateMetadata(id="model-free", provider="nous", is_free=True)
        with patch("hermes_cli.fallback_heuristics._gather_nous_candidates", return_value={"model-free": cand}):
            res = get_candidate_models(provider="nous", base_url=None, force_refresh=True)
            self.assertEqual([c.id for c in res], ["model-free"])

    # ─── N1 · P2: Global default parses to None & transport normalizes native ──

    def test_n1_global_default_parses_to_none_and_per_model_stops_before_global(self):
        """Global 'default' parses to None, while per-model override 'default' returns native dict."""
        self.assertIsNone(parse_reasoning_effort("default"))
        self.assertEqual(parse_reasoning_effort({"native": True}), {"native": True})

        cfg_global_default = {"agent": {"reasoning_effort": "default"}}
        self.assertIsNone(resolve_reasoning_config(cfg_global_default, "google/gemini-2.5-pro"))

        cfg_per_model_override = {
            "agent": {
                "reasoning_effort": "high",
                "reasoning_overrides": {"google/gemini-2.5-pro": "default"},
            }
        }
        self.assertEqual(
            resolve_reasoning_config(cfg_per_model_override, "google/gemini-2.5-pro"),
            {"native": True},
        )

        from agent.transports.chat_completions import _build_gemini_thinking_config
        self.assertIsNone(_build_gemini_thinking_config("google/gemini-2.5-pro", {"native": True}))
        self.assertIsNone(_build_gemini_thinking_config("google/gemini-2.5-pro", {"effort": "default"}))

    def test_n1_transport_normalizes_native_intent_per_profile(self):
        """Chat completions transport normalizes native reasoning_config to None for profiles other than nous/openrouter."""
        from agent.transports.chat_completions import ChatCompletionsTransport
        from providers.base import ProviderProfile

        transport = ChatCompletionsTransport()
        test_profile = ProviderProfile(name="zai")
        params = {"provider_profile": test_profile, "reasoning_config": {"native": True}}
        built = transport._build_kwargs_from_profile(test_profile, "glm-5", [], None, params)
        self.assertNotIn("thinking", built.get("extra_body", {}))

    # ─── N2 · P2: Pre-agent fallback reasoning on Gateway and TUI paths ─────────

    def test_n2_gateway_pre_agent_fallback_reasoning_propagated(self):
        """Gateway pre-agent fallback carries _fallback_entry and resolves reasoning config from it."""
        from gateway.run_config_loaders import GatewayConfigLoadersMixin
        from types import SimpleNamespace

        entry = {"provider": "nous", "model": "stepfun/step-3.7-flash:free", "_is_heuristic": True}
        loaders = GatewayConfigLoadersMixin()
        loaders._peek_session_state = lambda key: None
        loaders._resolve_session_key_or_none = lambda src, skey: None

        cfg = {"agent": {"reasoning_effort": "high"}}
        with patch("gateway.run._load_gateway_config", return_value=cfg):
            rc = loaders._resolve_session_reasoning_config(model="stepfun/step-3.7-flash:free", fallback_entry=entry)
            self.assertEqual(rc, {"native": True}, "Heuristic route must resolve to native default, not global high")

            static_entry = {"provider": "nous", "model": "stepfun/step-3.7-flash:free", "reasoning_effort": "low"}
            rc_low = loaders._resolve_session_reasoning_config(model="stepfun/step-3.7-flash:free", fallback_entry=static_entry)
            self.assertEqual(rc_low, {"enabled": True, "effort": "low"}, "Explicit entry effort must be honored")

    def test_n2_tui_pre_agent_fallback_entry_attached_and_resolved(self):
        """TUI _resolve_runtime_with_fallback attaches _fallback_entry."""
        from tui_gateway.server import _resolve_runtime_with_fallback
        from hermes_cli.auth import AuthError

        fallback_entry = {"provider": "openrouter", "model": "test/model:free", "reasoning_effort": "low"}
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=[AuthError("auth fail"), {"provider": "openrouter"}]), \
             patch("tui_gateway.server._load_fallback_model", return_value=[fallback_entry]):
            res = _resolve_runtime_with_fallback()
            self.assertTrue(res.used_fallback)
            self.assertEqual(res.runtime.get("_fallback_entry"), fallback_entry)

    # ─── N3 · P2: supports_tools default False and context window not 256k ─────

    def test_n3_supports_tools_default_false_and_context_window_no_fake_256k(self):
        """ModelCandidateMetadata defaults supports_tools to False and unknown context window is 0."""
        cand = ModelCandidateMetadata(id="unknown/unseen-model")
        self.assertFalse(cand.supports_tools)

        meta = enrich_model_metadata("completely-unknown-custom-slug-9999", provider="custom")
        self.assertEqual(meta.context_window, 0, "Unknown context window must not default to 256k")

        from hermes_cli.fallback_heuristics import _key_greatest_context
        c_known = ModelCandidateMetadata(id="known", context_window=200000)
        c_unknown = ModelCandidateMetadata(id="unknown", context_window=0)
        self.assertGreater(_key_greatest_context(c_known), _key_greatest_context(c_unknown))

    # ─── N4 · P2: Editor and runtime divergence over legacy fallback_model ─────

    def test_n4_stored_fallback_rules_and_write_chain_preserves_both_keys(self):
        """get_stored_fallback_rules reads both keys, marking legacy with _from_fallback_model, and _write_chain preserves both."""
        from hermes_cli.fallback_cmd import _write_chain, cmd_fallback_clear

        cfg = {
            "fallback_providers": [{"provider": "p1", "model": "m1"}],
            "fallback_model": [{"provider": "p2", "model": "m2"}],
        }
        rules = get_stored_fallback_rules(cfg)
        self.assertEqual(len(rules), 2)
        self.assertFalse(rules[0].get("_from_fallback_model", False))
        self.assertTrue(rules[1].get("_from_fallback_model", False))

        _write_chain(cfg, rules)
        self.assertEqual(cfg["fallback_providers"], [{"provider": "p1", "model": "m1"}])
        self.assertEqual(cfg["fallback_model"], [{"provider": "p2", "model": "m2"}])

        # Remove first rule (from fallback_providers) -> fallback_model remains intact
        remaining = [r for r in rules if r["model"] != "m1"]
        _write_chain(cfg, remaining)
        self.assertEqual(cfg["fallback_providers"], [])
        self.assertEqual(cfg["fallback_model"], [{"provider": "p2", "model": "m2"}])

        # Clear empties chain and removes fallback_model
        with patch("hermes_cli.config.load_config", return_value=cfg), \
             patch("builtins.input", return_value="y"), \
             patch("hermes_cli.config.save_config"):
            cmd_fallback_clear(SimpleNamespace())
            self.assertEqual(cfg.get("fallback_providers"), [])
            self.assertNotIn("fallback_model", cfg)

    # ─── N5 · P2: Curated OpenRouter emergency list removed ───────────────────

    def test_n5_curated_openrouter_emergency_removed_on_empty_catalog(self):
        """When all OpenRouter catalog sources fail/empty, returns empty dict without curated models."""
        from hermes_cli.fallback_heuristics import _gather_openrouter_candidates

        with patch("hermes_cli.fallback_heuristics._load_openrouter_live_catalog", return_value={}), \
             patch("hermes_cli.fallback_heuristics._load_openrouter_static_catalog", return_value={}), \
             patch("hermes_cli.fallback_heuristics._load_openrouter_disk_catalog", return_value={}):
            res = _gather_openrouter_candidates("openrouter")
            self.assertEqual(res, {})

    # ─── N6 · P2: Disabled static entry pops criteria & checks provenance ──────

    def test_n6_disabled_static_entry_pops_criteria_and_checks_provenance(self):
        """When heuristics disabled, static entry retains model and pops criteria/heuristic."""
        entry = {"provider": "nous", "model": "static-model", "criteria": {"free_only": True}}
        cfg = {"fallback_heuristics": False, "agent": {"reasoning_effort": "high"}}
        entries = _iter_fallback_entries([entry], config=cfg)
        self.assertEqual(len(entries), 1)
        self.assertNotIn("criteria", entries[0])
        self.assertNotIn("heuristic", entries[0])

        rc = resolve_fallback_entry_reasoning_config(cfg, "static-model", entries[0])
        self.assertEqual(rc, {"enabled": True, "effort": "high"}, "Static entry must not be classified as heuristic")

    # ─── N7 · P2: Delegation-scoped chains retain heuristics flag ─────────────

    def test_n7_scoped_fallback_chain_inherits_real_config_heuristics_flag(self):
        """scoped_fallback_chain inherits fallback_heuristics setting from config."""
        from hermes_cli.fallback_config import scoped_fallback_chain

        declared = [{"provider": "nous", "criteria": {"free_only": True}}]
        cfg = {"fallback_heuristics": True}
        cand = ModelCandidateMetadata(id="lab/free-match", provider="nous", is_free=True, supports_tools=True)
        with patch("hermes_cli.fallback_heuristics.get_candidate_models", return_value=[cand]):
            chain = scoped_fallback_chain(None, declared, pinned=False, owner="subagent", config=cfg)
            self.assertIsNotNone(chain)
            self.assertEqual(chain[0]["model"], "lab/free-match")

    # ─── N8 · P2: Live catalog negative caching ───────────────────────────────

    def test_n8_candidate_negative_caching(self):
        """Empty candidate results are negatively cached with _CANDIDATE_NEGATIVE_CACHE_TTL."""
        clear_candidate_cache()
        with patch("hermes_cli.fallback_heuristics._gather_generic_candidates", return_value={}):
            res = get_candidate_models(provider="empty-provider", force_refresh=False)
            self.assertEqual(res, [])

            from hermes_cli.fallback_heuristics import _CANDIDATE_CACHE
            self.assertTrue(any("empty-provider" in k for k in _CANDIDATE_CACHE))

    # ─── N9 · P3: Unknown sizes sorting & cross-vendor latest ─────────────────

    def test_n9_unknown_size_sorting_and_cross_vendor_latest(self):
        """Unknown param size ranks above nano (<1.0B) and dateless models do not sort by cross-vendor version tuple."""
        from hermes_cli.fallback_heuristics import _key_largest_params, _key_latest

        cand_nano = ModelCandidateMetadata(id="nano-0.5b", param_size_b=0.5)
        cand_unknown_size = ModelCandidateMetadata(id="unknown-size", param_size_b=None)
        self.assertGreater(_key_largest_params(cand_unknown_size), _key_largest_params(cand_nano))

        cand_v9_no_date = ModelCandidateMetadata(id="vendor-a/model-v9", version_tuple=(9, 0))
        cand_v1_no_date = ModelCandidateMetadata(id="vendor-b/model-v1", version_tuple=(1, 0))
        self.assertEqual(_key_latest(cand_v9_no_date)[:3], (0.0, 0.0, (0,)))
        self.assertEqual(_key_latest(cand_v1_no_date)[:3], (0.0, 0.0, (0,)))

    # ─── N10 · P3: Assertions test wiring and pass with networking disabled ────

    def test_n10_oneshot_pre_agent_fallback_wiring_failure_contract(self):
        """Deleting pre-agent fallback reasoning wiring falls back to global high and fails contract."""
        from hermes_cli.oneshot import _run_agent
        from hermes_constants import resolve_reasoning_config

        cfg = {"agent": {"reasoning_effort": "high"}}
        fallback_entry = {"provider": "openrouter", "model": "lab/fallback:free", "reasoning_effort": "none"}
        runtime = {"provider": "openrouter", "api_key": "dummy"}

        # Simulate absence of oneshot pre-agent fallback-entry reasoning resolution:
        unwired_rc = resolve_reasoning_config(cfg, fallback_entry["model"])
        self.assertEqual(unwired_rc, {"enabled": True, "effort": "high"})

        # Real oneshot wiring honors fallback_entry and resolves to enabled=False:
        with patch("hermes_cli.config.load_config", return_value=cfg), \
             patch("hermes_cli.runtime_provider.resolve_runtime_with_fallback", return_value=(runtime, fallback_entry)), \
             patch("hermes_cli.oneshot._create_session_db_for_oneshot"), \
             patch("hermes_cli.oneshot._load_resume_target", return_value=(None, [], None)), \
             patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build"), \
             patch("run_agent.AIAgent") as mock_agent:
            mock_inst = MagicMock()
            mock_inst.chat.return_value = "response"
            mock_agent.return_value = mock_inst

            _run_agent("test query")
            kwargs = mock_agent.call_args.kwargs
            self.assertNotEqual(kwargs.get("reasoning_config"), unwired_rc)
            self.assertEqual(kwargs.get("reasoning_config"), {"enabled": False})


if __name__ == "__main__":
    unittest.main()

