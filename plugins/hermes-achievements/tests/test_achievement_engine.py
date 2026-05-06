import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "plugin_api.py"
spec = importlib.util.spec_from_file_location("plugin_api", MODULE_PATH)
plugin_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin_api)


class AchievementEngineTests(unittest.TestCase):
    def test_tool_call_stats_detect_tool_names_and_errors(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"function": {"name": "terminal"}}]},
            {"role": "tool", "tool_name": "terminal", "content": "Error: port 3000 already in use"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "web_search"}}]},
        ]

        stats = plugin_api.analyze_messages("s1", "Fix dev server", messages)

        self.assertEqual(stats["tool_call_count"], 2)
        self.assertEqual(stats["tool_names"], {"terminal", "web_search"})
        self.assertEqual(stats["error_count"], 1)
        self.assertIs(stats["port_conflict"], True)

    def test_tiered_achievement_reaches_highest_matching_tier(self):
        definition = {
            "id": "let_him_cook",
            "threshold_metric": "max_tool_calls_in_session",
            "tiers": [
                {"name": "Copper", "threshold": 10},
                {"name": "Silver", "threshold": 25},
                {"name": "Gold", "threshold": 50},
            ],
        }
        aggregate = {"max_tool_calls_in_session": 28}

        result = plugin_api.evaluate_tiered(definition, aggregate)

        self.assertIs(result["unlocked"], True)
        self.assertEqual(result["tier"], "Silver")
        self.assertEqual(result["progress"], 28)
        self.assertEqual(result["next_tier"], "Gold")

    def test_tiered_achievement_can_be_discovered_without_unlocking(self):
        definition = {
            "id": "terminal_goblin",
            "threshold_metric": "total_terminal_calls",
            "tiers": [{"name": "Copper", "threshold": 50}],
        }
        aggregate = {"total_terminal_calls": 12}

        result = plugin_api.evaluate_tiered(definition, aggregate)

        self.assertIs(result["unlocked"], False)
        self.assertIs(result["discovered"], True)
        self.assertEqual(result["state"], "discovered")
        self.assertEqual(result["progress"], 12)
        self.assertEqual(result["next_threshold"], 50)

    def test_secret_achievement_stays_hidden_without_progress(self):
        definition = {
            "id": "permission_denied_any_percent",
            "name": "Permission Denied Any%",
            "secret": True,
            "requirements": [{"metric": "permission_denied_events", "gte": 3}],
        }
        aggregate = {"permission_denied_events": 0}

        result = plugin_api.evaluate_requirements(definition, aggregate)
        display = plugin_api.display_achievement({**definition, **result})

        self.assertEqual(result["state"], "secret")
        self.assertEqual(display["name"], "???")
        self.assertNotIn("Permission", display["description"])

    def test_multi_condition_unlock_requires_all_requirements(self):
        definition = {
            "id": "full_send",
            "requirements": [
                {"metric": "max_terminal_calls_in_session", "gte": 10},
                {"metric": "max_file_tool_calls_in_session", "gte": 5},
                {"metric": "max_web_calls_in_session", "gte": 2},
            ],
        }

        partial = plugin_api.evaluate_requirements(definition, {
            "max_terminal_calls_in_session": 12,
            "max_file_tool_calls_in_session": 2,
            "max_web_calls_in_session": 0,
        })
        complete = plugin_api.evaluate_requirements(definition, {
            "max_terminal_calls_in_session": 12,
            "max_file_tool_calls_in_session": 6,
            "max_web_calls_in_session": 2,
        })

        self.assertEqual(partial["state"], "discovered")
        self.assertIs(partial["unlocked"], False)
        self.assertLess(partial["progress_pct"], 100)
        self.assertEqual(complete["state"], "unlocked")
        self.assertIs(complete["unlocked"], True)

    def test_catalog_has_60_plus_unique_achievements(self):
        ids = [achievement["id"] for achievement in plugin_api.ACHIEVEMENTS]
        self.assertGreaterEqual(len(ids), 60)
        self.assertEqual(len(ids), len(set(ids)))

    def test_model_provider_metrics_are_aggregated(self):
        sessions = [
            {"model_names": {"openai/gpt-5", "anthropic/claude-sonnet-4"}},
            {"model_names": {"google/gemini-pro", "mistral/large"}},
            {"model_names": {"qwen/qwen3"}},
        ]

        aggregate = plugin_api.aggregate_stats(sessions)

        self.assertEqual(aggregate["distinct_model_count"], 5)
        self.assertEqual(aggregate["distinct_provider_count"], 5)
        result = plugin_api.evaluate_definition(
            next(a for a in plugin_api.ACHIEVEMENTS if a["id"] == "five_model_flight"),
            aggregate,
        )
        self.assertEqual(result["state"], "unlocked")
        self.assertEqual(result["tier"], "Copper")

    def test_removed_noisy_achievements_are_not_in_catalog(self):
        ids = {achievement["id"] for achievement in plugin_api.ACHIEVEMENTS}
        self.assertNotIn("fallback_pilot", ids)
        self.assertNotIn("browser_sleuth", ids)
        self.assertNotIn("release_ritualist", ids)

    def test_open_weights_pilgrim_counts_only_local_model_metadata(self):
        aggregate_mentions_only = plugin_api.aggregate_stats([
            {"model_names": {"openai/gpt-5"}, "local_model_events": 999},
        ])
        aggregate_local_chat = plugin_api.aggregate_stats([
            {"model_names": {"openai/gpt-5"}},
            {"model_names": {"ollama/llama3"}},
        ])
        definition = next(a for a in plugin_api.ACHIEVEMENTS if a["id"] == "open_weights_pilgrim")

        self.assertEqual(aggregate_mentions_only["local_model_chat_sessions"], 0)
        self.assertEqual(plugin_api.evaluate_definition(definition, aggregate_mentions_only)["state"], "discovered")
        self.assertEqual(aggregate_local_chat["local_model_chat_sessions"], 1)
        self.assertEqual(plugin_api.evaluate_definition(definition, aggregate_local_chat)["state"], "unlocked")

    def test_config_surgeon_ignores_generic_config_mentions(self):
        stats = plugin_api.analyze_messages("s1", "Config talk", [{"content": "config config configuration not configured"}])
        self.assertEqual(stats["config_events"], 0)
        stats = plugin_api.analyze_messages("s2", "Real config", [{"content": "edited config.yaml, manifest.json, and .env.local"}])
        self.assertGreaterEqual(stats["config_events"], 3)


if __name__ == "__main__":
    unittest.main()
