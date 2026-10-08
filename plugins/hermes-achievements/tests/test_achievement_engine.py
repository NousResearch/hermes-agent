import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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

    def test_finished_rescan_keeps_persisted_unlock_when_live_metric_shrinks(self):
        definition = {
            "id": "durable_unlock",
            "name": "Durable Unlock",
            "threshold_metric": "total_terminal_calls",
            "tiers": [{"name": "Copper", "threshold": 40}],
        }
        with TemporaryDirectory() as data_dir, patch.object(plugin_api, "ACHIEVEMENTS", [definition]), patch.object(plugin_api, "_data_dir", return_value=Path(data_dir)), patch.object(plugin_api, "get_hermes_home", return_value=Path(data_dir)):
            unlocked = plugin_api._compute_from_scan({"aggregate": {"total_terminal_calls": 40}, "sessions": []})
            rescanned = plugin_api._compute_from_scan({"aggregate": {"total_terminal_calls": 39}, "sessions": []})
            partial = plugin_api._compute_from_scan({"aggregate": {"total_terminal_calls": 39}, "sessions": []}, is_partial=True)
            persisted = plugin_api.load_state()["unlocks"]

        self.assertTrue(unlocked["achievements"][0]["unlocked"])
        self.assertTrue(rescanned["achievements"][0]["unlocked"])
        self.assertEqual(rescanned["achievements"][0]["state"], "unlocked")
        # In-flight snapshots are published to the cache during rescans: the floor applies there too.
        self.assertTrue(partial["achievements"][0]["unlocked"])
        self.assertEqual(partial["unlocked_count"], 1)
        self.assertEqual(list(persisted), ["durable_unlock"])

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

    def test_analyze_messages_handles_serialized_tool_calls_json_and_dict(self):
        # Stringified JSON tool_calls (from raw DB rows or legacy formats)
        messages_json = [
            {"role": "assistant", "tool_calls": '[{"function": {"name": "memory"}}]'},
            {"role": "tool", "tool_name": "memory", "content": "ok"},
        ]
        stats = plugin_api.analyze_messages("s1", "Memory test", messages_json)
        self.assertEqual(stats["tool_call_count"], 1)
        self.assertIn("memory", stats["tool_names"])
        self.assertEqual(stats["memory_events"], 1)

        # Single-dict tool_calls
        messages_dict = [
            {"role": "assistant", "tool_calls": {"function": {"name": "terminal"}}},
            {"role": "tool", "tool_name": "terminal", "content": "ok"},
        ]
        stats2 = plugin_api.analyze_messages("s2", "Dict test", messages_dict)
        self.assertEqual(stats2["tool_call_count"], 1)
        self.assertIn("terminal", stats2["tool_names"])

    def test_analyze_messages_dedupes_superseded_uid_generations(self):
        """A rewind/retry soft-archives the superseded rows and re-inserts the current generation
        under the same message_uid. A full-history scan must count only the newest row per uid, or
        the undone generation is counted twice."""
        messages = [
            {"id": 1, "role": "user", "content": "do the thing", "message_uid": "u1"},
            {"id": 2, "role": "assistant", "message_uid": "a1",
             "tool_calls": [{"function": {"name": "memory", "arguments": "{}"}}]},
            {"id": 3, "role": "tool", "tool_name": "memory", "content": "ok", "message_uid": "t1"},
            # Rewound/retried generation: same uids, newer ids, divergent tool.
            {"id": 4, "role": "user", "content": "do the thing", "message_uid": "u1"},
            {"id": 5, "role": "assistant", "message_uid": "a1",
             "tool_calls": [{"function": {"name": "terminal", "arguments": "{}"}}]},
            {"id": 6, "role": "tool", "tool_name": "terminal", "content": "ok", "message_uid": "t1"},
        ]

        stats = plugin_api.analyze_messages("s1", "Rewind", messages)

        self.assertEqual(stats["message_count"], 3)
        self.assertEqual(stats["tool_call_count"], 1)
        self.assertEqual(stats["terminal_calls"], 1)
        self.assertEqual(stats["memory_events"], 0)

    def test_dashboard_card_hover_does_not_move_click_target(self):
        style_css = (
            Path(__file__).resolve().parents[1]
            / "dashboard"
            / "dist"
            / "style.css"
        ).read_text(encoding="utf-8-sig")

        hover_rule = next(
            line for line in style_css.splitlines() if line.startswith(".ha-card:hover")
        )
        self.assertNotIn("transform:", hover_rule)
        self.assertIn("border-color: var(--ha-tier)", hover_rule)
        self.assertIn("box-shadow:", hover_rule)


if __name__ == "__main__":
    unittest.main()


class CompactionScanTests(unittest.TestCase):
    def test_scan_stats_survive_compaction_and_v1_checkpoint_is_rescanned(self):
        """#112273: compaction archives the active rows (active=0, compacted=1); the scan must
        keep counting them, and a schema-1 (active-only) checkpoint must not be reused."""
        import hermes_state
        from hermes_state import SessionDB

        with TemporaryDirectory() as tmp, patch.object(plugin_api, "_data_dir", return_value=Path(tmp) / "data"), patch.object(plugin_api, "get_hermes_home", return_value=Path(tmp)):
            db = SessionDB(Path(tmp) / "state.db")
            try:
                db.create_session("s1", "cli", model="m")
                for i in range(20):
                    db.append_message("s1", "assistant", tool_calls=[{"function": {"name": f"tool_{i}", "arguments": "{}"}}])
                    db.append_message("s1", "tool", content="ok", tool_name=f"tool_{i}")
                db.archive_and_compact("s1", [{"role": "user", "content": "[summary]"}])
            finally:
                db.close()
            # A schema-1 (active-only) checkpoint for this session must be ignored, not reused.
            stale = {"fingerprint": None, "stats": {"tool_call_count": 0}}
            plugin_api._write_json(plugin_api.CHECKPOINT_FILE, {"schema_version": 1, "generated_at": 1, "sessions": {"s1": stale}})
            with patch.object(hermes_state, "SessionDB", lambda read_only=True: SessionDB(Path(tmp) / "state.db", read_only=read_only)):
                scan = plugin_api.scan_sessions()

        self.assertEqual(scan["aggregate"]["max_distinct_tools_in_session"], 20)
        self.assertEqual(scan["scan_meta"]["sessions_reused"], 0)

    def test_scan_stats_include_inactive_messages_history(self):
        """#127626: scan_sessions must include inactive rows (active=0) in SessionDB
        so historical tool calls (e.g. memory_events) are not undercounted."""
        import hermes_state
        from hermes_state import SessionDB

        with TemporaryDirectory() as tmp, patch.object(plugin_api, "_data_dir", return_value=Path(tmp) / "data"), patch.object(plugin_api, "get_hermes_home", return_value=Path(tmp)):
            db = SessionDB(Path(tmp) / "state.db")
            try:
                db.create_session("s1", "cli", model="m")
                # Deactivated turn with memory tool
                db.append_message("s1", "assistant", tool_calls=[{"function": {"name": "memory", "arguments": "{}"}}])
                db.append_message("s1", "tool", content="ok", tool_name="memory")
                db._write_sql("UPDATE messages SET active = 0 WHERE session_id = 's1'")

                # Active turn with terminal tool
                db.append_message("s1", "assistant", tool_calls=[{"function": {"name": "terminal", "arguments": "{}"}}])
                db.append_message("s1", "tool", content="ok", tool_name="terminal")
            finally:
                db.close()

            with patch.object(hermes_state, "SessionDB", lambda read_only=True: SessionDB(Path(tmp) / "state.db", read_only=read_only)):
                scan = plugin_api.scan_sessions()

        self.assertEqual(scan["aggregate"]["total_tool_calls"], 2)
        self.assertGreaterEqual(scan["aggregate"]["memory_events"], 1)
        self.assertGreaterEqual(scan["aggregate"]["total_terminal_calls"], 1)

    def test_scan_dedupes_rewound_duplicate_generations(self):
        """#127626 follow-up: /retry soft-archives the superseded generation and re-inserts the
        current one under the same message_uid. ``include_inactive=True`` returns both, so the
        scan must dedupe per uid or the rewound work is counted twice."""
        import hermes_state
        from hermes_state import SessionDB

        with TemporaryDirectory() as tmp, patch.object(plugin_api, "_data_dir", return_value=Path(tmp) / "data"), patch.object(plugin_api, "get_hermes_home", return_value=Path(tmp)):
            db = SessionDB(Path(tmp) / "state.db")
            try:
                db.create_session("s1", "cli", model="m")
                db.append_message("s1", "user", content="text one")
                db.append_message("s1", "assistant", tool_calls=[{"function": {"name": "memory", "arguments": "{}"}}])
                db.append_message("s1", "tool", content="ok", tool_name="memory")
                warm = db.get_messages("s1", include_inactive=True)
                # Retry the turn with a divergent tool: the surface re-installs the warm dicts, so
                # the re-inserted rows keep the original uids while the superseded rows soft-archive.
                retry = [
                    dict(warm[0]),
                    {**warm[1], "content": None, "tool_calls": [{"function": {"name": "terminal", "arguments": "{}"}}]},
                    {**warm[2], "tool_name": "terminal", "content": "ok"},
                ]
                db.replace_messages("s1", retry, active_only=True, archive_dropped=True)
            finally:
                db.close()

            with patch.object(hermes_state, "SessionDB", lambda read_only=True: SessionDB(Path(tmp) / "state.db", read_only=read_only)):
                scan = plugin_api.scan_sessions()

        self.assertEqual(scan["aggregate"]["total_tool_calls"], 1)
        self.assertEqual(scan["aggregate"]["total_terminal_calls"], 1)
        self.assertEqual(scan["aggregate"]["memory_events"], 0)

    def test_scan_rescans_stale_schema_checkpoint_after_scan_basis_change(self):
        """#127626 follow-up: a checkpoint written under the old display-projection basis must be
        rescanned (cache miss), never reused, once the scan reads the full inactive history."""
        import hermes_state
        from hermes_state import SessionDB

        with TemporaryDirectory() as tmp, patch.object(plugin_api, "_data_dir", return_value=Path(tmp) / "data"), patch.object(plugin_api, "get_hermes_home", return_value=Path(tmp)):
            db = SessionDB(Path(tmp) / "state.db")
            try:
                db.create_session("s1", "cli", model="m")
                db.append_message("s1", "assistant", tool_calls=[{"function": {"name": "memory", "arguments": "{}"}}])
                db.append_message("s1", "tool", content="ok", tool_name="memory")
            finally:
                db.close()

            with patch.object(hermes_state, "SessionDB", lambda read_only=True: SessionDB(Path(tmp) / "state.db", read_only=read_only)):
                first = plugin_api.scan_sessions()
                rolled_back = plugin_api.load_checkpoint()
                rolled_back["schema_version"] = plugin_api._CHECKPOINT_SCHEMA_VERSION - 1
                plugin_api._write_json(plugin_api.CHECKPOINT_FILE, rolled_back)
                second = plugin_api.scan_sessions()

        self.assertEqual(first["scan_meta"]["sessions_rescanned"], 1)
        self.assertEqual(second["scan_meta"]["sessions_reused"], 0)
        self.assertEqual(second["scan_meta"]["sessions_rescanned"], 1)
