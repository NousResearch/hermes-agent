import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from _load_plugin_api import plugin_api


class MetricsSpineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.old_home = os.environ.get("HERMES_HOME")
        self.old_kanban = os.environ.get("HERMES_KANBAN_HOME")
        os.environ["HERMES_HOME"] = str(self.home)
        os.environ["HERMES_KANBAN_HOME"] = str(self.home)

    def tearDown(self):
        if self.old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self.old_home
        if self.old_kanban is None:
            os.environ.pop("HERMES_KANBAN_HOME", None)
        else:
            os.environ["HERMES_KANBAN_HOME"] = self.old_kanban
        self.tmp.cleanup()

    def _write_state_db(self, home: Path, rows):
        home.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(home / "state.db")
        con.executescript(
            """
            CREATE TABLE sessions (
              id TEXT, source TEXT, started_at REAL, ended_at REAL, end_reason TEXT,
              message_count INTEGER, tool_call_count INTEGER, api_call_count INTEGER,
              input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER,
              cache_read_tokens INTEGER, cache_write_tokens INTEGER,
              estimated_cost_usd REAL, actual_cost_usd REAL, title TEXT
            );
            """
        )
        con.executemany(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        con.commit()
        con.close()

    def _write_usage(self, home: Path, data):
        path = home / "skills" / ".usage.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_collect_profiles_discovers_siblings_when_home_is_profile_dir(self):
        root = self.home
        default_home = root / "profiles" / "default"
        alpha_home = root / "profiles" / "alpha"
        default_home.mkdir(parents=True, exist_ok=True)
        alpha_home.mkdir(parents=True, exist_ok=True)
        old_home = os.environ["HERMES_HOME"]
        os.environ["HERMES_HOME"] = str(default_home)
        try:
            profiles = plugin_api.collect_profiles()
        finally:
            os.environ["HERMES_HOME"] = old_home
        public = {profile.get("_public_name") for profile in profiles}
        self.assertIn("default", public)
        self.assertIn("profile_1", public)
        self.assertGreaterEqual(len(profiles), 2)

    def test_usage_rollup_is_profile_scoped_and_cost_confidence_is_partial(self):
        now = time.time()
        profile_home = self.home / "profiles" / "alpha"
        self._write_state_db(self.home, [
            ("raw-session-1", "cli", now - 60, now - 10, "completed", 4, 3, 2, 1000, 200, 0, 0, 0, 0.05, None, "private title"),
            ("raw-session-2", "cli", now - 60, None, None, 3, 2, 1, 900, 100, 0, 0, 0, 0.0, None, "open title"),
        ])
        self._write_state_db(profile_home, [
            ("raw-session-3", "telegram", now - 60, now - 5, "completed", 5, 4, 2, 0, 0, 0, 0, 0, 0.0, None, "zero usage"),
        ])
        profiles = [
            {"label": "Default", "_public_name": "default", "_path": str(self.home), "state": "active", "skill_count": 1},
            {"label": "Profile 1", "_public_name": "profile_1", "_path": str(profile_home), "state": "idle", "skill_count": 1},
        ]

        rollup = plugin_api.collect_usage_rollup(profiles, days=30)
        serialized = json.dumps(rollup)

        self.assertEqual(rollup["sessions"], 3)
        self.assertEqual(rollup["api_calls"], 5)
        self.assertEqual(rollup["total_tokens"], 2200)
        self.assertEqual(rollup["zero_usage_suspect_sessions"], 1)
        self.assertEqual(rollup["zero_cost_token_sessions"], 1)
        self.assertEqual(rollup["cost_confidence"], "partial")
        self.assertNotIn("raw-session", serialized)
        self.assertNotIn("private title", serialized)

    def test_skill_metadata_reads_profile_usage_without_raw_skill_names(self):
        now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        profile_home = self.home / "profiles" / "alpha"
        self._write_usage(self.home, {
            "private-default-skill": {"created_by": "agent", "created_at": now, "use_count": 2, "view_count": 3, "patch_count": 0, "last_used_at": now},
        })
        self._write_usage(profile_home, {
            "private-alpha-skill": {"created_at": now, "use_count": 0, "view_count": 1, "patch_count": 1, "last_patched_at": now},
        })
        profiles = [
            {"label": "Default", "_public_name": "default", "_path": str(self.home)},
            {"label": "Profile 1", "_public_name": "profile_1", "_path": str(profile_home)},
        ]

        metadata = plugin_api.collect_skill_metadata(profiles)
        serialized = json.dumps(plugin_api.strip_internal(metadata))

        self.assertEqual(metadata["summary"]["usage_sources"], 2)
        self.assertEqual(metadata["summary"]["total_skills"], 2)
        self.assertEqual(metadata["summary"]["agent_created"], 1)
        self.assertEqual(metadata["summary"]["never_used"], 1)
        self.assertIn("profile_1", serialized)
        self.assertNotIn("private-default-skill", serialized)
        self.assertNotIn("private-alpha-skill", serialized)

    def test_metrics_spine_exposes_operational_metrics_not_raw_ledgers(self):
        now = time.time()
        self._write_state_db(self.home, [
            ("raw-session-1", "cli", now - 60, now - 10, "completed", 4, 25, 2, 1000, 200, 0, 0, 0, 0.0, None, "private title"),
        ])
        self._write_usage(self.home, {
            "private-skill": {"created_by": "agent", "created_at": now, "use_count": 0, "view_count": 1, "patch_count": 0},
        })
        profiles = [{"label": "Default", "_public_name": "default", "_path": str(self.home), "state": "active", "skill_count": 1}]
        sessions = [{"state": "completed", "tool_call_count": 25, "message_count": 4, "avg_input_tokens": 500, "duration_seconds": 10}]
        kanban = {"boards": [], "totals": {"blocked": 1, "ready": 0, "running": 0}, "open": 1}
        skill_metadata = plugin_api.collect_skill_metadata(profiles)
        skill_coverage = {"summary": {"forced_skill_tasks": 0}}
        skill_hygiene = plugin_api.build_skill_hygiene(skill_metadata, skill_coverage, kanban)
        profile_fitness = {"summary": {"needs_review": 0, "average_score": 100}}
        performance = plugin_api.build_performance_tracking(sessions, kanban)
        ops_evals = {"summary": {"state": "ok", "score": 100}, "items": []}
        config_policy = {"summary": {"toolsets": 4, "max_turns": 90, "auxiliary_routes": 1}}
        rollup = plugin_api.collect_usage_rollup(profiles)

        spine = plugin_api.build_metrics_spine(
            profiles, sessions, kanban, skill_metadata, skill_coverage,
            skill_hygiene, profile_fitness, performance, ops_evals,
            config_policy, rollup,
        )
        serialized = json.dumps(plugin_api.strip_internal(spine))

        self.assertEqual(spine["schema_version"], "olympus.metrics_spine.v1")
        self.assertEqual(spine["window"]["ledger_owner"], "/analytics")
        self.assertEqual(spine["coverage"]["config_grounding"]["toolsets"], 4)
        self.assertEqual(spine["coverage"]["config_grounding"]["auxiliary_routes"], 1)
        self.assertNotIn("enabled_toolsets", spine["coverage"]["config_grounding"])
        self.assertEqual(spine["usage"]["cost_confidence"], "missing")
        self.assertEqual(spine["skills"]["agent_created"], 1)
        self.assertIn("Hermes Analytics owns raw usage and cost ledgers", spine["usage"]["note"])
        self.assertNotIn("raw-session-1", serialized)
        self.assertNotIn("private title", serialized)
        self.assertNotIn("private-skill", serialized)
        self.assertNotIn(str(self.home), serialized)
        self.assertNotIn("daily_bars", serialized)
        self.assertNotIn("leaderboard", serialized.lower())

    def test_metrics_spine_sources_stale_workers_from_orchestration(self):
        # Regression: stale_workers is produced by build_orchestration().summary,
        # never by the collect_kanban dict, so build_metrics_spine must read it
        # from the orchestration it is handed. With the bug it was always 0.
        profiles = [{"label": "Default", "_public_name": "default", "_path": str(self.home), "state": "active", "skill_count": 1}]
        sessions = [{"state": "completed", "tool_call_count": 1, "message_count": 1, "avg_input_tokens": 10, "duration_seconds": 5}]
        kanban = {"boards": [], "totals": {"blocked": 0, "ready": 1, "running": 1}, "open": 2}  # no stale_workers key
        orchestration = {"summary": {"stale_workers": 2, "blocked": 0}}

        spine = plugin_api.build_metrics_spine(
            profiles, sessions, kanban,
            plugin_api.collect_skill_metadata(profiles),
            {"summary": {"forced_skill_tasks": 0}},
            {"summary": {}},
            {"summary": {"needs_review": 0, "average_score": 100}},
            plugin_api.build_performance_tracking(sessions, kanban),
            {"summary": {"state": "ok", "score": 100}, "items": []},
            {"summary": {"toolsets": 0, "max_turns": 0, "auxiliary_routes": 0}},
            plugin_api.collect_usage_rollup(profiles),
            orchestration=orchestration,
        )

        self.assertEqual(spine["work"]["stale_workers"], 2)


if __name__ == "__main__":
    unittest.main()
