import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "plugin_api.py"
spec = importlib.util.spec_from_file_location("plugin_api", MODULE_PATH)
plugin_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin_api)


def sample_data():
    return {
        "generated_at": 1751328000,  # 2025-07-01 UTC
        "unlocked_count": 2,
        "total_count": 4,
        "aggregate": {"session_count": 12, "total_tool_calls": 340},
        "achievements": [
            {
                "id": "red_text",
                "name": "Red Text Connoisseur",
                "category": "Debugging",
                "state": "unlocked",
                "unlocked": True,
                "tier": "Gold",
                "progress_pct": 100,
                "unlocked_at": 1751200000,
            },
            {
                "id": "let_him_cook",
                "name": "Let Him Cook",
                "category": "Toolchain",
                "state": "discovered",
                "unlocked": False,
                "tier": None,
                "progress_pct": 40,
            },
            {
                "id": "night_owl",
                "name": "Night <Owl> & Friends",
                "category": "Lifestyle",
                "state": "unlocked",
                "unlocked": True,
                "tier": "Copper",
                "progress_pct": 100,
                "unlocked_at": 1751100000,
            },
            {
                "id": "model_hopper",
                "name": "Model Hopper",
                "category": "Models",
                "state": "secret",
                "unlocked": False,
                "tier": None,
                "progress_pct": 0,
            },
        ],
    }


class FilterTests(unittest.TestCase):
    def test_filter_by_state(self):
        items = sample_data()["achievements"]

        unlocked = plugin_api.filter_and_sort_achievements(items, state="unlocked")
        self.assertEqual([item["id"] for item in unlocked], ["red_text", "night_owl"])

        self.assertEqual(len(plugin_api.filter_and_sort_achievements(items)), 4)
        self.assertEqual(plugin_api.filter_and_sort_achievements(items, state="nope"), [])


class ExportJsonTests(unittest.TestCase):
    def test_export_json_structure_and_state_filter(self):
        payload = json.loads(plugin_api.export_json(sample_data(), state="unlocked"))

        self.assertEqual(payload["unlocked_count"], 2)
        self.assertEqual(payload["total_count"], 4)
        self.assertEqual(payload["generated_at"], 1751328000)
        self.assertEqual([item["id"] for item in payload["achievements"]], ["red_text", "night_owl"])

    def test_export_json_defaults_to_all_states(self):
        payload = json.loads(plugin_api.export_json(sample_data()))

        self.assertEqual(len(payload["achievements"]), 4)


class ExportMarkdownTests(unittest.TestCase):
    def test_markdown_has_header_categories_and_badges(self):
        content = plugin_api.export_markdown(sample_data())

        self.assertIn("# Hermes Achievements", content)
        self.assertIn("**2/4 unlocked** | Last scanned: 2025-07-01", content)
        # Defaults to unlocked-only.
        self.assertIn("## Debugging", content)
        self.assertIn("## Lifestyle", content)
        self.assertNotIn("## Toolchain", content)
        self.assertIn("| Achievement | Tier | Progress |", content)
        self.assertIn("img.shields.io/badge/Gold-100%25-FFD700", content)
        self.assertIn("██████████ 100%", content)

    def test_markdown_state_override_and_bad_timestamp(self):
        data = sample_data()
        data["generated_at"] = "not-a-timestamp"

        content = plugin_api.export_markdown(data, state="discovered")

        self.assertIn("Last scanned: unknown", content)
        self.assertIn("## Toolchain", content)
        self.assertIn("████░░░░░░ 40%", content)


class ExportSvgTests(unittest.TestCase):
    def test_svg_renders_one_row_per_unlocked_badge(self):
        content = plugin_api.export_svg(sample_data())

        self.assertTrue(content.startswith("<svg "))
        self.assertTrue(content.endswith("</svg>"))
        self.assertEqual(content.count('<rect class="badge"'), 2)
        self.assertIn("Red Text Connoisseur", content)
        self.assertIn("#FFD700", content)

    def test_svg_escapes_markup_in_names(self):
        content = plugin_api.export_svg(sample_data())

        self.assertNotIn("<Owl>", content)
        self.assertIn("Night &lt;Owl&gt; &amp; Frie", content)

    def test_svg_with_no_matching_badges_is_valid(self):
        content = plugin_api.export_svg(sample_data(), state="nope")

        self.assertTrue(content.startswith("<svg "))
        self.assertEqual(content.count('<rect class="badge"'), 0)


class AgentSummaryTests(unittest.TestCase):
    def test_summary_strengths_gaps_and_top_tier(self):
        summary = plugin_api._build_agent_summary(sample_data())

        self.assertEqual(summary["unlocked_count"], 2)
        self.assertEqual(summary["total_count"], 4)
        self.assertEqual(summary["total_sessions"], 12)
        self.assertEqual(summary["total_tool_calls"], 340)
        self.assertEqual(summary["strengths"], ["Debugging", "Lifestyle"])
        self.assertEqual(summary["gaps"], ["Toolchain", "Models"])
        self.assertEqual(summary["top_tier"], "Gold")
        self.assertEqual(summary["unlocked_ids"], ["red_text", "night_owl"])

    def test_summary_handles_empty_data(self):
        summary = plugin_api._build_agent_summary({})

        self.assertEqual(summary["unlocked_count"], 0)
        self.assertEqual(summary["strengths"], [])
        self.assertEqual(summary["gaps"], [])
        self.assertIsNone(summary["top_tier"])


class AgentSummaryFileTests(unittest.TestCase):
    def test_write_agent_summary_persists_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("HERMES_HOME")
            os.environ["HERMES_HOME"] = tmp
            original_home = plugin_api.get_hermes_home
            plugin_api.get_hermes_home = lambda: Path(tmp)
            try:
                plugin_api._write_agent_summary(sample_data())
                path = Path(tmp) / "plugins" / "hermes-achievements" / "agent_summary.json"
                self.assertTrue(path.exists())
                payload = json.loads(path.read_text())
                self.assertEqual(payload["top_tier"], "Gold")
                self.assertEqual(payload["unlocked_ids"], ["red_text", "night_owl"])
            finally:
                plugin_api.get_hermes_home = original_home
                if previous is None:
                    os.environ.pop("HERMES_HOME", None)
                else:
                    os.environ["HERMES_HOME"] = previous


if __name__ == "__main__":
    unittest.main()
