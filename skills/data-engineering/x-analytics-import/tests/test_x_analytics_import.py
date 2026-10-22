from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "x_analytics_import.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

spec = importlib.util.spec_from_file_location("x_analytics_import", SCRIPT)
xai = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = xai
spec.loader.exec_module(xai)


class ImporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = xai.load_config(None)

    def parse(self, filename: str, kind: str):
        return xai.parse_export(FIXTURES / filename, kind, self.config)

    def test_public_defaults_are_account_agnostic(self):
        self.assertEqual(xai.VERSION, "1.0.0")
        self.assertEqual(self.config["classification"]["fallback_lane"], "Unclassified")
        self.assertEqual(self.config["classification"]["lanes"], [])

    def test_public_example_config_compiles(self):
        config = xai.load_config(ROOT / "references" / "config.example.json")
        rules = xai.compile_lane_rules(config)
        self.assertEqual({rule["name"] for rule in rules}, {"Tutorials", "Releases"})

    def test_valid_overview_and_aliases(self):
        export = self.parse("overview-valid.csv", "overview")
        self.assertFalse(export.errors)
        self.assertEqual(export.kind, "overview")
        alias = self.parse("overview-aliases.csv", "overview")
        self.assertFalse(alias.errors)
        self.assertEqual(alias.rows[0]["new_follows"], 5)
        self.assertEqual(alias.rows[0]["unfollows"], 1)

    def test_video_two_row_header(self):
        export = self.parse("video-valid.csv", "video")
        self.assertFalse(export.errors)
        self.assertEqual(export.header_row, 2)
        self.assertEqual(len(export.rows), 3)
        self.assertEqual(export.rows[0]["completion_rate"], 0.25)
        self.assertGreater(export.rows[0]["estimated_revenue"], 0)

    def test_missing_columns_fail_validation(self):
        export = self.parse("missing-columns.csv", "overview")
        report = xai.validation_report([export])
        self.assertEqual(report["status"], "failed")
        self.assertGreater(report["error_count"], 0)
        self.assertTrue(
            any(issue["code"] == "missing_required_column" for issue in report["issues"])
        )

    def test_unknown_columns_warn_by_default(self):
        export = self.parse("unknown-columns.csv", "overview")
        report = xai.validation_report([export])
        self.assertEqual(report["status"], "passed_with_warnings")
        self.assertTrue(
            any(issue["code"] == "unknown_column" for issue in report["issues"])
        )

    def test_malformed_numeric_value_fails(self):
        export = self.parse("malformed-values.csv", "overview")
        report = xai.validation_report([export])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any(issue["code"] == "invalid_number" for issue in report["issues"])
        )

    def test_duplicate_post_id_fails(self):
        export = self.parse("duplicate-post-ids.csv", "content")
        report = xai.validation_report([export])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any(issue["code"] == "duplicate_post_id" for issue in report["issues"])
        )

    def test_partial_current_day_detection(self):
        overview = self.parse("overview-valid.csv", "overview")
        result = xai.identify_partial_days(overview.rows, self.config)
        self.assertEqual(len(result["partial_dates"]), 1)
        self.assertIsNotNone(result["latest_complete_date"])
        self.assertGreaterEqual(result["prior_sample_size"], 5)

    def test_post_type_separation(self):
        content = self.parse("content-valid.csv", "content")
        partial = {"partial_dates": [], "latest_complete_date": None}
        posts = xai.normalize_posts(content.rows, partial, self.config)
        types = {post["post_type"]["value"] for post in posts}
        self.assertIn("reply", types)
        self.assertIn("quote", types)
        self.assertIn("original", types)

    def test_uncertain_lane_classification(self):
        config = xai.load_config(None)
        config["classification"]["lanes"] = [
            {
                "name": "A",
                "priority": 100,
                "include_any": ["atlas"],
                "exclude_any": [],
                "post_types": [],
            },
            {
                "name": "B",
                "priority": 100,
                "include_any": ["project"],
                "exclude_any": [],
                "post_types": [],
            },
        ]
        rules = xai.compile_lane_rules(config)
        row = {"post_text": "Atlas Project"}
        result = xai.classify_lane(
            row,
            {"value": "original", "confidence": "medium", "method": "test"},
            rules,
            "Unclassified",
        )
        self.assertEqual(result["value"], "Uncertain")
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(set(result["candidates"]), {"A", "B"})

    def test_viral_outlier_is_flagged(self):
        overview = self.parse("overview-valid.csv", "overview")
        content = self.parse("content-valid.csv", "content")
        partial = xai.identify_partial_days(overview.rows, self.config)
        posts = xai.normalize_posts(content.rows, partial, self.config)
        stats = xai.calculate_statistics(posts, partial, self.config)
        self.assertGreaterEqual(stats["viral_outliers"]["count"], 1)
        self.assertGreater(stats["overall"]["metrics"]["impressions"]["p90"], 0)
        self.assertEqual(
            stats["overall"]["metrics"]["impressions"]["sample_size"],
            stats["overall"]["sample_size"],
        )

    def test_reconciliation_does_not_assume_equality(self):
        overview = self.parse("overview-valid.csv", "overview")
        content = self.parse("content-valid.csv", "content")
        report = xai.reconcile_exports(overview, content, self.config)
        self.assertIn(report["status"], {"passed", "passed_with_warnings"})
        self.assertTrue(
            any(
                check["code"]
                in {"content_date_coverage", "low_content_date_coverage"}
                for check in report["checks"]
            )
        )
        self.assertTrue(
            any(
                check["code"] == "metric_scopes_not_directly_comparable"
                for check in report["checks"]
            )
        )

    def test_idempotent_incremental_import(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            parser = xai.build_parser()
            argv = [
                "incremental-import",
                "--overview",
                str(FIXTURES / "overview-valid.csv"),
                "--content",
                str(FIXTURES / "content-valid.csv"),
                "--output-dir",
                str(output),
                "--json",
            ]
            code1, result1 = xai.run(parser.parse_args(argv))
            code2, result2 = xai.run(parser.parse_args(argv))
            self.assertEqual(code1, xai.EXIT_OK)
            self.assertEqual(result1["status"], "complete")
            self.assertEqual(code2, xai.EXIT_OK)
            self.assertEqual(result2["status"], "already_imported")
            entries = xai.read_manifest_entries(output)
            self.assertEqual(
                len([entry for entry in entries if entry["status"] == "complete"]), 1
            )
            required_artifacts = {
                "validation_report",
                "reconciliation_report",
                "classification_report",
                "analysis_report",
                "public_safe_report",
                "normalized_snapshot",
                "manifest",
            }
            self.assertTrue(required_artifacts.issubset(result1["artifacts"]))
            for name in required_artifacts:
                self.assertTrue(Path(result1["artifacts"][name]).is_file(), name)

    def test_validate_only_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "must-not-exist"
            parser = xai.build_parser()
            args = parser.parse_args(
                [
                    "validate-only",
                    "--overview",
                    str(FIXTURES / "overview-valid.csv"),
                    "--content",
                    str(FIXTURES / "content-valid.csv"),
                    "--output-dir",
                    str(output),
                ]
            )
            code, result = xai.run(args)
            self.assertEqual(code, xai.EXIT_OK)
            self.assertIn(result["status"], {"passed", "passed_with_warnings"})
            self.assertFalse(output.exists())

    def test_reconciliation_does_not_compare_metric_scopes(self):
        overview = self.parse("overview-valid.csv", "overview")
        content = self.parse("content-valid.csv", "content")
        content.rows[0]["impressions"] = 10**9
        report = xai.reconcile_exports(overview, content, self.config)
        self.assertIn(report["status"], {"passed", "passed_with_warnings"})
        self.assertTrue(
            any(
                check["code"] == "metric_scopes_not_directly_comparable"
                for check in report["checks"]
            )
        )

    def test_reconciliation_rejects_content_after_overview(self):
        overview = self.parse("overview-valid.csv", "overview")
        content = self.parse("content-valid.csv", "content")
        content.rows[0]["date"] = "2099-01-01"
        report = xai.reconcile_exports(overview, content, self.config)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(
            any(check["code"] == "content_after_overview" for check in report["checks"])
        )

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            parser = xai.build_parser()
            args = parser.parse_args(
                [
                    "dry-run",
                    "--overview",
                    str(FIXTURES / "overview-valid.csv"),
                    "--content",
                    str(FIXTURES / "content-valid.csv"),
                    "--output-dir",
                    str(output),
                ]
            )
            code, result = xai.run(args)
            self.assertEqual(code, xai.EXIT_OK)
            self.assertEqual(result["status"], "dry_run")
            self.assertEqual(list(output.iterdir()), [])

    def test_private_data_does_not_leak_to_public_safe_report(self):
        overview = self.parse("overview-valid.csv", "overview")
        content = self.parse("content-valid.csv", "content")
        video = self.parse("video-valid.csv", "video")
        import_id, combined_hash = xai.make_import_id([overview, content, video])
        snapshot = xai.build_snapshot(
            [overview, content, video], self.config, import_id, combined_hash
        )
        report = xai.render_public_safe_markdown(snapshot)
        secret_text = content.rows[0]["post_text"]
        self.assertNotIn(secret_text, report)
        self.assertNotIn(str(video.rows[0]["estimated_revenue"]), report)
        self.assertNotIn(overview.sha256, report)
        self.assertNotIn("lane summary", report.casefold())

    def test_snapshot_comparison_uses_matched_periods_and_post_ids(self):
        overview = self.parse("overview-valid.csv", "overview")
        content = self.parse("content-valid.csv", "content")
        import_id, combined_hash = xai.make_import_id([overview, content])
        left = xai.build_snapshot(
            [overview, content], self.config, import_id, combined_hash
        )
        right = json.loads(json.dumps(left))
        right["import_id"] = "right"
        right["data"]["overview"][0]["impressions"] += 100
        right["data"]["content"][0]["impressions"] += 50
        comparison = xai.compare_snapshot_dicts(left, right)
        self.assertGreater(
            comparison["overview_overlap"]["matched_period_metrics"]["impressions"][
                "delta"
            ],
            0,
        )
        self.assertEqual(comparison["content_match"]["matched_post_ids"], 10)

    def test_rebuild_classification_preserves_raw_metrics(self):
        overview = self.parse("overview-valid.csv", "overview")
        content = self.parse("content-valid.csv", "content")
        import_id, combined_hash = xai.make_import_id([overview, content])
        snapshot = xai.build_snapshot(
            [overview, content], self.config, import_id, combined_hash
        )
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            source = temp_path / "source.json"
            xai.atomic_write_json(source, snapshot)
            new_config = xai.load_config(None)
            new_config["classification"]["lanes"] = [
                {
                    "name": "Everything",
                    "priority": 1,
                    "include_any": [".+"],
                    "exclude_any": [],
                    "post_types": [],
                }
            ]
            result = xai.rebuild_classification(
                source, temp_path, new_config, dry_run=False
            )
            rebuilt = xai.load_snapshot(Path(result["snapshot"]))
            self.assertEqual(
                rebuilt["data"]["content"][0]["impressions"],
                snapshot["data"]["content"][0]["impressions"],
            )
            self.assertTrue(
                all(
                    post["lane"]["value"] == "Everything"
                    for post in rebuilt["data"]["content"]
                )
            )


if __name__ == "__main__":
    unittest.main()
