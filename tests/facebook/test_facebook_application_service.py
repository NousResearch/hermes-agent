from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


UBM_ROOT = Path(__file__).resolve().parents[2] / "universal_browser_manager"
sys.path.insert(0, str(UBM_ROOT))

from facebook_core.application import FacebookApplicationService  # noqa: E402
from facebook_core.repository import (  # noqa: E402
    AmbiguousFriendError,
    FacebookRepository,
)
from facebook_core.schema import create_schema  # noqa: E402
from facebook_core.storage import connect  # noqa: E402


class FacebookApplicationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "facebook_crm.db"
        connection = connect(self.db_path)
        create_schema(connection, applied_at="2026-07-21T00:00:00+00:00")
        connection.executemany(
            "INSERT INTO friends(name, profile_url, permission_tier) VALUES (?, ?, ?)",
            [
                ("Unique Contact", "https://www.facebook.com/unique", 1),
                ("Duplicate Contact", "https://www.facebook.com/duplicate-a", 1),
                ("Duplicate Contact", "https://www.facebook.com/duplicate-b", 1),
                ("Percent % Contact", "https://www.facebook.com/percent", 1),
            ],
        )
        connection.commit()
        connection.close()
        self.repository = FacebookRepository(self.db_path)
        self.service = FacebookApplicationService(self.repository)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_repository_resolves_id_and_exact_case_insensitive_name(self) -> None:
        by_name = self.repository.find_friend("unique contact")
        self.assertIsNotNone(by_name)
        assert by_name is not None

        by_id = self.repository.find_friend(by_name["id"])
        self.assertEqual(by_id, by_name)
        self.assertIsNone(self.repository.find_friend("Unique"))

    def test_repository_rejects_ambiguous_names(self) -> None:
        with self.assertRaises(AmbiguousFriendError) as context:
            self.repository.find_friend("Duplicate Contact")
        self.assertEqual(len(context.exception.matching_ids), 2)

    def test_application_service_returns_stable_result_contract(self) -> None:
        found = self.service.show_friend("Unique Contact")
        self.assertTrue(found["success"])
        self.assertEqual(found["friend"]["name"], "Unique Contact")

        ambiguous = self.service.show_friend("Duplicate Contact")
        self.assertFalse(ambiguous["success"])
        self.assertEqual(ambiguous["error"], "Ambiguous friend name; use CRM id")
        self.assertEqual(len(ambiguous["matching_ids"]), 2)

        missing = self.service.show_friend("Missing Contact")
        self.assertFalse(missing["success"])
        self.assertIn("not found", missing["error"])

    def test_repository_fails_closed_when_database_is_missing(self) -> None:
        missing_path = Path(self.temp_dir.name) / "missing.db"
        with self.assertRaises(sqlite3.OperationalError):
            FacebookRepository(missing_path).find_friend("Nobody")
        self.assertFalse(missing_path.exists())

    def test_local_list_and_search_are_deterministic_and_escape_patterns(self) -> None:
        listed = self.service.list_friends(limit=2)
        self.assertTrue(listed["success"])
        self.assertEqual(listed["count"], 2)
        self.assertEqual(
            [friend["name"] for friend in listed["friends"]],
            ["Duplicate Contact", "Duplicate Contact"],
        )

        searched = self.service.search_friends("unique", limit=10)
        self.assertEqual(searched["results_count"], 1)
        self.assertEqual(searched["friends"][0]["name"], "Unique Contact")

        literal_percent = self.service.search_friends("%", limit=10)
        self.assertEqual(literal_percent["results_count"], 1)
        self.assertEqual(literal_percent["friends"][0]["name"], "Percent % Contact")

        empty = self.service.search_friends("   ")
        self.assertFalse(empty["success"])

        with self.assertRaises(ValueError):
            self.repository.list_friends(limit=0)


if __name__ == "__main__":
    unittest.main()
