"""Cron aggregation must not perform full profile metadata scans."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hermes_cli import web_server
import hermes_cli.web_server_cron as _web_server_cron


class CronProfileEnumerationTests(unittest.TestCase):
    def test_uses_lightweight_name_path_enumerator(self):
        with tempfile.TemporaryDirectory() as root:
            homes = [
                ("default", Path(root)),
                ("coder-01", Path(root) / "profiles" / "coder-01"),
            ]
            with (
                mock.patch(
                    "hermes_cli.profiles.profiles_to_serve",
                    return_value=homes,
                ) as lightweight,
                mock.patch(
                    "hermes_cli.profiles.list_profiles",
                    side_effect=AssertionError("full profile scan is forbidden"),
                ),
            ):
                result = _web_server_cron._cron_profile_dicts()

        lightweight.assert_called_once_with(multiplex=True)
        self.assertEqual([item["name"] for item in result], ["default", "coder-01"])
        self.assertTrue(result[0]["is_default"])
        self.assertFalse(result[1]["is_default"])


if __name__ == "__main__":
    unittest.main()
