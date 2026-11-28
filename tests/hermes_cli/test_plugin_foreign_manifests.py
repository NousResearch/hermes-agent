"""Foreign harness metadata must not hide a repo's real Hermes plugin."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hermes_cli.agent_plugins import PLUGIN_SCHEMA_V1
from hermes_cli.plugins_discovery import scan_directory


class TestForeignHarnessManifests(unittest.TestCase):
    def test_foreign_metadata_skipped_but_hermes_manifest_discovered(self):
        for harness in (
            ".antigravity-plugin", ".claude-plugin", ".codex-plugin", ".cursor-plugin",
            ".devin-plugin", ".kimi-plugin", ".muse-plugin",
        ):
            with self.subTest(harness=harness), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repo = root / "multi-harness"
                foreign = repo / harness
                foreign.mkdir(parents=True)
                (foreign / "plugin.json").write_text(
                    json.dumps({"name": "multi-harness", "schemaVersion": 1}),
                    encoding="utf-8",
                )
                hermes = repo / ".hermes-plugin"
                hermes.mkdir()
                (hermes / "plugin.yaml").write_text(
                    "name: multi-harness\nversion: 1.0.0\n", encoding="utf-8"
                )

                with self.assertNoLogs("hermes_cli.plugins", level="WARNING"):
                    manifests = scan_directory(root, "user")

                self.assertEqual(
                    [(manifest.name, manifest.key) for manifest in manifests],
                    [("multi-harness", "multi-harness/.hermes-plugin")],
                )

    def test_other_portable_manifests_still_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, manifest in (
                ("portable", {"$schema": PLUGIN_SCHEMA_V1, "name": "portable"}),
                (".custom-plugin", {"name": "invalid"}),
                (".antigravity-plugin-extra", {"name": "invalid"}),
                (".muse-plugin-extra", {"name": "invalid"}),
            ):
                plugin = root / name
                plugin.mkdir()
                (plugin / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertLogs("hermes_cli.plugins", level="WARNING") as logs:
                manifests = scan_directory(root, "user")

            self.assertEqual([manifest.name for manifest in manifests], ["portable"])
            for name in (".custom-plugin", ".antigravity-plugin-extra", ".muse-plugin-extra"):
                self.assertTrue(any(
                    "Failed to parse" in message and name in message
                    for message in logs.output
                ))


if __name__ == "__main__":
    unittest.main()
