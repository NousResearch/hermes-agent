import json
import os
import tempfile
import unittest
from pathlib import Path

from _load_plugin_api import plugin_api


class SkillHygieneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.home)

    def tearDown(self):
        if self.old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self.old_home
        self.tmp.cleanup()

    def test_stored_skill_audits_are_safe_read_only_evidence(self):
        hub_dir = self.home / "skills" / ".hub"
        hub_dir.mkdir(parents=True)
        (hub_dir / "lock.json").write_text(json.dumps({
            "installed": {
                "github.com/private-org/visual-skill": {
                    "trust_level": "community",
                    "scan_verdict": "pass",
                    "skills_sh_audit": {
                        "agent-trust-hub": {"verdict": "Pass"},
                        "socket": {"verdict": "Warn"},
                        "snyk": {"verdict": "Fail"}
                    }
                }
            }
        }))

        metadata = plugin_api.collect_skill_metadata()
        payload = plugin_api.build_skill_hygiene(metadata, {"summary": {}})
        serialized = json.dumps(payload)

        self.assertEqual(payload["summary"]["hub_audit_pass"], 1)
        self.assertEqual(payload["summary"]["hub_audit_warn"], 1)
        self.assertEqual(payload["summary"]["hub_audit_fail"], 1)
        self.assertIn("Stored skill audit needs review", serialized)
        self.assertIn("agent-trust-hub: pass", serialized)
        self.assertIn("socket: warn", serialized)
        self.assertIn("snyk: fail", serialized)
        self.assertIn("skill:", serialized)
        self.assertNotIn("github.com/private-org/visual-skill", serialized)

    def test_malformed_skill_usage_metadata_does_not_break_scan(self):
        skills_dir = self.home / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / ".usage.json").write_text(json.dumps({
            "github.com/private-org/unstable-skill": {
                "use_count": "not-a-number",
                "view_count": "also-bad",
                "patch_count": "bad",
                "last_used_at": "not-a-date",
                "last_patched_at": {"bad": "timestamp"}
            }
        }))

        metadata = plugin_api.collect_skill_metadata()
        payload = plugin_api.build_skill_hygiene(metadata, {"summary": {}})
        serialized = json.dumps(payload)

        self.assertEqual(metadata["summary"]["total_skills"], 1)
        self.assertEqual(metadata["usage_items"][0]["use_count"], 0)
        self.assertEqual(metadata["usage_items"][0]["view_count"], 0)
        self.assertEqual(metadata["usage_items"][0]["patch_count"], 0)
        self.assertIsNone(metadata["usage_items"][0]["last_used_at"])
        self.assertIsNone(metadata["usage_items"][0]["last_patched_at"])
        self.assertIn("skill:", serialized)
        self.assertNotIn("github.com/private-org/unstable-skill", serialized)


if __name__ == "__main__":
    unittest.main()
