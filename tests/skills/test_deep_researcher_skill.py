"""Invariants: hostnames never count as independent origins; pruning keeps exact bytes recoverable."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "optional-skills/research/deep-researcher/scripts"
sys.path.insert(0, str(SCRIPTS))


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ledger = module("evidence_ledger")
run = module("research_run")


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "run"

    def cli(self, script, *args, success=True):
        result = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                                check=False, capture_output=True, text=True, encoding="utf-8")
        if success:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def init_run(self):
        self.cli("research_run.py", "init", self.root, "--question", "Synthetic question",
                 "--scope", "Synthetic fixtures", "--storage-mb", "1")

    def complete_run(self):
        self.init_run()
        self.cli("evidence_ledger.py", "init", self.root / "evidence.json",
                 "--topic", "Fixture", "--question", "Q1: Synthetic question")
        self.cli("evidence_ledger.py", "add-source", self.root / "evidence.json",
                 "--url", "https://example.org/data", "--title", "Dataset",
                 "--source-type", "primary", "--question", "Q1", "--summary", "Reports 10% growth.",
                 "--evidence", json.dumps({"excerpt": "The dataset reports growth of 10%.",
                                            "locator": "Results, paragraph 1"}))
        self.cli("evidence_ledger.py", "add-claim", self.root / "evidence.json",
                 "--text", "The dataset reports growth of 10%.", "--question", "Q1",
                 "--status", "single-source", "--source", "1", "--evidence", "S1E1")
        (self.root / "report.md").write_text(
            "The dataset reports growth of 10%. [1]\n\nSingle-origin evidence.\n\n"
            "## References\n1. [Dataset](https://example.org/data)\n", encoding="utf-8")
        brief = self.cli("evidence_ledger.py", "brief", self.root / "evidence.json").stdout
        (self.root / "evidence-brief.md").write_text(brief, encoding="utf-8")
        (self.root / "raw.txt").write_text("Synthetic retained text.\n" * 10000, encoding="utf-8")
        self.cli("research_run.py", "checkpoint", self.root, "--status", "complete",
                 "--artifact", "raw.txt")

    def init_ledger(self):
        self.path = self.base / "evidence.json"
        self.cli("evidence_ledger.py", "init", self.path, "--topic", "Fixture", "--question", "Q1")

    def add_source(self, url, *args):
        return self.cli("evidence_ledger.py", "add-source", self.path, "--url", url,
                        "--title", "Fixture", "--source-type", "primary", "--question", "Q1",
                        "--summary", "A test finding", *args)

    def claim(self, *args, success=True):
        return self.cli("evidence_ledger.py", "add-claim", self.path, "--text", "Narrow claim",
                        "--question", "Q1", "--status", "supported", "--source", "1,2", *args, success=success)

    def test_claim_validation_rejects_unretrieved_and_conflicting_sources(self):
        self.init_ledger()
        self.add_source("https://example.com/a")
        self.add_source("https://example.org/b", "--extraction-status", "failed")
        before = self.path.read_bytes()
        result = self.claim("--counter-source", "2", success=False)
        self.assertIn("both support and counter-evidence", result.stderr)
        self.assertIn("were not extracted: [2]", result.stderr)
        self.assertEqual(self.path.read_bytes(), before)

    def test_report_audit_reports_missing_and_duplicate_references(self):
        report = self.base / "report.md"
        report.write_text(
            "# Fixture\n\nA finding. [1] [3]\n\n## References\n\n"
            "1. [First](https://example.com/a)\n"
            "2. [Duplicate](https://example.com/a)\n", encoding="utf-8")
        result = self.cli("audit_report.py", report, "--no-live", success=False)
        self.assertIn("Citation [3] is used in the body but missing", result.stdout)
        self.assertIn("Duplicate bibliography URL in entries [1, 2]", result.stdout)
        self.assertIn("Bibliography entry [2] is never cited", result.stdout)

    def test_domains_are_not_independence(self):
        self.init_ledger()
        self.add_source("https://example.com/a")
        self.add_source("https://example.org/b")
        self.claim()
        data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        self.assertIsNone(data["sources"][0]["independence_group"])
        errors, warnings = ledger.validate(data)
        self.assertFalse(errors)
        self.assertTrue(any("lacks two independent" in item for item in warnings))
        # Legacy hostname defaults must also remain unverified.
        for index, source in enumerate(data["sources"]):
            source.pop("independence_assessed")
            source["independence_group"] = f"legacy-host-{index}"
        self.assertTrue(any("lacks two independent" in item for item in ledger.validate(data)[1]))

    def test_archive_prune_restore_exact_bytes(self):
        self.complete_run()
        (self.root / "unregistered.txt").write_text("Leave me alone", encoding="utf-8")
        before = {name: (self.root / name).read_bytes() for name in run.load(self.root)["artifacts"]}
        self.cli("research_run.py", "archive", self.root, "--prune")
        self.assertFalse((self.root / "evidence.json").exists())
        self.assertEqual((self.root / "report.md").read_bytes(), before["report.md"])
        self.assertTrue((self.root / "unregistered.txt").exists())
        self.assertLess((self.root / "artifacts.zip").stat().st_size, sum(map(len, before.values())))
        destination = self.base / "restored"
        self.cli("research_run.py", "restore", self.root, destination)
        for name, content in before.items():
            self.assertEqual((destination / name).read_bytes(), content)
        self.cli("research_run.py", "checkpoint", destination, "--status", "active", "--next-action", "Continue")
        self.assertEqual(run.load(destination)["status"], "active")
        self.cli("research_run.py", "restore", self.root, destination, success=False)
        self.cli("research_run.py", "archive", self.root, success=False)


if __name__ == "__main__":
    unittest.main()
