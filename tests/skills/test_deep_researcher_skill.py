"""Behavioral tests for evidence provenance and lossless run lifecycle."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import contextlib
import io
import pytest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / "optional-skills/research/deep-researcher/scripts"


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
                                capture_output=True, text=True)
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
        (self.root / "report.md").write_text("# Final report\nA synthetic finding.\n", encoding="utf-8")
        (self.root / "evidence.json").write_text(json.dumps({"fixture": "evidence " * 10000}), encoding="utf-8")
        self.cli("research_run.py", "checkpoint", self.root, "--status", "complete",
                 "--artifact", "evidence.json")

    def init_ledger(self):
        self.path = self.base / "evidence.json"
        self.cli("evidence_ledger.py", "init", self.path, "--topic", "Fixture", "--question", "Q1")

    def add_source(self, url, *args):
        return self.cli("evidence_ledger.py", "add-source", self.path, "--url", url,
                        "--title", "Fixture", "--source-type", "primary", "--question", "Q1",
                        "--summary", "A test finding", *args)

    def claim(self, *args):
        return self.cli("evidence_ledger.py", "add-claim", self.path, "--text", "Narrow claim",
                        "--question", "Q1", "--status", "supported", "--source", "1,2", *args)

    def test_domains_are_not_independence(self):
        self.init_ledger()
        self.add_source("https://example.com/a")
        self.add_source("https://example.org/b")
        self.claim()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsNone(data["sources"][0]["independence_group"])
        errors, warnings = ledger.validate(data)
        self.assertFalse(errors)
        self.assertTrue(any("lacks two independent" in item for item in warnings))
        # Legacy hostname defaults must also remain unverified.
        for index, source in enumerate(data["sources"]):
            source.pop("independence_assessed")
            source["independence_group"] = f"legacy-host-{index}"
        self.assertTrue(any("lacks two independent" in item for item in ledger.validate(data)[1]))

    def test_assessed_origins_and_locators(self):
        self.init_ledger()
        for index, host in enumerate(("example.com", "example.org")):
            self.add_source(f"https://{host}/a", "--independence-group", f"dataset-{index}",
                            "--independence-note", "Distinct original dataset reviewed",
                            "--evidence", json.dumps({"excerpt": "A short finding", "locator": "Results, p. 3"}))
        self.claim("--evidence", "S1E1", "--evidence", "S2E1", "--caveat", "Limited population")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(ledger.validate(data), ([], []))
        brief = self.cli("evidence_ledger.py", "brief", self.path).stdout
        self.assertIn("Results, p. 3", brief)
        self.assertIn("Limited population", brief)
        data["sources"][1]["independence_group"] = "dataset-0"
        self.assertTrue(any("lacks two independent" in item for item in ledger.validate(data)[1]))
        data["claims"][0]["evidence_ids"].append("S99E1")
        self.assertTrue(any("unknown or unrelated" in item for item in ledger.validate(data)[0]))
        data["claims"][0]["evidence_ids"] = ["S2E1"]
        data["claims"][0]["source_ids"] = [1]
        self.assertTrue(any("unknown or unrelated" in item for item in ledger.validate(data)[0]))

    def test_invalid_evidence_does_not_write_source(self):
        self.init_ledger()
        before = self.path.read_bytes()
        self.cli("evidence_ledger.py", "add-source", self.path, "--url", "https://example.com",
                 "--title", "Bad", "--source-type", "primary", "--evidence", '{"excerpt":"Missing location"}', success=False)
        self.assertEqual(self.path.read_bytes(), before)

    def test_brief_budget_does_not_discard_stored_evidence(self):
        self.init_ledger()
        self.add_source("https://example.com/a", "--evidence",
                        json.dumps({"excerpt": "finding " * 1000, "locator": "Section 1"}))
        self.cli("evidence_ledger.py", "add-claim", self.path, "--text", "Claim", "--question", "Q1",
                 "--status", "single-source", "--source", "1", "--evidence", "S1E1")
        before = self.path.read_bytes()
        brief = self.cli("evidence_ledger.py", "brief", self.path, "--max-chars", "300").stdout
        self.assertLessEqual(len(brief), 300)
        self.assertIn("BRIEF TRUNCATED", brief)
        self.assertEqual(self.path.read_bytes(), before)

    def test_existing_report_audit_offline(self):
        self.init_ledger()
        self.add_source("https://example.com/a")
        report = self.base / "report.md"
        report.write_text("# Fixture\n\n## Findings\nThe source reports a result. [1]\n\n## Citation Links\n\n1. [Fixture](https://example.com/a)\n", encoding="utf-8")
        self.cli("audit_report.py", report, "--ledger", self.path, "--no-live")

    def test_checkpoint_and_resume_state(self):
        self.init_run()
        self.cli("research_run.py", "checkpoint", self.root, "--completed", "Q1",
                 "--gaps", "Q2", "--next-action", "Retrieve original")
        data = run.load(self.root)
        self.assertEqual(data["next_action"], "Retrieve original")
        self.assertEqual(data["gaps"], ["Q2"])
        self.cli("research_run.py", "checkpoint", self.root, "--clear-gaps")
        self.assertEqual(run.load(self.root)["completed"], ["Q1"])
        self.assertEqual(run.load(self.root)["gaps"], [])
        self.cli("research_run.py", "checkpoint", self.root, "--status", "complete", success=False)
        self.cli("research_run.py", "archive", self.root, "--prune", success=False)

    def test_archive_prune_restore_exact_bytes(self):
        self.complete_run()
        (self.root / "unregistered.txt").write_text("Leave me alone", encoding="utf-8")
        before = {name: (self.root / name).read_bytes() for name in run.load(self.root)["artifacts"]}
        self.cli("research_run.py", "archive", self.root, "--prune")
        self.assertFalse((self.root / "evidence.json").exists())
        self.assertEqual((self.root / "report.md").read_bytes(), before["report.md"])
        self.assertTrue((self.root / "unregistered.txt").exists())
        self.assertLess((self.root / "artifacts.zip").stat().st_size, len(before["evidence.json"]))
        destination = self.base / "restored"
        self.cli("research_run.py", "restore", self.root, destination)
        for name, content in before.items():
            self.assertEqual((destination / name).read_bytes(), content)
        self.cli("research_run.py", "checkpoint", destination, "--status", "active", "--next-action", "Continue")
        self.assertEqual(run.load(destination)["status"], "active")
        self.cli("research_run.py", "restore", self.root, destination, success=False)
        self.cli("research_run.py", "archive", self.root, success=False)

    def test_archive_without_prune_retains_files(self):
        self.complete_run()
        self.cli("research_run.py", "archive", self.root)
        self.assertTrue((self.root / "evidence.json").exists())

    def test_archive_without_optional_compression_modules(self):
        for omit_zlib, method in ((False, "deflate"), (True, "stored")):
            with self.subTest(method=method):
                self.root = self.base / method
                self.complete_run()
                original = (self.root / "evidence.json").read_bytes()
                args = type("Args", (), {"run": str(self.root), "prune": True,
                                         "destination": str(self.base / (method + "-restored"))})()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(run.zipfile, "lzma", None))
                    if omit_zlib:
                        stack.enter_context(patch.object(run.zipfile, "zlib", None))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    run.archive(args)
                    self.assertEqual(run.load(self.root)["archive"]["compression"], method)
                    run.restore(args)
                self.assertEqual((Path(args.destination) / "evidence.json").read_bytes(), original)

    def test_verification_failure_never_prunes(self):
        self.complete_run()
        args = type("Args", (), {"run": str(self.root), "prune": True})()
        with patch.object(run, "verify", side_effect=ValueError("Simulated corruption")):
            with self.assertRaises(ValueError):
                run.archive(args)
        self.assertTrue((self.root / "evidence.json").exists())
        self.assertFalse((self.root / "artifacts.zip").exists())
        self.assertEqual(run.load(self.root)["status"], "complete")

    def test_corruption_blocks_restore(self):
        self.complete_run()
        self.cli("research_run.py", "archive", self.root)
        with (self.root / "artifacts.zip").open("ab") as stream:
            stream.write(b"corrupted")
        destination = self.base / "bad-restore"
        self.cli("research_run.py", "restore", self.root, destination, success=False)
        self.assertFalse(destination.exists())

    def test_unsafe_paths_and_existing_directory(self):
        self.init_run()
        self.cli("research_run.py", "init", self.root, "--question", "Q", "--scope", "S", success=False)
        outside = self.base / "private.txt"
        outside.write_text("Private fixture", encoding="utf-8")
        for name in ("../private.txt", str(outside), "a/../private.txt", "C:/private.txt"):
            self.cli("research_run.py", "checkpoint", self.root, "--artifact", name, success=False)
        self.assertEqual(outside.read_text(encoding="utf-8"), "Private fixture")

    @pytest.mark.linux_only
    def test_symlink_artifact_is_rejected(self):
        self.init_run()
        outside = self.base / "private.txt"
        outside.write_text("Private fixture", encoding="utf-8")
        (self.root / "link").symlink_to(outside)
        self.cli("research_run.py", "checkpoint", self.root, "--artifact", "link", success=False)
        self.assertEqual(outside.read_text(encoding="utf-8"), "Private fixture")

    def test_storage_warning_is_explicit(self):
        self.init_run()
        (self.root / "large.txt").write_bytes(b"x" * (1024 * 1024 + 1))
        result = self.cli("research_run.py", "checkpoint", self.root, "--artifact", "large.txt")
        self.assertIn("storage budget exceeded", result.stdout)


if __name__ == "__main__":
    unittest.main()
