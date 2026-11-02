"""Behaviour contracts for the systematic-debugging tracer helper."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "software-development" / "systematic-debugging"
SCRIPT = SKILL_DIR / "scripts" / "debug_trace.py"


def _load():
    spec = importlib.util.spec_from_file_location("debug_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TraceCrashTests(unittest.TestCase):
    def _target(self, body: str) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        tmp.write(body)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_clean_exit_reports_no_crash(self) -> None:
        target = self._target("x = 1\n")
        report = _load().trace_crash(target, [])
        self.assertEqual(report, {"crashed": False})

    def test_exception_carries_innermost_locals(self) -> None:
        target = self._target(
            "def inner(secret):\n"
            "    raise ValueError('boom')\n"
            "def outer():\n"
            "    token = 'tok-123'\n"
            "    inner(token)\n"
            "outer()\n"
        )
        report = _load().trace_crash(target, [])
        self.assertTrue(report["crashed"])
        self.assertIn("ValueError: boom", report["exception"])
        functions = [frame["function"] for frame in report["frames"]]
        self.assertIn("inner", functions)
        self.assertIn("outer", functions)
        outer = next(f for f in report["frames"] if f["function"] == "outer")
        self.assertEqual(outer["locals"].get("token"), "'tok-123'")

    def test_large_locals_are_truncated(self) -> None:
        target = self._target("big = 'x' * 5000\nraise RuntimeError('nope')\n")
        report = _load().trace_crash(target, [])
        (frame,) = [f for f in report["frames"] if f["function"] == "<module>"]
        self.assertIn("truncated", frame["locals"]["big"])

    def test_argv_reaches_the_target(self) -> None:
        target = self._target("import sys\nassert sys.argv[1:] == ['a', 'b']\n")
        report = _load().trace_crash(target, ["a", "b"])
        self.assertEqual(report, {"crashed": False})


class CliTests(unittest.TestCase):
    def test_missing_target_exits_2(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "no-such-file.py"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 2)

    def test_json_flag_emits_parseable_report(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write("raise KeyError('k')\n")
            name = tmp.name
        self.addCleanup(lambda: Path(name).unlink(missing_ok=True))
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), name, "--json"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        self.assertTrue(report["crashed"])
        self.assertIn("KeyError", report["exception"])


if __name__ == "__main__":
    unittest.main()
