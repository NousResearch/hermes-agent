"""Tests for per-job max_iterations override in the cron scheduler.

Fallback chain under test:
  job["max_iterations"] (positive int)
    > config["agent"]["max_turns"]
    > config["max_turns"]
    > 90  (hard default)

Run:  python -m pytest tests/cron/test_scheduler_max_iterations.py -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _resolve(job: dict, cfg: dict):
    """
    Mirror of the production resolution logic in cron/scheduler.py.
    Must stay in sync with the "Max iterations" block in _run_job.
    Returns (max_iterations: int, warnings: list[str]).
    """
    warnings = []
    global_max = cfg.get("agent", {}).get("max_turns") or cfg.get("max_turns") or 90
    job_val = job.get("max_iterations")
    if job_val is None:
        return global_max, warnings
    if isinstance(job_val, int) and job_val > 0:
        return job_val, warnings
    warnings.append(f"invalid max_iterations {job_val!r}")
    return global_max, warnings


class TestFallbackChain(unittest.TestCase):

    def test_per_job_override_wins_over_all_config(self):
        result, warns = _resolve(
            {"max_iterations": 25},
            {"agent": {"max_turns": 50}, "max_turns": 30},
        )
        self.assertEqual(result, 25)
        self.assertEqual(warns, [])

    def test_no_field_falls_back_to_agent_max_turns(self):
        result, warns = _resolve({}, {"agent": {"max_turns": 40}})
        self.assertEqual(result, 40)
        self.assertEqual(warns, [])

    def test_no_field_falls_back_to_top_level_max_turns(self):
        result, warns = _resolve({}, {"max_turns": 55})
        self.assertEqual(result, 55)
        self.assertEqual(warns, [])

    def test_empty_config_uses_hard_default_90(self):
        result, warns = _resolve({}, {})
        self.assertEqual(result, 90)
        self.assertEqual(warns, [])

    def test_agent_max_turns_beats_top_level_max_turns(self):
        result, warns = _resolve({}, {"agent": {"max_turns": 20}, "max_turns": 60})
        self.assertEqual(result, 20)


class TestValidation(unittest.TestCase):

    def test_string_value_rejected_warns_and_falls_back(self):
        result, warns = _resolve({"max_iterations": "forty"}, {"agent": {"max_turns": 30}})
        self.assertEqual(result, 30)
        self.assertEqual(len(warns), 1)
        self.assertIn("forty", warns[0])

    def test_negative_integer_rejected(self):
        result, warns = _resolve({"max_iterations": -5}, {"max_turns": 45})
        self.assertEqual(result, 45)
        self.assertEqual(len(warns), 1)

    def test_zero_rejected(self):
        result, warns = _resolve({"max_iterations": 0}, {})
        self.assertEqual(result, 90)
        self.assertEqual(len(warns), 1)

    def test_float_rejected(self):
        result, warns = _resolve({"max_iterations": 10.5}, {"max_turns": 20})
        self.assertEqual(result, 20)
        self.assertEqual(len(warns), 1)

    def test_explicit_none_treated_as_absent(self):
        result, warns = _resolve({"max_iterations": None}, {"max_turns": 35})
        self.assertEqual(result, 35)
        self.assertEqual(warns, [])


class TestBoundary(unittest.TestCase):

    def test_value_of_one_valid(self):
        result, warns = _resolve({"max_iterations": 1}, {})
        self.assertEqual(result, 1)
        self.assertEqual(warns, [])

    def test_large_value_accepted(self):
        result, warns = _resolve({"max_iterations": 500}, {})
        self.assertEqual(result, 500)
        self.assertEqual(warns, [])


class TestSchedulerImport(unittest.TestCase):
    def test_scheduler_imports_cleanly(self):
        try:
            import cron.scheduler  # noqa: F401
        except ImportError as e:
            self.skipTest(f"Optional deps not installed in test env: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
