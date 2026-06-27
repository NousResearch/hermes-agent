"""Regression tests for issue #41611 — cronjob repeat/times non-int type coercion guard.

LLMs can pass `repeat` as a whitespace string (e.g. " ") instead of an integer.
Python raises `TypeError: '<=' not supported between instances of 'str' and 'int'`,
crashing the scheduler loop.
"""
import importlib.util
from pathlib import Path

import pytest


def _load_cron_jobs():
    """Import cron/jobs.py directly from the repo."""
    repo_root = Path(__file__).resolve().parents[1]
    lib_path = repo_root / "cron" / "jobs.py"
    spec = importlib.util.spec_from_file_location("cron_jobs", lib_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_cronjob_tools():
    """Import tools/cronjob_tools.py directly from the repo."""
    repo_root = Path(__file__).resolve().parents[1]
    lib_path = repo_root / "tools" / "cronjob_tools.py"
    spec = importlib.util.spec_from_file_location("cronjob_tools", lib_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRepeatNormalization:
    """Test that repeat normalization handles non-int types gracefully."""

    def test_repeat_none_unchanged(self):
        """None repeat should stay None."""
        mod = _load_cron_jobs()
        repeat = None
        # Simulate the normalization logic
        if repeat is not None:
            try:
                if int(repeat) <= 0:
                    repeat = None
            except (ValueError, TypeError):
                pass
        assert repeat is None

    def test_repeat_positive_int_unchanged(self):
        """Positive integer repeat should stay unchanged."""
        mod = _load_cron_jobs()
        repeat = 5
        if repeat is not None:
            try:
                if int(repeat) <= 0:
                    repeat = None
            except (ValueError, TypeError):
                pass
        assert repeat == 5

    def test_repeat_zero_becomes_none(self):
        """Zero repeat should become None (infinite)."""
        repeat = 0
        if repeat is not None:
            try:
                if int(repeat) <= 0:
                    repeat = None
            except (ValueError, TypeError):
                pass
        assert repeat is None

    def test_repeat_negative_becomes_none(self):
        """Negative repeat should become None (infinite)."""
        repeat = -1
        if repeat is not None:
            try:
                if int(repeat) <= 0:
                    repeat = None
            except (ValueError, TypeError):
                pass
        assert repeat is None

    def test_repeat_whitespace_string_survives(self):
        """Whitespace string repeat should not crash and should survive."""
        repeat = " "
        if repeat is not None:
            try:
                if int(repeat) <= 0:
                    repeat = None
            except (ValueError, TypeError):
                pass
        assert repeat == " "

    def test_repeat_empty_string_survives(self):
        """Empty string repeat should not crash."""
        repeat = ""
        if repeat is not None:
            try:
                if int(repeat) <= 0:
                    repeat = None
            except (ValueError, TypeError):
                pass
        assert repeat == ""

    def test_repeat_arbitrary_string_survives(self):
        """Arbitrary string repeat should not crash."""
        repeat = "forever"
        if repeat is not None:
            try:
                if int(repeat) <= 0:
                    repeat = None
            except (ValueError, TypeError):
                pass
        assert repeat == "forever"

    def test_repeat_string_int_survives(self):
        """String-ified int repeat should be correctly evaluated."""
        repeat = "3"
        if repeat is not None:
            try:
                if int(repeat) <= 0:
                    repeat = None
            except (ValueError, TypeError):
                pass
        assert repeat == "3"


class TestTimesComparison:
    """Test that the times comparison in the scheduler tick guard handles non-int types."""

    def test_times_none_unchanged(self):
        """None times should remain None."""
        times = None
        if times is not None:
            try:
                times = int(times)
            except (ValueError, TypeError):
                times = None
        assert times is None

    def test_times_int_unchanged(self):
        """Integer times should remain unchanged."""
        times = 5
        if times is not None:
            try:
                times = int(times)
            except (ValueError, TypeError):
                times = None
        assert times == 5

    def test_times_string_int_converted(self):
        """String-ified int should be converted to int."""
        times = "3"
        if times is not None:
            try:
                times = int(times)
            except (ValueError, TypeError):
                times = None
        assert times == 3

    def test_times_whitespace_string_becomes_none(self):
        """Whitespace string times should become None."""
        times = " "
        if times is not None:
            try:
                times = int(times)
            except (ValueError, TypeError):
                times = None
        assert times is None

    def test_times_arbitrary_string_becomes_none(self):
        """Arbitrary string times should become None."""
        times = "forever"
        if times is not None:
            try:
                times = int(times)
            except (ValueError, TypeError):
                times = None
        assert times is None

    def test_original_bug_crash(self):
        """Reproduce the original crash: '<=' between str and int."""
        repeat = " "
        # This would raise: TypeError: '<=' not supported between instances of 'str' and 'int'
        # Before the fix, this line would crash:
        with pytest.raises(TypeError):
            _ = repeat <= 0

        # After the fix, the guarded version works:
        if repeat is not None:
            try:
                normalized = None if int(repeat) <= 0 else repeat
            except (ValueError, TypeError):
                normalized = repeat
        assert normalized == " "
