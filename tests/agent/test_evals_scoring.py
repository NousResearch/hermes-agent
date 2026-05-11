"""Tests for the eval scoring primitives."""

import json
import os
import pytest

from agent.evals.types import CheckType, DeterministicCheck, CheckResult
from agent.evals.scoring import run_check, score_checks


@pytest.fixture
def workdir(tmp_path):
    return str(tmp_path)


class TestRunCheck:
    """Tests for individual deterministic checks."""

    def test_file_exists_pass(self, workdir):
        path = os.path.join(workdir, "hello.txt")
        with open(path, "w") as f:
            f.write("hi")
        check = DeterministicCheck(CheckType.FILE_EXISTS, "hello.txt")
        result = run_check(check, workdir)
        assert result.passed is True

    def test_file_exists_fail(self, workdir):
        check = DeterministicCheck(CheckType.FILE_EXISTS, "missing.txt")
        result = run_check(check, workdir)
        assert result.passed is False

    def test_file_not_exists_pass(self, workdir):
        check = DeterministicCheck(CheckType.FILE_NOT_EXISTS, "nope.txt")
        result = run_check(check, workdir)
        assert result.passed is True

    def test_file_not_exists_fail(self, workdir):
        with open(os.path.join(workdir, "exists.txt"), "w") as f:
            f.write("x")
        check = DeterministicCheck(CheckType.FILE_NOT_EXISTS, "exists.txt")
        result = run_check(check, workdir)
        assert result.passed is False

    def test_content_contains_pass(self, workdir):
        with open(os.path.join(workdir, "data.txt"), "w") as f:
            f.write("hello world foo bar")
        check = DeterministicCheck(CheckType.CONTENT_CONTAINS, "data.txt", "foo")
        result = run_check(check, workdir)
        assert result.passed is True

    def test_content_contains_fail(self, workdir):
        with open(os.path.join(workdir, "data.txt"), "w") as f:
            f.write("hello world")
        check = DeterministicCheck(CheckType.CONTENT_CONTAINS, "data.txt", "missing")
        result = run_check(check, workdir)
        assert result.passed is False

    def test_content_not_contains_pass(self, workdir):
        with open(os.path.join(workdir, "data.txt"), "w") as f:
            f.write("hello world")
        check = DeterministicCheck(CheckType.CONTENT_NOT_CONTAINS, "data.txt", "absent")
        result = run_check(check, workdir)
        assert result.passed is True

    def test_content_equals_pass(self, workdir):
        with open(os.path.join(workdir, "exact.txt"), "w") as f:
            f.write("exactly this")
        check = DeterministicCheck(CheckType.CONTENT_EQUALS, "exact.txt", "exactly this")
        result = run_check(check, workdir)
        assert result.passed is True

    def test_content_equals_fail(self, workdir):
        with open(os.path.join(workdir, "exact.txt"), "w") as f:
            f.write("not this")
        check = DeterministicCheck(CheckType.CONTENT_EQUALS, "exact.txt", "exactly this")
        result = run_check(check, workdir)
        assert result.passed is False

    def test_json_valid_pass(self, workdir):
        with open(os.path.join(workdir, "data.json"), "w") as f:
            json.dump({"key": "value"}, f)
        check = DeterministicCheck(CheckType.JSON_VALID, "data.json")
        result = run_check(check, workdir)
        assert result.passed is True

    def test_json_valid_fail(self, workdir):
        with open(os.path.join(workdir, "bad.json"), "w") as f:
            f.write("{broken")
        check = DeterministicCheck(CheckType.JSON_VALID, "bad.json")
        result = run_check(check, workdir)
        assert result.passed is False

    def test_json_key_exists_pass(self, workdir):
        with open(os.path.join(workdir, "meta.json"), "w") as f:
            json.dump({"name": "hermes", "meta": {"version": "1.0"}}, f)
        check = DeterministicCheck(CheckType.JSON_KEY_EXISTS, "meta.json", "meta.version")
        result = run_check(check, workdir)
        assert result.passed is True

    def test_json_key_exists_fail(self, workdir):
        with open(os.path.join(workdir, "meta.json"), "w") as f:
            json.dump({"name": "hermes"}, f)
        check = DeterministicCheck(CheckType.JSON_KEY_EXISTS, "meta.json", "missing_key")
        result = run_check(check, workdir)
        assert result.passed is False

    def test_regex_match_pass(self, workdir):
        with open(os.path.join(workdir, "log.txt"), "w") as f:
            f.write("error: something failed at line 42")
        check = DeterministicCheck(CheckType.REGEX_MATCH, "log.txt", r"line \d+")
        result = run_check(check, workdir)
        assert result.passed is True

    def test_regex_match_fail(self, workdir):
        with open(os.path.join(workdir, "log.txt"), "w") as f:
            f.write("all good")
        check = DeterministicCheck(CheckType.REGEX_MATCH, "log.txt", r"error")
        result = run_check(check, workdir)
        assert result.passed is False

    def test_exit_code_pass(self, workdir):
        with open(os.path.join(workdir, "exit.txt"), "w") as f:
            f.write("0")
        check = DeterministicCheck(CheckType.EXIT_CODE, "exit.txt", 0)
        result = run_check(check, workdir)
        assert result.passed is True

    def test_exit_code_fail(self, workdir):
        with open(os.path.join(workdir, "exit.txt"), "w") as f:
            f.write("1")
        check = DeterministicCheck(CheckType.EXIT_CODE, "exit.txt", 0)
        result = run_check(check, workdir)
        assert result.passed is False

    def test_check_on_missing_file(self, workdir):
        check = DeterministicCheck(CheckType.CONTENT_CONTAINS, "nope.txt", "x")
        result = run_check(check, workdir)
        assert result.passed is False
        assert "not found" in result.message


class TestScoreChecks:
    """Tests for the aggregate scoring function."""

    def test_all_pass(self):
        checks = [
            DeterministicCheck(CheckType.FILE_EXISTS, "a"),
            DeterministicCheck(CheckType.FILE_EXISTS, "b"),
        ]
        results = [
            CheckResult(check=checks[0], passed=True),
            CheckResult(check=checks[1], passed=True),
        ]
        assert score_checks(results) == 1.0

    def test_all_fail(self):
        checks = [DeterministicCheck(CheckType.FILE_EXISTS, "a")]
        results = [CheckResult(check=checks[0], passed=False)]
        assert score_checks(results) == 0.0

    def test_partial(self):
        c1 = DeterministicCheck(CheckType.FILE_EXISTS, "a", weight=1.0)
        c2 = DeterministicCheck(CheckType.FILE_EXISTS, "b", weight=1.0)
        results = [
            CheckResult(check=c1, passed=True),
            CheckResult(check=c2, passed=False),
        ]
        assert score_checks(results) == 0.5

    def test_weighted(self):
        c1 = DeterministicCheck(CheckType.FILE_EXISTS, "a", weight=3.0)
        c2 = DeterministicCheck(CheckType.FILE_EXISTS, "b", weight=1.0)
        results = [
            CheckResult(check=c1, passed=True),
            CheckResult(check=c2, passed=False),
        ]
        assert score_checks(results) == 0.75

    def test_empty(self):
        assert score_checks([]) == 0.0
