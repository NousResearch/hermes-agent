"""巡检基类单元测试。"""

import json
import sys
import pytest

from scripts.base_checker import (
    BaseChecker, ExitCode, CheckResult, InspectionReport
)


class MockChecker(BaseChecker):
    """用于测试的模拟巡检器。"""

    @property
    def component_name(self):
        return "mock"

    def check(self):
        checks = [
            self.make_check("check1", ExitCode.NORMAL, value=10, threshold=100, message="正常"),
            self.make_check("check2", ExitCode.WARNING, value=85, threshold=80, message="告警"),
        ]
        return InspectionReport(
            component=self.component_name,
            timestamp="2026-06-04T00:00:00",
            status=self.overall_status(checks),
            checks=checks,
            summary="测试报告",
        )


class TestExitCode:
    def test_values(self):
        assert ExitCode.NORMAL == 0
        assert ExitCode.WARNING == 1
        assert ExitCode.CRITICAL == 2

    def test_comparison(self):
        assert ExitCode.CRITICAL > ExitCode.WARNING > ExitCode.NORMAL


class TestCheckResult:
    def test_creation(self):
        r = CheckResult(name="test", status=ExitCode.NORMAL, value=42, threshold=100, message="ok")
        assert r.name == "test"
        assert r.status == ExitCode.NORMAL
        assert r.value == 42


class TestInspectionReport:
    def test_to_dict(self):
        checks = [CheckResult(name="c1", status=ExitCode.NORMAL)]
        report = InspectionReport(component="test", timestamp="2026-01-01", status=ExitCode.NORMAL, checks=checks)
        d = report.to_dict()
        assert d["component"] == "test"
        assert d["status"] == "NORMAL"
        assert d["status_code"] == 0
        assert len(d["checks"]) == 1

    def test_to_json(self):
        checks = [CheckResult(name="c1", status=ExitCode.WARNING)]
        report = InspectionReport(component="test", timestamp="2026-01-01", status=ExitCode.WARNING, checks=checks)
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["status"] == "WARNING"


class TestBaseChecker:
    def test_overall_status(self):
        checker = MockChecker({})
        checks = [
            CheckResult(name="a", status=ExitCode.NORMAL),
            CheckResult(name="b", status=ExitCode.WARNING),
        ]
        assert checker.overall_status(checks) == ExitCode.WARNING

    def test_overall_status_all_normal(self):
        checker = MockChecker({})
        checks = [CheckResult(name="a", status=ExitCode.NORMAL)]
        assert checker.overall_status(checks) == ExitCode.NORMAL

    def test_overall_status_empty(self):
        checker = MockChecker({})
        assert checker.overall_status([]) == ExitCode.NORMAL

    def test_make_check(self):
        checker = MockChecker({})
        c = checker.make_check("test", ExitCode.CRITICAL, value=95, threshold=90, message="严重")
        assert c.name == "test"
        assert c.status == ExitCode.CRITICAL
