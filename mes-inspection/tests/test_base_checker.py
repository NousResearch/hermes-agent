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


class MockNodeChecker(BaseChecker):
    """支持 check_node 的模拟巡检器。"""

    @property
    def component_name(self):
        return "mock_node"

    def check(self):
        return self.check_all_targets()

    def check_node(self, target):
        name = target.get("name", "unknown")
        checks = [
            self.make_check("连通性", ExitCode.NORMAL, value=True, message=f"{name} 正常"),
        ]
        if name == "bad-node":
            checks = [self.make_check("连通性", ExitCode.CRITICAL, value=False, message=f"{name} 异常")]
        return InspectionReport(
            component=self.component_name,
            timestamp=self._now_iso(),
            status=self.overall_status(checks),
            checks=checks,
            summary=f"{name} 巡检完成",
            metadata={"node_name": name},
        )


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

    def test_now_iso(self):
        checker = MockChecker({})
        iso = checker._now_iso()
        assert "T" in iso


class TestGetTargets:
    def test_no_targets_returns_single_element(self):
        checker = MockChecker({"host": "10.0.0.1", "name": "node1"})
        targets = checker.get_targets()
        assert len(targets) == 1
        assert targets[0]["host"] == "10.0.0.1"

    def test_targets_list_returns_list(self):
        config = {
            "targets": [
                {"name": "n1", "host": "10.0.0.1"},
                {"name": "n2", "host": "10.0.0.2"},
            ]
        }
        checker = MockChecker(config)
        targets = checker.get_targets()
        assert len(targets) == 2

    def test_empty_targets_returns_single_element(self):
        checker = MockChecker({"targets": []})
        targets = checker.get_targets()
        assert len(targets) == 1


class TestCheckNode:
    def test_not_implemented_raises(self):
        checker = MockChecker({})
        with pytest.raises(NotImplementedError, match="未实现 check_node"):
            checker.check_node({"name": "test"})


class TestCheckAllTargets:
    def test_single_target(self):
        config = {"targets": [{"name": "node1", "host": "10.0.0.1"}]}
        checker = MockNodeChecker(config)
        report = checker.check_all_targets()
        assert report.status == ExitCode.NORMAL
        assert len(report.checks) == 1
        assert report.checks[0].details["node_name"] == "node1"

    def test_multi_target_all_normal(self):
        config = {
            "targets": [
                {"name": "node1", "host": "10.0.0.1"},
                {"name": "node2", "host": "10.0.0.2"},
            ]
        }
        checker = MockNodeChecker(config)
        report = checker.check_all_targets()
        assert report.status == ExitCode.NORMAL
        assert len(report.checks) == 2
        assert report.metadata["node_count"] == 2
        assert report.metadata["nodes"]["node1"] == "NORMAL"
        assert report.metadata["nodes"]["node2"] == "NORMAL"

    def test_multi_target_one_abnormal(self):
        config = {
            "targets": [
                {"name": "good-node", "host": "10.0.0.1"},
                {"name": "bad-node", "host": "10.0.0.2"},
            ]
        }
        checker = MockNodeChecker(config)
        report = checker.check_all_targets()
        assert report.status == ExitCode.CRITICAL
        assert report.metadata["nodes"]["good-node"] == "NORMAL"
        assert report.metadata["nodes"]["bad-node"] == "CRITICAL"
        assert "严重" in report.summary

    def test_multi_target_node_exception(self):
        """check_node 抛异常时，该节点标记 CRITICAL，不影响其他节点。"""
        config = {
            "targets": [
                {"name": "good-node", "host": "10.0.0.1"},
                {"name": "error-node", "host": "10.0.0.2"},
            ]
        }
        checker = MockNodeChecker(config)

        original_check_node = checker.check_node
        def patched_check_node(target):
            if target["name"] == "error-node":
                raise ConnectionError("SSH 连接超时")
            return original_check_node(target)

        checker.check_node = patched_check_node
        report = checker.check_all_targets()
        assert report.status == ExitCode.CRITICAL
        assert len(report.checks) == 2
        error_check = [c for c in report.checks if "异常" in c.message][0]
        assert "SSH 连接超时" in error_check.message

    def test_check_delegates_to_check_all_targets(self):
        """check() 默认调用 check_all_targets()。"""
        config = {"targets": [{"name": "node1", "host": "10.0.0.1"}]}
        checker = MockNodeChecker(config)
        report = checker.check()
        assert report.status == ExitCode.NORMAL
        assert report.checks[0].details["node_name"] == "node1"


class TestMergeNodeReports:
    def test_merge(self):
        checker = MockNodeChecker({})
        r1 = InspectionReport(
            component="mock", timestamp="2026-01-01", status=ExitCode.NORMAL,
            checks=[CheckResult(name="c1", status=ExitCode.NORMAL)],
            metadata={"node_name": "n1"},
        )
        r2 = InspectionReport(
            component="mock", timestamp="2026-01-01", status=ExitCode.WARNING,
            checks=[CheckResult(name="c2", status=ExitCode.WARNING)],
            metadata={"node_name": "n2"},
        )
        merged = checker._merge_node_reports([r1, r2])
        assert merged.status == ExitCode.WARNING
        assert len(merged.checks) == 2
        assert merged.metadata["node_count"] == 2
        assert merged.metadata["nodes"]["n1"] == "NORMAL"
        assert merged.metadata["nodes"]["n2"] == "WARNING"
