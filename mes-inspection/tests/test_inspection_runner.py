"""巡检执行器单元测试。"""

import json
import pytest
from unittest.mock import patch, MagicMock
from scripts.inspection_runner import (
    run_inspections,
    output_gate_result,
    HEARTBEAT_COMPONENTS,
    FULL_CHECK_COMPONENTS,
    DEEP_ANALYSIS_COMPONENTS,
)
from scripts.base_checker import ExitCode, InspectionReport, CheckResult


def _make_report(component, status):
    return InspectionReport(
        component=component,
        timestamp="2026-06-04T00:00:00",
        status=status,
        checks=[],
        summary=f"{component} {status.name}",
    )


class TestRunInspections:
    @patch("scripts.inspection_runner.create_checker")
    def test_all_normal(self, mock_create):
        mock_checker = MagicMock()
        mock_checker.check.return_value = _make_report("nginx", ExitCode.NORMAL)
        mock_create.return_value = mock_checker
        result = run_inspections(["nginx"])
        assert result["abnormal"] is False
        assert result["summary"]["normal"] == 1
        assert result["reports"] == []

    @patch("scripts.inspection_runner.create_checker")
    def test_warning_triggers_abnormal(self, mock_create):
        mock_checker = MagicMock()
        mock_checker.check.return_value = _make_report("nginx", ExitCode.WARNING)
        mock_create.return_value = mock_checker
        result = run_inspections(["nginx"])
        assert result["abnormal"] is True
        assert result["summary"]["warning"] == 1
        assert len(result["reports"]) == 1

    @patch("scripts.inspection_runner.create_checker")
    def test_critical_triggers_abnormal(self, mock_create):
        mock_checker = MagicMock()
        mock_checker.check.return_value = _make_report("jvm", ExitCode.CRITICAL)
        mock_create.return_value = mock_checker
        result = run_inspections(["jvm"])
        assert result["abnormal"] is True
        assert result["summary"]["critical"] == 1

    @patch("scripts.inspection_runner.create_checker")
    def test_mixed_results(self, mock_create):
        def side_effect(component):
            m = MagicMock()
            if component == "nginx":
                m.check.return_value = _make_report("nginx", ExitCode.NORMAL)
            else:
                m.check.return_value = _make_report("jvm", ExitCode.CRITICAL)
            return m
        mock_create.side_effect = side_effect
        result = run_inspections(["nginx", "jvm"])
        assert result["abnormal"] is True
        assert result["summary"]["normal"] == 1
        assert result["summary"]["critical"] == 1
        assert len(result["reports"]) == 1

    @patch("scripts.inspection_runner.create_checker")
    def test_checker_exception(self, mock_create):
        mock_create.side_effect = RuntimeError("连接失败")
        result = run_inspections(["nginx"])
        assert result["abnormal"] is True
        assert result["summary"]["critical"] == 1


class TestOutputGateResult:
    def test_normal_outputs_wake_false(self, capsys):
        result = {"abnormal": False, "reports": [], "summary": {}}
        output_gate_result(result)
        captured = capsys.readouterr()
        last_line = captured.out.strip().split("\n")[-1]
        gate = json.loads(last_line)
        assert gate["wakeAgent"] is False

    def test_abnormal_outputs_report_and_wake_true(self, capsys):
        result = {
            "abnormal": True,
            "reports": [{"component": "nginx", "status": "CRITICAL"}],
            "summary": {},
        }
        output_gate_result(result)
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        last_line = lines[-1]
        gate = json.loads(last_line)
        assert gate["wakeAgent"] is True
        assert "===INSPECTION_REPORT===" in captured.out


class TestComponentLists:
    def test_heartbeat_has_upstream_only(self):
        assert HEARTBEAT_COMPONENTS == ["upstream"]
        assert "jvm" not in HEARTBEAT_COMPONENTS

    def test_full_check_has_all_6(self):
        assert len(FULL_CHECK_COMPONENTS) == 6

    def test_deep_analysis_has_oracle_elk(self):
        assert "oracle" in DEEP_ANALYSIS_COMPONENTS
        assert "elk" in DEEP_ANALYSIS_COMPONENTS


class TestEndToEnd:
    @patch("scripts.inspection_runner.create_checker")
    def test_heartbeat_normal_gate(self, mock_create):
        mock_checker = MagicMock()
        mock_checker.check.return_value = _make_report("upstream", ExitCode.NORMAL)
        mock_create.return_value = mock_checker
        result = run_inspections(["upstream"])
        assert result["abnormal"] is False

    @patch("scripts.inspection_runner.create_checker")
    def test_heartbeat_abnormal_gate(self, mock_create):
        mock_checker = MagicMock()
        report = _make_report("upstream", ExitCode.CRITICAL)
        report.checks = [CheckResult(name="端点 [app-1]", status=ExitCode.CRITICAL, value=None, message="连接失败")]
        mock_checker.check.return_value = report
        mock_create.return_value = mock_checker
        result = run_inspections(["upstream"])
        assert result["abnormal"] is True
        assert len(result["reports"]) == 1
        assert result["reports"][0]["component"] == "upstream"

    @patch("scripts.inspection_runner.create_checker")
    def test_full_check_mixed(self, mock_create):
        def side_effect(component):
            m = MagicMock()
            status_map = {
                "nginx": ExitCode.NORMAL,
                "jvm": ExitCode.WARNING,
                "rabbitmq": ExitCode.NORMAL,
                "oracle": ExitCode.CRITICAL,
                "elk": ExitCode.NORMAL,
                "skywalking": ExitCode.WARNING,
            }
            m.check.return_value = _make_report(component, status_map[component])
            return m
        mock_create.side_effect = side_effect
        result = run_inspections(FULL_CHECK_COMPONENTS)
        assert result["abnormal"] is True
        assert result["summary"]["normal"] == 3
        assert result["summary"]["warning"] == 2
        assert result["summary"]["critical"] == 1
        assert len(result["reports"]) == 3
