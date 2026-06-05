"""自愈引擎模块测试。"""

import sys
import pytest

sys.path.insert(0, "C:/project/hermes-agent/mes-inspection")

from self_healing.decision_matrix import DecisionMatrix, HealLevel, FaultScenario, KNOWN_SCENARIOS
from self_healing.self_healer import SelfHealer
from self_healing.code_analyzer import CodeAnalyzer


@pytest.fixture
def config():
    return {
        "data_risk_weight": 0.30,
        "reversibility_weight": 0.25,
        "impact_scope_weight": 0.25,
        "success_rate_weight": 0.15,
        "config_change_weight": 0.05,
        "l1_threshold": 30,
        "l2_threshold": 70,
        "enabled": True,
    }


@pytest.fixture
def matrix(config):
    return DecisionMatrix(config)


@pytest.fixture
def healer(config):
    return SelfHealer(config)


@pytest.fixture
def analyzer():
    return CodeAnalyzer()


class TestDecisionMatrix:
    def test_nginx_process_down_is_l1(self, matrix):
        scenario = KNOWN_SCENARIOS[("nginx", "process_down")]
        result = matrix.score_scenario(scenario)
        assert result.level == HealLevel.L1
        assert result.total_score < 30

    def test_oracle_tablespace_is_l3(self, matrix):
        scenario = KNOWN_SCENARIOS[("oracle", "tablespace_full")]
        result = matrix.score_scenario(scenario)
        assert result.level == HealLevel.L3
        assert result.total_score == 100

    def test_jvm_oom_is_l1(self, matrix):
        scenario = KNOWN_SCENARIOS[("jvm", "oom")]
        result = matrix.score_scenario(scenario)
        assert result.level == HealLevel.L1

    def test_jvm_deadlock_is_l2(self, matrix):
        scenario = KNOWN_SCENARIOS[("jvm", "deadlock")]
        result = matrix.score_scenario(scenario)
        assert result.level == HealLevel.L2

    def test_score_from_report(self, matrix):
        report = {
            "component": "nginx",
            "checks": [
                {"name": "进程存活", "status_code": 2, "status": "CRITICAL"},
                {"name": "5xx错误率", "status_code": 1, "status": "WARNING"},
            ],
        }
        results = matrix.score_from_report(report)
        assert "process_down" in results
        assert "5xx_critical" in results

    def test_score_from_report_normal_check_skipped(self, matrix):
        report = {
            "component": "nginx",
            "checks": [
                {"name": "进程存活", "status_code": 0, "status": "NORMAL"},
            ],
        }
        results = matrix.score_from_report(report)
        assert len(results) == 0

    def test_unknown_scenario_defaults_l3(self, matrix):
        report = {
            "component": "unknown_component",
            "checks": [
                {"name": "unknown_check", "status_code": 2, "status": "CRITICAL"},
            ],
        }
        results = matrix.score_from_report(report)
        assert len(results) == 1
        result = list(results.values())[0]
        assert result.level == HealLevel.L3

    def test_custom_thresholds(self):
        config = {
            "l1_threshold": 50,
            "l2_threshold": 80,
        }
        matrix = DecisionMatrix(config)
        assert matrix.l1_threshold == 50
        assert matrix.l2_threshold == 80

    def test_data_risk_overrides_to_l3(self, matrix):
        scenario = FaultScenario(
            component="test", fault_type="data_corruption",
            description="数据损坏",
            data_risk=90, reversibility=0, impact_scope=0,
            historical_success=0.99, config_change=0,
        )
        result = matrix.score_scenario(scenario)
        assert result.level == HealLevel.L3

    def test_actions_populated(self, matrix):
        scenario = KNOWN_SCENARIOS[("nginx", "process_down")]
        result = matrix.score_scenario(scenario)
        assert len(result.recommended_actions) > 0

    def test_safety_notes_populated(self, matrix):
        scenario = KNOWN_SCENARIOS[("nginx", "process_down")]
        result = matrix.score_scenario(scenario)
        assert len(result.safety_notes) > 0


class TestSelfHealer:
    def test_disabled_healer(self, config):
        config["enabled"] = False
        healer = SelfHealer(config)
        result = healer.process_report({"component": "nginx", "status_code": 2})
        assert result["healed"] is False
        assert result["reason"] == "自愈功能已禁用"

    def test_normal_report_no_heal(self, healer):
        report = {"component": "nginx", "status_code": 0}
        result = healer.process_report(report)
        assert result["healed"] is False
        assert result["reason"] == "无异常"

    def test_l1_heal_executes(self, healer):
        report = {
            "component": "nginx",
            "status_code": 2,
            "checks": [
                {"name": "进程存活", "status_code": 2, "status": "CRITICAL"},
            ],
        }
        result = healer.process_report(report)
        assert "faults" in result
        assert len(result["faults"]) > 0

    def test_l2_heal_not_executed(self, healer):
        report = {
            "component": "jvm",
            "status_code": 2,
            "checks": [
                {"name": "死锁", "status_code": 2, "status": "CRITICAL"},
            ],
        }
        result = healer.process_report(report)
        for fault in result.get("faults", []):
            if fault["level"] == "L2":
                assert fault["executed"] is False

    def test_format_heal_report(self, healer):
        heal_result = {
            "healed": True,
            "faults": [
                {
                    "component": "nginx",
                    "fault_type": "process_down",
                    "level": "L1",
                    "score": 7.8,
                    "actions": ["systemctl restart nginx"],
                    "executed": True,
                    "output": "restarted",
                    "success": True,
                },
            ],
        }
        report = healer.format_heal_report(heal_result)
        assert "L1" in report
        assert "nginx" in report

    def test_format_empty_report(self, healer):
        report = healer.format_heal_report({"healed": False, "faults": []})
        assert report == ""


class TestCodeAnalyzer:
    def test_analyze_null_pointer_exception(self, analyzer):
        log = "java.lang.NullPointerException: Cannot invoke method\n    at com.example.Service.method(Service.java:42)"
        result = analyzer.analyze_log(log)
        assert result.exception_type == "java.lang.NullPointerException"
        assert len(result.stack_frames) == 1
        assert result.stack_frames[0].line == 42

    def test_analyze_unknown_exception(self, analyzer):
        log = "Something went wrong"
        result = analyzer.analyze_log(log)
        assert result.exception_type == "Unknown"

    def test_skip_framework_frames(self, analyzer):
        log = (
            "java.lang.NullPointerException: test\n"
            "    at com.example.Service.method(Service.java:10)\n"
            "    at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:100)\n"
            "    at java.base/java.lang.Thread.run(Thread.java:829)"
        )
        result = analyzer.analyze_log(log)
        assert len(result.stack_frames) == 1
        assert "example" in result.stack_frames[0].file

    def test_max_10_frames(self, analyzer):
        frames = "\n".join(
            [f"    at com.example.Class{i}.method(Class{i}.java:{i})" for i in range(20)]
        )
        log = f"java.lang.Exception: test\n{frames}"
        result = analyzer.analyze_log(log)
        assert len(result.stack_frames) <= 10

    def test_npe_diff_suggestion(self, analyzer):
        log = "java.lang.NullPointerException: test\n    at com.example.Service.method(Service.java:42)"
        result = analyzer.analyze_log(log)
        assert "if (obj == null)" in result.diff_suggestion

    def test_format_analysis(self, analyzer):
        log = "java.lang.NullPointerException: test\n    at com.example.Service.method(Service.java:42)"
        analysis = analyzer.analyze_log(log)
        formatted = analyzer.format_analysis(analysis)
        assert "NullPointerException" in formatted
        assert "Service.java" in formatted
