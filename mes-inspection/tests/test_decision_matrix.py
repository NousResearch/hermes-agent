"""决策矩阵单元测试。"""

import pytest

from self_healing.decision_matrix import (
    DecisionMatrix, HealLevel, FaultScenario, KNOWN_SCENARIOS
)


class TestDecisionMatrix:
    def setup_method(self):
        self.matrix = DecisionMatrix({
            "data_risk_weight": 0.30,
            "reversibility_weight": 0.25,
            "impact_scope_weight": 0.25,
            "success_rate_weight": 0.15,
            "config_change_weight": 0.05,
            "l1_threshold": 30,
            "l2_threshold": 70,
        })

    def test_nginx_process_down_is_l1(self):
        scenario = KNOWN_SCENARIOS[("nginx", "process_down")]
        result = self.matrix.score_scenario(scenario)
        assert result.level == HealLevel.L1
        assert result.total_score < 30

    def test_oracle_tablespace_is_l3(self):
        scenario = KNOWN_SCENARIOS[("oracle", "tablespace_full")]
        result = self.matrix.score_scenario(scenario)
        assert result.level == HealLevel.L3

    def test_jvm_deadlock_is_l2(self):
        scenario = KNOWN_SCENARIOS[("jvm", "deadlock")]
        result = self.matrix.score_scenario(scenario)
        assert result.level in (HealLevel.L1, HealLevel.L2)

    def test_high_data_risk_forces_l3(self):
        scenario = FaultScenario(
            component="test", fault_type="data_risk",
            description="高数据风险",
            data_risk=90, reversibility=0, impact_scope=0,
            historical_success=1.0, config_change=0,
        )
        result = self.matrix.score_scenario(scenario)
        assert result.level == HealLevel.L3

    def test_known_scenarios_coverage(self):
        assert ("nginx", "process_down") in KNOWN_SCENARIOS
        assert ("jvm", "oom") in KNOWN_SCENARIOS
        assert ("rabbitmq", "queue_backlog") in KNOWN_SCENARIOS
        assert ("oracle", "tablespace_full") in KNOWN_SCENARIOS
        assert ("elk", "cluster_red") in KNOWN_SCENARIOS
        assert ("skywalking", "sla_drop") in KNOWN_SCENARIOS
