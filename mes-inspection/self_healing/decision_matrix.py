"""故障自愈决策矩阵 - 五维评分模型。"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class HealLevel(str, Enum):
    L1 = "L1"  # 全自动
    L2 = "L2"  # 半自动
    L3 = "L3"  # 人工处理


@dataclass
class FaultScenario:
    """故障场景描述。"""
    component: str
    fault_type: str
    description: str
    data_risk: float        # 0-100: 无数据风险=0, 可能影响日志=30, 可能影响业务数据=100
    reversibility: float    # 0-100: 完全可逆=0, 需要回滚=50, 不可逆=100
    impact_scope: float     # 0-100: 单组件=0, 影响关联组件=50, 影响整个系统=100
    historical_success: float  # 0-1: 历史成功率
    config_change: float    # 0-100: 不需要=0, 需要但可回滚=50, 需要且不可回滚=100


@dataclass
class ScoreResult:
    """评分结果。"""
    total_score: float
    level: HealLevel
    breakdown: Dict[str, float]
    recommended_actions: List[str]
    safety_notes: List[str]


# 预定义故障场景评分
KNOWN_SCENARIOS = {
    # Nginx
    ("nginx", "process_down"): FaultScenario(
        component="nginx", fault_type="process_down",
        description="Nginx 进程挂掉",
        data_risk=0, reversibility=0, impact_scope=30,
        historical_success=0.98, config_change=0,
    ),
    ("nginx", "5xx_critical"): FaultScenario(
        component="nginx", fault_type="5xx_critical",
        description="Nginx 5xx 错误率 > 5%",
        data_risk=10, reversibility=0, impact_scope=50,
        historical_success=0.85, config_change=10,
    ),
    # JVM
    ("jvm", "oom"): FaultScenario(
        component="jvm", fault_type="oom",
        description="Tomcat OOM",
        data_risk=20, reversibility=0, impact_scope=50,
        historical_success=0.90, config_change=0,
    ),
    ("jvm", "deadlock"): FaultScenario(
        component="jvm", fault_type="deadlock",
        description="线程死锁",
        data_risk=30, reversibility=30, impact_scope=60,
        historical_success=0.70, config_change=0,
    ),
    ("jvm", "full_gc"): FaultScenario(
        component="jvm", fault_type="full_gc",
        description="Full GC 频繁",
        data_risk=0, reversibility=0, impact_scope=20,
        historical_success=0.95, config_change=0,
    ),
    # RabbitMQ
    ("rabbitmq", "queue_backlog"): FaultScenario(
        component="rabbitmq", fault_type="queue_backlog",
        description="队列堆积 > 10K",
        data_risk=0, reversibility=0, impact_scope=30,
        historical_success=0.85, config_change=0,
    ),
    ("rabbitmq", "no_consumers"): FaultScenario(
        component="rabbitmq", fault_type="no_consumers",
        description="消费者数为 0",
        data_risk=10, reversibility=0, impact_scope=50,
        historical_success=0.80, config_change=0,
    ),
    # Oracle
    ("oracle", "tablespace_full"): FaultScenario(
        component="oracle", fault_type="tablespace_full",
        description="表空间 > 90%",
        data_risk=80, reversibility=50, impact_scope=70,
        historical_success=0.60, config_change=50,
    ),
    ("oracle", "lock_wait"): FaultScenario(
        component="oracle", fault_type="lock_wait",
        description="锁等待 > 10",
        data_risk=50, reversibility=30, impact_scope=60,
        historical_success=0.65, config_change=0,
    ),
    ("oracle", "slow_sql"): FaultScenario(
        component="oracle", fault_type="slow_sql",
        description="慢 SQL",
        data_risk=40, reversibility=20, impact_scope=40,
        historical_success=0.50, config_change=30,
    ),
    # ELK
    ("elk", "cluster_red"): FaultScenario(
        component="elk", fault_type="cluster_red",
        description="ELK 集群 red",
        data_risk=30, reversibility=20, impact_scope=60,
        historical_success=0.70, config_change=10,
    ),
    ("elk", "disk_full"): FaultScenario(
        component="elk", fault_type="disk_full",
        description="ELK 磁盘 > 90%",
        data_risk=20, reversibility=0, impact_scope=40,
        historical_success=0.90, config_change=0,
    ),
    # SkyWalking
    ("skywalking", "sla_drop"): FaultScenario(
        component="skywalking", fault_type="sla_drop",
        description="SLA < 95%",
        data_risk=10, reversibility=0, impact_scope=50,
        historical_success=0.40, config_change=0,
    ),
}


class DecisionMatrix:
    """自愈决策矩阵。"""

    def __init__(self, config: Dict[str, Any]):
        self.weights = {
            "data_risk": config.get("data_risk_weight", 0.30),
            "reversibility": config.get("reversibility_weight", 0.25),
            "impact_scope": config.get("impact_scope_weight", 0.25),
            "success_rate": config.get("success_rate_weight", 0.15),
            "config_change": config.get("config_change_weight", 0.05),
        }
        self.l1_threshold = config.get("l1_threshold", 30)
        self.l2_threshold = config.get("l2_threshold", 70)
        self.default_first_level = config.get("default_first_occurrence_level", "L3")
        self.data_involved_level = config.get("data_involved_level", "L3")

    def score_scenario(self, scenario: FaultScenario) -> ScoreResult:
        """对故障场景评分，返回自愈级别。"""
        # 首次出现 → 默认 L3
        # 涉及数据 → 永远 L3
        if scenario.data_risk >= 80:
            return ScoreResult(
                total_score=100,
                level=HealLevel.L3,
                breakdown={"data_risk": scenario.data_risk},
                recommended_actions=["通知 DBA", "记录详细诊断报告"],
                safety_notes=["涉及业务数据，必须人工处理"],
            )

        breakdown = {
            "data_risk": scenario.data_risk * self.weights["data_risk"],
            "reversibility": scenario.reversibility * self.weights["reversibility"],
            "impact_scope": scenario.impact_scope * self.weights["impact_scope"],
            "success_rate": (1 - scenario.historical_success) * 100 * self.weights["success_rate"],
            "config_change": scenario.config_change * self.weights["config_change"],
        }
        total = sum(breakdown.values())

        if total < self.l1_threshold:
            level = HealLevel.L1
        elif total < self.l2_threshold:
            level = HealLevel.L2
        else:
            level = HealLevel.L3

        return ScoreResult(
            total_score=round(total, 2),
            level=level,
            breakdown=breakdown,
            recommended_actions=self._get_actions(scenario, level),
            safety_notes=self._get_safety_notes(scenario, level),
        )

    def score_from_report(self, report: Dict[str, Any]) -> Dict[str, ScoreResult]:
        """从巡检报告中识别故障场景并评分。"""
        component = report.get("component", "")
        checks = report.get("checks", [])
        results = {}

        for check in checks:
            if check.get("status_code", 0) == 0:
                continue
            fault_type = self._identify_fault_type(component, check)
            key = (component, fault_type)
            if key in KNOWN_SCENARIOS:
                results[fault_type] = self.score_scenario(KNOWN_SCENARIOS[key])
            else:
                # 未知场景，默认 L3
                results[fault_type] = ScoreResult(
                    total_score=70,
                    level=HealLevel(self.default_first_level),
                    breakdown={},
                    recommended_actions=["收集更多信息", "通知运维人员"],
                    safety_notes=["首次出现的故障类型，建议人工确认"],
                )

        return results

    def _identify_fault_type(self, component: str, check: Dict[str, Any]) -> str:
        """从检查项识别故障类型。"""
        name = check.get("name", "").lower()
        mapping = {
            "nginx": {"进程存活": "process_down", "5xx错误率": "5xx_critical"},
            "jvm": {"进程存活": "process_down", "堆内存": "oom", "死锁": "deadlock", "gc频率": "full_gc"},
            "rabbitmq": {"服务存活": "process_down", "队列深度": "queue_backlog", "消费者": "no_consumers"},
            "oracle": {"表空间": "tablespace_full", "锁等待": "lock_wait", "慢sql": "slow_sql"},
            "elk": {"集群状态": "cluster_red", "磁盘": "disk_full"},
            "skywalking": {"sla": "sla_drop", "响应时间": "response_slow"},
        }
        component_map = mapping.get(component, {})
        for keyword, fault_type in component_map.items():
            if keyword in name:
                return fault_type
        return "unknown"

    def _get_actions(self, scenario: FaultScenario, level: HealLevel) -> List[str]:
        """获取推荐修复动作。"""
        actions = {
            ("nginx", "process_down"): ["systemctl restart nginx"],
            ("nginx", "5xx_critical"): ["nginx -t 检查配置", "nginx -s reload"],
            ("jvm", "oom"): ["jmap -dump:format=b,file=/tmp/heap.hprof <PID>", "systemctl restart tomcat"],
            ("jvm", "deadlock"): ["jstack <PID> > /tmp/thread_dump.txt", "通知开发团队"],
            ("jvm", "full_gc"): ["jcmd <PID> GC.run"],
            ("rabbitmq", "queue_backlog"): ["检查消费者状态", "考虑扩容消费者"],
            ("rabbitmq", "no_consumers"): ["检查消费者进程", "重启消费者服务"],
            ("oracle", "tablespace_full"): ["通知 DBA", "ALTER TABLESPACE ... ADD DATAFILE"],
            ("oracle", "lock_wait"): ["列出锁持有者", "评估是否 kill 会话"],
            ("oracle", "slow_sql"): ["收集执行计划", "通知 DBA 优化"],
            ("elk", "cluster_red"): ["检查未分配分片", "尝试 reroute allocate"],
            ("elk", "disk_full"): ["清理 7 天前索引: curl -X DELETE localhost:9200/mes-logs-*"],
            ("skywalking", "sla_drop"): ["生成链路分析报告", "定位慢接口"],
        }
        key = (scenario.component, scenario.fault_type)
        return actions.get(key, ["收集更多信息"])

    def _get_safety_notes(self, scenario: FaultScenario, level: HealLevel) -> List[str]:
        """获取安全注意事项。"""
        notes = []
        if scenario.data_risk > 0:
            notes.append(f"数据风险: {scenario.data_risk}/100")
        if scenario.reversibility > 50:
            notes.append("操作不可逆，请谨慎执行")
        if level == HealLevel.L1:
            notes.append("L1 全自动，已验证的安全操作")
        elif level == HealLevel.L2:
            notes.append("L2 半自动，需人工确认后执行")
        else:
            notes.append("L3 人工处理，需运维人员介入")
        return notes
