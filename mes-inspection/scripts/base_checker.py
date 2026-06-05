"""MES 巡检基类 - 所有巡检脚本的基础。"""

import json
import sys
import time
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional


class ExitCode(IntEnum):
    """巡检退出码。"""
    NORMAL = 0      # 一切正常
    WARNING = 1     # 告警
    CRITICAL = 2    # 严重


@dataclass
class CheckResult:
    """单个检查项的结果。"""
    name: str
    status: ExitCode
    value: Any = None
    threshold: Any = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InspectionReport:
    """巡检报告。"""
    component: str
    timestamp: str
    status: ExitCode
    checks: List[CheckResult]
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "timestamp": self.timestamp,
            "status": self.status.name,
            "status_code": int(self.status),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.name,
                    "status_code": int(c.status),
                    "value": c.value,
                    "threshold": c.threshold,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "summary": self.summary,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class BaseChecker(ABC):
    """巡检基类。

    子类需要实现：
    - component_name: 组件名称
    - check(): 执行巡检逻辑，返回 InspectionReport

    多节点支持：
    - get_targets(): 从配置获取目标节点列表
    - check_node(target): 对单个节点执行巡检（可选实现）
    - check_all_targets(): 遍历所有 target 调用 check_node() 并聚合
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @property
    @abstractmethod
    def component_name(self) -> str:
        """组件名称，如 'nginx', 'jvm'。"""
        ...

    @abstractmethod
    def check(self) -> InspectionReport:
        """执行巡检，返回报告。"""
        ...

    def check_node(self, target: Dict[str, Any]) -> InspectionReport:
        """对单个节点执行巡检。子类可覆盖此方法支持多节点。

        Args:
            target: 节点配置字典，包含 host, name 等字段。

        Returns:
            该节点的巡检报告。

        Raises:
            NotImplementedError: 子类未实现时抛出。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 check_node()，无法执行多节点巡检"
        )

    def get_targets(self) -> List[Dict[str, Any]]:
        """获取目标节点列表。

        如果配置中有 targets 字段，返回该列表。
        否则返回单元素列表（用顶层 config 构造），保证单节点兼容。

        Returns:
            目标节点配置列表。
        """
        targets = self.config.get("targets")
        if targets and isinstance(targets, list):
            return targets
        return [{"name": self.config.get("name", "default"), **self.config}]

    def check_all_targets(self) -> InspectionReport:
        """遍历所有 target 调用 check_node() 并聚合结果。

        当只有一个 target 时直接返回 check_node() 的结果。
        当有多个 target 时合并所有检查项，按节点标记 node_name。

        Returns:
            聚合后的巡检报告。
        """
        targets = self.get_targets()

        if len(targets) == 1:
            report = self.check_node(targets[0])
            node_name = targets[0].get("name", "default")
            for c in report.checks:
                c.details.setdefault("node_name", node_name)
            report.metadata.setdefault("node_name", node_name)
            return report

        # 多节点：逐个检查，聚合
        reports = []
        for target in targets:
            node_name = target.get("name", target.get("host", "unknown"))
            try:
                report = self.check_node(target)
                for c in report.checks:
                    c.details.setdefault("node_name", node_name)
                report.metadata.setdefault("node_name", node_name)
                reports.append(report)
            except Exception as e:
                # 单节点异常不影响其他节点
                err_report = InspectionReport(
                    component=self.component_name,
                    timestamp=self._now_iso(),
                    status=ExitCode.CRITICAL,
                    checks=[CheckResult(
                        name="节点连通性", status=ExitCode.CRITICAL,
                        value=None, message=f"节点 {node_name} 巡检异常: {e}",
                        details={"node_name": node_name, "error": str(e)},
                    )],
                    summary=f"节点 {node_name} 巡检异常: {e}",
                    metadata={"node_name": node_name, "error": str(e)},
                )
                reports.append(err_report)

        return self._merge_node_reports(reports)

    def _merge_node_reports(self, reports: List[InspectionReport]) -> InspectionReport:
        """合并多个节点的报告为一个聚合报告。"""
        all_checks = []
        node_statuses = {}
        for r in reports:
            node_name = r.metadata.get("node_name", "unknown")
            node_statuses[node_name] = r.status.name
            all_checks.extend(r.checks)

        status = self.overall_status(all_checks)
        total = len(reports)
        normal = sum(1 for r in reports if r.status == ExitCode.NORMAL)
        warning = sum(1 for r in reports if r.status == ExitCode.WARNING)
        critical = sum(1 for r in reports if r.status == ExitCode.CRITICAL)

        summary_parts = [f"多节点巡检完成，{total} 个节点"]
        if critical:
            summary_parts.append(f"{critical} 个严重")
        if warning:
            summary_parts.append(f"{warning} 个告警")
        if not critical and not warning:
            summary_parts.append("全部正常")
        summary = "，".join(summary_parts)

        return InspectionReport(
            component=self.component_name,
            timestamp=self._now_iso(),
            status=status,
            checks=all_checks,
            summary=summary,
            metadata={
                "node_count": total,
                "nodes": node_statuses,
            },
        )

    def run(self) -> int:
        """运行巡检并输出 JSON 到 stdout，返回退出码。"""
        start = time.time()
        try:
            report = self.check()
        except Exception as e:
            report = InspectionReport(
                component=self.component_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ExitCode.CRITICAL,
                checks=[],
                summary=f"巡检执行异常: {e}",
                metadata={"error": str(e), "error_type": type(e).__name__},
            )
        elapsed = time.time() - start
        report.metadata["elapsed_seconds"] = round(elapsed, 2)
        print(report.to_json())
        return int(report.status)

    def run_command(self, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
        """安全执行 shell 命令。"""
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )

    def make_check(
        self,
        name: str,
        status: ExitCode,
        value: Any = None,
        threshold: Any = None,
        message: str = "",
        **details,
    ) -> CheckResult:
        """创建检查结果的便捷方法。"""
        return CheckResult(
            name=name,
            status=status,
            value=value,
            threshold=threshold,
            message=message,
            details=details if details else {},
        )

    def overall_status(self, checks: List[CheckResult]) -> ExitCode:
        """根据所有检查项计算整体状态（取最严重的）。"""
        if not checks:
            return ExitCode.NORMAL
        return ExitCode(max(int(c.status) for c in checks))

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
