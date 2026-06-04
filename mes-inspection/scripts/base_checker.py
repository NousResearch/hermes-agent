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
