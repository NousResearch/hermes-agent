"""Nginx 巡检脚本 - 检查进程、错误率、连接数、上游可达性、响应时间。"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.base_checker import BaseChecker, CheckResult, ExitCode, InspectionReport
from scripts.ssh_executor import create_executor, SSHExecutor, LocalExecutor


class NginxChecker(BaseChecker):
    """Nginx 巡检器。"""

    @property
    def component_name(self) -> str:
        return "nginx"

    def check(self) -> InspectionReport:
        return self.check_all_targets()

    def check_node(self, target: Dict[str, Any]) -> InspectionReport:
        """对单个节点执行 Nginx 巡检。

        Args:
            target: 节点配置字典，包含 host, name, status_url 等字段。

        Returns:
            该节点的巡检报告。
        """
        executor = create_executor(target)
        checks: List[CheckResult] = []

        checks.append(self._check_process(target, executor))
        checks.append(self._check_5xx_error_rate(target, executor))
        checks.append(self._check_active_connections(target, executor))
        checks.append(self._check_upstream(target, executor))
        checks.append(self._check_response_time(target, executor))

        status = self.overall_status(checks)
        node_name = target.get("name", target.get("host", "default"))
        summary = f"Nginx 巡检完成，节点 {node_name}，共 {len(checks)} 项，状态: {status.name}"
        return InspectionReport(
            component=self.component_name,
            timestamp=self._now_iso(),
            status=status,
            checks=checks,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # 检查项
    # ------------------------------------------------------------------

    def _check_process(self, target: Dict[str, Any], executor) -> CheckResult:
        """检查 Nginx 进程是否存活。"""
        try:
            result = executor.run("ps aux | grep nginx | grep -v grep", timeout=10)
            alive = bool(result.stdout.strip())
            if alive:
                return self.make_check(
                    "进程存活", ExitCode.NORMAL,
                    value=True, threshold=True,
                    message="Nginx 进程正在运行",
                )
            return self.make_check(
                "进程存活", ExitCode.CRITICAL,
                value=False, threshold=True,
                message="未检测到 Nginx 进程",
            )
        except Exception as e:
            return self.make_check(
                "进程存活", ExitCode.CRITICAL,
                value=None, message=f"进程检查异常: {e}",
            )

    def _check_5xx_error_rate(self, target: Dict[str, Any], executor) -> CheckResult:
        """解析 access.log 最近 10000 行，计算 5xx 错误率。"""
        warn = self.config.get("error_5xx_threshold_warn", 1.0)
        crit = self.config.get("error_5xx_threshold_critical", 5.0)
        log_path = target.get("access_log_path") or self.config.get(
            "access_log_path", "/var/log/nginx/access.log"
        )
        try:
            cmd = f"tail -n 10000 {log_path}"
            result = executor.run(cmd, timeout=30)
            lines = result.stdout.strip().splitlines()
            if not lines:
                return self.make_check(
                    "5xx错误率", ExitCode.NORMAL,
                    value=0.0, threshold=f"warn={warn}%,critical={crit}%",
                    message="日志为空，跳过检查",
                )
            count_5xx = sum(1 for line in lines if self._is_5xx(line))
            rate = count_5xx / len(lines) * 100
            if rate >= crit:
                status = ExitCode.CRITICAL
            elif rate >= warn:
                status = ExitCode.WARNING
            else:
                status = ExitCode.NORMAL
            return self.make_check(
                "5xx错误率", status,
                value=round(rate, 2), threshold=f"warn={warn}%,critical={crit}%",
                message=f"最近 {len(lines)} 行中 5xx 占比 {rate:.2f}%（{count_5xx} 条）",
            )
        except Exception as e:
            return self.make_check(
                "5xx错误率", ExitCode.CRITICAL,
                value=None, message=f"5xx 错误率检查异常: {e}",
            )

    def _check_active_connections(self, target: Dict[str, Any], executor) -> CheckResult:
        """从 stub_status 页面解析活跃连接数。"""
        warn = self.config.get("active_connections_warn", 10000)
        status_url = target.get("status_url") or self.config.get(
            "status_url", "http://localhost/nginx_status"
        )
        try:
            result = executor.run(f"curl -s {status_url}", timeout=10)
            match = re.search(r"Active connections:\s*(\d+)", result.stdout)
            if not match:
                return self.make_check(
                    "活跃连接数", ExitCode.WARNING,
                    value=None, threshold=f"warn={warn}",
                    message=f"无法从 {status_url} 解析活跃连接数",
                )
            conns = int(match.group(1))
            if conns >= warn:
                status = ExitCode.WARNING
            else:
                status = ExitCode.NORMAL
            return self.make_check(
                "活跃连接数", status,
                value=conns, threshold=f"warn={warn}",
                message=f"当前活跃连接数: {conns}",
            )
        except Exception as e:
            return self.make_check(
                "活跃连接数", ExitCode.CRITICAL,
                value=None, message=f"连接数检查异常: {e}",
            )

    def _check_upstream(self, target: Dict[str, Any], executor) -> CheckResult:
        """检查上游服务可达性。"""
        health_url = target.get("health_check_url") or self.config.get(
            "health_check_url", "http://localhost:8080/health"
        )
        try:
            cmd = f'curl -s -o /dev/null -w "%{{http_code}}" {health_url}'
            result = executor.run(cmd, timeout=10)
            code = result.stdout.strip()
            if code.startswith("2"):
                return self.make_check(
                    "上游可达性", ExitCode.NORMAL,
                    value=code, threshold="2xx",
                    message=f"上游返回 HTTP {code}",
                )
            elif code.startswith("5"):
                return self.make_check(
                    "上游可达性", ExitCode.CRITICAL,
                    value=code, threshold="2xx",
                    message=f"上游返回 HTTP {code}",
                )
            else:
                return self.make_check(
                    "上游可达性", ExitCode.WARNING,
                    value=code, threshold="2xx",
                    message=f"上游返回 HTTP {code}",
                )
        except Exception as e:
            return self.make_check(
                "上游可达性", ExitCode.CRITICAL,
                value=None, message=f"上游检查异常: {e}",
            )

    def _check_response_time(self, target: Dict[str, Any], executor) -> CheckResult:
        """测量上游平均响应时间。"""
        warn = self.config.get("response_time_warn", 2.0)
        crit = self.config.get("response_time_critical", 5.0)
        health_url = target.get("health_check_url") or self.config.get(
            "health_check_url", "http://localhost:8080/health"
        )
        try:
            cmd = f'curl -s -o /dev/null -w "%{{time_total}}" {health_url}'
            result = executor.run(cmd, timeout=30)
            elapsed = float(result.stdout.strip())
            if elapsed >= crit:
                status = ExitCode.CRITICAL
            elif elapsed >= warn:
                status = ExitCode.WARNING
            else:
                status = ExitCode.NORMAL
            return self.make_check(
                "响应时间", status,
                value=round(elapsed, 3), threshold=f"warn={warn}s,critical={crit}s",
                message=f"响应时间: {elapsed:.3f}s",
            )
        except Exception as e:
            return self.make_check(
                "响应时间", ExitCode.CRITICAL,
                value=None, message=f"响应时间检查异常: {e}",
            )

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _is_5xx(line: str) -> bool:
        """判断日志行是否包含 5xx 状态码（假设标准 combined/common 格式）。"""
        parts = line.split()
        if len(parts) < 9:
            return False
        try:
            code = int(parts[8])
            return 500 <= code < 600
        except ValueError:
            return False

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    from config.config_manager import ConfigManager

    cfg = ConfigManager()
    cfg.load()
    checker = NginxChecker(cfg.get_component_config("nginx"))
    sys.exit(checker.run())
