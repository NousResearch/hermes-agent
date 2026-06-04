"""JVM/Tomcat 巡检脚本 - 检查进程、堆内存、GC、线程、死锁、日志错误。"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.base_checker import BaseChecker, CheckResult, ExitCode, InspectionReport
from scripts.ssh_executor import create_executor, SSHExecutor, LocalExecutor


class JvmChecker(BaseChecker):
    """JVM/Tomcat 巡检器。"""

    @property
    def component_name(self) -> str:
        return "jvm"

    def check(self) -> InspectionReport:
        return self.check_all_targets()

    def check_node(self, target: Dict[str, Any]) -> InspectionReport:
        """对单个节点执行 JVM 巡检。"""
        executor = create_executor(target)
        checks: List[CheckResult] = []

        pid = self._get_pid(executor)
        if pid is None:
            checks.append(self.make_check(
                "进程存活", ExitCode.CRITICAL,
                value=None, message="未找到 catalina 进程",
            ))
            for name in ["堆内存使用率", "GC频率", "GC耗时", "线程状态", "死锁检测", "日志错误"]:
                checks.append(self.make_check(
                    name, ExitCode.CRITICAL,
                    value=None, message="进程不存在，跳过检查",
                ))
            status = ExitCode.CRITICAL
            summary = "JVM 进程未运行"
        else:
            checks.append(self._check_process(pid))
            gc_data = self._get_jstat_data(pid, executor)
            checks.append(self._check_heap(gc_data))
            checks.append(self._check_gc_freq(gc_data))
            checks.append(self._check_gc_time(gc_data))
            checks.append(self._check_threads(pid, executor))
            checks.append(self._check_deadlock(pid, executor))
            checks.append(self._check_log_errors(target, executor))
            status = self.overall_status(checks)
            summary = f"JVM 巡检完成，PID={pid}，共 {len(checks)} 项，状态: {status.name}"

        return InspectionReport(
            component=self.component_name,
            timestamp=self._now_iso(),
            status=status,
            checks=checks,
            summary=summary,
            metadata={"pid": pid},
        )

    def _get_pid(self, executor) -> Optional[int]:
        """通过 jps 获取 catalina 进程 PID。"""
        proc_name = self.config.get("tomcat_process_name", "catalina")
        try:
            result = executor.run(f"jps -l | grep {proc_name}", timeout=10)
            if not result.stdout.strip():
                return None
            first_line = result.stdout.strip().splitlines()[0]
            pid = int(first_line.split()[0])
            return pid
        except Exception:
            return None

    def _get_jstat_data(self, pid: int, executor) -> Dict[str, float]:
        """获取 jstat -gcutil 数据，返回解析后的字典。"""
        data: Dict[str, float] = {}
        try:
            cmd = f"jstat -gcutil {pid} 1000 1"
            result = executor.run(cmd, timeout=15)
            lines = result.stdout.strip().splitlines()
            if len(lines) < 2:
                return data
            header = lines[1].split()
            values = lines[2].split() if len(lines) > 2 else []
            for h, v in zip(header, values):
                try:
                    data[h] = float(v)
                except ValueError:
                    pass
        except Exception:
            pass
        return data

    def _check_process(self, pid: int) -> CheckResult:
        """确认 catalina 进程存活。"""
        return self.make_check(
            "进程存活", ExitCode.NORMAL,
            value=pid, message=f"catalina 进程 PID={pid} 正在运行",
        )

    def _check_heap(self, gc_data: Dict[str, float]) -> CheckResult:
        """检查 Old 区使用率（近似堆内存使用率）。"""
        warn = self.config.get("heap_usage_warn", 80.0)
        crit = self.config.get("heap_usage_critical", 90.0)
        old_usage = gc_data.get("O")
        if old_usage is None:
            return self.make_check(
                "堆内存使用率", ExitCode.WARNING,
                value=None, threshold=f"warn={warn}%,critical={crit}%",
                message="无法解析 Old 区使用率",
            )
        if old_usage >= crit:
            status = ExitCode.CRITICAL
        elif old_usage >= warn:
            status = ExitCode.WARNING
        else:
            status = ExitCode.NORMAL
        return self.make_check(
            "堆内存使用率", status,
            value=round(old_usage, 1), threshold=f"warn={warn}%,critical={crit}%",
            message=f"Old 区使用率: {old_usage:.1f}%",
        )

    def _check_gc_freq(self, gc_data: Dict[str, float]) -> CheckResult:
        """检查 Full GC 频率。"""
        warn = self.config.get("full_gc_freq_warn", 5)
        fgc = gc_data.get("FGC")
        if fgc is None:
            return self.make_check(
                "GC频率", ExitCode.WARNING,
                value=None, threshold=f"warn={warn}",
                message="无法解析 Full GC 次数",
            )
        fgc = int(fgc)
        if fgc >= warn:
            status = ExitCode.WARNING
        else:
            status = ExitCode.NORMAL
        return self.make_check(
            "GC频率", status,
            value=fgc, threshold=f"warn={warn}",
            message=f"Full GC 次数: {fgc}",
        )

    def _check_gc_time(self, gc_data: Dict[str, float]) -> CheckResult:
        """检查 Full GC 平均耗时。"""
        warn_time = self.config.get("full_gc_time_critical", 10.0)
        fgc = gc_data.get("FGC", 0)
        fgct = gc_data.get("FGCT", 0)
        if fgc is None or fgct is None or fgc == 0:
            return self.make_check(
                "GC耗时", ExitCode.NORMAL,
                value=0.0, threshold=f"critical={warn_time}s",
                message="无 Full GC 记录",
            )
        avg_time = fgct / fgc
        if avg_time >= warn_time:
            status = ExitCode.CRITICAL
        else:
            status = ExitCode.NORMAL
        return self.make_check(
            "GC耗时", status,
            value=round(avg_time, 2), threshold=f"critical={warn_time}s",
            message=f"Full GC 平均耗时: {avg_time:.2f}s（总 {fgct:.2f}s / {int(fgc)} 次）",
        )

    def _check_threads(self, pid: int, executor) -> CheckResult:
        """通过 jstack 统计线程状态。"""
        warn_active = self.config.get("active_threads_warn", 500)
        crit_blocked = self.config.get("blocked_threads_critical", 10)
        try:
            result = executor.run(f"jstack {pid}", timeout=15)
            output = result.stdout
            if not output.strip():
                return self.make_check(
                    "线程状态", ExitCode.WARNING,
                    value=None, message="jstack 输出为空",
                )
            state_counts: Dict[str, int] = {}
            for match in re.finditer(r'Thread.State:\s*(\S+)', output):
                state = match.group(1)
                state_counts[state] = state_counts.get(state, 0) + 1
            total = sum(state_counts.values())
            blocked = state_counts.get("BLOCKED", 0)

            if total >= warn_active:
                status = ExitCode.WARNING
            elif blocked >= crit_blocked:
                status = ExitCode.CRITICAL
            else:
                status = ExitCode.NORMAL

            detail_parts = [f"{s}={c}" for s, c in sorted(state_counts.items())]
            return self.make_check(
                "线程状态", status,
                value=total,
                threshold=f"active_warn={warn_active},blocked_crit={crit_blocked}",
                message=f"总线程数: {total}，BLOCKED: {blocked}",
                state_counts=state_counts,
                detail=", ".join(detail_parts),
            )
        except Exception as e:
            return self.make_check(
                "线程状态", ExitCode.CRITICAL,
                value=None, message=f"线程检查异常: {e}",
            )

    def _check_deadlock(self, pid: int, executor) -> CheckResult:
        """通过 jstack 检测死锁。"""
        try:
            result = executor.run(f"jstack {pid}", timeout=15)
            if re.search(r'Found.*deadlock', result.stdout, re.IGNORECASE):
                return self.make_check(
                    "死锁检测", ExitCode.CRITICAL,
                    value=True, threshold=False,
                    message="检测到死锁！",
                    deadlock_detail=self._extract_deadlock_detail(result.stdout),
                )
            return self.make_check(
                "死锁检测", ExitCode.NORMAL,
                value=False, threshold=False,
                message="未检测到死锁",
            )
        except Exception as e:
            return self.make_check(
                "死锁检测", ExitCode.CRITICAL,
                value=None, message=f"死锁检测异常: {e}",
            )

    def _check_log_errors(self, target: Dict[str, Any], executor) -> CheckResult:
        """检查 catalina.out 最近 N 行中的错误数。"""
        log_lines = self.config.get("catalina_log_lines", 1000)
        log_path = target.get("catalina_log_path") or self.config.get(
            "catalina_log_path", "logs/catalina.out"
        )
        try:
            cmd = f"tail -n {log_lines} {log_path}"
            result = executor.run(cmd, timeout=15)
            output = result.stdout
            if not output.strip():
                return self.make_check(
                    "日志错误", ExitCode.NORMAL,
                    value=0, message="日志为空",
                )
            error_count = 0
            for line in output.splitlines():
                if re.search(r'ERROR|Exception', line, re.IGNORECASE):
                    error_count += 1
            if error_count > 0:
                status = ExitCode.WARNING
            else:
                status = ExitCode.NORMAL
            return self.make_check(
                "日志错误", status,
                value=error_count, message=f"最近 {log_lines} 行中错误/异常数: {error_count}",
            )
        except Exception as e:
            return self.make_check(
                "日志错误", ExitCode.CRITICAL,
                value=None, message=f"日志检查异常: {e}",
            )

    @staticmethod
    def _extract_deadlock_detail(output: str) -> str:
        """提取死锁详细信息（Found deadlock 后 20 行）。"""
        lines = output.splitlines()
        for i, line in enumerate(lines):
            if re.search(r'Found.*deadlock', line, re.IGNORECASE):
                return "\n".join(lines[i:i + 20])
        return ""

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    from config.config_manager import ConfigManager

    cfg = ConfigManager()
    cfg.load()
    checker = JvmChecker(cfg.get_component_config("jvm"))
    sys.exit(checker.run())
