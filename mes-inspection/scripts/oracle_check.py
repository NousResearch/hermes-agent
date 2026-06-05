"""Oracle 巡检脚本 — 检查连接、慢 SQL、表空间、锁等待、活跃会话。"""

import os
import re
import sys
from datetime import datetime, timezone
from typing import List

from scripts.base_checker import BaseChecker, CheckResult, ExitCode, InspectionReport


class OracleChecker(BaseChecker):

    @property
    def component_name(self) -> str:
        return "oracle"

    def check(self) -> InspectionReport:
        checks: List[CheckResult] = []

        # 1. 连接测试
        conn_check = self._check_connection()
        checks.append(conn_check)
        if conn_check.status == ExitCode.CRITICAL:
            return self._report(checks, "Oracle 数据库不可用")

        # 2-5: 每个查询独立 try/except，失败降级为 WARNING
        checks.append(self._check_slow_sql())
        checks.append(self._check_tablespace())
        checks.append(self._check_lock_wait())
        checks.append(self._check_active_sessions())

        return self._report(checks, self._build_summary(checks))

    # ── SQL 执行 ─────────────────────────────────────────────

    def _run_sql(self, sql: str) -> str:
        """通过 sqlplus 执行 SQL，返回 stdout。"""
        conn_str = self.config.get("connection_string", "")
        username = self.config.get("username", "")
        password = os.getenv(self.config.get("password_env", "ORACLE_PASSWORD"), "")
        if not all([conn_str, username, password]):
            raise RuntimeError("Oracle 连接配置不完整 (connection_string / username / password_env)")

        # sqlplus -s 静默模式，SET LINESIZE/PAGESIZE 减少噪音
        wrapped = (
            "SET LINESIZE 32767\n"
            "SET PAGESIZE 0\n"
            "SET FEEDBACK OFF\n"
            "SET HEADING OFF\n"
            "SET TRIMSPOOL ON\n"
            f"{sql}\n"
            "EXIT;"
        )
        # 写入临时文件避免 shell 转义问题
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False, encoding="utf-8") as f:
            f.write(wrapped)
            sql_file = f.name

        try:
            cmd = f'sqlplus -s {username}/{password}@{conn_str} @{sql_file}'
            result = self.run_command(cmd, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"sqlplus 执行失败: {result.stderr.strip()}")
            return result.stdout
        finally:
            try:
                os.unlink(sql_file)
            except OSError:
                pass

    # ── 1. 连接测试 ──────────────────────────────────────────

    def _check_connection(self) -> CheckResult:
        try:
            output = self._run_sql("SELECT 1 FROM dual;")
            if "1" in output:
                return self.make_check("数据库连接", ExitCode.NORMAL, value="ok", message="Oracle 连接正常")
            return self.make_check("数据库连接", ExitCode.CRITICAL, value="fail", message=f"连接异常: {output.strip()[:200]}")
        except Exception as e:
            return self.make_check("数据库连接", ExitCode.CRITICAL, value="error", message=f"连接失败: {e}")

    # ── 2. 慢 SQL ────────────────────────────────────────────

    def _check_slow_sql(self) -> CheckResult:
        threshold_sec = self.config.get("slow_sql_threshold_sec", 5.0)
        warn_count = self.config.get("slow_sql_count_warn", 5)
        threshold_us = int(threshold_sec * 1_000_000)

        sql = (
            f"SELECT sql_id, ROUND(elapsed_time/1000000,1) sec, "
            f"SUBSTR(sql_text,1,100) txt "
            f"FROM v$sql "
            f"WHERE elapsed_time > {threshold_us} "
            f"ORDER BY elapsed_time DESC "
            f"FETCH FIRST 10 ROWS ONLY;"
        )

        try:
            output = self._run_sql(sql)
            rows = self._parse_sql_output(output, ["sql_id", "sec", "txt"])
            count = len(rows)

            if count == 0:
                return self.make_check(
                    "慢 SQL", ExitCode.NORMAL, value=0,
                    threshold=f">{threshold_sec}s",
                    message=f"无执行超过 {threshold_sec}s 的 SQL",
                )

            status = ExitCode.CRITICAL if count >= warn_count else ExitCode.WARNING
            details_rows = []
            for r in rows:
                details_rows.append({"sql_id": r.get("sql_id", ""), "elapsed_sec": r.get("sec", ""), "text": r.get("txt", "")})

            return self.make_check(
                "慢 SQL", status,
                value=count,
                threshold=f">{threshold_sec}s, >={warn_count} 条告警",
                message=f"发现 {count} 条慢 SQL (>{threshold_sec}s)",
                slow_sqls=details_rows,
            )
        except Exception as e:
            return self.make_check("慢 SQL", ExitCode.WARNING, message=f"慢 SQL 查询失败: {e}")

    # ── 3. 表空间使用率 ──────────────────────────────────────

    def _check_tablespace(self) -> CheckResult:
        warn = self.config.get("tablespace_usage_warn", 80.0)
        crit = self.config.get("tablespace_usage_critical", 90.0)

        sql = (
            "SELECT a.tablespace_name, ROUND((a.total - NVL(b.free,0)) / a.total * 100, 1) used_pct "
            "FROM (SELECT tablespace_name, SUM(bytes) total FROM dba_data_files GROUP BY tablespace_name) a "
            "LEFT JOIN (SELECT tablespace_name, SUM(bytes) free FROM dba_free_space GROUP BY tablespace_name) b "
            "ON a.tablespace_name = b.tablespace_name "
            "ORDER BY used_pct DESC;"
        )

        try:
            output = self._run_sql(sql)
            rows = self._parse_sql_output(output, ["tablespace_name", "used_pct"])

            if not rows:
                return self.make_check("表空间使用率", ExitCode.WARNING, message="未获取到表空间信息")

            worst_ts = None
            worst_pct = 0.0
            alert_rows = []
            for r in rows:
                ts_name = r.get("tablespace_name", "?")
                try:
                    pct = float(r.get("used_pct", 0))
                except (ValueError, TypeError):
                    continue
                if pct > worst_pct:
                    worst_pct = pct
                    worst_ts = ts_name
                if pct >= warn:
                    alert_rows.append({"tablespace": ts_name, "used_pct": pct})

            if worst_pct >= crit:
                status = ExitCode.CRITICAL
            elif worst_pct >= warn:
                status = ExitCode.WARNING
            else:
                status = ExitCode.NORMAL

            return self.make_check(
                "表空间使用率", status,
                value=f"{worst_pct:.1f}% ({worst_ts})",
                threshold=f"warn={warn}%,crit={crit}%",
                message=f"最高表空间 {worst_ts} 使用 {worst_pct:.1f}%",
                alert_tablespaces=alert_rows,
            )
        except Exception as e:
            return self.make_check("表空间使用率", ExitCode.WARNING, message=f"表空间查询失败: {e}")

    # ── 4. 锁等待 ────────────────────────────────────────────

    def _check_lock_wait(self) -> CheckResult:
        crit = self.config.get("lock_wait_critical", 10)

        sql = (
            "SELECT s.sid, s.serial#, s.username, l.locked_mode "
            "FROM v$session s JOIN v$lock l ON s.sid = l.sid "
            "WHERE l.block = 1;"
        )

        try:
            output = self._run_sql(sql)
            rows = self._parse_sql_output(output, ["sid", "serial#", "username", "locked_mode"])
            count = len(rows)

            if count == 0:
                return self.make_check("锁等待", ExitCode.NORMAL, value=0, message="无阻塞锁")

            status = ExitCode.CRITICAL if count >= crit else ExitCode.WARNING
            blocker_details = [{"sid": r.get("sid"), "username": r.get("username"), "mode": r.get("locked_mode")} for r in rows]

            return self.make_check(
                "锁等待", status,
                value=count,
                threshold=f"crit={crit}",
                message=f"检测到 {count} 个阻塞锁",
                blockers=blocker_details,
            )
        except Exception as e:
            return self.make_check("锁等待", ExitCode.WARNING, message=f"锁等待查询失败: {e}")

    # ── 5. 活跃会话数 ───────────────────────────────────────

    def _check_active_sessions(self) -> CheckResult:
        warn = self.config.get("active_sessions_warn", 100)

        sql = "SELECT COUNT(*) cnt FROM v$session WHERE status = 'ACTIVE';"

        try:
            output = self._run_sql(sql)
            rows = self._parse_sql_output(output, ["cnt"])
            count = int(rows[0].get("cnt", 0)) if rows else 0

            if count >= warn:
                status = ExitCode.WARNING
            else:
                status = ExitCode.NORMAL

            return self.make_check(
                "活跃会话数", status,
                value=count,
                threshold=f"warn={warn}",
                message=f"当前活跃会话 {count}" + (f", 超过阈值 {warn}" if status != ExitCode.NORMAL else ""),
            )
        except Exception as e:
            return self.make_check("活跃会话数", ExitCode.WARNING, message=f"活跃会话查询失败: {e}")

    # ── 辅助方法 ─────────────────────────────────────────────

    def _parse_sql_output(self, output: str, columns: List[str]) -> List[dict]:
        """解析 sqlplus 的制表符/空格分隔输出为字典列表。"""
        rows = []
        for line in output.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # 尝试按多个空格或制表符分割
            parts = re.split(r"\s{2,}|\t+", line)
            if len(parts) < len(columns):
                continue
            row = {}
            for i, col in enumerate(columns):
                row[col] = parts[i].strip() if i < len(parts) else ""
            rows.append(row)
        return rows

    def _report(self, checks: List[CheckResult], summary: str) -> InspectionReport:
        return InspectionReport(
            component=self.component_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=self.overall_status(checks),
            checks=checks,
            summary=summary,
            metadata={"mode": "sqlplus"},
        )

    def _build_summary(self, checks: List[CheckResult]) -> str:
        crits = sum(1 for c in checks if c.status == ExitCode.CRITICAL)
        warns = sum(1 for c in checks if c.status == ExitCode.WARNING)
        if crits:
            return f"发现 {crits} 项严重问题, {warns} 项告警"
        if warns:
            return f"发现 {warns} 项告警"
        return "所有检查项正常"


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config_manager import ConfigManager

    cfg = ConfigManager()
    cfg.load()
    checker = OracleChecker(cfg.get_component_config("oracle"))
    sys.exit(checker.run())
