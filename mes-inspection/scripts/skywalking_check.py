"""SkyWalking 巡检脚本。"""

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# 确保项目根目录在 sys.path 中，支持独立运行
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.base_checker import BaseChecker, CheckResult, ExitCode, InspectionReport

RESP_TIME_QUERY = '''
query {{
  readMetricsValues(condition: {{
    name: "service_resp_time",
    entity: {{ scope: Service, normal: true, serviceName: "{service}" }}
  }}) {{
    values {{ values {{ id value }} }}
  }}
}}
'''

SLA_QUERY = '''
query {{
  readMetricsValues(condition: {{
    name: "service_sla",
    entity: {{ scope: Service, normal: true, serviceName: "{service}" }}
  }}) {{
    values {{ values {{ id value }} }}
  }}
}}
'''

TOP_SLOW_ENDPOINTS_QUERY = '''
query {{
  readTopNRecords(condition: {{
    name: "service_resp_time",
    topN: 10,
    order: DES,
    entity: {{ scope: Service, normal: true, serviceName: "{service}" }}
  }}, duration: {{
    start: "{start}",
    end: "{end}",
    step: MINUTE
  }}) {{
    name value
  }}
}}
'''

ALARMS_QUERY = '''
query {
  readAlarms(duration: { start: "2026-01-01", end: "2026-12-31", step: MINUTE }) {
    msgs { scopeId name startTime message }
  }
}
'''


class SkyWalkingChecker(BaseChecker):

    @property
    def component_name(self) -> str:
        return "skywalking"

    def _sw_graphql(self, query: str) -> dict:
        """发送 SkyWalking GraphQL 请求。"""
        base_url = self.config.get("oap_url", "http://localhost:12800")
        url = f"{base_url}/graphql"
        payload = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())

    def _check_oap_health(self) -> CheckResult:
        """检查 OAP 服务存活。"""
        base_url = self.config.get("oap_url", "http://localhost:12800")
        url = f"{base_url}/healthcheck"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                code = resp.getcode()
                if code == 200:
                    return self.make_check(
                        "OAP服务存活", ExitCode.NORMAL, "healthy", None,
                        "OAP 服务正常运行",
                    )
                else:
                    return self.make_check(
                        "OAP服务存活", ExitCode.CRITICAL, code, None,
                        f"OAP 服务返回状态码 {code}",
                    )
        except Exception as e:
            return self.make_check(
                "OAP服务存活", ExitCode.CRITICAL, "unreachable", None,
                f"OAP 服务不可达: {e}",
            )

    def _check_response_time(self) -> CheckResult:
        """检查服务 P95 响应时间。"""
        service = self.config.get("service_name", "mes-service")
        query = RESP_TIME_QUERY.format(service=service)
        try:
            data = self._sw_graphql(query)
            values = (data.get("data", {})
                          .get("readMetricsValues", {})
                          .get("values", {})
                          .get("values", []))
            if not values:
                return self.make_check(
                    "P95响应时间", ExitCode.WARNING, None, None,
                    f"未获取到服务 {service} 的响应时间数据",
                )
            resp_time = values[0].get("value", 0)
            warn = self.config.get("p95_response_time_warn", 2000)
            critical = self.config.get("p95_response_time_critical", 5000)

            if resp_time >= critical:
                status = ExitCode.CRITICAL
                msg = f"P95 响应时间 {resp_time}ms 超过严重阈值 {critical}ms"
            elif resp_time >= warn:
                status = ExitCode.WARNING
                msg = f"P95 响应时间 {resp_time}ms 超过告警阈值 {warn}ms"
            else:
                status = ExitCode.NORMAL
                msg = f"P95 响应时间 {resp_time}ms 正常"

            return self.make_check(
                "P95响应时间", status, resp_time, warn,
                msg, service=service,
            )
        except Exception as e:
            return self.make_check(
                "P95响应时间", ExitCode.WARNING, None, None,
                f"查询响应时间失败: {e}",
            )

    def _check_sla(self) -> CheckResult:
        """检查服务 SLA。"""
        service = self.config.get("service_name", "mes-service")
        query = SLA_QUERY.format(service=service)
        try:
            data = self._sw_graphql(query)
            values = (data.get("data", {})
                          .get("readMetricsValues", {})
                          .get("values", {})
                          .get("values", []))
            if not values:
                return self.make_check(
                    "服务SLA", ExitCode.WARNING, None, None,
                    f"未获取到服务 {service} 的 SLA 数据",
                )
            raw_sla = values[0].get("value", 0)
            # SkyWalking SLA 是千分比，转换为百分比
            sla_pct = round(raw_sla / 100, 2)
            warn = self.config.get("sla_warn", 99.0)
            critical = self.config.get("sla_critical", 95.0)

            if sla_pct < critical:
                status = ExitCode.CRITICAL
                msg = f"服务 SLA {sla_pct}% 低于严重阈值 {critical}%"
            elif sla_pct < warn:
                status = ExitCode.WARNING
                msg = f"服务 SLA {sla_pct}% 低于告警阈值 {warn}%"
            else:
                status = ExitCode.NORMAL
                msg = f"服务 SLA {sla_pct}% 正常"

            return self.make_check(
                "服务SLA", status, sla_pct, warn,
                msg, service=service, raw_value=raw_sla,
            )
        except Exception as e:
            return self.make_check(
                "服务SLA", ExitCode.WARNING, None, None,
                f"查询 SLA 失败: {e}",
            )

    def _check_slow_endpoints(self) -> CheckResult:
        """检查 Top N 慢接口。"""
        service = self.config.get("service_name", "mes-service")
        now = datetime.utcnow()
        start = (now.replace(minute=0, second=0, microsecond=0)).strftime("%Y-%m-%d %H%M")
        end = now.strftime("%Y-%m-%d %H%M")
        query = TOP_SLOW_ENDPOINTS_QUERY.format(
            service=service, start=start, end=end
        )
        try:
            data = self._sw_graphql(query)
            records = data.get("data", {}).get("readTopNRecords", [])
            if not records:
                return self.make_check(
                    "慢接口Top10", ExitCode.NORMAL, 0, None,
                    "无慢接口记录",
                )
            endpoints = []
            for r in records:
                endpoints.append({
                    "name": r.get("name", "unknown"),
                    "resp_time_ms": r.get("value", 0),
                })
            max_time = max(e["resp_time_ms"] for e in endpoints)
            critical = self.config.get("p95_response_time_critical", 5000)
            status = ExitCode.WARNING if max_time >= critical else ExitCode.NORMAL
            return self.make_check(
                "慢接口Top10", status, len(endpoints), None,
                f"Top {len(endpoints)} 慢接口，最慢 {max_time}ms",
                endpoints=endpoints,
            )
        except Exception as e:
            return self.make_check(
                "慢接口Top10", ExitCode.WARNING, None, None,
                f"查询慢接口失败: {e}",
            )

    def _check_alarms(self) -> CheckResult:
        """检查活跃告警。"""
        try:
            year = datetime.utcnow().year
            query = ALARMS_QUERY.replace("2026-01-01", f"{year}-01-01").replace("2026-12-31", f"{year}-12-31")
            data = self._sw_graphql(query)
            msgs = (data.get("data", {})
                        .get("readAlarms", {})
                        .get("msgs", []))
            if not msgs:
                return self.make_check(
                    "活跃告警", ExitCode.NORMAL, 0, None,
                    "无活跃告警",
                )
            critical_count = len(msgs)
            return self.make_check(
                "活跃告警", ExitCode.WARNING, critical_count, None,
                f"当前有 {critical_count} 条活跃告警",
                alarms=msgs[:10],
            )
        except Exception as e:
            return self.make_check(
                "活跃告警", ExitCode.WARNING, None, None,
                f"查询告警失败: {e}",
            )

    def check(self) -> InspectionReport:
        checks: List[CheckResult] = []
        metadata: Dict[str, Any] = {}

        # OAP 服务存活
        checks.append(self._check_oap_health())

        # P95 响应时间
        checks.append(self._check_response_time())

        # 服务 SLA
        checks.append(self._check_sla())

        # 慢接口
        checks.append(self._check_slow_endpoints())

        # 活跃告警
        checks.append(self._check_alarms())

        status = self.overall_status(checks)
        summary = f"共 {len(checks)} 项检查，" + (
            "全部正常" if status == ExitCode.NORMAL else
            f"存在 {'严重' if status == ExitCode.CRITICAL else '告警'} 问题"
        )

        return InspectionReport(
            component=self.component_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            checks=checks,
            summary=summary,
            metadata=metadata,
        )


if __name__ == "__main__":
    from config.config_manager import ConfigManager

    cfg = ConfigManager()
    cfg.load()
    checker = SkyWalkingChecker(cfg.get_component_config("skywalking"))
    sys.exit(checker.run())
