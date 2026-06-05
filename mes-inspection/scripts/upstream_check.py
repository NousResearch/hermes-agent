"""轻量上游节点存活检查器 — 通过 nginx_upstream_check_module 获取后端节点状态。

心跳巡检专用，一次 HTTP 请求获取所有 upstream 后端节点的存活状态。
依赖 nginx_upstream_check_module（需在 Nginx 配置中启用 upstream_check 状态页）。
"""

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.base_checker import BaseChecker, CheckResult, ExitCode, InspectionReport


class UpstreamChecker(BaseChecker):

    @property
    def component_name(self) -> str:
        return "upstream"

    def check(self) -> InspectionReport:
        status_url = self.config.get("status_url", "")
        if not status_url:
            return InspectionReport(
                component=self.component_name,
                timestamp=self._now_iso(),
                status=ExitCode.CRITICAL,
                checks=[self.make_check("配置", ExitCode.CRITICAL, value=None, message="未配置 upstream status_url")],
                summary="未配置 upstream 状态页 URL",
                metadata={"node_count": 0, "nodes": {}},
            )

        try:
            nodes = self._fetch_upstream_nodes(status_url)
        except Exception as e:
            return InspectionReport(
                component=self.component_name,
                timestamp=self._now_iso(),
                status=ExitCode.CRITICAL,
                checks=[self.make_check("Nginx 连通性", ExitCode.CRITICAL, value=None, message=f"无法获取 upstream 状态: {e}")],
                summary=f"upstream 状态页不可达: {e}",
                metadata={"node_count": 0, "nodes": {}},
            )

        if not nodes:
            return InspectionReport(
                component=self.component_name,
                timestamp=self._now_iso(),
                status=ExitCode.WARNING,
                checks=[self.make_check("upstream 节点", ExitCode.WARNING, value=0, message="upstream 状态页无节点数据")],
                summary="无 upstream 节点",
                metadata={"node_count": 0, "nodes": {}},
            )

        checks = []
        for node in nodes:
            name = node.get("name", node.get("server", "unknown"))
            node_status = node.get("status", "unknown")
            upstream = node.get("upstream", "")

            if node_status == "up":
                status = ExitCode.NORMAL
                message = f"{name} 正常运行（upstream: {upstream}）"
            else:
                status = ExitCode.CRITICAL
                message = f"{name} 不可用（upstream: {upstream}，状态: {node_status}）"

            checks.append(self.make_check(
                f"节点 [{name}]", status,
                value=node_status,
                threshold="up",
                message=message,
                upstream=upstream,
                server=name,
            ))

        overall = self.overall_status(checks)
        total = len(checks)
        down = sum(1 for c in checks if c.status == ExitCode.CRITICAL)

        node_statuses = {}
        for c in checks:
            server = c.details.get("server", c.name)
            node_statuses[server] = c.status.name

        if down:
            summary = f"{total} 个后端节点，{down} 个不可用"
        else:
            summary = f"{total} 个后端节点全部正常"

        return InspectionReport(
            component=self.component_name,
            timestamp=self._now_iso(),
            status=overall,
            checks=checks,
            summary=summary,
            metadata={"node_count": total, "nodes": node_statuses},
        )

    def _fetch_upstream_nodes(self, status_url: str) -> List[Dict[str, str]]:
        """从 nginx_upstream_check_module 状态页获取节点列表。

        支持 JSON 格式（?format=json）和文本格式。
        返回 [{"name": "10.0.0.1:8080", "status": "up", "upstream": "mes-backend"}, ...]
        """
        req = urllib.request.Request(status_url)
        with urllib.request.urlopen(req, timeout=self.config.get("timeout", 5)) as resp:
            body = resp.read().decode("utf-8", errors="replace")

        if "format=json" in status_url or body.strip().startswith("{"):
            return self._parse_json_status(body)
        return self._parse_text_status(body)

    def _parse_json_status(self, body: str) -> List[Dict[str, str]]:
        """解析 JSON 格式的 upstream 状态。

        nginx_upstream_check_module JSON 格式:
        {"upstream_name": [{"server": "ip:port", "status": "up/down"}, ...], ...}
        """
        data = json.loads(body)
        nodes = []
        for upstream_name, servers in data.items():
            if not isinstance(servers, list):
                continue
            for srv in servers:
                nodes.append({
                    "name": srv.get("server", "unknown"),
                    "status": srv.get("status", "unknown"),
                    "upstream": upstream_name,
                })
        return nodes

    def _parse_text_status(self, body: str) -> List[Dict[str, str]]:
        """解析文本格式的 upstream 状态。

        nginx_upstream_check_module 文本格式:
        upstream mes-backend
            server 10.0.0.1:8080 up
            server 10.0.0.2:8080 down
        """
        nodes = []
        current_upstream = ""
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("upstream "):
                current_upstream = line.split(None, 1)[1] if len(line.split()) > 1 else ""
            elif line.startswith("server "):
                parts = line.split()
                if len(parts) >= 3:
                    nodes.append({
                        "name": parts[1],
                        "status": parts[2],
                        "upstream": current_upstream,
                    })
        return nodes

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    from config.config_manager import ConfigManager
    cfg = ConfigManager()
    cfg.load()
    checker = UpstreamChecker(cfg.get_component_config("upstream"))
    sys.exit(checker.run())
