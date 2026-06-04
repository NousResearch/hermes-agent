"""ELK（Elasticsearch）巡检脚本。"""

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


class ELKChecker(BaseChecker):

    @property
    def component_name(self) -> str:
        return "elk"

    def _es_request(self, path: str) -> dict:
        """发送 Elasticsearch API 请求。"""
        base_url = self.config.get("elasticsearch_url", "http://localhost:9200")
        url = f"{base_url}{path}"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def _check_unassigned_shards(self, health_data: dict) -> CheckResult:
        """检查未分配分片数。"""
        unassigned = health_data.get("unassigned_shards", 0)
        warn = self.config.get("unassigned_shards_warn", 5)
        status = ExitCode.CRITICAL if unassigned >= warn * 2 else (
            ExitCode.WARNING if unassigned >= warn else ExitCode.NORMAL
        )
        return self.make_check(
            "未分配分片", status, unassigned, warn,
            f"未分配分片 {unassigned} 个（阈值 {warn}）",
        )

    def _check_index_write_rate(self) -> CheckResult:
        """检查索引写入速率（取 top 5 索引）。"""
        data = self._es_request(
            "/_cat/indices?format=json&s=write.indexing.index_total:desc"
            "&h=index,docs.count,store.size,pri.write.indexing.index_total,pri.write.indexing.index_time"
        )
        if not data:
            return self.make_check(
                "索引写入速率", ExitCode.NORMAL, 0, None,
                "无索引数据",
            )
        rows = data[:5]
        return self.make_check(
            "索引写入速率", ExitCode.NORMAL, len(rows), None,
            f"写入最频繁的 {len(rows)} 个索引",
            top_indices=rows,
        )

    def _check_jvm_and_disk(self) -> List[CheckResult]:
        """检查节点 JVM 堆内存和磁盘使用率。"""
        data = self._es_request("/_nodes/stats/jvm,fs,os?pretty")
        nodes = data.get("nodes", {})
        jvm_warn = self.config.get("jvm_heap_warn", 75.0)
        disk_warn = self.config.get("disk_usage_warn", 80.0)
        disk_critical = self.config.get("disk_usage_critical", 90.0)

        checks: List[CheckResult] = []
        for node_id, node in nodes.items():
            node_name = node.get("name", node_id)

            # JVM 堆内存
            jvm = node.get("jvm", {}).get("mem", {})
            heap_used = jvm.get("heap_used_percent", 0)
            heap_max = jvm.get("heap_max_in_bytes", 0)
            heap_used_bytes = jvm.get("heap_used_in_bytes", 0)
            heap_status = ExitCode.WARNING if heap_used >= jvm_warn else ExitCode.NORMAL
            checks.append(self.make_check(
                f"JVM堆内存 [{node_name}]", heap_status, heap_used, jvm_warn,
                f"JVM 堆使用率 {heap_used}%（阈值 {jvm_warn}%）",
                heap_used_bytes=heap_used_bytes,
                heap_max_bytes=heap_max,
            ))

            # 磁盘使用率
            fs = node.get("fs", {}).get("total", {})
            total = fs.get("total_in_bytes", 0)
            available = fs.get("available_in_bytes", 0)
            if total > 0:
                used_pct = round((1 - available / total) * 100, 1)
                if used_pct >= disk_critical:
                    disk_status = ExitCode.CRITICAL
                elif used_pct >= disk_warn:
                    disk_status = ExitCode.WARNING
                else:
                    disk_status = ExitCode.NORMAL
                checks.append(self.make_check(
                    f"磁盘使用率 [{node_name}]", disk_status, used_pct, disk_warn,
                    f"磁盘使用率 {used_pct}%（告警 {disk_warn}%，严重 {disk_critical}%）",
                    total_bytes=total,
                    available_bytes=available,
                ))

        if not nodes:
            checks.append(self.make_check(
                "节点信息", ExitCode.WARNING, 0, None,
                "未获取到节点信息",
            ))
        return checks

    def check(self) -> InspectionReport:
        checks: List[CheckResult] = []
        metadata: Dict[str, Any] = {}

        # 集群健康
        try:
            health = self._es_request("/_cluster/health?pretty")
            metadata["cluster_name"] = health.get("cluster_name", "")
            checks.append(self._check_cluster_health_from_data(health))
            checks.append(self._check_unassigned_shards(health))
        except Exception as e:
            checks.append(self.make_check(
                "集群健康", ExitCode.CRITICAL, None, None,
                f"无法连接 Elasticsearch: {e}",
            ))

        # 索引写入速率
        try:
            checks.append(self._check_index_write_rate())
        except Exception as e:
            checks.append(self.make_check(
                "索引写入速率", ExitCode.WARNING, None, None,
                f"获取索引信息失败: {e}",
            ))

        # JVM 与磁盘
        try:
            checks.extend(self._check_jvm_and_disk())
        except Exception as e:
            checks.append(self.make_check(
                "节点状态", ExitCode.WARNING, None, None,
                f"获取节点统计失败: {e}",
            ))

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

    def _check_cluster_health_from_data(self, data: dict) -> CheckResult:
        """从已有健康数据创建集群状态检查结果。"""
        status = data.get("status", "unknown")
        warn = self.config.get("cluster_status_warn", "yellow")
        critical = self.config.get("cluster_status_critical", "red")

        if status == critical:
            return self.make_check(
                "集群状态", ExitCode.CRITICAL, status, critical,
                f"集群状态为 {status}，存在不可用分片",
                cluster_name=data.get("cluster_name", ""),
                active_shards=data.get("active_shards"),
                relocating_shards=data.get("relocating_shards"),
                initializing_shards=data.get("initializing_shards"),
            )
        elif status == warn:
            return self.make_check(
                "集群状态", ExitCode.WARNING, status, warn,
                f"集群状态为 {status}，存在未分配分片",
                cluster_name=data.get("cluster_name", ""),
                active_shards=data.get("active_shards"),
            )
        else:
            return self.make_check(
                "集群状态", ExitCode.NORMAL, status, "green",
                "集群状态正常",
                cluster_name=data.get("cluster_name", ""),
                active_shards=data.get("active_shards"),
            )


if __name__ == "__main__":
    from config.config_manager import ConfigManager

    cfg = ConfigManager()
    cfg.load()
    checker = ELKChecker(cfg.get_component_config("elk"))
    sys.exit(checker.run())
