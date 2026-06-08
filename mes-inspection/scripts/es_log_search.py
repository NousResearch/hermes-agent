"""ES 日志检索器 — 从 Elasticsearch 查询 MES 应用日志。"""

import json
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class EsLogSearcher:
    """从 Elasticsearch 查询 MES 应用日志，支持按主机名、时间范围、关键字、日志级别过滤。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _es_request(self, index: str, body: str) -> dict:
        """发送 ES 搜索请求。"""
        from config.default_thresholds import DEFAULT_THRESHOLDS
        base_url = self.config.get("elasticsearch_url") or DEFAULT_THRESHOLDS["elk"]["elasticsearch_url"]
        url = f"{base_url}/{index}/_search"
        data = body.encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.config.get("timeout", 10)) as resp:
            return json.loads(resp.read())

    def _resolve_index_pattern(self, start_time: str, end_time: str) -> str:
        """根据时间范围生成索引列表，跨天查询用逗号分隔。"""
        from config.default_thresholds import DEFAULT_THRESHOLDS
        prefix = self.config.get("index_prefix") or DEFAULT_THRESHOLDS["debug"]["es_index_prefix"]
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

        dates = []
        current = start_dt.date()
        end_date = end_dt.date()
        while current <= end_date:
            dates.append(f"{prefix}-{current.strftime('%Y.%m.%d')}")
            current += timedelta(days=1)

        return ",".join(dates)

    def _build_query_body(self, host_name: str, start_time: str, end_time: str,
                          keyword: str = None, level: str = None,
                          size: int = 100) -> str:
        """构建 ES 查询 JSON。"""
        filters = [
            {"term": {"host.name.keyword": host_name}},
            {"range": {"@timestamp": {"gte": start_time, "lte": end_time}}},
        ]

        must = []
        if keyword:
            must.append({"query_string": {"query": keyword}})
        if level:
            must.append({"query_string": {"query": level}})

        query: Dict[str, Any] = {"bool": {"filter": filters}}
        if must:
            query["bool"]["must"] = must

        body = {
            "query": query,
            "sort": [{"@timestamp": "asc"}],
            "size": size,
            "_source": [
                "@timestamp", "message", "host.name",
                "host.ip", "log.file.path", "tags",
            ],
        }
        return json.dumps(body)

    def _parse_hits(self, data: list) -> List[Dict]:
        """解析 hits 为日志条目列表。"""
        logs = []
        for hit in data:
            source = hit.get("_source", {})
            host = source.get("host", {})
            log_info = source.get("log", {})
            file_info = log_info.get("file", {})
            logs.append({
                "index": hit.get("_index", ""),
                "doc_id": hit.get("_id", ""),
                "timestamp": source.get("@timestamp", ""),
                "message": source.get("message", ""),
                "host_name": host.get("name", ""),
                "host_ip": host.get("ip", []),
                "log_file": file_info.get("path", ""),
                "tags": source.get("tags", []),
            })
        return logs

    def search(self, host_name: str, start_time: str, end_time: str,
               keyword: str = None, level: str = None, size: int = 100) -> Dict:
        """搜索 ES 日志。异常时返回 error 字典而非抛出异常。"""
        try:
            index = self._resolve_index_pattern(start_time, end_time)
            body = self._build_query_body(host_name, start_time, end_time, keyword, level, size)
            raw = self._es_request(index, body)
            hits = raw.get("hits", {}).get("hits", [])
            total_info = raw.get("hits", {}).get("total", {})
            total = total_info.get("value", 0) if isinstance(total_info, dict) else total_info

            logs = self._parse_hits(hits)

            result = {
                "logs": logs,
                "total": total,
                "config": {
                    "elasticsearch_url": self.config.get("elasticsearch_url", ""),
                    "index": index,
                    "host_name": host_name,
                    "start_time": start_time,
                    "end_time": end_time,
                },
            }
            return result

        except Exception as e:
            return {
                "logs": [],
                "total": 0,
                "error": str(e),
                "config": {
                    "elasticsearch_url": self.config.get("elasticsearch_url", ""),
                    "host_name": host_name,
                    "start_time": start_time,
                    "end_time": end_time,
                },
            }
