"""ES 日志检索器测试。"""

import json
from unittest.mock import patch, MagicMock
import pytest

from scripts.es_log_search import EsLogSearcher


# ── 测试用 ES 响应数据 ────────────────────────────────────────────────

def _make_hit(timestamp="2026-06-05T05:54:11.816Z",
              message="2026-06-05 01:21:43.028 WARN [taskExecutor1-307] DataRecordServiceImpl - 采集项重复",
              host_name="39QEMES-Tomcat-Crontab01",
              ip="10.52.2.26",
              log_path="/u01/app/mes-app/logs/catalina.2026060501.log",
              tags=None,
              index="39qjmes-2026.06.05",
              doc_id="abc123"):
    """构造单条 ES hit。"""
    return {
        "_index": index,
        "_id": doc_id,
        "_source": {
            "@timestamp": timestamp,
            "message": message,
            "host": {
                "name": host_name,
                "ip": [ip],
            },
            "log": {
                "file": {"path": log_path},
                "offset": 233990255,
            },
            "tags": tags or ["tomcat-mes"],
            "agent": {"name": host_name},
        }
    }


def _make_es_response(hits=None, total=None):
    """构造 ES 搜索响应。"""
    if hits is None:
        hits = []
    if total is None:
        total = {"value": len(hits), "relation": "eq"}
    return {
        "hits": {
            "total": total,
            "hits": hits,
        }
    }


# ── 查询构建测试 ──────────────────────────────────────────────────────

class TestEsLogSearcherBuildQuery:
    """_build_query_body 和 _resolve_index_pattern 测试。"""

    def _make_searcher(self, **overrides):
        config = {"elasticsearch_url": "http://localhost:9200", "index_prefix": "39qjmes"}
        config.update(overrides)
        return EsLogSearcher(config)

    def test_basic_query(self):
        """基本查询包含 host.name 过滤和时间范围。"""
        searcher = self._make_searcher()
        body = searcher._build_query_body(
            host_name="39QEMES-Tomcat-Crontab01",
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-05T23:59:59Z",
        )
        parsed = json.loads(body)
        # 验证 bool.filter 存在
        assert "bool" in parsed["query"]
        assert "filter" in parsed["query"]["bool"]
        filters = parsed["query"]["bool"]["filter"]
        # 应有 term 和 range 两个过滤条件
        assert len(filters) == 2
        term_filter = [f for f in filters if "term" in f][0]
        range_filter = [f for f in filters if "range" in f][0]
        assert term_filter["term"]["host.name.keyword"] == "39QEMES-Tomcat-Crontab01"
        assert "@timestamp" in range_filter["range"]
        # 验证排序
        assert parsed["sort"] == [{"@timestamp": "asc"}]

    def test_query_with_keyword(self):
        """带关键字的查询添加 must.query_string。"""
        searcher = self._make_searcher()
        body = searcher._build_query_body(
            host_name="39QEMES-Tomcat-Crontab01",
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-05T23:59:59Z",
            keyword="采集项重复",
        )
        parsed = json.loads(body)
        must = parsed["query"]["bool"].get("must", [])
        assert len(must) == 1
        assert "query_string" in must[0]
        assert must[0]["query_string"]["query"] == "采集项重复"

    def test_query_with_level(self):
        """带日志级别的查询添加 must.query_string。"""
        searcher = self._make_searcher()
        body = searcher._build_query_body(
            host_name="39QEMES-Tomcat-Crontab01",
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-05T23:59:59Z",
            level="WARN",
        )
        parsed = json.loads(body)
        must = parsed["query"]["bool"].get("must", [])
        assert len(must) == 1
        assert must[0]["query_string"]["query"] == "WARN"

    def test_index_pattern_single_day(self):
        """单天索引模式。"""
        searcher = self._make_searcher()
        pattern = searcher._resolve_index_pattern(
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-05T23:59:59Z",
        )
        assert pattern == "39qjmes-2026.06.05"

    def test_index_pattern_cross_days(self):
        """跨天索引模式用逗号分隔。"""
        searcher = self._make_searcher()
        pattern = searcher._resolve_index_pattern(
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-07T23:59:59Z",
        )
        assert pattern == "39qjmes-2026.06.05,39qjmes-2026.06.06,39qjmes-2026.06.07"


# ── 结果解析测试 ──────────────────────────────────────────────────────

class TestEsLogSearcherParseResults:
    """_parse_hits 测试。"""

    def _make_searcher(self):
        return EsLogSearcher({"elasticsearch_url": "http://localhost:9200"})

    def test_parse_hits(self):
        """正常解析 hits。"""
        searcher = self._make_searcher()
        hits = [_make_hit(), _make_hit(timestamp="2026-06-05T06:00:00.000Z", message="test log 2")]
        result = searcher._parse_hits(hits)
        assert len(result) == 2
        assert result[0]["timestamp"] == "2026-06-05T05:54:11.816Z"
        assert result[0]["host_name"] == "39QEMES-Tomcat-Crontab01"
        assert result[0]["host_ip"] == ["10.52.2.26"]
        assert result[0]["log_file"] == "/u01/app/mes-app/logs/catalina.2026060501.log"
        assert result[0]["index"] == "39qjmes-2026.06.05"
        assert result[0]["doc_id"] == "abc123"
        assert "host_hostname" not in result[0]
        assert result[1]["message"] == "test log 2"

    def test_parse_empty_hits(self):
        """空 hits 返回空列表。"""
        searcher = self._make_searcher()
        result = searcher._parse_hits([])
        assert result == []

    def test_parse_tags(self):
        """tags 字段正确解析。"""
        searcher = self._make_searcher()
        hits = [_make_hit(tags=["tomcat-mes", "production"])]
        result = searcher._parse_hits(hits)
        assert result[0]["tags"] == ["tomcat-mes", "production"]


# ── 搜索集成测试 ──────────────────────────────────────────────────────

class TestEsLogSearcherSearch:
    """search() 方法测试，mock urllib.request.urlopen。"""

    def _make_searcher(self, **overrides):
        config = {"elasticsearch_url": "http://localhost:9200", "index_prefix": "39qjmes"}
        config.update(overrides)
        return EsLogSearcher(config)

    @patch("scripts.es_log_search.urllib.request.urlopen")
    def test_search_returns_logs(self, mock_urlopen):
        """正常搜索返回日志列表。"""
        resp_data = _make_es_response(hits=[
            _make_hit(),
            _make_hit(timestamp="2026-06-05T06:00:00.000Z", message="second log"),
        ])
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        searcher = self._make_searcher()
        result = searcher.search(
            host_name="39QEMES-Tomcat-Crontab01",
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-05T23:59:59Z",
        )
        assert "logs" in result
        assert len(result["logs"]) == 2
        assert result["total"] == 2
        assert "error" not in result

    @patch("scripts.es_log_search.urllib.request.urlopen")
    def test_search_empty_result(self, mock_urlopen):
        """空结果返回空列表。"""
        resp_data = _make_es_response(hits=[])
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        searcher = self._make_searcher()
        result = searcher.search(
            host_name="39QEMES-Tomcat-Crontab01",
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-05T23:59:59Z",
        )
        assert result["logs"] == []
        assert result["total"] == 0

    @patch("scripts.es_log_search.urllib.request.urlopen")
    def test_search_connection_error(self, mock_urlopen):
        """连接异常返回 error 字典，不抛异常。"""
        mock_urlopen.side_effect = ConnectionError("Connection refused")

        searcher = self._make_searcher()
        result = searcher.search(
            host_name="39QEMES-Tomcat-Crontab01",
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-05T23:59:59Z",
        )
        assert "error" in result
        assert "logs" in result
        assert result["logs"] == []

    @patch("scripts.es_log_search.urllib.request.urlopen")
    def test_search_with_keyword(self, mock_urlopen):
        """带关键字的搜索。"""
        resp_data = _make_es_response(hits=[_make_hit()])
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        searcher = self._make_searcher()
        result = searcher.search(
            host_name="39QEMES-Tomcat-Crontab01",
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-05T23:59:59Z",
            keyword="采集项重复",
        )
        # 验证请求体包含关键字
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        must = body["query"]["bool"]["must"]
        assert any("query_string" in m for m in must)

    @patch("scripts.es_log_search.urllib.request.urlopen")
    def test_search_with_size(self, mock_urlopen):
        """自定义 size 参数。"""
        resp_data = _make_es_response(hits=[_make_hit()])
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        searcher = self._make_searcher()
        searcher.search(
            host_name="39QEMES-Tomcat-Crontab01",
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-05T23:59:59Z",
            size=50,
        )
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        assert body["size"] == 50

    @patch("scripts.es_log_search.urllib.request.urlopen")
    def test_search_default_index(self, mock_urlopen):
        """默认索引前缀为 39qjmes。"""
        resp_data = _make_es_response(hits=[])
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        searcher = self._make_searcher()
        searcher.search(
            host_name="39QEMES-Tomcat-Crontab01",
            start_time="2026-06-05T00:00:00Z",
            end_time="2026-06-05T23:59:59Z",
        )
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "39qjmes-2026.06.05" in req.full_url


# ── 配置默认值测试 ──────────────────────────────────────────────────────

class TestEsLogSearchConfigFromThresholds:
    """验证 ES 搜索器从 config 读取默认值。"""

    def test_default_es_url_from_thresholds(self):
        from config.default_thresholds import DEFAULT_THRESHOLDS
        expected = DEFAULT_THRESHOLDS["elk"]["elasticsearch_url"]
        assert expected == "http://localhost:9200"

    def test_default_index_prefix_from_thresholds(self):
        from config.default_thresholds import DEFAULT_THRESHOLDS
        expected = DEFAULT_THRESHOLDS["debug"]["es_index_prefix"]
        assert expected == "39qjmes"

