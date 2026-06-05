"""上游节点存活检查器单元测试。"""

import json
import pytest
from unittest.mock import patch, MagicMock
from scripts.base_checker import ExitCode


JSON_STATUS = json.dumps({
    "mes-backend": [
        {"server": "10.0.0.1:8080", "status": "up"},
        {"server": "10.0.0.2:8080", "status": "up"},
    ]
})

JSON_STATUS_ONE_DOWN = json.dumps({
    "mes-backend": [
        {"server": "10.0.0.1:8080", "status": "up"},
        {"server": "10.0.0.2:8080", "status": "down"},
    ]
})

TEXT_STATUS = """upstream mes-backend
    server 10.0.0.1:8080 up
    server 10.0.0.2:8080 up
"""

TEXT_STATUS_ONE_DOWN = """upstream mes-backend
    server 10.0.0.1:8080 up
    server 10.0.0.2:8080 down
"""


def _mock_urlopen(body: str, url_filter: str = ""):
    def factory(req, timeout=5):
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = body.encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp
    return factory


class TestUpstreamChecker:
    def _make_checker(self, config=None):
        from scripts.upstream_check import UpstreamChecker
        return UpstreamChecker(config or {
            "status_url": "http://localhost/upstream_status?format=json",
            "timeout": 5,
        })

    @patch("scripts.upstream_check.urllib.request.urlopen")
    def test_all_nodes_up_json(self, mock_urlopen):
        mock_urlopen.side_effect = _mock_urlopen(JSON_STATUS)
        checker = self._make_checker()
        report = checker.check()
        assert report.status == ExitCode.NORMAL
        assert len(report.checks) == 2
        assert report.metadata["node_count"] == 2
        assert "全部正常" in report.summary

    @patch("scripts.upstream_check.urllib.request.urlopen")
    def test_one_node_down_json(self, mock_urlopen):
        mock_urlopen.side_effect = _mock_urlopen(JSON_STATUS_ONE_DOWN)
        checker = self._make_checker()
        report = checker.check()
        assert report.status == ExitCode.CRITICAL
        assert report.metadata["nodes"]["10.0.0.1:8080"] == "NORMAL"
        assert report.metadata["nodes"]["10.0.0.2:8080"] == "CRITICAL"
        assert "1 个不可用" in report.summary

    @patch("scripts.upstream_check.urllib.request.urlopen")
    def test_all_nodes_up_text(self, mock_urlopen):
        mock_urlopen.side_effect = _mock_urlopen(TEXT_STATUS)
        checker = self._make_checker({"status_url": "http://localhost/upstream_status", "timeout": 5})
        report = checker.check()
        assert report.status == ExitCode.NORMAL
        assert len(report.checks) == 2

    @patch("scripts.upstream_check.urllib.request.urlopen")
    def test_one_node_down_text(self, mock_urlopen):
        mock_urlopen.side_effect = _mock_urlopen(TEXT_STATUS_ONE_DOWN)
        checker = self._make_checker({"status_url": "http://localhost/upstream_status", "timeout": 5})
        report = checker.check()
        assert report.status == ExitCode.CRITICAL
        assert report.metadata["nodes"]["10.0.0.2:8080"] == "CRITICAL"

    @patch("scripts.upstream_check.urllib.request.urlopen")
    def test_multiple_upstreams(self, mock_urlopen):
        multi = json.dumps({
            "web-backend": [
                {"server": "10.0.0.1:8080", "status": "up"},
            ],
            "api-backend": [
                {"server": "10.0.1.1:9090", "status": "down"},
                {"server": "10.0.1.2:9090", "status": "up"},
            ],
        })
        mock_urlopen.side_effect = _mock_urlopen(multi)
        checker = self._make_checker()
        report = checker.check()
        assert report.status == ExitCode.CRITICAL
        assert len(report.checks) == 3
        assert report.metadata["nodes"]["10.0.0.1:8080"] == "NORMAL"
        assert report.metadata["nodes"]["10.0.1.1:9090"] == "CRITICAL"
        assert report.metadata["nodes"]["10.0.1.2:9090"] == "NORMAL"

    def test_no_status_url_returns_critical(self):
        checker = self._make_checker({"timeout": 5})
        report = checker.check()
        assert report.status == ExitCode.CRITICAL
        assert "未配置" in report.checks[0].message

    @patch("scripts.upstream_check.urllib.request.urlopen")
    def test_connection_error_returns_critical(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError("Connection refused")
        checker = self._make_checker()
        report = checker.check()
        assert report.status == ExitCode.CRITICAL
        assert "无法获取" in report.checks[0].message

    @patch("scripts.upstream_check.urllib.request.urlopen")
    def test_empty_response_returns_warning(self, mock_urlopen):
        mock_urlopen.side_effect = _mock_urlopen("{}")
        checker = self._make_checker()
        report = checker.check()
        assert report.status == ExitCode.WARNING
        assert "无" in report.summary

    def test_component_name(self):
        checker = self._make_checker()
        assert checker.component_name == "upstream"

    @patch("scripts.upstream_check.urllib.request.urlopen")
    def test_check_result_contains_upstream_name(self, mock_urlopen):
        mock_urlopen.side_effect = _mock_urlopen(JSON_STATUS)
        checker = self._make_checker()
        report = checker.check()
        for c in report.checks:
            assert "upstream" in c.details
            assert c.details["upstream"] == "mes-backend"
