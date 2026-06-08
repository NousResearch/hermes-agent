"""debug_tools.py 的单元测试。"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.debug_tools import DebugToolsCLI


class TestDebugToolsCLI:
    """DebugToolsCLI 参数解析测试。"""

    def test_parse_gc_args(self):
        """gc 子命令参数解析。"""
        cli = DebugToolsCLI()
        args = cli.parse_args([
            "gc",
            "--host", "192.168.1.100",
            "--ssh-user", "admin",
            "--ssh-key", "/path/to/key",
            "--ssh-port", "2222",
            "--gc-log-path", "/var/log/gc.log",
            "--start", "2024-01-01T00:00:00+0800",
            "--end", "2024-01-01T01:00:00+0800",
        ])
        assert args.subcommand == "gc"
        assert args.host == "192.168.1.100"
        assert args.ssh_user == "admin"
        assert args.ssh_key == "/path/to/key"
        assert args.ssh_port == 2222
        assert args.gc_log_path == "/var/log/gc.log"
        assert args.start == "2024-01-01T00:00:00+0800"
        assert args.end == "2024-01-01T01:00:00+0800"

    def test_parse_stack_args(self):
        """stack 子命令参数解析。"""
        cli = DebugToolsCLI()
        args = cli.parse_args([
            "stack",
            "--host", "10.0.0.1",
            "--ssh-user", "root",
            "--pid", "12345",
            "--process-name", "myapp",
            "--keyword", "BLOCKED",
            "--context-lines", "5",
        ])
        assert args.subcommand == "stack"
        assert args.host == "10.0.0.1"
        assert args.ssh_user == "root"
        assert args.pid == 12345
        assert args.process_name == "myapp"
        assert args.keyword == "BLOCKED"
        assert args.context_lines == 5

    def test_parse_log_args(self):
        """log 子命令参数解析。"""
        cli = DebugToolsCLI()
        args = cli.parse_args([
            "log",
            "--es-url", "http://es:9200",
            "--host-name", "web-server-01",
            "--start", "2024-01-01T00:00:00Z",
            "--end", "2024-01-01T23:59:59Z",
            "--keyword", "timeout",
            "--level", "ERROR",
            "--size", "50",
        ])
        assert args.subcommand == "log"
        assert args.es_url == "http://es:9200"
        assert args.host_name == "web-server-01"
        assert args.start == "2024-01-01T00:00:00Z"
        assert args.end == "2024-01-01T23:59:59Z"
        assert args.keyword == "timeout"
        assert args.level == "ERROR"
        assert args.size == 50


class TestDebugToolsCLIRun:
    """DebugToolsCLI.run 方法测试（mock 分析器）。"""

    @patch("scripts.debug_tools.GcLogAnalyzer")
    def test_run_gc(self, mock_gc_cls):
        """run gc 调用 GcLogAnalyzer.analyze。"""
        mock_instance = MagicMock()
        mock_instance.analyze.return_value = {"total_count": 10, "full_count": 2}
        mock_gc_cls.return_value = mock_instance

        cli = DebugToolsCLI()
        result = cli.run([
            "gc",
            "--host", "192.168.1.100",
            "--start", "2024-01-01T00:00:00+0800",
            "--end", "2024-01-01T01:00:00+0800",
        ])

        mock_gc_cls.assert_called_once()
        mock_instance.analyze.assert_called_once_with(
            "2024-01-01T00:00:00+0800", "2024-01-01T01:00:00+0800"
        )
        assert result == {"total_count": 10, "full_count": 2}

    @patch("scripts.debug_tools.ThreadStackAnalyzer")
    def test_run_stack(self, mock_stack_cls):
        """run stack 调用 ThreadStackAnalyzer.analyze。"""
        mock_instance = MagicMock()
        mock_instance.analyze.return_value = {"summary": {"total_threads": 50}}
        mock_stack_cls.return_value = mock_instance

        cli = DebugToolsCLI()
        result = cli.run([
            "stack",
            "--host", "10.0.0.1",
            "--keyword", "BLOCKED",
            "--context-lines", "5",
        ])

        mock_stack_cls.assert_called_once()
        mock_instance.analyze.assert_called_once_with("BLOCKED", 5)
        assert result == {"summary": {"total_threads": 50}}

    @patch("scripts.debug_tools.EsLogSearcher")
    def test_run_log(self, mock_es_cls):
        """run log 调用 EsLogSearcher.search。"""
        mock_instance = MagicMock()
        mock_instance.search.return_value = {"total": 100, "logs": []}
        mock_es_cls.return_value = mock_instance

        cli = DebugToolsCLI()
        result = cli.run([
            "log",
            "--host-name", "web-01",
            "--start", "2024-01-01T00:00:00Z",
            "--end", "2024-01-01T23:59:59Z",
            "--keyword", "error",
            "--level", "ERROR",
            "--size", "200",
        ])

        mock_es_cls.assert_called_once()
        mock_instance.search.assert_called_once_with(
            "web-01", "2024-01-01T00:00:00Z", "2024-01-01T23:59:59Z",
            keyword="error", level="ERROR", size=200,
        )
        assert result == {"total": 100, "logs": []}

    def test_run_missing_subcommand_raises_system_exit(self):
        """无子命令时 sys.exit(1)。"""
        cli = DebugToolsCLI()
        with pytest.raises(SystemExit) as exc_info:
            cli.run([])
        assert exc_info.value.code == 1


class TestDebugToolsConfigDefaults:
    """验证 CLI 默认值从 config 读取。"""

    def test_default_ssh_user_from_config(self):
        from config.default_thresholds import DEFAULT_THRESHOLDS
        assert DEFAULT_THRESHOLDS["ssh"]["default_user"] == "root"

    def test_default_ssh_port_from_config(self):
        from config.default_thresholds import DEFAULT_THRESHOLDS
        assert DEFAULT_THRESHOLDS["ssh"]["default_port"] == 22

    def test_default_process_name_from_config(self):
        from config.default_thresholds import DEFAULT_THRESHOLDS
        assert DEFAULT_THRESHOLDS["jvm"]["tomcat_process_name"] == "catalina"
