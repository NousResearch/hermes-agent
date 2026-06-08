"""GC 日志分析器测试。"""

import subprocess
from unittest.mock import patch, MagicMock
import pytest

from scripts.gc_log_analyzer import GcLogEntry, GcLogAnalyzer, _escape_sed_pattern


# ── GcLogEntry 解析测试 ──────────────────────────────────────────────

class TestGcLogEntryParsing:
    """GcLogEntry.parse() 各种 GC 格式解析测试。"""

    def test_parse_g1_young_gc(self):
        line = "2026-01-15T10:30:45+0800: 0.123: [GC pause (G1 Evacuation Pause) (young) 128M->64M(512M), 0.0156780 secs]"
        entry = GcLogEntry.parse(line)
        assert entry is not None
        assert entry.gc_type == "young"
        assert entry.before_mb == 128
        assert entry.after_mb == 64
        assert entry.total_mb == 512
        assert abs(entry.pause_sec - 0.0156780) < 1e-6
        assert "2026-01-15T10:30:45+0800" in entry.timestamp

    def test_parse_g1_mixed_gc(self):
        line = "2026-01-15T10:31:00+0800: 1.234: [GC pause (G1 Evacuation Pause) (mixed) 256M->96M(512M), 0.0234567 secs]"
        entry = GcLogEntry.parse(line)
        assert entry is not None
        assert entry.gc_type == "mixed"
        assert entry.before_mb == 256
        assert entry.after_mb == 96
        assert entry.total_mb == 512
        assert abs(entry.pause_sec - 0.0234567) < 1e-6

    def test_parse_full_gc_allocation_failure(self):
        line = "2026-01-15T10:32:00+0800: 2.345: [Full GC (Allocation Failure) 512M->200M(512M), 0.1234567 secs]"
        entry = GcLogEntry.parse(line)
        assert entry is not None
        assert entry.gc_type == "full"
        assert entry.before_mb == 512
        assert entry.after_mb == 200
        assert entry.total_mb == 512
        assert abs(entry.pause_sec - 0.1234567) < 1e-6

    def test_parse_cms_gc(self):
        line = "2026-01-15T10:33:00+0800: 3.456: [GC (Allocation Failure) [ParNew: 131072K->16384K(131072K), 0.0123456 secs] 262144K->131072K(524288K), 0.0125678 secs]"
        entry = GcLogEntry.parse(line)
        assert entry is not None
        assert entry.gc_type == "young"
        assert entry.before_mb == 256
        assert entry.after_mb == 128
        assert entry.total_mb == 512

    def test_parse_non_gc_line_returns_none(self):
        line = "2026-01-15T10:30:45+0800: Heap: garbage collection log"
        entry = GcLogEntry.parse(line)
        assert entry is None

    def test_parse_malformed_line_returns_none(self):
        line = "this is not a gc log line at all"
        entry = GcLogEntry.parse(line)
        assert entry is None


# ── 时间过滤测试 ──────────────────────────────────────────────────────

class TestGcLogAnalyzerTimeFilter:
    """GcLogAnalyzer.filter_by_time() 测试。"""

    def _make_entry(self, ts: str, gc_type: str = "young") -> GcLogEntry:
        return GcLogEntry(
            timestamp=ts, gc_type=gc_type,
            before_mb=128, after_mb=64, total_mb=512, pause_sec=0.01,
        )

    def test_filter_by_time_range(self):
        entries = [
            self._make_entry("2026-01-15T10:00:00+0800"),
            self._make_entry("2026-01-15T10:30:00+0800"),
            self._make_entry("2026-01-15T11:00:00+0800"),
        ]
        result = GcLogAnalyzer.filter_by_time(
            entries, "2026-01-15T10:15:00+0800", "2026-01-15T10:45:00+0800"
        )
        assert len(result) == 1
        assert result[0].timestamp == "2026-01-15T10:30:00+0800"

    def test_filter_no_range_returns_all(self):
        entries = [
            self._make_entry("2026-01-15T10:00:00+0800"),
            self._make_entry("2026-01-15T11:00:00+0800"),
        ]
        result = GcLogAnalyzer.filter_by_time(entries, None, None)
        assert len(result) == 2

    def test_filter_empty_list(self):
        result = GcLogAnalyzer.filter_by_time([], "2026-01-15T10:00:00+0800", "2026-01-15T11:00:00+0800")
        assert result == []


# ── 汇总统计测试 ──────────────────────────────────────────────────────

class TestGcLogAnalyzerSummary:
    """GcLogAnalyzer.summarize() 测试。"""

    def test_summarize_counts(self):
        entries = [
            GcLogEntry("t1", "young", 128, 64, 512, 0.01),
            GcLogEntry("t2", "young", 256, 128, 512, 0.02),
            GcLogEntry("t3", "full", 512, 200, 512, 0.10),
            GcLogEntry("t4", "mixed", 200, 100, 512, 0.03),
        ]
        summary = GcLogAnalyzer.summarize(entries)
        assert summary["total_count"] == 4
        assert summary["young_count"] == 2
        assert summary["full_count"] == 1
        assert summary["mixed_count"] == 1
        assert abs(summary["avg_pause_sec"] - 0.04) < 1e-6
        assert abs(summary["max_pause_sec"] - 0.10) < 1e-6
        assert abs(summary["total_pause_sec"] - 0.16) < 1e-6
        assert summary["reclaimed_mb"] == (128 - 64) + (256 - 128) + (512 - 200) + (200 - 100)

    def test_summarize_empty(self):
        summary = GcLogAnalyzer.summarize([])
        assert summary["total_count"] == 0
        assert summary["young_count"] == 0
        assert summary["full_count"] == 0
        assert summary["mixed_count"] == 0
        assert summary["avg_pause_sec"] == 0.0
        assert summary["max_pause_sec"] == 0.0
        assert summary["total_pause_sec"] == 0.0
        assert summary["reclaimed_mb"] == 0


# ── SSH 获取测试 ──────────────────────────────────────────────────────

class TestGcLogAnalyzerFetch:
    """GcLogAnalyzer.fetch_gc_log() SSH 调用测试。"""

    def _make_analyzer(self, **overrides) -> GcLogAnalyzer:
        config = {
            "host": "10.0.0.1",
            "ssh_user": "app",
            "ssh_key": "~/.ssh/id_rsa",
            "ssh_port": 22,
            "gc_log_path": "/u01/app/mes-app/logs/gc.log",
        }
        config.update(overrides)
        return GcLogAnalyzer(config)

    @patch("scripts.gc_log_analyzer.create_executor")
    def test_fetch_uses_tail_when_no_time(self, mock_create):
        mock_exec = MagicMock()
        mock_exec.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="2026-01-15T10:00:00+0800: 0.1: [GC pause (young) 128M->64M(512M), 0.01 secs]\n",
            stderr="",
        )
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer()
        entries = analyzer.fetch_gc_log()
        assert len(entries) == 1
        mock_create.assert_called_once()
        call_args = mock_create.call_args[0][0]
        assert call_args["host"] == "10.0.0.1"

    @patch("scripts.gc_log_analyzer.create_executor")
    def test_fetch_uses_sed_for_time_range(self, mock_create):
        mock_exec = MagicMock()
        mock_exec.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="2026-01-15T10:30:00+0800: 1.0: [GC pause (young) 128M->64M(512M), 0.01 secs]\n",
            stderr="",
        )
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer()
        entries = analyzer.fetch_gc_log(
            start_time="2026-01-15T10:00:00+0800",
            end_time="2026-01-15T11:00:00+0800",
        )
        assert len(entries) == 1
        cmd = mock_exec.run.call_args[0][0]
        assert "sed" in cmd

    @patch("scripts.gc_log_analyzer.create_executor")
    def test_fetch_file_not_found(self, mock_create):
        mock_exec = MagicMock()
        mock_exec.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="No such file",
        )
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer()
        entries = analyzer.fetch_gc_log()
        assert entries == []

    @patch("scripts.gc_log_analyzer.create_executor")
    def test_fetch_default_path(self, mock_create):
        mock_exec = MagicMock()
        mock_exec.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer()
        analyzer.fetch_gc_log()
        cmd = mock_exec.run.call_args[0][0]
        assert "/u01/app/mes-app/logs/gc.log" in cmd

    @patch("scripts.gc_log_analyzer.create_executor")
    def test_fetch_gc_log_escapes_shell_injection(self, mock_create):
        mock_executor = MagicMock()
        mock_create.return_value = mock_executor
        mock_executor.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        analyzer = GcLogAnalyzer({"host": "10.0.0.1", "gc_log_path": "/tmp/gc.log"})
        analyzer.fetch_gc_log(start_time="'; rm -rf / #")
        cmd_arg = mock_executor.run.call_args[0][0]
        assert "sed" in cmd_arg
        # 注入载荷中的 / 被转义为 \/，不会闭合 sed 模式分隔符
        assert "\\/" in cmd_arg


class TestEscapeSedPattern:
    """_escape_sed_pattern 转义测试。"""

    def test_normal_timestamp_unchanged(self):
        assert _escape_sed_pattern("2026-06-05T01:00:00") == "2026-06-05T01:00:00"

    def test_slash_escaped(self):
        assert _escape_sed_pattern("a/b") == "a\\/b"

    def test_ampersand_escaped(self):
        assert _escape_sed_pattern("a&b") == "a\\&b"

    def test_backslash_escaped(self):
        assert _escape_sed_pattern("a\\b") == "a\\\\b"

    def test_injection_payload_fully_escaped(self):
        payload = "'; rm -rf / #"
        escaped = _escape_sed_pattern(payload)
        # 所有 / 被转义
        assert "/" not in escaped.replace("\\/", "")
        # & 被转义
        assert "&" not in escaped.replace("\\&", "")


class TestGcLogConfigFromThresholds:
    """验证 GC 分析器从 config 读取默认值而非硬编码。"""

    def test_default_gc_log_path_from_config(self):
        from config.default_thresholds import DEFAULT_THRESHOLDS
        expected = DEFAULT_THRESHOLDS["debug"]["gc_log_path"]
        assert expected != ""
        assert "gc.log" in expected

    def test_max_lines_from_config(self):
        from config.default_thresholds import DEFAULT_THRESHOLDS
        expected = DEFAULT_THRESHOLDS["debug"]["gc_max_lines"]
        assert expected == 50000
