"""GC 日志分析器 — 通过 SSH 获取远程 JVM GC 日志并解析指标。"""

import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.ssh_executor import create_executor

def _get_default(key: str):
    """从 DEFAULT_THRESHOLDS 读取默认值。"""
    from config.default_thresholds import DEFAULT_THRESHOLDS
    return DEFAULT_THRESHOLDS.get("debug", {}).get(key)

# G1/Mixed/Full GC: [GC pause ... (young|mixed) NNNM->NNNM(NNNM), N.NNN secs]
# Full GC: [Full GC ... NNNM->NNNM(NNNM), N.NNN secs]
# CMS: [GC ... [ParNew: NNNK->NNNK(NNNK), N.NNN secs] NNNK->NNNK(NNNK), N.NNN secs]
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+\-]\d{4}):")
_GC_RE = re.compile(
    r"\[(?:GC pause|Full GC|GC)\b"       # GC 类型起始
    r".*?"                                 # 中间描述
    r"(?:\(young\)|\(mixed\))?"            # 可选 young/mixed 标记
    r".*?"
    r"(\d+)([MK])->(\d+)([MK])\((\d+)([MK])\)"  # 内存: before->after(total)
    r".*?(\d+\.\d+)\s+secs\]"             # 暂停时间
)
_FULL_GC_RE = re.compile(
    r"\[Full GC\b"
    r".*?"
    r"(\d+)([MK])->(\d+)([MK])\((\d+)([MK])\)"
    r".*?(\d+\.\d+)\s+secs\]"
)
_CMS_RE = re.compile(
    r"\[GC\b.*?\[.*?(\d+)([MK])->(\d+)([MK])\((\d+)([MK])\).*?\]"
    r".*?(\d+)([MK])->(\d+)([MK])\((\d+)([MK])\)"
    r".*?(\d+\.\d+)\s+secs\]"
)


def _escape_sed_pattern(s: str) -> str:
    """转义 sed 模式中的特殊字符。"""
    return s.replace("\\", "\\\\").replace("/", "\\/").replace("&", "\\&")


def _to_mb(value: int, unit: str) -> int:
    return value if unit == "M" else value // 1024


@dataclass
class GcLogEntry:
    """单条 GC 日志记录。"""
    timestamp: str
    gc_type: str      # young / mixed / full
    before_mb: int
    after_mb: int
    total_mb: int
    pause_sec: float

    @staticmethod
    def parse(line: str) -> Optional["GcLogEntry"]:
        """解析单行 GC 日志，非 GC 行返回 None。"""
        ts_m = _TS_RE.match(line)
        if not ts_m:
            return None
        timestamp = ts_m.group(1)

        # Full GC 优先匹配
        m = _FULL_GC_RE.search(line)
        if m and "Full GC" in line:
            before = _to_mb(int(m.group(1)), m.group(2))
            after = _to_mb(int(m.group(3)), m.group(4))
            total = _to_mb(int(m.group(5)), m.group(6))
            pause = float(m.group(7))
            return GcLogEntry(timestamp, "full", before, after, total, pause)

        # CMS GC（含 ParNew 子串）
        if "ParNew" in line:
            m = _CMS_RE.search(line)
            if m:
                before = _to_mb(int(m.group(7)), m.group(8))
                after = _to_mb(int(m.group(9)), m.group(10))
                total = _to_mb(int(m.group(11)), m.group(12))
                pause = float(m.group(13))
                return GcLogEntry(timestamp, "young", before, after, total, pause)

        # G1 GC (young / mixed)
        m = _GC_RE.search(line)
        if m:
            before = _to_mb(int(m.group(1)), m.group(2))
            after = _to_mb(int(m.group(3)), m.group(4))
            total = _to_mb(int(m.group(5)), m.group(6))
            pause = float(m.group(7))
            gc_type = "mixed" if "(mixed)" in line else "young"
            return GcLogEntry(timestamp, gc_type, before, after, total, pause)

        return None


class GcLogAnalyzer:
    """GC 日志分析器 — 通过 SSH 获取并解析远程 JVM GC 日志。"""

    SSH_TIMEOUT = 60

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def fetch_gc_log(
        self, start_time: str = None, end_time: str = None
    ) -> List[GcLogEntry]:
        """通过 SSH 获取 GC 日志并解析为条目列表。"""
        executor = create_executor(self.config)
        log_path = self.config.get("gc_log_path") or _get_default("gc_log_path") or "/u01/app/mes-app/logs/gc.log"
        cmd = self._build_fetch_cmd(log_path, start_time, end_time)
        result = executor.run(cmd, timeout=self.SSH_TIMEOUT)
        if result.returncode != 0:
            return []
        return self._parse_lines(result.stdout)

    def _build_fetch_cmd(
        self, log_path: str, start_time: str = None, end_time: str = None
    ) -> str:
        # log_path 使用 shlex.quote（shell 顶层参数）；
        # start_time/end_time 使用 _escape_sed_pattern（sed /pattern/ 内部）
        if start_time and end_time:
            return f"sed -n '/{_escape_sed_pattern(start_time)}/,/{_escape_sed_pattern(end_time)}/p' {shlex.quote(log_path)}"
        if start_time:
            return f"sed -n '/{_escape_sed_pattern(start_time)}/,$p' {shlex.quote(log_path)}"
        max_lines = self.config.get("gc_max_lines") or _get_default("gc_max_lines") or 50000
        return f"tail -n {max_lines} {shlex.quote(log_path)}"

    @staticmethod
    def _parse_lines(raw: str) -> List[GcLogEntry]:
        entries = []
        for line in raw.splitlines():
            entry = GcLogEntry.parse(line.strip())
            if entry:
                entries.append(entry)
        return entries

    @staticmethod
    def filter_by_time(
        entries: List[GcLogEntry],
        start_time: str = None,
        end_time: str = None,
    ) -> List[GcLogEntry]:
        """按时间范围精确过滤。"""
        if not start_time and not end_time:
            return entries
        result = []
        for e in entries:
            if start_time and e.timestamp < start_time:
                continue
            if end_time and e.timestamp > end_time:
                continue
            result.append(e)
        return result

    @staticmethod
    def summarize(entries: List[GcLogEntry]) -> Dict[str, Any]:
        """汇总 GC 统计指标。"""
        if not entries:
            return {
                "total_count": 0, "young_count": 0, "full_count": 0,
                "mixed_count": 0, "avg_pause_sec": 0.0, "max_pause_sec": 0.0,
                "total_pause_sec": 0.0, "reclaimed_mb": 0,
            }
        pauses = [e.pause_sec for e in entries]
        return {
            "total_count": len(entries),
            "young_count": sum(1 for e in entries if e.gc_type == "young"),
            "full_count": sum(1 for e in entries if e.gc_type == "full"),
            "mixed_count": sum(1 for e in entries if e.gc_type == "mixed"),
            "avg_pause_sec": sum(pauses) / len(pauses),
            "max_pause_sec": max(pauses),
            "total_pause_sec": sum(pauses),
            "reclaimed_mb": sum(e.before_mb - e.after_mb for e in entries),
        }

    def analyze(
        self, start_time: str = None, end_time: str = None
    ) -> Dict[str, Any]:
        """完整分析流程：获取→过滤→汇总。"""
        entries = self.fetch_gc_log(start_time, end_time)
        # sed 做粗筛（远程减少传输量），filter_by_time 做本地精确过滤（更可靠）
        entries = self.filter_by_time(entries, start_time, end_time)
        return self.summarize(entries)
