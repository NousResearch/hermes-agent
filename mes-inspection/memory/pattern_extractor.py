"""模式提取器 - 从历史案例中归纳故障模式。"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


class PatternExtractor:
    """从历史案例中提取故障模式。"""

    def __init__(self, memory_logger):
        self.logger = memory_logger

    def extract_patterns(self, days: int = 30) -> Dict[str, Any]:
        """提取最近 N 天的故障模式。"""
        cutoff = datetime.now() - timedelta(days=days)
        cases = []
        for entry in self.logger._index["cases"]:
            ts = entry.get("timestamp", "")
            if ts:
                try:
                    case_time = datetime.fromisoformat(ts)
                    if case_time >= cutoff:
                        case_path = self.logger.log_dir / entry["file"]
                        if case_path.exists():
                            with open(case_path, "r", encoding="utf-8") as f:
                                cases.append(json.load(f))
                except (ValueError, FileNotFoundError):
                    continue

        return {
            "component_frequency": self._component_frequency(cases),
            "fault_co_occurrence": self._fault_co_occurrence(cases),
            "peak_hours": self._peak_hours(cases),
            "fix_effectiveness": self._fix_effectiveness(cases),
            "trend": self._trend(cases, days),
        }

    def _component_frequency(self, cases: List[Dict]) -> Dict[str, int]:
        """统计各组件故障频率。"""
        counter = Counter(c.get("component", "unknown") for c in cases)
        return dict(counter.most_common())

    def _fault_co_occurrence(self, cases: List[Dict]) -> List[Tuple[str, str, int]]:
        """分析故障共现关系。"""
        pair_counter = Counter()
        for case in cases:
            checks = case.get("failed_checks", [])
            names = [c.get("name", "") for c in checks if c.get("status_code", 0) > 0]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pair = tuple(sorted([names[i], names[j]]))
                    pair_counter[pair] += 1
        return [(a, b, count) for (a, b), count in pair_counter.most_common(10)]

    def _peak_hours(self, cases: List[Dict]) -> Dict[int, int]:
        """分析故障高发时段。"""
        hour_counter = Counter()
        for case in cases:
            ts = case.get("timestamp", "")
            if ts:
                try:
                    hour = datetime.fromisoformat(ts).hour
                    hour_counter[hour] += 1
                except ValueError:
                    continue
        return dict(sorted(hour_counter.items()))

    def _fix_effectiveness(self, cases: List[Dict]) -> Dict[str, Any]:
        """分析修复有效性。"""
        total = len(cases)
        fixed = sum(1 for c in cases if c.get("fix_action"))
        by_level = Counter(c.get("heal_level", "unknown") for c in cases)
        return {
            "total": total,
            "fixed": fixed,
            "fix_rate": round(fixed / total * 100, 1) if total else 0,
            "by_level": dict(by_level),
        }

    def _trend(self, cases: List[Dict], days: int) -> Dict[str, int]:
        """故障趋势（按天统计）。"""
        day_counter = Counter()
        for case in cases:
            ts = case.get("timestamp", "")
            if ts:
                try:
                    day = datetime.fromisoformat(ts).strftime("%Y-%m-%d")
                    day_counter[day] += 1
                except ValueError:
                    continue
        return dict(sorted(day_counter.items()))

    def format_patterns(self, patterns: Dict[str, Any]) -> str:
        """格式化模式分析结果。"""
        lines = [
            "🧬 故障模式分析报告",
            "━" * 20,
            "",
            "📊 组件故障频率:",
        ]
        for comp, count in patterns.get("component_frequency", {}).items():
            lines.append(f"  {comp}: {count} 次")

        if patterns.get("peak_hours"):
            peak = max(patterns["peak_hours"], key=patterns["peak_hours"].get)
            lines.extend(["", f"⏰ 高发时段: {peak}:00 ({patterns['peak_hours'][peak]} 次)"])

        eff = patterns.get("fix_effectiveness", {})
        if eff:
            lines.extend([
                "",
                f"🔧 修复率: {eff.get('fix_rate', 0)}%",
                f"   总案例: {eff.get('total', 0)}, 已修复: {eff.get('fixed', 0)}",
            ])

        return "\n".join(lines)
