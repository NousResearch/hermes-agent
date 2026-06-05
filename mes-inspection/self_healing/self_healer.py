"""自愈执行器。"""

import json
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from self_healing.decision_matrix import DecisionMatrix, HealLevel, ScoreResult


class SelfHealer:
    """自愈执行器。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.matrix = DecisionMatrix(config)
        self.enabled = config.get("enabled", True)

    def process_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """处理巡检报告，决定并执行自愈动作。

        返回自愈结果摘要。
        """
        if not self.enabled:
            return {"healed": False, "reason": "自愈功能已禁用"}

        component = report.get("component", "")
        status_code = report.get("status_code", 0)

        if status_code == 0:
            return {"healed": False, "reason": "无异常"}

        # 评分
        scores = self.matrix.score_from_report(report)
        if not scores:
            return {"healed": False, "reason": "无法识别故障场景"}

        # 取最高级别的故障处理
        results = []
        for fault_type, score_result in scores.items():
            result = self._execute_heal(component, fault_type, score_result, report)
            results.append(result)

        return {
            "healed": any(r.get("executed") for r in results),
            "faults": results,
            "timestamp": datetime.now().isoformat(),
        }

    def _execute_heal(
        self,
        component: str,
        fault_type: str,
        score: ScoreResult,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行单个故障的自愈。"""
        result = {
            "component": component,
            "fault_type": fault_type,
            "level": score.level.value,
            "score": score.total_score,
            "actions": score.recommended_actions,
            "executed": False,
            "output": "",
            "success": False,
        }

        if score.level == HealLevel.L1:
            # L1 全自动执行
            for action in score.recommended_actions:
                try:
                    proc = subprocess.run(
                        action, shell=True, capture_output=True, text=True, timeout=60
                    )
                    result["executed"] = True
                    result["output"] = proc.stdout + proc.stderr
                    result["success"] = proc.returncode == 0
                except Exception as e:
                    result["output"] = str(e)
                    result["success"] = False
        elif score.level == HealLevel.L2:
            # L2 输出方案，等待人工确认（在 Hermes 中通过飞书推送）
            result["executed"] = False
            result["output"] = "L2 半自动：需人工确认后执行"
        else:
            # L3 人工处理
            result["executed"] = False
            result["output"] = "L3 人工处理：需运维人员介入"

        return result

    def format_heal_report(self, heal_result: Dict[str, Any]) -> str:
        """格式化自愈结果为可读文本。"""
        if not heal_result.get("healed") and not heal_result.get("faults"):
            return ""

        lines = [
            "🔧 自愈执行结果",
            "━" * 20,
        ]

        for fault in heal_result.get("faults", []):
            level = fault.get("level", "?")
            icon = {"L1": "🤖", "L2": "👤", "L3": "🆘"}.get(level, "?")
            lines.append(f"{icon} [{level}] {fault.get('component')}/{fault.get('fault_type')}")
            lines.append(f"   评分: {fault.get('score')}")
            if fault.get("executed"):
                status = "✅ 成功" if fault.get("success") else "❌ 失败"
                lines.append(f"   执行: {status}")
                if fault.get("output"):
                    lines.append(f"   输出: {fault['output'][:200]}")
            else:
                lines.append(f"   状态: 等待人工确认")
            if fault.get("actions"):
                lines.append(f"   建议: {', '.join(fault['actions'])}")
            lines.append("")

        return "\n".join(lines)
