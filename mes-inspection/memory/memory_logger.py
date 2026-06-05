"""记忆日志器 - 故障案例记录与检索。"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_log_dir() -> Path:
    """获取默认日志目录。"""
    home = os.getenv("MES_INSPECTION_HOME", "")
    if home:
        return Path(home) / "memory" / "cases"
    return Path.home() / ".mes-inspection" / "memory" / "cases"


class MemoryLogger:
    """故障案例记忆日志器。

    存储结构：
    ~/.mes-inspection/memory/cases/
    ├── index.json                    # 索引文件
    ├── 2026-06/
    │   ├── 001_nginx_process.json
    │   ├── 002_jvm_oom.json
    │   └── ...
    └── 2026-07/
        └── ...
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else _default_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.log_dir.parent / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        if self.index_path.exists():
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"cases": [], "next_id": 1}

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

    def log_case(
        self,
        component: str,
        status_code: int,
        summary: str,
        checks: List[Dict[str, Any]],
        root_cause: str = "",
        fix_action: str = "",
        heal_level: str = "",
        elapsed_seconds: float = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """记录故障案例，返回案例 ID。"""
        case_id = self._index["next_id"]
        self._index["next_id"] = case_id + 1

        now = datetime.now()
        month_dir = self.log_dir / now.strftime("%Y-%m")
        month_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{case_id:03d}_{component}_{now.strftime('%Y%m%d_%H%M%S')}.json"
        case_path = month_dir / filename

        failed_checks = [c for c in checks if c.get("status_code", 0) > 0]

        case = {
            "id": case_id,
            "timestamp": now.isoformat(),
            "component": component,
            "status_code": status_code,
            "status": ["NORMAL", "WARNING", "CRITICAL"][min(status_code, 2)],
            "summary": summary,
            "failed_checks": failed_checks,
            "root_cause": root_cause,
            "fix_action": fix_action,
            "heal_level": heal_level,
            "elapsed_seconds": elapsed_seconds,
            "metadata": metadata or {},
        }

        with open(case_path, "w", encoding="utf-8") as f:
            json.dump(case, f, ensure_ascii=False, indent=2)

        # 更新索引
        self._index["cases"].append({
            "id": case_id,
            "timestamp": now.isoformat(),
            "component": component,
            "status_code": status_code,
            "file": str(case_path.relative_to(self.log_dir)),
        })
        self._save_index()

        return f"#{case_id}"

    def search_cases(
        self,
        component: Optional[str] = None,
        fault_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """搜索历史案例。"""
        results = []
        for entry in reversed(self._index["cases"]):
            if component and entry.get("component") != component:
                continue
            case_path = self.log_dir / entry["file"]
            if case_path.exists():
                with open(case_path, "r", encoding="utf-8") as f:
                    case = json.load(f)
                if fault_type:
                    names = [c.get("name", "") for c in case.get("failed_checks", [])]
                    if not any(fault_type.lower() in n.lower() for n in names):
                        continue
                results.append(case)
                if len(results) >= limit:
                    break
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息。"""
        total = len(self._index["cases"])
        by_component = {}
        for entry in self._index["cases"]:
            comp = entry.get("component", "unknown")
            by_component[comp] = by_component.get(comp, 0) + 1
        return {"total_cases": total, "by_component": by_component}

    def format_for_hermes_memory(self, case_id: int) -> str:
        """将案例格式化为 Hermes Memory 条目。"""
        for entry in self._index["cases"]:
            if entry.get("id") == case_id:
                case_path = self.log_dir / entry["file"]
                if case_path.exists():
                    with open(case_path, "r", encoding="utf-8") as f:
                        case = json.load(f)
                    from alerter.formatter import format_memory_entry
                    return format_memory_entry(
                        component=case["component"],
                        status_code=case["status_code"],
                        summary=case["summary"],
                        checks=case.get("failed_checks", []),
                        root_cause=case.get("root_cause", ""),
                        fix_action=case.get("fix_action", ""),
                        elapsed_seconds=case.get("elapsed_seconds", 0),
                    )
        return ""
