"""MES 进化工具 — 注册到 Hermes Agent 的工具系统。"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from evolution.evolution_runner import EvolutionRunner, MES_SKILLS


EVOLUTION_SCHEMA = {
    "name": "mes_evolution",
    "description": "MES 巡检技能进化工具 — 使用 DSPy + GEPA 自动优化巡检 Skills。支持单个进化和批量进化。",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "evolve", "evolve_all"],
                "description": "操作类型: list(列出可进化技能), evolve(进化单个), evolve_all(批量进化)"
            },
            "skill_name": {
                "type": "string",
                "description": "evolve 操作必需。技能名称，如 mes-nginx-check"
            },
            "dry_run": {
                "type": "boolean",
                "description": "仅验证配置，不执行优化",
                "default": False,
            },
        },
        "required": ["action"],
    },
}


def _load_config() -> Dict[str, Any]:
    try:
        from config.config_manager import ConfigManager
        cm = ConfigManager()
        cm.load()
        return cm.get_component_config("evolution")
    except ImportError:
        return {}


def mes_evolution(action: str = "list", skill_name: Optional[str] = None, dry_run: bool = False, task_id: Optional[str] = None) -> str:
    config = _load_config()

    if action == "list":
        return json.dumps({"success": True, "skills": MES_SKILLS, "count": len(MES_SKILLS)}, ensure_ascii=False)

    if action == "evolve":
        if not skill_name:
            return json.dumps({"success": False, "error": "evolve 操作需要 skill_name 参数"}, ensure_ascii=False)
        if skill_name not in MES_SKILLS:
            return json.dumps({"success": False, "error": f"未知技能: {skill_name}。可选: {', '.join(MES_SKILLS)}"}, ensure_ascii=False)
        runner = EvolutionRunner(config)
        result = runner.evolve_skill(skill_name, dry_run=dry_run)
        return json.dumps(result, ensure_ascii=False)

    if action == "evolve_all":
        runner = EvolutionRunner(config)
        results = runner.evolve_all(dry_run=dry_run)
        report = runner.format_report(results)
        return json.dumps({"success": any(r["success"] for r in results), "results": results, "report": report}, ensure_ascii=False)

    return json.dumps({"success": False, "error": f"未知操作: {action}"}, ensure_ascii=False)


def check_evolution_requirements() -> bool:
    vendor_path = Path(__file__).parent.parent / "vendor" / "hermes-agent-self-evolution"
    return vendor_path.exists()


def register_evolution_tool():
    try:
        from tools.registry import registry
        registry.register(
            name="mes_evolution", toolset="mes_inspection", schema=EVOLUTION_SCHEMA,
            handler=lambda args, **kw: mes_evolution(action=args.get("action", "list"), skill_name=args.get("skill_name"), dry_run=args.get("dry_run", False), task_id=kw.get("task_id")),
            check_fn=check_evolution_requirements, emoji="🧬",
        )
    except ImportError:
        pass
