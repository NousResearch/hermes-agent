"""统一巡检执行器 — 运行多个 checker，输出 wakeAgent 门控格式。

输出规范：
- 全部正常：最后一行 {"wakeAgent": false}
- 有异常：异常报告 + 最后一行 {"wakeAgent": true}

异常报告格式：
===INSPECTION_REPORT===
{"component":"nginx","status":"CRITICAL","checks":[...]}
===END===
{"wakeAgent": true}
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.base_checker import BaseChecker, ExitCode, InspectionReport


def load_component_config(component: str) -> Dict[str, Any]:
    """加载单个组件的配置。"""
    from config.config_manager import ConfigManager
    cm = ConfigManager()
    cm.load()
    return cm.get_component_config(component)


def create_checker(component: str) -> BaseChecker:
    """根据组件名创建对应的 checker 实例。"""
    config = load_component_config(component)

    checker_map = {
        "nginx": "scripts.nginx_check.NginxChecker",
        "jvm": "scripts.jvm_check.JvmChecker",
        "rabbitmq": "scripts.rabbitmq_check.RabbitMQChecker",
        "oracle": "scripts.oracle_check.OracleChecker",
        "elk": "scripts.elk_check.ElkChecker",
        "skywalking": "scripts.skywalking_check.SkyWalkingChecker",
        "upstream": "scripts.upstream_check.UpstreamChecker",
    }

    if component not in checker_map:
        raise ValueError(f"未知组件: {component}，可选: {list(checker_map.keys())}")

    module_path, class_name = checker_map[component].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    checker_cls = getattr(module, class_name)
    return checker_cls(config)


def run_inspections(components: List[str]) -> Dict[str, Any]:
    """运行多个组件的巡检，返回汇总结果。"""
    reports = []
    summary = {
        "total": len(components),
        "normal": 0, "warning": 0, "critical": 0,
        "components": {},
        "nodes": {},
    }

    for component in components:
        try:
            checker = create_checker(component)
            report = checker.check()
            status_name = report.status.name
            summary["components"][component] = status_name

            # 按节点统计
            node_statuses = report.metadata.get("nodes", {})
            for node_name, node_status in node_statuses.items():
                if node_name not in summary["nodes"]:
                    summary["nodes"][node_name] = {}
                summary["nodes"][node_name][component] = node_status

            if report.status == ExitCode.NORMAL:
                summary["normal"] += 1
            elif report.status == ExitCode.WARNING:
                summary["warning"] += 1
            else:
                summary["critical"] += 1

            if report.status != ExitCode.NORMAL:
                reports.append(report.to_dict())
        except Exception as e:
            summary["critical"] += 1
            summary["components"][component] = "ERROR"
            reports.append({
                "component": component,
                "status": "CRITICAL",
                "status_code": 2,
                "checks": [],
                "summary": f"巡检执行异常: {e}",
                "metadata": {"error": str(e)},
            })

    abnormal = summary["warning"] > 0 or summary["critical"] > 0
    return {"abnormal": abnormal, "reports": reports, "summary": summary}


def output_gate_result(result: Dict[str, Any]):
    """输出 wakeAgent 门控格式到 stdout。"""
    if not result["abnormal"]:
        print(json.dumps({"wakeAgent": False}, ensure_ascii=False))
        return

    print("===INSPECTION_REPORT===")
    for report in result["reports"]:
        print(json.dumps(report, ensure_ascii=False))
    print("===END===")
    print(json.dumps({"wakeAgent": True}, ensure_ascii=False))


HEARTBEAT_COMPONENTS = ["upstream"]
FULL_CHECK_COMPONENTS = ["nginx", "jvm", "rabbitmq", "oracle", "elk", "skywalking"]
DEEP_ANALYSIS_COMPONENTS = ["oracle", "elk"]
