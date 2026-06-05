"""心跳巡检入口 — Nginx + JVM，高频检测（2 分钟）。"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.inspection_runner import run_inspections, output_gate_result, HEARTBEAT_COMPONENTS

if __name__ == "__main__":
    result = run_inspections(HEARTBEAT_COMPONENTS)
    output_gate_result(result)
    sys.exit(0 if not result["abnormal"] else 2)
