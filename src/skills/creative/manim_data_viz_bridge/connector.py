"""
Orchestrates chart rendering and Manim script emission (stateless).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .chart_engine import render_chart_png
from .interface import DataVizBridgeProtocol
from .manim_emitter import emit_scene_script
from .models import DataVizBridgeRequest, DataVizBridgeResult


class ManimDataVizBridge(DataVizBridgeProtocol):
    """Concrete bridge implementation."""

    CHART_NAME = "bridge_chart.png"
    SCRIPT_NAME = "bridge_scene.py"

    def build(self, request: DataVizBridgeRequest) -> DataVizBridgeResult:
        work = Path(tempfile.mkdtemp(prefix="manim_dviz_"))
        chart_path = work / self.CHART_NAME
        render_chart_png(request.viz, chart_path)
        source = emit_scene_script(request, self.CHART_NAME)
        script_path = work / self.SCRIPT_NAME
        script_path.write_text(source, encoding="utf-8")
        q = request.manim.resolution
        cls = request.manim.scene_class_name
        cmd = f"manim -p{q} {self.SCRIPT_NAME} {cls}"
        return DataVizBridgeResult(
            work_dir=work,
            chart_path=chart_path,
            manim_script_path=script_path,
            manim_source=source,
            render_suggestion=cmd,
        )
