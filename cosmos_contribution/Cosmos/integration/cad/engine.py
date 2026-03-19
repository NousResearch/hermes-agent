"""
Cosmos CAD Engine.

CadQuery-based parametric modeling.
"""

import logging
from typing import Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for CadQuery
try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False
    cq = None


class CosmosCAD:
    """
    CAD modeling engine using CadQuery.

    Provides a high-level API for creating 3D models.
    """

    def __init__(self):
        if not CADQUERY_AVAILABLE:
            logger.warning("CadQuery not installed: pip install cadquery")

        self.current_model = None
        self.history: List[str] = []

    def _check_available(self):
        """Ensure CadQuery is available."""
        if not CADQUERY_AVAILABLE:
            raise ImportError("CadQuery is required: pip install cadquery")

    def create_box(
        self,
        length: float,
        width: float,
        height: float
    ) -> "cq.Workplane":
        """Create a box primitive."""
        self._check_available()
        self.current_model = cq.Workplane("XY").box(length, width, height)
        self.history.append(f"box({length}, {width}, {height})")
        return self.current_model

    def create_cylinder(
        self,
        radius: float,
        height: float
    ) -> "cq.Workplane":
        """Create a cylinder primitive."""
        self._check_available()
        self.current_model = cq.Workplane("XY").cylinder(height, radius)
        self.history.append(f"cylinder(r={radius}, h={height})")
        return self.current_model

    def create_sphere(self, radius: float) -> "cq.Workplane":
        """Create a sphere primitive."""
        self._check_available()
        self.current_model = cq.Workplane("XY").sphere(radius)
        self.history.append(f"sphere(r={radius})")
        return self.current_model

    def extrude(self, height: float) -> "cq.Workplane":
        """Extrude the current 2D sketch."""
        self._check_available()
        if self.current_model:
            self.current_model = self.current_model.extrude(height)
            self.history.append(f"extrude({height})")
        return self.current_model

    def fillet(self, radius: float) -> "cq.Workplane":
        """Apply fillet to all edges."""
        self._check_available()
        if self.current_model:
            self.current_model = self.current_model.edges().fillet(radius)
            self.history.append(f"fillet({radius})")
        return self.current_model

    def chamfer(self, size: float) -> "cq.Workplane":
        """Apply chamfer to all edges."""
        self._check_available()
        if self.current_model:
            self.current_model = self.current_model.edges().chamfer(size)
            self.history.append(f"chamfer({size})")
        return self.current_model

    def cut(self, tool: "cq.Workplane") -> "cq.Workplane":
        """Boolean subtract tool from current model."""
        self._check_available()
        if self.current_model:
            self.current_model = self.current_model.cut(tool)
            self.history.append("cut(tool)")
        return self.current_model

    def union(self, other: "cq.Workplane") -> "cq.Workplane":
        """Boolean union with another model."""
        self._check_available()
        if self.current_model:
            self.current_model = self.current_model.union(other)
            self.history.append("union(other)")
        return self.current_model

    def intersect(self, other: "cq.Workplane") -> "cq.Workplane":
        """Boolean intersection with another model."""
        self._check_available()
        if self.current_model:
            self.current_model = self.current_model.intersect(other)
            self.history.append("intersect(other)")
        return self.current_model

    def translate(self, x: float, y: float, z: float) -> "cq.Workplane":
        """Translate the current model."""
        self._check_available()
        if self.current_model:
            self.current_model = self.current_model.translate((x, y, z))
            self.history.append(f"translate({x}, {y}, {z})")
        return self.current_model

    def rotate(
        self,
        axis: Tuple[float, float, float],
        angle: float
    ) -> "cq.Workplane":
        """Rotate the current model."""
        self._check_available()
        if self.current_model:
            self.current_model = self.current_model.rotate(
                (0, 0, 0), axis, angle
            )
            self.history.append(f"rotate({axis}, {angle})")
        return self.current_model

    def export_stl(self, filename: str):
        """Export current model to STL."""
        self._check_available()
        if self.current_model:
            cq.exporters.export(self.current_model, filename)
            logger.info(f"Exported STL: {filename}")

    def export_step(self, filename: str):
        """Export current model to STEP."""
        self._check_available()
        if self.current_model:
            cq.exporters.export(self.current_model, filename, exportType="STEP")
            logger.info(f"Exported STEP: {filename}")

    def export_svg(self, filename: str):
        """Export 2D projection to SVG."""
        self._check_available()
        if self.current_model:
            cq.exporters.export(self.current_model, filename, exportType="SVG")
            logger.info(f"Exported SVG: {filename}")

    def get_volume(self) -> float:
        """Get volume of current model."""
        self._check_available()
        if self.current_model:
            return self.current_model.val().Volume()
        return 0.0

    def get_surface_area(self) -> float:
        """Get surface area of current model."""
        self._check_available()
        if self.current_model:
            return self.current_model.val().Area()
        return 0.0

    def reset(self):
        """Reset the CAD engine."""
        self.current_model = None
        self.history = []
