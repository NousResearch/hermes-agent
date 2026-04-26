"""
Pydantic models for the Manim data-visualization bridge.

Hermes tool-calling and self-reflection rely on precise field descriptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


ChartKind = Literal["line", "scatter", "bar", "heatmap", "distribution", "regression"]


class TabularPayload(BaseModel):
    """Numeric or categorical rows for seaborn/matplotlib."""

    records: Optional[list[dict[str, float | int | str]]] = Field(
        default=None,
        description="Row-wise data; keys become columns for wide-form plots.",
    )
    x: Optional[list[float]] = Field(default=None, description="1D series for simple XY charts.")
    y: Optional[list[float]] = Field(default=None, description="1D series paired with x.")
    matrix: Optional[list[list[float]]] = Field(
        default=None,
        description="2D numeric grid for heatmaps; rows are y, columns are x.",
    )

    @model_validator(mode="after")
    def _one_mode(self) -> TabularPayload:
        has_rec = bool(self.records)
        has_xy = self.x is not None and self.y is not None
        has_mx = self.matrix is not None
        if sum(1 for v in (has_rec, has_xy, has_mx) if v) != 1:
            raise ValueError("Provide exactly one of: records, (x and y), or matrix.")
        if has_xy and len(self.x or []) != len(self.y or []):
            raise ValueError("x and y must have equal length.")
        return self


class VizSpecification(BaseModel):
    """Structured intent produced from a single natural-language prompt (by the agent)."""

    chart_type: ChartKind = Field(description="High-level chart family to render.")
    title: str = Field(max_length=500, description="Figure title; also used in the Manim scene.")
    x_label: str = Field(default="", max_length=120)
    y_label: str = Field(default="", max_length=120)
    hue_column: Optional[str] = Field(
        default=None,
        description="Optional column name in records for color grouping.",
    )
    seaborn_style: str = Field(
        default="darkgrid",
        description="Seaborn axes style: darkgrid, whitegrid, dark, white, ticks.",
    )
    palette: str = Field(default="deep", description="Seaborn/matplotlib color palette name.")
    x_field: str = Field(default="x", description="Column name for X in records-based charts.")
    y_field: str = Field(default="y", description="Column name for Y in records-based charts.")
    category_field: str = Field(default="category", description="X/categories for bar charts.")
    value_field: str = Field(default="value", description="Heights/values for bar charts.")
    figure_size: tuple[float, float] = Field(default=(10.0, 6.0))
    dpi: int = Field(default=200, ge=72, le=600)
    data: TabularPayload


class ManimSceneOptions(BaseModel):
    """How the static chart is staged inside Manim."""

    scene_class_name: str = Field(default="DataVizBridgeScene", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    resolution: Literal["ql", "qm", "qh"] = Field(
        default="ql",
        description="Suggested manim quality flag: ql draft, qh production.",
    )
    tex_explanations: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Short LaTeX fragments (no $ delimiters); shown after the chart reveal.",
    )


class DataVizBridgeRequest(BaseModel):
    """
    Full bridge input: original user prompt plus a machine-readable viz spec.

    The agent should map one user prompt into `viz` before calling the bridge.
    """

    user_prompt: str = Field(description="Verbatim user request; embedded in script comments only.")
    viz: VizSpecification
    manim: ManimSceneOptions = Field(default_factory=ManimSceneOptions)


class DataVizBridgeResult(BaseModel):
    """Artifacts and commands for rendering outside this stateless bridge."""

    work_dir: Path = Field(description="Temporary directory holding assets (caller may delete).")
    chart_path: Path = Field(description="High-res PNG written by matplotlib/seaborn.")
    manim_script_path: Path = Field(description="Ready-to-run Manim Community Edition script.")
    manim_source: str = Field(description="Same script content for inline inspection.")
    render_suggestion: str = Field(
        description="Example manim CLI invocation; paths are relative to work_dir.",
    )
