from __future__ import annotations

from pathlib import Path

from .models import InspectionReport, PromotionDecision


def inspect_mcp_candidate(path: Path) -> InspectionReport:
    target = Path(path)
    warnings: list[str] = []
    capabilities: list[str] = []
    if (target / "package.json").exists():
        capabilities.append("node_package")
    if (target / "Dockerfile").exists():
        warnings.append("contains Dockerfile")
    return InspectionReport(
        summary="Static preflight report for MCP server candidate.",
        decision=PromotionDecision.HOLD,
        capabilities=capabilities,
        warnings=warnings,
        reasons=["Manual approval is required before activation."],
    )


def inspect_skill_candidate(path: Path) -> InspectionReport:
    target = Path(path)
    warnings: list[str] = []
    if any(file.suffix == ".py" for file in target.rglob("*") if file.is_file()):
        warnings.append("contains executable Python source")
    return InspectionReport(
        summary="Static preflight report for skill candidate.",
        decision=PromotionDecision.HOLD,
        warnings=warnings,
        reasons=["Skill packages remain disabled until explicitly approved."],
    )
