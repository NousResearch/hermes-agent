"""Availability checks for Cube-backed high-risk tools."""

from __future__ import annotations

from tools.environments.cube_sandbox import check_cube_sandbox_requirements


def check_cube_high_risk_requirements() -> bool:
    """Verify Cube control plane, template, SDK, and credentials are reachable."""
    return check_cube_sandbox_requirements()
