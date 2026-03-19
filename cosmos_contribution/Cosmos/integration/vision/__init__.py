"""
Vision integration package.

Exposes the high-level VisionModule API and related dataclasses so that
imports like `from Cosmos.integration.vision import VisionModule` work
correctly across the codebase (tool router, MCP server, visual debugging).
"""

from .core import (
    VisionModule,
    VisionTask,
    VisionResult,
    ImageInput,
    SceneGraph,
)

__all__ = [
    "VisionModule",
    "VisionTask",
    "VisionResult",
    "ImageInput",
    "SceneGraph",
]
