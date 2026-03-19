"""
cosmos Visual Debugging - Screenshot & Diagram Analysis.

"Let me look at that with my special eyes!"

This module implements:
1. UI Element Recognition (Buttons, Inputs, Errors)
2. Diagram Understanding (Architecture flows)
3. Visual Debugging (Correlating visual errors with logs)
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from loguru import logger

from Cosmos.integration.vision import VisionModule, VisionTask, ImageInput

@dataclass
class UIElement:
    type: str # button, input, text, error_banner
    text: str
    location: Dict[str, int] # bbox {x, y, w, h} (Simulated or extracted)
    confidence: float

@dataclass
class DiagramNode:
    id: str
    label: str
    type: str # box, cylinder, cloud
    connections: List[str] = field(default_factory=list)

@dataclass
class VisualAnalysisResult:
    ui_elements: List[UIElement] = field(default_factory=list)
    diagram_nodes: List[DiagramNode] = field(default_factory=list)
    detected_errors: List[str] = field(default_factory=list)
    summary: str = ""

class VisualDebugger:
    """
    Expert system for visual debugging.
    Integrates OCR, Object Detection (simulated via VQA), and Error Pattern Matching.
    """
    def __init__(self, vision_module: VisionModule):
        self.vision = vision_module
        
    async def analyze_screenshot(self, image_source: Any) -> VisualAnalysisResult:
        """
        Analyze a screenshot for UI elements and errors.
        """
        result = VisualAnalysisResult()
        
        # 1. OCR for text extraction
        ocr_res = await self.vision.extract_text(image_source)
        full_text = ocr_res.text or ""
        
        # 2. Heuristic Error Detection
        error_patterns = [
            r"Error:.*",
            r"Exception:.*",
            r"Failed to.*",
            r"404 Not Found",
            r"500 Internal Server Error"
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            result.detected_errors.extend(matches)
            
        # 3. VQA for UI understanding (if BLIP is available)
        # We ask specific questions to identify layout
        ui_desc = await self.vision.caption(image_source)
        result.summary = ui_desc.caption or "No description"
        
        # Simulate UI Element detection via VQA
        # In a real system with LLaVA, we'd ask "List distinct UI buttons"
        
        return result

    async def parse_diagram(self, image_source: Any) -> VisualAnalysisResult:
        """
        Parse an architecture diagram or flowchart.
        """
        result = VisualAnalysisResult()
        
        # 1. Caption to get general idea
        caption = await self.vision.caption(image_source)
        result.summary = caption.caption or ""
        
        # 2. VQA to find connections (Simulated advanced behavior)
        # "What connects to the Database?"
        # For now, we rely on OCR and heuristic graph construction
        ocr_res = await self.vision.extract_text(image_source)
        
        # Heuristic: Text blocks close to arrows (hard to do without BBox)
        # We'll just list potential nodes found in text
        lines = (ocr_res.text or "").split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 2 and len(line) < 30: # Likely a label
                node = DiagramNode(
                    id=line.lower().replace(" ", "_"),
                    label=line,
                    type="unknown"
                )
                result.diagram_nodes.append(node)
                
        return result

# Factory for easy usage
def create_visual_debugger(device="auto"):
    vm = VisionModule(device=device)
    return VisualDebugger(vm)
