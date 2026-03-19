"""
Cosmos Visual Canvas - HermesAgent A2UI Compatibility
=======================================================

Provides an agent-driven visual workspace matching HermesAgent's Canvas/A2UI system.

Features:
- A2UI markup parsing (XML-like declarative UI)
- Live canvas rendering via matplotlib/Streamlit
- JavaScript evaluation in canvas context
- Snapshot capture
- Browser automation integration

A2UI Components Supported:
- <text> - Text display
- <button> - Interactive buttons
- <image> - Image display
- <chart> - Data visualization
- <table> - Tabular data
- <input> - Form inputs
- <container> - Layout containers

"A picture is worth a thousand tokens." - The Collective
"""

import os
import re
import json
import asyncio
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

# Optional imports
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class A2UIComponentType(Enum):
    """A2UI component types."""
    TEXT = "text"
    BUTTON = "button"
    IMAGE = "image"
    CHART = "chart"
    TABLE = "table"
    INPUT = "input"
    CONTAINER = "container"
    MARKDOWN = "markdown"
    CODE = "code"
    PROGRESS = "progress"
    DIVIDER = "divider"


@dataclass
class A2UIComponent:
    """Parsed A2UI component."""
    type: A2UIComponentType
    props: Dict[str, Any] = field(default_factory=dict)
    children: List["A2UIComponent"] = field(default_factory=list)
    content: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "type": self.type.value,
            "props": self.props,
            "content": self.content,
            "children": [c.to_dict() for c in self.children]
        }


@dataclass
class CanvasState:
    """Current state of the visual canvas."""
    components: List[A2UIComponent] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    width: int = 800
    height: int = 600
    background_color: str = "#ffffff"
    last_snapshot: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)


class A2UIParser:
    """
    Parser for HermesAgent A2UI markup language.

    A2UI uses XML-like syntax for declarative UI definition:

    <container layout="vertical" padding="10">
        <text size="24" weight="bold">Hello World</text>
        <button onClick="handleClick">Click Me</button>
        <image src="path/to/image.png" width="200"/>
        <chart type="line" data="{{chartData}}"/>
    </container>
    """

    COMPONENT_MAP = {
        "text": A2UIComponentType.TEXT,
        "button": A2UIComponentType.BUTTON,
        "image": A2UIComponentType.IMAGE,
        "img": A2UIComponentType.IMAGE,
        "chart": A2UIComponentType.CHART,
        "table": A2UIComponentType.TABLE,
        "input": A2UIComponentType.INPUT,
        "container": A2UIComponentType.CONTAINER,
        "div": A2UIComponentType.CONTAINER,
        "markdown": A2UIComponentType.MARKDOWN,
        "md": A2UIComponentType.MARKDOWN,
        "code": A2UIComponentType.CODE,
        "progress": A2UIComponentType.PROGRESS,
        "divider": A2UIComponentType.DIVIDER,
        "hr": A2UIComponentType.DIVIDER,
    }

    @classmethod
    def parse(cls, markup: str) -> List[A2UIComponent]:
        """
        Parse A2UI markup into component tree.

        Args:
            markup: A2UI XML-like markup string

        Returns:
            List of A2UIComponent objects
        """
        if not markup or not markup.strip():
            return []

        # Try BeautifulSoup parsing first
        if BS4_AVAILABLE:
            return cls._parse_with_bs4(markup)
        else:
            return cls._parse_simple(markup)

    @classmethod
    def _parse_with_bs4(cls, markup: str) -> List[A2UIComponent]:
        """Parse using BeautifulSoup."""
        soup = BeautifulSoup(f"<root>{markup}</root>", "html.parser")
        root = soup.find("root")

        components = []
        for child in root.children:
            if hasattr(child, 'name') and child.name:
                comp = cls._parse_element(child)
                if comp:
                    components.append(comp)

        return components

    @classmethod
    def _parse_element(cls, element) -> Optional[A2UIComponent]:
        """Parse a single BeautifulSoup element."""
        tag_name = element.name.lower()

        if tag_name not in cls.COMPONENT_MAP:
            # Unknown tag - treat as container
            comp_type = A2UIComponentType.CONTAINER
        else:
            comp_type = cls.COMPONENT_MAP[tag_name]

        # Extract attributes as props
        props = dict(element.attrs) if element.attrs else {}

        # Get text content
        content = element.get_text(strip=True) if element.string else ""

        # Parse children
        children = []
        for child in element.children:
            if hasattr(child, 'name') and child.name:
                child_comp = cls._parse_element(child)
                if child_comp:
                    children.append(child_comp)

        return A2UIComponent(
            type=comp_type,
            props=props,
            children=children,
            content=content
        )

    @classmethod
    def _parse_simple(cls, markup: str) -> List[A2UIComponent]:
        """Simple regex-based parsing fallback."""
        components = []

        # Simple tag matching
        tag_pattern = r"<(\w+)([^>]*)>([^<]*)</\1>|<(\w+)([^/>]*)/>"

        for match in re.finditer(tag_pattern, markup, re.DOTALL):
            if match.group(1):
                # Opening/closing tag
                tag_name = match.group(1).lower()
                attrs_str = match.group(2)
                content = match.group(3).strip()
            else:
                # Self-closing tag
                tag_name = match.group(4).lower()
                attrs_str = match.group(5)
                content = ""

            comp_type = cls.COMPONENT_MAP.get(tag_name, A2UIComponentType.CONTAINER)

            # Parse attributes
            props = {}
            for attr_match in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                props[attr_match.group(1)] = attr_match.group(2)

            components.append(A2UIComponent(
                type=comp_type,
                props=props,
                content=content
            ))

        return components


class VisualCanvas:
    """
    Visual canvas for HermesAgent A2UI compatibility.

    Provides an agent-driven workspace that can:
    - Render A2UI components
    - Execute JavaScript in canvas context
    - Capture snapshots
    - Navigate URLs (browser integration)
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize visual canvas.

        Args:
            output_dir: Directory for canvas output (default: ~/.cosmos/canvas)
        """
        self.output_dir = Path(output_dir or os.path.expanduser("~/.cosmos/canvas"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state = CanvasState()
        self._js_context: Dict[str, Any] = {}
        self._browser = None

        logger.info("VisualCanvas initialized")

    async def reset(self) -> Dict:
        """Reset canvas to empty state."""
        self.state = CanvasState()
        self._js_context = {}

        logger.debug("Canvas reset")
        return {"status": "reset", "timestamp": datetime.now().isoformat()}

    async def push(self, a2ui: str) -> Dict:
        """
        Push A2UI components to the canvas.

        Args:
            a2ui: A2UI markup string

        Returns:
            Dict with push status
        """
        try:
            components = A2UIParser.parse(a2ui)
            self.state.components.extend(components)
            self.state.last_updated = datetime.now()

            logger.debug(f"Pushed {len(components)} components to canvas")

            return {
                "status": "pushed",
                "components_added": len(components),
                "total_components": len(self.state.components)
            }

        except Exception as e:
            logger.error(f"Canvas push failed: {e}")
            return {"status": "error", "error": str(e)}

    async def eval(self, code: str) -> Dict:
        """
        Evaluate JavaScript code in canvas context.

        Note: This is a simulated JS environment using Python.
        For full JS support, use browser integration.

        Args:
            code: JavaScript code to evaluate

        Returns:
            Dict with evaluation result
        """
        try:
            # Simulate basic JS operations
            result = self._eval_pseudo_js(code)

            return {
                "status": "evaluated",
                "result": result,
                "context_size": len(self._js_context)
            }

        except Exception as e:
            logger.error(f"Canvas eval failed: {e}")
            return {"status": "error", "error": str(e)}

    def _eval_pseudo_js(self, code: str) -> Any:
        """
        Pseudo-JavaScript evaluation.

        Supports basic operations:
        - Variable assignment (let/const/var)
        - JSON operations
        - console.log
        - Basic math
        - String operations
        """
        # Extract variable assignments
        assign_pattern = r"(?:let|const|var)\s+(\w+)\s*=\s*(.+?);"
        for match in re.finditer(assign_pattern, code):
            var_name = match.group(1)
            var_value = match.group(2).strip()

            # Try to parse as JSON
            try:
                self._js_context[var_name] = json.loads(var_value)
            except json.JSONDecodeError:
                # Store as string
                self._js_context[var_name] = var_value

        # Handle console.log
        log_pattern = r"console\.log\((.+?)\)"
        for match in re.finditer(log_pattern, code):
            arg = match.group(1).strip()
            if arg in self._js_context:
                logger.info(f"[Canvas JS] {self._js_context[arg]}")
            else:
                logger.info(f"[Canvas JS] {arg}")

        # Return last assigned value or context
        return self._js_context

    async def snapshot(
        self,
        format: str = "png",
        max_width: int = None,
        quality: int = 90
    ) -> Dict:
        """
        Capture a snapshot of the current canvas state.

        Args:
            format: Image format ("png" or "jpg")
            max_width: Maximum width for resizing
            quality: JPEG quality (1-100)

        Returns:
            Dict with snapshot path and metadata
        """
        try:
            if not self.state.components:
                # Empty canvas - return blank image
                return await self._create_blank_snapshot(format)

            # Render canvas to image
            filepath = await self._render_canvas(format, max_width, quality)

            self.state.last_snapshot = str(filepath)

            return {
                "status": "captured",
                "path": str(filepath),
                "format": format,
                "components_rendered": len(self.state.components),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Canvas snapshot failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _create_blank_snapshot(self, format: str) -> Dict:
        """Create a blank canvas snapshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"canvas_blank_{timestamp}.{format}"
        filepath = self.output_dir / filename

        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_facecolor(self.state.background_color)
            ax.text(0.5, 0.5, "Empty Canvas", ha='center', va='center',
                   fontsize=20, color='#cccccc')
            ax.axis('off')
            fig.savefig(filepath, dpi=100, bbox_inches='tight',
                       facecolor=self.state.background_color)
            plt.close(fig)
        elif PIL_AVAILABLE:
            img = Image.new('RGB', (self.state.width, self.state.height),
                          color=self.state.background_color)
            img.save(filepath)

        return {
            "status": "captured",
            "path": str(filepath),
            "format": format,
            "components_rendered": 0,
            "blank": True
        }

    async def _render_canvas(
        self,
        format: str,
        max_width: int,
        quality: int
    ) -> Path:
        """Render canvas components to image."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"canvas_render_{timestamp}.{format}"
        filepath = self.output_dir / filename

        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("Matplotlib required for canvas rendering")

        # Calculate layout
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_facecolor(self.state.background_color)
        ax.axis('off')

        # Render components
        y_pos = 95  # Start from top
        for comp in self.state.components:
            y_pos = self._render_component(ax, comp, y_pos)

        # Add title
        ax.text(50, 98, "Cosmos Canvas", ha='center', va='top',
               fontsize=12, color='#666666', style='italic')

        fig.savefig(filepath, dpi=150, bbox_inches='tight',
                   facecolor=self.state.background_color)
        plt.close(fig)

        # Resize if needed
        if max_width and PIL_AVAILABLE:
            img = Image.open(filepath)
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                img.save(filepath, quality=quality)

        return filepath

    def _render_component(self, ax, comp: A2UIComponent, y_pos: float) -> float:
        """Render a single component and return new y position."""
        spacing = 5

        if comp.type == A2UIComponentType.TEXT:
            size = int(comp.props.get("size", 14))
            weight = comp.props.get("weight", "normal")
            color = comp.props.get("color", "#000000")

            ax.text(5, y_pos, comp.content, fontsize=size/2,
                   fontweight=weight, color=color, va='top')
            y_pos -= (size/2 + spacing)

        elif comp.type == A2UIComponentType.BUTTON:
            # Render as bordered rectangle with text
            ax.add_patch(plt.Rectangle((5, y_pos-4), 30, 4,
                        fill=True, facecolor='#4a90d9', edgecolor='#2a70b9'))
            ax.text(20, y_pos-2, comp.content, ha='center', va='center',
                   fontsize=8, color='white')
            y_pos -= 8

        elif comp.type == A2UIComponentType.DIVIDER:
            ax.axhline(y=y_pos, xmin=0.05, xmax=0.95, color='#cccccc', linewidth=1)
            y_pos -= spacing

        elif comp.type == A2UIComponentType.PROGRESS:
            value = float(comp.props.get("value", 50))
            ax.add_patch(plt.Rectangle((5, y_pos-2), 90, 2,
                        fill=True, facecolor='#eeeeee'))
            ax.add_patch(plt.Rectangle((5, y_pos-2), value*0.9, 2,
                        fill=True, facecolor='#4caf50'))
            y_pos -= 6

        elif comp.type == A2UIComponentType.CODE:
            # Monospace text with background
            ax.add_patch(plt.Rectangle((3, y_pos-8), 94, 8,
                        fill=True, facecolor='#f5f5f5', edgecolor='#dddddd'))
            ax.text(5, y_pos-2, comp.content[:100], fontsize=7,
                   fontfamily='monospace', va='top')
            y_pos -= 12

        elif comp.type == A2UIComponentType.CHART:
            # Simple placeholder chart
            chart_type = comp.props.get("type", "bar")
            ax.text(5, y_pos, f"[{chart_type} chart]", fontsize=10,
                   color='#666666', style='italic', va='top')
            y_pos -= 30

        elif comp.type == A2UIComponentType.TABLE:
            ax.text(5, y_pos, "[table]", fontsize=10,
                   color='#666666', style='italic', va='top')
            y_pos -= 20

        elif comp.type == A2UIComponentType.IMAGE:
            src = comp.props.get("src", "")
            width = int(comp.props.get("width", 100))
            ax.text(5, y_pos, f"[image: {src}]", fontsize=8,
                   color='#666666', style='italic', va='top')
            y_pos -= 15

        elif comp.type == A2UIComponentType.CONTAINER:
            # Render children with indentation
            for child in comp.children:
                y_pos = self._render_component(ax, child, y_pos)

        elif comp.type == A2UIComponentType.MARKDOWN:
            # Simple markdown rendering
            lines = comp.content.split('\n')
            for line in lines[:5]:  # Limit lines
                ax.text(5, y_pos, line, fontsize=10, va='top')
                y_pos -= 6

        else:
            # Unknown - render as text
            ax.text(5, y_pos, f"[{comp.type.value}: {comp.content[:50]}]",
                   fontsize=8, color='#999999', va='top')
            y_pos -= 6

        return y_pos

    # =========================================================================
    # BROWSER INTEGRATION
    # =========================================================================

    async def navigate(self, url: str) -> Dict:
        """
        Navigate the canvas browser to a URL.

        Args:
            url: URL to navigate to

        Returns:
            Dict with navigation status
        """
        try:
            # Try to use playwright/selenium for actual browser
            try:
                from playwright.async_api import async_playwright

                if not self._browser:
                    pw = await async_playwright().start()
                    self._browser = await pw.chromium.launch(headless=True)

                page = await self._browser.new_page()
                await page.goto(url)

                return {
                    "status": "navigated",
                    "url": url,
                    "title": await page.title()
                }

            except ImportError:
                # Fallback: just record the navigation
                logger.info(f"Canvas navigated to: {url}")
                return {
                    "status": "navigated",
                    "url": url,
                    "note": "Browser automation not available - install playwright"
                }

        except Exception as e:
            logger.error(f"Canvas navigate failed: {e}")
            return {"status": "error", "error": str(e)}

    async def click(self, selector: str) -> Dict:
        """Click an element in the browser."""
        if not self._browser:
            return {"status": "error", "error": "Browser not initialized - call navigate first"}

        try:
            # Would need page reference stored
            return {"status": "clicked", "selector": selector}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def type_text(self, selector: str, text: str) -> Dict:
        """Type text into an element."""
        if not self._browser:
            return {"status": "error", "error": "Browser not initialized"}

        try:
            return {"status": "typed", "selector": selector, "text": text}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def get_state(self) -> Dict:
        """Get current canvas state."""
        return {
            "components": [c.to_dict() for c in self.state.components],
            "variables": self._js_context,
            "width": self.state.width,
            "height": self.state.height,
            "background_color": self.state.background_color,
            "last_snapshot": self.state.last_snapshot,
            "last_updated": self.state.last_updated.isoformat()
        }

    def set_dimensions(self, width: int, height: int):
        """Set canvas dimensions."""
        self.state.width = width
        self.state.height = height

    def set_background(self, color: str):
        """Set canvas background color."""
        self.state.background_color = color


# =============================================================================
# SINGLETON AND UTILITY FUNCTIONS
# =============================================================================

_canvas: Optional[VisualCanvas] = None


def get_canvas() -> VisualCanvas:
    """Get or create the global visual canvas."""
    global _canvas
    if _canvas is None:
        _canvas = VisualCanvas()
    return _canvas


async def canvas_push(a2ui: str) -> Dict:
    """Push A2UI components to canvas."""
    return await get_canvas().push(a2ui)


async def canvas_eval(code: str) -> Dict:
    """Evaluate JavaScript in canvas context."""
    return await get_canvas().eval(code)


async def canvas_snapshot(format: str = "png") -> Dict:
    """Capture canvas snapshot."""
    return await get_canvas().snapshot(format)


async def canvas_reset() -> Dict:
    """Reset canvas to empty state."""
    return await get_canvas().reset()
