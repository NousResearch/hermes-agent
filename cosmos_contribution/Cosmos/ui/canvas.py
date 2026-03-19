"""
Cosmos Canvas Manager - Matplotlib Output for CLI and GUI.

AGI v1.8.4 Feature: Manages matplotlib canvas outputs for visualization
in both CLI (Rich terminal) and GUI (Streamlit) environments.

Features:
- Render figures to terminal (via Rich/sixel)
- Render figures to Streamlit
- Save figures to session storage
- Track and retrieve recent figures
- Emit canvas render signals
"""

import asyncio
import base64
import io
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable

from loguru import logger

# Optional imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = Any  # Type alias for when matplotlib isn't available

try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CanvasOutput:
    """A rendered canvas output."""
    output_id: str
    figure: Optional[Figure]
    title: str
    source: str  # Agent or component that generated it
    created_at: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None

    # Rendered outputs
    png_data: Optional[bytes] = None
    svg_data: Optional[str] = None
    base64_png: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_id": self.output_id,
            "title": self.title,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "session_id": self.session_id,
            "has_png": self.png_data is not None,
            "has_svg": self.svg_data is not None,
            "metadata": self.metadata,
        }


# =============================================================================
# CANVAS MANAGER
# =============================================================================

class CanvasManager:
    """
    Manages matplotlib canvas outputs for CLI and GUI.

    Provides unified interface for:
    - Rendering figures to different outputs (terminal, Streamlit, file)
    - Storing figures in session
    - Tracking figure history
    - Emitting canvas render signals
    """

    def __init__(
        self,
        data_dir: str = "./data/canvas",
        max_history: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._max_history = max_history
        self._outputs: Dict[str, CanvasOutput] = {}
        self._history: List[str] = []  # Output IDs in order
        self._session_outputs: Dict[str, List[str]] = {}  # session_id -> output_ids

        # Callbacks
        self._render_callbacks: List[Callable[[CanvasOutput], Awaitable[None]]] = []

        # Nexus integration
        self._nexus = None

        # Rich console for terminal output
        self._console = Console() if RICH_AVAILABLE else None

        logger.info("CanvasManager initialized")

    def connect_nexus(self, nexus) -> None:
        """Connect to Nexus for signal emission."""
        self._nexus = nexus

    # =========================================================================
    # FIGURE RENDERING
    # =========================================================================

    def render_figure(
        self,
        fig: Figure,
        title: str = "Canvas Output",
        source: str = "system",
        session_id: Optional[str] = None,
        dpi: int = 100,
        format: str = "png",
    ) -> CanvasOutput:
        """
        Render a matplotlib figure and store it.

        Args:
            fig: Matplotlib Figure to render
            title: Title for the output
            source: Source agent/component
            session_id: Optional session to associate with
            dpi: Resolution for PNG output
            format: Output format ("png", "svg", "both")

        Returns:
            CanvasOutput with rendered data
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib not available")

        output_id = f"canvas_{uuid.uuid4().hex[:12]}"

        output = CanvasOutput(
            output_id=output_id,
            figure=fig,
            title=title,
            source=source,
            session_id=session_id,
        )

        # Render to PNG
        if format in ("png", "both"):
            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=dpi, bbox_inches="tight")
            output.png_data = png_buffer.getvalue()
            output.base64_png = base64.b64encode(output.png_data).decode("utf-8")
            png_buffer.close()

        # Render to SVG
        if format in ("svg", "both"):
            svg_buffer = io.StringIO()
            fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
            output.svg_data = svg_buffer.getvalue()
            svg_buffer.close()

        # Store
        self._outputs[output_id] = output
        self._history.append(output_id)
        if len(self._history) > self._max_history:
            old_id = self._history.pop(0)
            self._outputs.pop(old_id, None)

        # Track by session
        if session_id:
            if session_id not in self._session_outputs:
                self._session_outputs[session_id] = []
            self._session_outputs[session_id].append(output_id)

        logger.debug(f"Rendered canvas {output_id}: {title}")
        return output

    async def render_and_emit(
        self,
        fig: Figure,
        title: str = "Canvas Output",
        source: str = "system",
        session_id: Optional[str] = None,
        dpi: int = 100,
    ) -> CanvasOutput:
        """
        Render figure and emit canvas render signal.

        Args:
            fig: Matplotlib Figure to render
            title: Title for the output
            source: Source agent/component
            session_id: Optional session ID
            dpi: Resolution

        Returns:
            CanvasOutput with rendered data
        """
        output = self.render_figure(
            fig=fig,
            title=title,
            source=source,
            session_id=session_id,
            dpi=dpi,
            format="both",
        )

        # Emit signal
        await self._emit_signal("GUI_CANVAS_RENDER", {
            "output_id": output.output_id,
            "title": title,
            "source": source,
            "session_id": session_id,
        })

        # Notify callbacks
        for callback in self._render_callbacks:
            try:
                await callback(output)
            except Exception as e:
                logger.error(f"Canvas callback error: {e}")

        return output

    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================

    def render_to_terminal(
        self,
        output: CanvasOutput,
        width: int = 80,
    ) -> str:
        """
        Render canvas output to terminal (ASCII/Unicode art).

        Note: Full image rendering requires a terminal with image support.
        This provides a text-based representation.

        Args:
            output: CanvasOutput to render
            width: Target width in characters

        Returns:
            Text representation of the canvas
        """
        lines = [
            f"┌{'─' * (width - 2)}┐",
            f"│ Canvas: {output.title[:width-12]:<{width-11}}│",
            f"│ Source: {output.source[:width-12]:<{width-11}}│",
            f"│ Time: {output.created_at.strftime('%H:%M:%S'):<{width-9}}│",
            f"├{'─' * (width - 2)}┤",
        ]

        # Note about image data
        if output.png_data:
            size_kb = len(output.png_data) / 1024
            lines.append(f"│ [PNG: {size_kb:.1f}KB available]{'':>{width-26}}│")
        if output.svg_data:
            lines.append(f"│ [SVG available]{'':>{width-18}}│")

        lines.append(f"│ ID: {output.output_id:<{width-7}}│")
        lines.append(f"└{'─' * (width - 2)}┘")

        return "\n".join(lines)

    def render_to_streamlit(self, output: CanvasOutput) -> None:
        """
        Render canvas output to Streamlit.

        Must be called within a Streamlit context.

        Args:
            output: CanvasOutput to render
        """
        try:
            import streamlit as st

            if output.figure:
                st.pyplot(output.figure)
            elif output.png_data:
                st.image(output.png_data, caption=output.title)
            elif output.base64_png:
                st.image(
                    f"data:image/png;base64,{output.base64_png}",
                    caption=output.title,
                )
            else:
                st.warning(f"No renderable data for canvas: {output.output_id}")

        except ImportError:
            logger.warning("Streamlit not available for canvas render")
        except Exception as e:
            logger.error(f"Streamlit render error: {e}")

    def save_to_file(
        self,
        output: CanvasOutput,
        filepath: Optional[Path] = None,
        format: str = "png",
    ) -> Path:
        """
        Save canvas output to file.

        Args:
            output: CanvasOutput to save
            filepath: Target path (auto-generated if None)
            format: File format ("png" or "svg")

        Returns:
            Path to saved file
        """
        if filepath is None:
            filename = f"{output.output_id}.{format}"
            filepath = self.data_dir / filename

        filepath = Path(filepath)

        if format == "png" and output.png_data:
            filepath.write_bytes(output.png_data)
        elif format == "svg" and output.svg_data:
            filepath.write_text(output.svg_data)
        elif output.figure:
            output.figure.savefig(str(filepath), format=format, bbox_inches="tight")
        else:
            raise ValueError(f"No data available for format: {format}")

        logger.debug(f"Saved canvas to {filepath}")
        return filepath

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def get_output(self, output_id: str) -> Optional[CanvasOutput]:
        """Get a canvas output by ID."""
        return self._outputs.get(output_id)

    def get_recent_outputs(self, limit: int = 10) -> List[CanvasOutput]:
        """Get recent canvas outputs."""
        recent_ids = self._history[-limit:]
        return [self._outputs[oid] for oid in reversed(recent_ids) if oid in self._outputs]

    def get_session_outputs(self, session_id: str) -> List[CanvasOutput]:
        """Get all canvas outputs for a session."""
        output_ids = self._session_outputs.get(session_id, [])
        return [self._outputs[oid] for oid in output_ids if oid in self._outputs]

    def get_outputs_by_source(self, source: str) -> List[CanvasOutput]:
        """Get all canvas outputs from a specific source."""
        return [o for o in self._outputs.values() if o.source == source]

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_render(
        self,
        callback: Callable[[CanvasOutput], Awaitable[None]],
    ) -> None:
        """Register a callback for canvas renders."""
        self._render_callbacks.append(callback)

    def remove_render_callback(
        self,
        callback: Callable[[CanvasOutput], Awaitable[None]],
    ) -> bool:
        """Remove a render callback."""
        try:
            self._render_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    async def _emit_signal(self, signal_type: str, payload: Dict[str, Any]) -> None:
        """Emit a signal to Nexus."""
        if not self._nexus:
            return

        try:
            from Cosmos.core.nexus import SignalType

            signal_enum = getattr(SignalType, signal_type, None)
            if signal_enum:
                await self._nexus.emit(
                    type=signal_enum,
                    payload=payload,
                    source="canvas_manager",
                    urgency=0.4,
                )
        except Exception as e:
            logger.debug(f"Failed to emit signal {signal_type}: {e}")

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get canvas manager statistics."""
        total_size = sum(
            len(o.png_data or b"") + len((o.svg_data or "").encode())
            for o in self._outputs.values()
        )

        return {
            "total_outputs": len(self._outputs),
            "history_size": len(self._history),
            "max_history": self._max_history,
            "sessions_tracked": len(self._session_outputs),
            "total_size_bytes": total_size,
            "render_callbacks": len(self._render_callbacks),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_canvas_manager(
    data_dir: str = "./data/canvas",
    max_history: int = 50,
) -> CanvasManager:
    """Factory function to create a CanvasManager instance."""
    return CanvasManager(data_dir=data_dir, max_history=max_history)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_canvas_instance: Optional[CanvasManager] = None


def get_canvas_manager() -> CanvasManager:
    """Get the global CanvasManager instance."""
    global _canvas_instance
    if _canvas_instance is None:
        _canvas_instance = create_canvas_manager()
    return _canvas_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def render_and_show(
    fig: Figure,
    title: str = "Plot",
    source: str = "system",
) -> CanvasOutput:
    """
    Convenience function to render and emit a figure.

    Args:
        fig: Matplotlib Figure
        title: Title for the output
        source: Source identifier

    Returns:
        CanvasOutput
    """
    manager = get_canvas_manager()
    return await manager.render_and_emit(fig=fig, title=title, source=source)


def quick_plot(
    data: List[float],
    title: str = "Quick Plot",
    source: str = "system",
) -> CanvasOutput:
    """
    Quick plot helper for simple data visualization.

    Args:
        data: List of values to plot
        title: Plot title
        source: Source identifier

    Returns:
        CanvasOutput
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for quick_plot")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    manager = get_canvas_manager()
    output = manager.render_figure(fig=fig, title=title, source=source)

    plt.close(fig)
    return output
