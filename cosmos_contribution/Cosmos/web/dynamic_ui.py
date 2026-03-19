"""
Cosmos Dynamic UI Manager
=============================

Manages dynamic, expandable UI sections for the Cosmos web interface.
Supports collapsible panels, real-time updates, and interactive content.

"Good news everyone! The UI now responds to your every whim!"
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import   Optional, Callable
from enum import Enum
from loguru import logger


class SectionState(Enum):
    """State of a UI section."""
    COLLAPSED = "collapsed"
    EXPANDED = "expanded"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class UISection:
    """A dynamic UI section with content and state."""
    section_id: str
    title: str
    content: str = ""
    state: SectionState = SectionState.COLLAPSED
    priority: int = 5  # 1-10, higher = more important
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_any(self) -> dict:
        return {
            "id": self.section_id,
            "title": self.title,
            "content": self.content,
            "state": self.state.value,
            "priority": self.priority,
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class InteractivePanel:
    """A panel containing multiple sections."""
    panel_id: str
    title: str
    sections: dict[str, UISection] = field(default_factory=dict)
    layout: str = "vertical"  # vertical, horizontal, grid
    created_at: datetime = field(default_factory=datetime.now)

    def add_section(self, section: UISection) -> None:
        """Add a section to the panel."""
        self.sections[section.section_id] = section
        logger.debug(f"Added section {section.section_id} to panel {self.panel_id}")

    def remove_section(self, section_id: str) -> bool:
        """Remove a section from the panel."""
        if section_id in self.sections:
            del self.sections[section_id]
            return True
        return False

    def get_section(self, section_id: str) -> Optional[UISection]:
        """Get a section by ID."""
        return self.sections.get(section_id)

    def to_any(self) -> dict:
        return {
            "id": self.panel_id,
            "title": self.title,
            "layout": self.layout,
            "sections": [s.to_any() for s in self.sections.values()],
            "created_at": self.created_at.isoformat()
        }


class DynamicUIManager:
    """
    Manages dynamic UI panels and sections for Cosmos.

    Features:
    - Create/update/delete panels and sections
    - Real-time state changes via WebSocket
    - Priority-based rendering
    - Content caching
    """

    def __init__(self):
        self.panels: dict[str, InteractivePanel] = {}
        self._update_callbacks: list[Callable] = []
        self._lock = asyncio.Lock()

        # Initialize default panels
        self._init_default_panels()

    def _init_default_panels(self):
        """Initialize default UI panels."""
        # Main conversation panel
        main_panel = InteractivePanel(
            panel_id="main",
            title="Cosmos Swarm",
            layout="vertical"
        )
        main_panel.add_section(UISection(
            section_id="chat",
            title="Swarm Chat",
            state=SectionState.EXPANDED,
            priority=10
        ))
        main_panel.add_section(UISection(
            section_id="thinking",
            title="Thinking Process",
            state=SectionState.COLLAPSED,
            priority=7
        ))
        main_panel.add_section(UISection(
            section_id="memory",
            title="Memory Recall",
            state=SectionState.COLLAPSED,
            priority=5
        ))
        self.panels["main"] = main_panel

        # Status panel
        status_panel = InteractivePanel(
            panel_id="status",
            title="System Status",
            layout="horizontal"
        )
        status_panel.add_section(UISection(
            section_id="evolution",
            title="Evolution Status",
            state=SectionState.COLLAPSED,
            priority=6
        ))
        status_panel.add_section(UISection(
            section_id="workers",
            title="Active Workers",
            state=SectionState.COLLAPSED,
            priority=5
        ))
        self.panels["status"] = status_panel

    def on_update(self, callback: Callable):
        """Register a callback for UI updates."""
        self._update_callbacks.append(callback)

    async def _notify_update(self, panel_id: str, section_id: Optional[str] = None):
        """Notify callbacks of UI update."""
        for callback in self._update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(panel_id, section_id)
                else:
                    callback(panel_id, section_id)
            except Exception as e:
                logger.error(f"UI update callback error: {e}")

    async def create_panel(self, panel_id: str, title: str, layout: str = "vertical") -> InteractivePanel:
        """Create a new panel."""
        async with self._lock:
            panel = InteractivePanel(panel_id=panel_id, title=title, layout=layout)
            self.panels[panel_id] = panel
            logger.info(f"Created panel: {panel_id}")
            await self._notify_update(panel_id)
            return panel

    async def add_section(
        self,
        panel_id: str,
        section_id: str,
        title: str,
        content: str = "",
        state: SectionState = SectionState.COLLAPSED,
        priority: int = 5
    ) -> Optional[UISection]:
        """Add a section to a panel."""
        async with self._lock:
            panel = self.panels.get(panel_id)
            if not panel:
                logger.warning(f"Panel not found: {panel_id}")
                return None

            section = UISection(
                section_id=section_id,
                title=title,
                content=content,
                state=state,
                priority=priority
            )
            panel.add_section(section)
            await self._notify_update(panel_id, section_id)
            return section

    async def update_section_content(
        self,
        panel_id: str,
        section_id: str,
        content: str,
        append: bool = False
    ) -> bool:
        """Update section content."""
        async with self._lock:
            panel = self.panels.get(panel_id)
            if not panel:
                return False

            section = panel.get_section(section_id)
            if not section:
                return False

            if append:
                section.content += content
            else:
                section.content = content

            section.updated_at = datetime.now()
            await self._notify_update(panel_id, section_id)
            return True

    async def set_section_state(
        self,
        panel_id: str,
        section_id: str,
        state: SectionState
    ) -> bool:
        """Set section state (expanded/collapsed/loading/error)."""
        async with self._lock:
            panel = self.panels.get(panel_id)
            if not panel:
                return False

            section = panel.get_section(section_id)
            if not section:
                return False

            section.state = state
            section.updated_at = datetime.now()
            await self._notify_update(panel_id, section_id)
            return True

    async def toggle_section(self, panel_id: str, section_id: str) -> Optional[SectionState]:
        """Toggle section between expanded and collapsed."""
        async with self._lock:
            panel = self.panels.get(panel_id)
            if not panel:
                return None

            section = panel.get_section(section_id)
            if not section:
                return None

            if section.state == SectionState.EXPANDED:
                section.state = SectionState.COLLAPSED
            else:
                section.state = SectionState.EXPANDED

            section.updated_at = datetime.now()
            await self._notify_update(panel_id, section_id)
            return section.state

    def get_panel(self, panel_id: str) -> Optional[InteractivePanel]:
        """Get a panel by ID."""
        return self.panels.get(panel_id)

    def get_all_panels(self) -> list[dict]:
        """Get all panels as dictionaries."""
        return [panel.to_any() for panel in self.panels.values()]

    def render_html(self, panel_id: str) -> str:
        """Render a panel as HTML."""
        panel = self.panels.get(panel_id)
        if not panel:
            return "<div class='error'>Panel not found</div>"

        # Sort sections by priority (higher first)
        sorted_sections = sorted(
            panel.sections.values(),
            key=lambda s: s.priority,
            reverse=True
        )

        sections_html = ""
        for section in sorted_sections:
            state_class = section.state.value
            expanded_attr = "open" if section.state == SectionState.EXPANDED else ""

            sections_html += f"""
            <details class="ui-section {state_class}" id="{section.section_id}" {expanded_attr}>
                <summary class="section-header">
                    <span class="section-title">{section.title}</span>
                    <span class="section-toggle"></span>
                </summary>
                <div class="section-content">
                    {section.content}
                </div>
            </details>
            """

        layout_class = f"layout-{panel.layout}"
        return f"""
        <div class="interactive-panel {layout_class}" id="{panel.panel_id}">
            <h2 class="panel-title">{panel.title}</h2>
            <div class="panel-sections">
                {sections_html}
            </div>
        </div>
        """


# Global instance
_ui_manager: Optional[DynamicUIManager] = None


def get_ui_manager() -> DynamicUIManager:
    """Get or create the global UI manager."""
    global _ui_manager
    if _ui_manager is None:
        _ui_manager = DynamicUIManager()
    return _ui_manager


# Convenience functions
async def update_thinking_panel(content: str, append: bool = True):
    """Update the thinking process panel."""
    manager = get_ui_manager()
    await manager.set_section_state("main", "thinking", SectionState.EXPANDED)
    await manager.update_section_content("main", "thinking", content, append=append)


async def update_memory_panel(content: str):
    """Update the memory recall panel."""
    manager = get_ui_manager()
    await manager.set_section_state("main", "memory", SectionState.EXPANDED)
    await manager.update_section_content("main", "memory", content)


async def show_loading(panel_id: str, section_id: str):
    """Show loading state for a section."""
    manager = get_ui_manager()
    await manager.set_section_state(panel_id, section_id, SectionState.LOADING)


async def show_error(panel_id: str, section_id: str, error_msg: str):
    """Show error state for a section."""
    manager = get_ui_manager()
    await manager.set_section_state(panel_id, section_id, SectionState.ERROR)
    await manager.update_section_content(panel_id, section_id, f"Error: {error_msg}")
