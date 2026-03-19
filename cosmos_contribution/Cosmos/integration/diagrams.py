"""
cosmos Diagram Specialist - Mermaid & Visualization.

"A picture is worth a thousand lines of code, especially if it compiles."

Features:
- Advanced Mermaid Support (Flowchart, Sequence, Gantt, State, Pie)
- ASCII Art Generation for Terminal Fallback
- Codebase Visualization (Directory Trees)
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Union
from loguru import logger

class DiagramSkill:
    """
    Diagram generation skill for creating Mermaid diagrams and ASCII visualizations.

    Supports flowcharts, sequence diagrams, Gantt charts, state diagrams,
    and directory tree visualizations.
    """

    def __init__(self):
        """Initialize DiagramSkill with optional configuration."""
        self.default_direction = "TD"  # Top-Down for flowcharts
        self.theme = "cosmos"  # Custom dark theme
        logger.debug("DiagramSkill initialized")

    # --- Mermaid Generators ---

    def generate_mermaid_flowchart(self, nodes: List[Dict[str, str]], edges: List[Dict[str, str]], direction: str = "TD") -> str:
        """Generate Mermaid syntax for a flowchart (TD, LR, etc.)."""
        lines = [f"graph {direction}"]
        
        # Style definitions (The "cosmos Theme")
        lines.append("    classDef normal fill:#111,stroke:#333,color:#fff")
        lines.append("    classDef active fill:#003300,stroke:#00ff88,color:#fff")
        lines.append("    classDef error fill:#330000,stroke:#ff0000,color:#fff")

        for node in nodes:
            nid = node.get("id")
            label = node.get("label", nid)
            shape = node.get("shape", "box") # box, round, diamond
            style_class = node.get("class", "normal")
            
            # Shape logic
            if shape == "round":
                line = f"    {nid}(\"{label}\")"
            elif shape == "diamond":
                line = f"    {nid}{{\"{label}\"}}"
            elif shape == "circle":
                line = f"    {nid}((\"{label}\"))"
            else:
                line = f"    {nid}[\"{label}\"]"
            
            lines.append(f"{line}:::{style_class}")
        
        for edge in edges:
            lines.append(f"    {edge['from']} -->|{edge.get('label', '')}| {edge['to']}")
            
        return "\n".join(lines)

    def generate_sequence_diagram(self, participants: List[Dict[str, str]], messages: List[Dict[str, str]]) -> str:
        """Generate a complex sequence diagram with activations and notes."""
        lines = ["sequenceDiagram", "    autonumber"]
        
        for p in participants:
            lines.append(f"    participant {p['id']} as {p.get('alias', p['id'])}")
        
        for m in messages:
            # Note support
            if "note" in m:
                lines.append(f"    Note over {m['from']},{m['to']}: {m['note']}")
            
            arrow = "-->>" if m.get("dotted") else "->>"
            suffix = "+" if m.get("activate") else "-" if m.get("deactivate") else ""
            
            lines.append(f"    {m['from']}{arrow}{suffix}{m['to']}: {m['text']}")
                
        return "\n".join(lines)

    def generate_gantt_chart(self, title: str, sections: Dict[str, List[Dict]]) -> str:
        """Generate a Project Timeline/Gantt chart."""
        lines = [f"gantt", f"    title {title}", "    dateFormat YYYY-MM-DD", "    axisFormat %m-%d"]
        
        for section, tasks in sections.items():
            lines.append(f"    section {section}")
            for t in tasks:
                crit = "crit," if t.get("critical") else ""
                done = "done," if t.get("done") else "active," if t.get("active") else ""
                lines.append(f"    {t['name']} :{crit}{done} {t.get('id', '')}, {t['start']}, {t['duration']}")
                
        return "\n".join(lines)

    def generate_state_diagram(self, states: List[str], transitions: List[Dict[str, str]]) -> str:
        """Generate a State Machine diagram."""
        lines = ["stateDiagram-v2"]
        for s in states:
            lines.append(f"    {s}")
        for t in transitions:
            lines.append(f"    {t['from']} --> {t['to']}: {t.get('event', '')}")
        return "\n".join(lines)

    # --- ASCII Art Generators (The "Retro Future") ---

    def generate_directory_tree(self, root_path: str, max_depth: int = 2) -> str:
        """Generate a visual ASCII tree of a directory structure."""
        root = Path(root_path)
        if not root.exists():
            return "Error: Path not found."
            
        tree_lines = [f"📂 {root.name}/"]
        
        def _build_tree(directory, prefix="", depth=0):
            if depth >= max_depth:
                return
            
            items = list(directory.glob("*"))
            items.sort(key=lambda x: (not x.is_dir(), x.name))
            
            for index, item in enumerate(items):
                is_last = (index == len(items) - 1)
                connector = "└── " if is_last else "├── "
                
                if item.is_dir():
                    tree_lines.append(f"{prefix}{connector}📂 {item.name}/")
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _build_tree(item, new_prefix, depth + 1)
                else:
                    tree_lines.append(f"{prefix}{connector}📄 {item.name}")

        _build_tree(root)
        return "\n".join(tree_lines)

diagram_skill = DiagramSkill()
