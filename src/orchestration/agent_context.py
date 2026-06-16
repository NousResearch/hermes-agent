"""Agent Context — the Task Control Block (TCB) equivalent.

Following the IBM CICS model, an AgentContext is the unit of isolation
within the single-daemon architecture.  Each agent gets its own context:
workspace, memory, skills, and pseudo-conversational state — without
needing a separate OS process.

CICS analogue
-------------
* CICS Task Control Block  →  AgentContext
* CICS Address Space       →  Hermes Gateway (single process)
* CICS pseudo-conversational → state snapshot / restore

Fields
------
id
    Stable agent identifier, e.g. ``"coding-agent"``.
workspace
    Root directory for this agent's files
    (``~/.hermes/agents/<id>/``).  Contains MEMORY.md, skills/, state.json.
memory
    Loaded durable facts (MEMORY.md content keyed by entry).
skills
    List of skill names available to this agent.
state
    Pseudo-conversational state snapshot — serialised to state.json so an
    agent can be paused and resumed without holding a live process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_state import get_hermes_home


# ── helpers ──────────────────────────────────────────────────────────

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


# ── AgentContext ──────────────────────────────────────────────────────


@dataclass
class AgentContext:
    """Isolated runtime context for one logical agent inside the Gateway."""

    id: str
    workspace: Path

    # Durable memory — loaded from MEMORY.md (one entry per line)
    memory: dict[str, Any] = field(default_factory=dict)

    # Skill names available to this agent
    skills: list[str] = field(default_factory=list)

    # Pseudo-conversational state (CICS-style)
    state: dict[str, Any] = field(default_factory=dict)

    # ── factory ──────────────────────────────────────────────────────

    @classmethod
    def from_disk(cls, agent_id: str) -> "AgentContext":
        """Load (or create) an agent context from disk.

        Directory layout (under ``~/.hermes/agents/<id>/``)::

            MEMORY.md   — durable facts
            state.json  — pseudo-conversational state snapshot
            skills/     — skill definitions (optional, future)

        If the directory does not exist it is created with sensible
        defaults.
        """
        base = _ensure_dir(get_hermes_home() / "agents" / agent_id)

        # memory
        mem_raw = _read_text_safe(base / "MEMORY.md")
        memory: dict[str, Any] = {}
        if mem_raw:
            # Simple key-value per non-empty line:  "key: value"
            for line in mem_raw.splitlines():
                line = line.strip()
                if line and ":" in line:
                    k, _, v = line.partition(":")
                    memory[k.strip()] = v.strip()

        # state
        state: dict[str, Any] = {}
        state_file = base / "state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        # skills (directory listing of SKILL.md files)
        skills_dir = base / "skills"
        skills: list[str] = []
        if skills_dir.is_dir():
            for skill_md in sorted(skills_dir.rglob("SKILL.md")):
                # skill name = parent directory name
                skills.append(skill_md.parent.name)

        return cls(
            id=agent_id,
            workspace=base,
            memory=memory,
            skills=skills,
            state=state,
        )

    # ── persistence ──────────────────────────────────────────────────

    def save_state(self) -> None:
        """Persist the pseudo-conversational state to disk."""
        state_file = self.workspace / "state.json"
        state_file.write_text(
            json.dumps(self.state, indent=2, default=str),
            encoding="utf-8",
        )

    def save_memory(self) -> None:
        """Persist the memory dict to MEMORY.md."""
        mem_file = self.workspace / "MEMORY.md"
        lines = [f"{k}: {v}" for k, v in self.memory.items()]
        mem_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
