"""Machine-readable specification models (.samagent/spec.yaml and .samagent/brief.md)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


class AutonomyLevel(str, Enum):
    PLAN_ONLY = "plan_only"
    MILESTONES = "milestones"
    HANDS_OFF = "hands_off"


@dataclass
class UserStory:
    id: str
    as_role: str
    can: str
    accept: str
    route: str = "/"
    method: str = "GET"
    auth_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "as": self.as_role,
            "can": self.can,
            "accept": self.accept,
            "route": self.route,
            "method": self.method,
            "auth_required": self.auth_required,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UserStory":
        return cls(
            id=str(raw.get("id", "S1")),
            as_role=str(raw.get("as") or raw.get("as_role") or "visitor"),
            can=str(raw.get("can", "")),
            accept=str(raw.get("accept", "")),
            route=str(raw.get("route", "/")),
            method=str(raw.get("method", "GET")).upper(),
            auth_required=bool(
                raw.get(
                    "auth_required",
                    str(raw.get("as") or raw.get("as_role") or "visitor").lower()
                    not in ("visitor", "anonymous", "public", "guest"),
                )
            ),
        )


@dataclass
class AssumptionEntry:
    id: str
    text: str
    source: str = "default"  # "default" | "user" | "conductor"
    question_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "text": self.text, "source": self.source}
        if self.question_id:
            d["question_id"] = self.question_id
        return d

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AssumptionEntry":
        return cls(
            id=str(raw.get("id", "X1")),
            text=str(raw.get("text", "")),
            source=str(raw.get("source", "default")),
            question_id=raw.get("question_id"),
        )


@dataclass
class BudgetSpec:
    max_usd: float = 6.0
    max_minutes: int = 45

    def to_dict(self) -> Dict[str, Any]:
        return {"max_usd": float(self.max_usd), "max_minutes": int(self.max_minutes)}

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "BudgetSpec":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            max_usd=float(raw.get("max_usd", 6.0)),
            max_minutes=int(raw.get("max_minutes", 45)),
        )


@dataclass
class ModuleSpec:
    name: str
    description: str
    owned_globs: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    estimated_minutes: float = 8.0
    sensitivity: str = "public"  # "public" | "internal" | "private"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModuleSpec":
        return cls(
            name=str(raw.get("name", "core")),
            description=str(raw.get("description", "")),
            owned_globs=list(raw.get("owned_globs") or []),
            depends_on=list(raw.get("depends_on") or []),
            estimated_minutes=float(raw.get("estimated_minutes", 8.0)),
            sensitivity=str(raw.get("sensitivity", "public")),
        )


@dataclass
class SpecDocument:
    goal: str
    roles: List[str] = field(default_factory=lambda: ["visitor", "member", "admin"])
    stories: List[UserStory] = field(default_factory=list)
    non_goals: List[str] = field(default_factory=list)
    stack: str = "web-auth-crud"
    assumptions: List[AssumptionEntry] = field(default_factory=list)
    modules: List[ModuleSpec] = field(default_factory=list)
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    autonomy: str = AutonomyLevel.MILESTONES.value
    router_policy: str = "default"  # "default" | "local_strict"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "roles": list(self.roles),
            "stories": [s.to_dict() for s in self.stories],
            "non_goals": list(self.non_goals),
            "stack": self.stack,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "modules": [m.to_dict() for m in self.modules],
            "budget": self.budget.to_dict(),
            "autonomy": self.autonomy,
            "router_policy": self.router_policy,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SpecDocument":
        autonomy_val = str(raw.get("autonomy", AutonomyLevel.MILESTONES.value))
        if autonomy_val not in {e.value for e in AutonomyLevel}:
            autonomy_val = AutonomyLevel.MILESTONES.value
        return cls(
            goal=str(raw.get("goal", "")).strip(),
            roles=[str(r) for r in (raw.get("roles") or ["visitor"])],
            stories=[UserStory.from_dict(s) for s in (raw.get("stories") or [])],
            non_goals=[str(n) for n in (raw.get("non_goals") or [])],
            stack=str(raw.get("stack", "web-auth-crud")),
            assumptions=[AssumptionEntry.from_dict(a) for a in (raw.get("assumptions") or [])],
            modules=[ModuleSpec.from_dict(m) for m in (raw.get("modules") or [])],
            budget=BudgetSpec.from_dict(raw.get("budget")),
            autonomy=autonomy_val,
            router_policy=str(raw.get("router_policy", "default")),
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> "SpecDocument":
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("spec.yaml must deserialize to a mapping")
        return cls.from_dict(data)

    def render_brief_markdown(self) -> str:
        """Render the human-readable PRD (.samagent/brief.md) that the user approves."""
        lines = [
            f"# Project Brief: {self.goal}",
            "",
            f"- **Template / Stack:** `{self.stack}`",
            f"- **Autonomy Mode:** `{self.autonomy}`",
            f"- **Routing Policy:** `{self.router_policy}`",
            f"- **Budget Cap:** ${self.budget.max_usd:.2f} / {self.budget.max_minutes} min",
            f"- **Roles:** {', '.join(self.roles) if self.roles else 'visitor'}",
            "",
            "## User Stories & Executable Acceptance Criteria",
            "",
        ]
        for s in self.stories:
            auth_tag = " *(auth required)*" if s.auth_required else " *(public)*"
            lines.append(f"- **{s.id}** (`{s.method} {s.route}`){auth_tag}: As **{s.as_role}**, I can {s.can}.")
            lines.append(f"  - *Acceptance:* {s.accept}")
        if self.modules:
            lines.extend(["", "## Modules & Ownership", ""])
            for m in self.modules:
                globs = ", ".join(f"`{g}`" for g in m.owned_globs) or "`*`"
                deps = f" (depends on: {', '.join(m.depends_on)})" if m.depends_on else ""
                lines.append(f"- **{m.name}** ({globs}){deps}: {m.description}")
        if self.assumptions:
            lines.extend(["", "## Explicit Assumptions Ledger", ""])
            for a in self.assumptions:
                lines.append(f"- **{a.id}** [{a.source}]: {a.text}")
        if self.non_goals:
            lines.extend(["", "## Out of Scope (Non-Goals)", ""])
            for ng in self.non_goals:
                lines.append(f"- {ng}")
        lines.append("")
        return "\n".join(lines)

    def save(self, project_dir: Path) -> Tuple[Path, Path]:
        sam_dir = Path(project_dir) / ".samagent"
        sam_dir.mkdir(parents=True, exist_ok=True)
        spec_path = sam_dir / "spec.yaml"
        brief_path = sam_dir / "brief.md"
        spec_path.write_text(self.to_yaml(), encoding="utf-8")
        brief_path.write_text(self.render_brief_markdown(), encoding="utf-8")
        return spec_path, brief_path

    @classmethod
    def load(cls, project_dir: Path) -> "SpecDocument":
        spec_path = Path(project_dir) / ".samagent" / "spec.yaml"
        return cls.from_yaml(spec_path.read_text(encoding="utf-8"))
