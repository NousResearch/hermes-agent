"""PolicySpec — the tenant's declaration of what its agents may do.

Business actions are named in the customer's language ("refund"), not the runtime's
("crm_refund"), because the person who decides that refunds need approval is not the
person who knows which tool issues one. The mapping between the two lives here, declared
once per tenant and reused by every agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from nova._fields import Doc
from nova.errors import SpecError

#: What to do with a tool no rule mentions. ``allow`` suits a deployment still being
#: mapped out; ``deny`` is the posture a security review will ask for.
UNLISTED_CHOICES = ("allow", "deny")

#: Tools a dispatched worker needs to report its own outcome. Without these an allow-list
#: would silently produce agents that run work and never close it — the failure mode that
#: stopped positive tool scoping shipping in Phase 1.
DEFAULT_BASELINE_TOOLS: tuple[str, ...] = (
    "kanban_complete",
    "kanban_block",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_request_review",
    "kanban_show",
    "clarify",
    "todo_list",
)


@dataclass(frozen=True)
class ActionSpec:
    """One business action, and the tools that perform it."""

    name: str
    description: str = ""
    tools: tuple[str, ...] = ()
    #: Tenant-wide default. An agent may still require approval for an action whose
    #: tenant default is False, via its own ``approval.required_for``.
    requires_approval: bool = False

    @classmethod
    def parse(cls, name: str, doc: Doc) -> "ActionSpec":
        spec = cls(
            name=name,
            description=doc.str_("description"),
            tools=tuple(doc.str_list("tools")),
            requires_approval=doc.bool_("requires_approval", default=False),
        )
        doc.reject_unknown()
        if not spec.tools:
            raise SpecError(
                f"action {name!r} names no tools, so nothing can trigger it — list the "
                "tools that perform this action",
                field=f"actions.{name}.tools",
                source=doc.source,
            )
        return spec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tools": list(self.tools)}
        if self.description:
            out["description"] = self.description
        if self.requires_approval:
            out["requires_approval"] = True
        return out


@dataclass(frozen=True)
class PermissionSpec:
    """One named permission, and the tools it grants."""

    name: str
    description: str = ""
    tools: tuple[str, ...] = ()

    @classmethod
    def parse(cls, name: str, doc: Doc) -> "PermissionSpec":
        spec = cls(
            name=name,
            description=doc.str_("description"),
            tools=tuple(doc.str_list("tools")),
        )
        doc.reject_unknown()
        if not spec.tools:
            raise SpecError(
                f"permission {name!r} grants no tools — list the tools it should allow",
                field=f"permissions.{name}.tools",
                source=doc.source,
            )
        return spec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tools": list(self.tools)}
        if self.description:
            out["description"] = self.description
        return out


@dataclass(frozen=True)
class PolicySpec:
    """One tenant's policy. Absent from a bundle means no policy, not an empty one."""

    actions: Mapping[str, ActionSpec] = field(default_factory=dict)
    permissions: Mapping[str, PermissionSpec] = field(default_factory=dict)
    baseline_tools: tuple[str, ...] = DEFAULT_BASELINE_TOOLS
    unlisted_tool: str = "allow"
    source: Optional[Path] = None

    @classmethod
    def parse(
        cls,
        data: Mapping[str, Any],
        *,
        source: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "PolicySpec":
        doc = Doc(data, source=source, env=env)

        actions: dict[str, ActionSpec] = {}
        actions_doc = doc.child("actions")
        if actions_doc is not None:
            for name in actions_doc.keys():
                entry = actions_doc.child(name)
                if entry is None:
                    raise SpecError(f"action {name!r} must be a mapping", source=source)
                actions[name] = ActionSpec.parse(name, entry)
            actions_doc.reject_unknown()

        permissions: dict[str, PermissionSpec] = {}
        permissions_doc = doc.child("permissions")
        if permissions_doc is not None:
            for name in permissions_doc.keys():
                entry = permissions_doc.child(name)
                if entry is None:
                    raise SpecError(f"permission {name!r} must be a mapping", source=source)
                permissions[name] = PermissionSpec.parse(name, entry)
            permissions_doc.reject_unknown()

        defaults = doc.child("defaults")
        unlisted = "allow"
        if defaults is not None:
            unlisted = defaults.choice("unlisted_tool", UNLISTED_CHOICES, default="allow")
            defaults.reject_unknown()

        spec = cls(
            actions=actions,
            permissions=permissions,
            baseline_tools=tuple(doc.str_list("baseline_tools", default=DEFAULT_BASELINE_TOOLS)),
            unlisted_tool=unlisted or "allow",
            source=source,
        )
        doc.reject_unknown()
        return spec

    def action_for_tool(self, tool: str) -> Optional[ActionSpec]:
        """The business action a tool performs, if any.

        A tool listed under two actions is a declaration error caught at load, so this
        can return the first match without ambiguity.
        """
        for action in self.actions.values():
            if tool in action.tools:
                return action
        return None

    def tools_for_permissions(self, names) -> set[str]:
        granted: set[str] = set()
        for name in names:
            permission = self.permissions.get(name)
            if permission is not None:
                granted.update(permission.tools)
        return granted

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": {name: spec.to_dict() for name, spec in sorted(self.actions.items())},
            "permissions": {
                name: spec.to_dict() for name, spec in sorted(self.permissions.items())
            },
            "baseline_tools": list(self.baseline_tools),
            "defaults": {"unlisted_tool": self.unlisted_tool},
        }
