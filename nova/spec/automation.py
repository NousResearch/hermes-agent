"""A scheduled piece of work, declared as a governed object rather than a loose prompt.

Phase 11 surfaced the runtime's scheduler and deliberately withheld *create*. The reason
was not effort: ``cron.jobs.create_job`` takes a free-text prompt and a schedule, and
calling it from a control plane would hand an agent a recurring instruction that NOVA
never compiled and no policy ever reviewed. Every other agent behaviour in this platform
goes through spec → policy → ``pre_tool_call``; an automation created that way would go
around all three, on a timer, forever.

So an automation is a spec, like an agent or an objective:

    Tenant → AutomationSpec → NOVA validation → policy check → Hermes cron job

What the declaration buys, concretely:

* **An owner.** ``agent`` names who runs it, so the work executes inside that agent's
  profile and therefore under that agent's compiled policy. The agent's policy is the
  automation's ceiling, enforced by the runtime, not by this file.
* **A reviewable instruction.** ``objective`` is the prompt, in the bundle, diffable —
  "why did it do that last month" has an answer that is a file.
* **A declared reach.** ``permissions``, ``knowledge`` and ``channels`` say what the work
  is expected to need. NOVA refuses at compile time anything the owning agent was not
  already granted, so an automation can never be a privilege-escalation path.
* **Governance metadata.** Who declared it and why, carried with it.

The validation lives in :mod:`nova.automations.compile`, because it needs the whole
bundle. This module is the shape and the parse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from nova._fields import Doc
from nova.errors import SpecError

#: The directory in a tenant bundle holding one automation per file.
AUTOMATIONS_DIR = "automations"

#: Schedule kinds NOVA will declare. Mirrors what ``cron.jobs.parse_schedule`` produces;
#: the runtime is still the one that parses the expression, so a schedule NOVA accepts
#: and the runtime rejects fails at compile time with the runtime's own message rather
#: than being stored half-valid.
SCHEDULE_KINDS = ("cron", "interval", "once")

#: Longest declared objective. Not a safety boundary — the agent's policy is that — but a
#: recurring instruction that runs to thousands of characters is almost always a
#: procedure that belongs in the knowledge base, referenced by a short objective.
MAX_OBJECTIVE_CHARS = 4000


@dataclass(frozen=True)
class AutomationSpec:
    """Recurring work a tenant has declared, owned by one of its agents."""

    id: str
    title: str
    #: The agent that runs this. Its compiled policy is the runtime ceiling for
    #: everything the automation does.
    agent: str
    #: A human schedule phrase ("every day at 07:00", "every 30 minutes"). Parsed by the
    #: runtime, never by NOVA — a second schedule parser would drift from the one that
    #: actually decides when things run.
    schedule: str
    #: What the agent should do. Reviewable, diffable, and bounded by the agent's policy.
    objective: str
    enabled: bool = True
    #: Business actions this work is expected to need. Validated at compile time against
    #: the owning agent's grants; never *widens* anything.
    permissions: tuple[str, ...] = ()
    #: Corpora the work may read. Validated against the agent's declared knowledge.
    knowledge: tuple[str, ...] = ()
    #: Channels the work may reach. Validated against channels that grant the agent.
    channels: tuple[str, ...] = ()
    #: Why this exists and who asked for it. Free text, carried into the audit record.
    reason: str = ""
    owner: str = ""
    source: Optional[Path] = None

    @classmethod
    def parse(
        cls,
        data: Any,
        *,
        source: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "AutomationSpec":
        doc = Doc(data, source=source, env=env)
        spec = cls(
            id=doc.identifier("id"),
            title=doc.str_("title", required=True),
            agent=doc.identifier("agent"),
            schedule=doc.str_("schedule", required=True),
            objective=doc.str_("objective", required=True),
            enabled=doc.bool_("enabled", default=True),
            permissions=tuple(doc.str_list("permissions")),
            knowledge=tuple(doc.str_list("knowledge")),
            channels=tuple(doc.str_list("channels")),
            reason=doc.str_("reason", default=""),
            owner=doc.str_("owner", default=""),
            source=source,
        )
        doc.reject_unknown()
        spec.validate()
        return spec

    def validate(self) -> None:
        """Shape checks that need no other part of the bundle.

        Anything requiring the agent, the policy or the catalog is a *compile* check and
        lives in :mod:`nova.automations.compile` — this method is safe to call on a spec
        that arrived over the API before any bundle has been consulted.
        """
        if not self.title.strip():
            raise SpecError("title is required", source=self.source)
        if not self.schedule.strip():
            raise SpecError(
                f"automation {self.id!r}: a schedule is required — "
                "for example 'every day at 07:00' or 'every 30 minutes'",
                source=self.source,
            )
        objective = self.objective.strip()
        if not objective:
            raise SpecError(
                f"automation {self.id!r}: an objective is required. An automation with "
                "no instruction is a timer that wakes an agent up to do nothing",
                source=self.source,
            )
        if len(objective) > MAX_OBJECTIVE_CHARS:
            raise SpecError(
                f"automation {self.id!r}: the objective is {len(objective)} characters, "
                f"over the {MAX_OBJECTIVE_CHARS} limit. A recurring instruction this "
                "long is usually a procedure — put it in the knowledge base and have the "
                "objective reference it",
                source=self.source,
            )
        for label, values in (
            ("permissions", self.permissions),
            ("knowledge", self.knowledge),
            ("channels", self.channels),
        ):
            if len(set(values)) != len(values):
                raise SpecError(
                    f"automation {self.id!r}: duplicate entries in {label}",
                    source=self.source,
                )

    def to_dict(self) -> dict[str, Any]:
        """The declaration, as the control plane returns it.

        Includes the objective: it is the reviewable part, and withholding it would
        make the screen unable to answer the one question a governed automation exists
        to answer — *what is this agent being told to do on a timer?*
        """
        return {
            "id": self.id,
            "title": self.title,
            "agent": self.agent,
            "schedule": self.schedule,
            "objective": self.objective,
            "enabled": self.enabled,
            "permissions": list(self.permissions),
            "knowledge": list(self.knowledge),
            "channels": list(self.channels),
            "reason": self.reason,
            "owner": self.owner,
        }


def load_automations(
    root: Path, *, env: Optional[Mapping[str, str]] = None
) -> tuple[AutomationSpec, ...]:
    """Every automation declared in a bundle, sorted by id.

    Absent directory means none declared, which is the common and valid case — the same
    rule objectives and channels follow.
    """
    from nova.spec.bundle import _load_yaml  # local: avoids a cycle at import time

    directory = root / AUTOMATIONS_DIR
    if not directory.is_dir():
        return ()

    specs: list[AutomationSpec] = []
    seen: dict[str, Path] = {}
    for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
        spec = AutomationSpec.parse(_load_yaml(path) or {}, source=path, env=env)
        if spec.id in seen:
            raise SpecError(
                f"automation id {spec.id!r} is declared twice: "
                f"{seen[spec.id].name} and {path.name}",
                source=path,
            )
        seen[spec.id] = path
        specs.append(spec)
    return tuple(sorted(specs, key=lambda s: s.id))
