"""A business objective and the plan that carries it out, declared rather than inferred.

An objective here is a **repeatable business process** — the monthly close, a quarterly
refund audit, an onboarding run — not a one-off prompt. It names its steps, who owns each,
and what each depends on. That is a deliberate choice against having NOVA ask a model to
decompose the work:

- The runtime already has an LLM decomposer, and a second one would be duplication.
- NOVA depends on the standard library and PyYAML. An LLM decomposer needs a model provider,
  which is a dependency this layer does not take.
- A declared plan is reproducible, reviewable and diffable. "Why did it do that last month"
  has an answer that is a file, not a sampled generation.

An objective that genuinely needs decomposing can still have it: a step may be marked
``decompose: true``, which lands on the runtime's board in triage and lets the runtime's own
decomposer fan it out. NOVA records that the routing of those children was delegated and is
therefore ungoverned, because claiming otherwise would be the kind of control that looks
present in a review and does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from nova._fields import Doc
from nova.errors import SpecError

#: The directory in a tenant bundle holding one objective per file.
OBJECTIVES_DIR = "objectives"

#: Priority range the runtime accepts. Mirrored rather than imported: importing it would
#: pull the runtime into NOVA's import graph for the sake of one integer.
PRIORITY_RANGE = (0, 9)


@dataclass(frozen=True)
class PlanStep:
    """One unit of work in an objective's plan."""

    id: str
    title: str
    assignee: str
    body: str = ""
    #: Ids of steps that must complete first. An empty list means the step starts at once,
    #: which is how parallelism is expressed — there is no "parallel" keyword.
    depends_on: tuple[str, ...] = ()
    priority: int = 0
    max_runtime_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    #: Hand this step to the runtime's own decomposer instead of running it as one task.
    #: The children it creates are routed by the runtime, not by NOVA.
    decompose: bool = False

    @classmethod
    def parse(cls, data: Any, *, source: Optional[Path], prefix: str, env) -> "PlanStep":
        doc = Doc(data, source=source, prefix=prefix, env=env)
        step = cls(
            id=doc.identifier("id"),
            title=doc.str_("title", required=True),
            assignee=doc.identifier("assignee"),
            body=doc.str_("body"),
            depends_on=tuple(doc.str_list("depends_on")),
            priority=doc.int_("priority", default=0, minimum=PRIORITY_RANGE[0],
                              maximum=PRIORITY_RANGE[1]) or 0,
            max_runtime_seconds=doc.int_("max_runtime_seconds", minimum=1),
            max_retries=doc.int_("max_retries", minimum=1),
            decompose=doc.bool_("decompose"),
        )
        doc.reject_unknown()
        if step.id in step.depends_on:
            raise SpecError("depends on itself", field=f"{prefix}.depends_on", source=source)
        return step

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "assignee": self.assignee,
            "depends_on": list(self.depends_on),
            "priority": self.priority,
            "decompose": self.decompose,
        }
        if self.body:
            out["body"] = self.body
        if self.max_runtime_seconds is not None:
            out["max_runtime_seconds"] = self.max_runtime_seconds
        if self.max_retries is not None:
            out["max_retries"] = self.max_retries
        return out


@dataclass(frozen=True)
class ObjectiveSpec:
    """One declared business objective."""

    id: str
    title: str
    #: The agent accountable for the objective. Every step's assignee must be this agent or
    #: one it is permitted to delegate to — which is the whole point of the routing check.
    owner: str
    description: str = ""
    steps: tuple[PlanStep, ...] = ()
    #: What "done" means, in the customer's words. Carried onto the owner's own work item so
    #: the agent judging completion is reading the same sentence the customer wrote.
    acceptance: str = ""
    enabled: bool = True
    source: Optional[Path] = None

    @property
    def step_ids(self) -> frozenset[str]:
        return frozenset(step.id for step in self.steps)

    def step(self, step_id: str) -> PlanStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        known = ", ".join(sorted(self.step_ids)) or "(none)"
        raise SpecError(f"no step {step_id!r} in objective {self.id!r}; steps: {known}")

    @property
    def assignees(self) -> tuple[str, ...]:
        """Every agent this objective would put work on, in declaration order."""
        seen: dict[str, None] = {}
        for step in self.steps:
            seen.setdefault(step.assignee, None)
        return tuple(seen)

    @classmethod
    def parse(
        cls,
        data: Any,
        *,
        source: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "ObjectiveSpec":
        doc = Doc(data, source=source, env=env)
        identifier = doc.identifier("id")

        raw_steps = doc._raw("steps", []) or []
        if not isinstance(raw_steps, list):
            raise SpecError("must be a list of steps", field="steps", source=source)
        if not raw_steps:
            raise SpecError(
                "declares no steps; an objective with no work is not a valid objective",
                field="steps",
                source=source,
            )

        steps: list[PlanStep] = []
        seen: set[str] = set()
        for index, entry in enumerate(raw_steps):
            step = PlanStep.parse(entry, source=source, prefix=f"steps[{index}]", env=env)
            if step.id in seen:
                raise SpecError(
                    f"duplicate step id {step.id!r}", field=f"steps[{index}].id", source=source
                )
            seen.add(step.id)
            steps.append(step)

        spec = cls(
            id=identifier,
            title=doc.str_("title", required=True),
            owner=doc.identifier("owner"),
            description=doc.str_("description"),
            steps=tuple(steps),
            acceptance=doc.str_("acceptance"),
            enabled=doc.bool_("enabled", default=True),
            source=source,
        )
        doc.reject_unknown()
        _check_dependencies(spec)
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "owner": self.owner,
            "description": self.description,
            "acceptance": self.acceptance,
            "enabled": self.enabled,
            "steps": [step.to_dict() for step in self.steps],
        }


def _check_dependencies(spec: ObjectiveSpec) -> None:
    """Every dependency resolves, and the plan is a DAG.

    A cycle is caught here rather than at submission because the runtime would accept it:
    a task whose parent never completes simply waits forever, which looks like a stuck
    worker rather than a malformed plan. Finding it at load costs a traversal and turns a
    silent hang into a message naming the steps involved.
    """
    known = spec.step_ids
    for step in spec.steps:
        unknown = [name for name in step.depends_on if name not in known]
        if unknown:
            raise SpecError(
                f"depends on unknown step(s): {', '.join(sorted(unknown))}; "
                f"steps in this objective: {', '.join(sorted(known))}",
                field=f"steps.{step.id}.depends_on",
                source=spec.source,
            )

    cycle = _find_cycle(spec)
    if cycle:
        raise SpecError(
            f"dependencies form a cycle: {' -> '.join(cycle)}. Every step in it would wait "
            "for another step in it, forever",
            field="steps",
            source=spec.source,
        )


def _find_cycle(spec: ObjectiveSpec) -> list[str]:
    """One cycle in the dependency graph, named in order, or an empty list."""
    edges = {step.id: tuple(step.depends_on) for step in spec.steps}
    visiting: set[str] = set()
    done: set[str] = set()
    stack: list[str] = []

    def walk(node: str) -> list[str]:
        if node in done:
            return []
        if node in visiting:
            # Report from the first occurrence, so the message names the loop itself
            # rather than the path that happened to reach it.
            start = stack.index(node)
            return [*stack[start:], node]
        visiting.add(node)
        stack.append(node)
        for parent in edges.get(node, ()):
            found = walk(parent)
            if found:
                return found
        stack.pop()
        visiting.discard(node)
        done.add(node)
        return []

    for step in spec.steps:
        found = walk(step.id)
        if found:
            return found
    return []


def load_objectives(
    bundle_root: Path,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> tuple[ObjectiveSpec, ...]:
    """Read ``objectives/*.yaml`` from a tenant bundle. An absent directory means none.

    Absent is valid: a tenant with no declared processes gets exactly the behaviour every
    deployment had before this phase existed.
    """
    import yaml

    directory = bundle_root / OBJECTIVES_DIR
    if not directory.is_dir():
        return ()

    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix in (".yaml", ".yml")
    )
    objectives: list[ObjectiveSpec] = []
    seen: dict[str, Path] = {}
    for path in paths:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SpecError(f"could not be read: {exc.strerror or exc}", source=path) from exc
        except yaml.YAMLError as exc:
            raise SpecError(f"is not valid YAML: {exc}", source=path) from exc

        spec = ObjectiveSpec.parse(data or {}, source=path, env=env)
        if spec.id in seen:
            raise SpecError(
                f"duplicate objective id {spec.id!r} — already declared in {seen[spec.id].name}",
                field="id",
                source=path,
            )
        seen[spec.id] = path
        objectives.append(spec)
    return tuple(objectives)
