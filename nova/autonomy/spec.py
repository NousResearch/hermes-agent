"""The ``autonomy:`` block of a tenant's ``policy.yaml``.

::

    autonomy:
      provider: typesafe        # typesafe | fake | none (none = triage off)
      mode: shadow              # shadow | enforce
      data: metadata_only       # metadata_only | full_args
      timeout_seconds: 2.0
      min_confidence: 0.90      # optional; overrides the default sets' Score/Choice minimum
      actions:
        send_external_email:
          questions: default    # or an inline list
          state: supervised     # supervised | graduated — written by promotion, not by hand
          model_version: ...    # the provider model a graduated action earned trust on

**What may relax an escalation, and only that.** Triage runs only for a call ``decide()``
already sent to a person, and the most it can do is let that one call run — and only when
the tenant chose ``enforce``, the action is ``graduated`` on the exact model version that
is answering, and every question passed. Everything else is checked here, at build time:

* ``enforce`` with the ``fake`` or ``none`` provider is refused — a demo provider must never
  be the thing that lets real calls through.
* An action named here must be a declared business action.
* A ``graduated`` action must say which model version it graduated on, so a model change
  is a demotion rather than a silent continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from nova._fields import Doc
from nova.autonomy.questions import QuestionSet, question_set_for
from nova.errors import SpecError

PROVIDERS = ("typesafe", "fake", "none")
MODES = ("shadow", "enforce")
DATA_MODES = ("metadata_only", "full_args")
STATES = ("supervised", "graduated")

DEFAULT_TIMEOUT_SECONDS = 2.0
#: Bounds on the hook's deadline: long enough for one provider call, short enough that an
#: escalated tool call is never held for a noticeable time.
MIN_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS = 0.2, 10.0


@dataclass(frozen=True)
class ActionAutonomy:
    name: str
    questions: QuestionSet
    #: As written in the policy, for round-tripping: ``"default"`` or the inline list.
    questions_source: Any = "default"
    state: str = "supervised"
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"questions": self.questions_source, "state": self.state}
        if self.model_version:
            out["model_version"] = self.model_version
        return out

    def compiled(self) -> dict[str, Any]:
        return {"state": self.state, "model_version": self.model_version,
                "questions": self.questions.to_list()}


@dataclass(frozen=True)
class AutonomySpec:
    provider: str = "none"
    mode: str = "shadow"
    data: str = "metadata_only"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    min_confidence: Optional[float] = None
    actions: Mapping[str, ActionAutonomy] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.provider != "none" and bool(self.actions)

    @classmethod
    def parse(cls, doc: Doc, *, known_actions: Mapping[str, Any]) -> "AutonomySpec":
        provider = doc.choice("provider", PROVIDERS, default="none")
        mode = doc.choice("mode", MODES, default="shadow")
        data = doc.choice("data", DATA_MODES, default="metadata_only")
        timeout = doc.number("timeout_seconds", default=DEFAULT_TIMEOUT_SECONDS,
                             minimum=MIN_TIMEOUT_SECONDS, maximum=MAX_TIMEOUT_SECONDS)
        minimum = doc.number("min_confidence", minimum=0.0, maximum=1.0)
        if mode == "enforce" and provider in ("fake", "none"):
            raise SpecError(
                f"autonomy.mode enforce needs a real provider; {provider!r} can only run in shadow "
                "mode, where it never lets a call through",
                field="autonomy.mode", source=doc.source,
            )
        actions: dict[str, ActionAutonomy] = {}
        actions_doc = doc.child("actions")
        if actions_doc is not None:
            for name in actions_doc.keys():
                entry = actions_doc.child(name)
                if entry is None:
                    raise SpecError(f"autonomy.actions.{name} must be a mapping", source=doc.source)
                if name not in known_actions:
                    raise SpecError(
                        f"autonomy.actions.{name} is not a business action this policy declares "
                        f"(declared: {', '.join(sorted(known_actions)) or 'none'})",
                        field=f"autonomy.actions.{name}", source=doc.source,
                    )
                raw = entry.value("questions")
                source = "default" if raw is None else raw
                try:
                    questions = question_set_for(name, source, min_confidence=minimum)
                except SpecError as exc:
                    raise SpecError(str(exc), field=f"autonomy.actions.{name}.questions",
                                    source=doc.source) from None
                state = entry.choice("state", STATES, default="supervised")
                version = entry.str_("model_version")
                if state == "graduated" and not version:
                    raise SpecError(
                        f"autonomy.actions.{name} is graduated but does not say which provider model "
                        "version it graduated on; a graduation is only valid on that model",
                        field=f"autonomy.actions.{name}.model_version", source=doc.source,
                    )
                entry.reject_unknown()
                actions[name] = ActionAutonomy(name, questions, source, state, version)
            actions_doc.reject_unknown()
        doc.reject_unknown()
        return cls(provider, mode, data, timeout, minimum, actions)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"provider": self.provider, "mode": self.mode, "data": self.data,
                               "timeout_seconds": self.timeout_seconds,
                               "actions": {n: a.to_dict() for n, a in sorted(self.actions.items())}}
        if self.min_confidence is not None:
            out["min_confidence"] = self.min_confidence
        return out

    def compiled_for(self, approval_actions: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        """The worker's view: only the actions this agent must escalate. None when nothing applies."""
        if not self.active:
            return None
        actions = {name: spec.compiled() for name, spec in sorted(self.actions.items())
                   if name in approval_actions}
        if not actions:
            return None
        return {"provider": self.provider, "mode": self.mode, "data": self.data,
                "timeout_seconds": self.timeout_seconds, "actions": actions}
