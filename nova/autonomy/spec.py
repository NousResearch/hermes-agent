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

from dataclasses import dataclass, field, replace
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
class PromotionRules:
    """When an action has earned graduation. Conservative by default: 50 reviewed decisions
    triage called safe, 98% of them approved unchanged, and not one rejected."""

    min_shadow_decisions: int = 50
    min_agreement: float = 0.98
    max_false_safe: int = 0
    window: int = 100

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "PromotionRules":
        if doc is None:
            return cls()
        rules = cls(
            min_shadow_decisions=doc.int_("min_shadow_decisions", default=50, minimum=1),
            min_agreement=doc.number("min_agreement", default=0.98, minimum=0.0, maximum=1.0),
            max_false_safe=doc.int_("max_false_safe", default=0, minimum=0),
            window=doc.int_("window", default=100, minimum=1),
        )
        doc.reject_unknown()
        if rules.window < rules.min_shadow_decisions:
            raise SpecError("autonomy.promotion.window must be at least min_shadow_decisions, or "
                            "promotion could never be reached", field="autonomy.promotion.window",
                            source=doc.source)
        return rules

    def to_dict(self) -> dict[str, Any]:
        return {"min_shadow_decisions": self.min_shadow_decisions, "min_agreement": self.min_agreement,
                "max_false_safe": self.max_false_safe, "window": self.window}


@dataclass(frozen=True)
class DemotionRules:
    """What sends a graduated action back to supervised. All on by default."""

    on_rejection_of_autonomous: bool = True
    on_false_safe_rate_above: float = 0.02
    on_provider_model_change: bool = True

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "DemotionRules":
        if doc is None:
            return cls()
        rules = cls(
            on_rejection_of_autonomous=doc.bool_("on_rejection_of_autonomous", default=True),
            on_false_safe_rate_above=doc.number("on_false_safe_rate_above", default=0.02,
                                                minimum=0.0, maximum=1.0),
            on_provider_model_change=doc.bool_("on_provider_model_change", default=True),
        )
        doc.reject_unknown()
        return rules

    def to_dict(self) -> dict[str, Any]:
        return {"on_rejection_of_autonomous": self.on_rejection_of_autonomous,
                "on_false_safe_rate_above": self.on_false_safe_rate_above,
                "on_provider_model_change": self.on_provider_model_change}


@dataclass(frozen=True)
class AutonomySpec:
    provider: str = "none"
    mode: str = "shadow"
    data: str = "metadata_only"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    min_confidence: Optional[float] = None
    actions: Mapping[str, ActionAutonomy] = field(default_factory=dict)
    promotion: PromotionRules = field(default_factory=PromotionRules)
    demotion: DemotionRules = field(default_factory=DemotionRules)

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
        promotion = PromotionRules.parse(doc.child("promotion"))
        demotion = DemotionRules.parse(doc.child("demotion"))
        doc.reject_unknown()
        return cls(provider, mode, data, timeout, minimum, actions, promotion, demotion)

    def with_states(self, states: Mapping[str, Mapping[str, Any]]) -> "AutonomySpec":
        """This spec with each action's state taken from the machine-written state file.

        The file wins over ``policy.yaml``: it is where a confirmed promotion and an automatic
        demotion are recorded. An entry for an action the policy no longer configures is
        ignored. A graduated entry without a model version is read as supervised — a
        graduation is only valid on the model it was earned on.
        """
        actions = dict(self.actions)
        for name, entry in (states or {}).items():
            if name not in actions or not isinstance(entry, Mapping):
                continue
            state = entry.get("state")
            version = str(entry.get("model_version") or "")
            if state not in STATES or (state == "graduated" and not version):
                state, version = "supervised", ""
            actions[name] = replace(actions[name], state=state,
                                    model_version=version if state == "graduated" else "")
        return replace(self, actions=actions)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"provider": self.provider, "mode": self.mode, "data": self.data,
                               "timeout_seconds": self.timeout_seconds,
                               "actions": {n: a.to_dict() for n, a in sorted(self.actions.items())}}
        if self.min_confidence is not None:
            out["min_confidence"] = self.min_confidence
        out["promotion"] = self.promotion.to_dict()
        out["demotion"] = self.demotion.to_dict()
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
