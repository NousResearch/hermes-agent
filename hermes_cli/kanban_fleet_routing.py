"""Q2 of the dispatch preflight: is this the right agent for this class of work?

:mod:`hermes_cli.kanban_skill_preflight` answers Q1 — can the profile start a worker turn at all.
Q1 is mechanical: hand the resolver the identifiers the worker would receive and see whether any
of them loads. Q2 is not mechanical. It is a policy lookup against ``fleet-routing-rule.json``, a
named record. Nothing in dispatch asked it, so the review lane was filled by whichever profile
looked free: two profiles that cannot start a review turn were handed review gates, and four that
were never granted review duty could start them.

The rule denies by default:

* a profile absent from ``profiles`` cannot be routed at all;
* ``must_not_serve: ["*"]`` retires a profile from dispatch;
* a declared ``task_class`` must appear in ``may_serve``;
* the review lane additionally needs ``review_allowed: true``, and admits a class listed in
  ``review_classes`` *or* one the record grants in ``review_competence`` — see below.

A card's ``task_class`` names the WORK being handed over, not the duty the reviewer performs. The
review lane used to judge it against ``review_classes`` alone, so a reviewer whose ``may_serve``
held ``execution`` was refused an ``execution``-class card, parked, unblocked, refused again, and
after three rounds auto-escalated to ``triage`` and decomposed into duplicate children — one real
bug became three concurrent edits to the same file (t_2e0cd7e4, events 7154/7155: ``review_refused``
for ``engineering-lead`` on class ``execution``). The repair is a recorded grant, not a widened
predicate: ``review_competence`` names the work-classes that profile may review and
``review_competence_decided_by`` names the decision that granted them, so the widening is visible in
the canon instead of implied by code. A grant can only repeat a class already in ``may_serve``, and
``must_not_serve``, ``review_allowed`` and ``excluded_from_general_review`` still bound it — no
profile acquires a class or a lane its own record does not state (t_d90101d2 review: treating all of
``may_serve`` as review competence silently granted hottr-bot general review, researcher a research
review lane and ux-ui-audit-agent ``evidence-review``).

A refusal is a policy verdict, not an unblock loop: dispatch parks it with
``counts_toward_loop=False``, so a policy that will never change cannot itself drive a card into
``triage`` and decomposition. Every class refusal also names ``capable_profiles`` — the profiles
this rule would accept for that class and lane — so the owner of a parked card has somewhere to
send it.

An explicit prohibition always wins: a class listed in ``must_not_serve`` is refused even if the
same entry also lists it as permitted, so a contradictory record fails closed.

Enforcement is switched on by the record itself (``enforcement.enabled``). A PROPOSED rule is
evaluated but never refuses, so this module can gate dispatch the day the rule is adopted without
a second code change. Q1 refusals do not depend on the rule and are always active — a card whose
worker cannot start is a wasted spawn whichever record is in force.

A card with no recorded ``task_class`` is admitted under the profile-level rule and flagged
``unclassified``: the class fit cannot be asserted, and inferring one from the title is exactly
the guessing this rule exists to stop. New cards should declare a class at creation or at review
handoff; ``hermes kanban create --class`` and ``kanban_create(task_class=...)`` both do.

Every refusal names the profile, the class, the lane, the decision that forbids it, the rule path,
the rule status and the config version, so a refusal is traceable to the record it enforces. A
refusal that only said "skills unavailable" is what made the review-lane failures opaque.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROUTING_REFUSAL_MARKER = "fleet routing refusal"

#: Environment override for the rule path (tests, alternate boards).
RULE_PATH_ENV = "HERMES_FLEET_ROUTING_RULE"

#: Rule location relative to the Hermes home. The audit that produced the rule owns the path.
DEFAULT_RULE_RELPATH = os.path.join("audit", "fleet-routing", "fleet-routing-rule.json")

#: A task class is a lowercase slug: ``code-review``, ``security-review``, ``execution``.
_CLASS_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class RoutingRuleError(RuntimeError):
    """The routing rule exists but cannot be read as one."""


# ---------------------------------------------------------------------------------------------
# The rule record
# ---------------------------------------------------------------------------------------------

def rule_path(path: Optional[str] = None) -> Path:
    """Resolved path of the routing rule: explicit > ``HERMES_FLEET_ROUTING_RULE`` > Hermes root.

    Rooted at the *fleet* home, not the calling profile's home: the rule governs every profile, so
    a dispatcher running as ``profiles/<name>`` must read the same record as one running as root.
    """
    if path:
        return Path(path).expanduser()
    override = (os.environ.get(RULE_PATH_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root

    return Path(get_default_hermes_root()) / DEFAULT_RULE_RELPATH


def load_rule(path: Optional[str] = None) -> tuple[dict, Path]:
    """``(rule, path)``. Raises :class:`RoutingRuleError` when the record is unusable.

    Read fresh on every call: the file is ~10 KB and a stale cached policy is a far worse failure
    than a re-read.
    """
    resolved = rule_path(path)
    try:
        with open(resolved, encoding="utf-8-sig") as handle:
            rule = json.load(handle)
    except FileNotFoundError as exc:
        raise RoutingRuleError(f"fleet routing rule not found at {resolved}") from exc
    except (OSError, ValueError) as exc:
        raise RoutingRuleError(f"fleet routing rule at {resolved} is unreadable: {exc}") from exc
    if not isinstance(rule, dict) or not isinstance(rule.get("profiles"), dict):
        raise RoutingRuleError(
            f"fleet routing rule at {resolved} has no 'profiles' mapping; refusing to guess")
    return rule, resolved


def enforcement_state(rule: Mapping[str, Any]) -> tuple[bool, str, Optional[str]]:
    """``(enabled, status_text, adoption_decision_id)`` for a loaded rule.

    ``enabled`` is the single explicit switch: absent, false or malformed all mean "not in force",
    so a rule that is still PROPOSED can be evaluated for reporting without refusing anything.
    """
    block = rule.get("enforcement")
    if not isinstance(block, Mapping):
        return False, str(rule.get("status") or "PROPOSED"), None
    enabled = block.get("enabled") is True
    status = str(block.get("status") or rule.get("status") or ("ENFORCED" if enabled else "PROPOSED"))
    decision = block.get("decision_id")
    return enabled, status, (str(decision) if decision else None)


def rule_config_version(rule: Mapping[str, Any]) -> Optional[int]:
    """Config version the rule was written against (top-level, else the first profile's route)."""
    block = rule.get("enforcement")
    if isinstance(block, Mapping) and block.get("config_version") is not None:
        try:
            return int(block["config_version"])
        except (TypeError, ValueError):
            pass
    if rule.get("config_version") is not None:
        try:
            return int(rule["config_version"])
        except (TypeError, ValueError):
            pass
    for entry in (rule.get("profiles") or {}).values():
        route = entry.get("route") if isinstance(entry, Mapping) else None
        if isinstance(route, Mapping) and route.get("config_version") is not None:
            try:
                return int(route["config_version"])
            except (TypeError, ValueError):
                pass
    return None


def task_classes(rule: Mapping[str, Any]) -> list[str]:
    classes = rule.get("task_classes")
    return [str(name) for name in classes] if isinstance(classes, (list, tuple)) else []


# ---------------------------------------------------------------------------------------------
# Task class on a card
# ---------------------------------------------------------------------------------------------

def normalize_task_class(value: Any) -> Optional[str]:
    """``None``/blank -> ``None``; otherwise the class slug (shape-checked only)."""
    if value is None:
        return None
    name = str(value).strip()
    if not name:
        return None
    return name


def check_task_class(value: Any) -> Optional[str]:
    """Validate a class for storage on a card. Raises ``ValueError`` on an unknown class.

    The vocabulary is the rule's ``task_classes``. A missing or unreadable rule file does not
    block card creation (a policy record is not a schema constraint) — dispatch still refuses an
    unrecognised class because it cannot be in any ``may_serve`` list.
    """
    name = normalize_task_class(value)
    if name is None:
        return None
    if not _CLASS_RE.match(name):
        raise ValueError(
            f"task_class {name!r} is not a lowercase slug (e.g. 'code-review', 'security-review')")
    try:
        rule, _path = load_rule()
    except RoutingRuleError:
        return name
    known = task_classes(rule)
    if known and name not in known:
        raise ValueError(
            f"unknown task_class {name!r}; the fleet routing rule defines: {', '.join(known)}")
    return name


# ---------------------------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingRefusal:
    """A refused assignment, traceable to the record that forbids it."""

    question: str                      # "Q1" (can it start) | "Q2" (is it the right agent)
    profile: str
    task_class: Optional[str]
    lane: str
    reason: str
    missing: tuple[str, ...] = ()
    #: Profiles this rule would accept for the same class and lane; empty when none would.
    capable_profiles: tuple[str, ...] = ()
    decision_id: Optional[str] = None
    rule_path: Optional[str] = None
    rule_status: Optional[str] = None
    config_version: Optional[int] = None

    def as_text(self) -> str:
        """One greppable line: question, profile, class, lane, decision, rule, config version."""
        parts = [
            f"{ROUTING_REFUSAL_MARKER} ({self.question})",
            f"profile={self.profile!r}",
            f"task_class={(self.task_class or '(unclassified)')!r}",
            f"lane={self.lane!r}",
            f"decision={self.decision_id or '(unrecorded)'!r}",
            f"rule={str(self.rule_path) if self.rule_path else '(none)'!r}",
            f"rule_status={self.rule_status or '(unknown)'!r}",
            f"config_version={self.config_version if self.config_version is not None else 'unknown'}",
        ]
        if self.missing:
            parts.append("missing=" + repr(list(self.missing)))
        if self.capable_profiles:
            parts.append("capable=" + repr(list(self.capable_profiles)))
        return " ".join(parts) + f": {self.reason}"

    def as_event(self) -> dict:
        """JSON-safe payload for a ``routing_refused`` board event."""
        return {
            "question": self.question,
            "profile": self.profile,
            "task_class": self.task_class,
            "lane": self.lane,
            "reason": self.reason,
            "missing": list(self.missing) or None,
            "capable_profiles": list(self.capable_profiles) or None,
            "decision_id": self.decision_id,
            "rule_path": str(self.rule_path) if self.rule_path else None,
            "rule_status": self.rule_status,
            "config_version": self.config_version,
        }


@dataclass
class Assessment:
    """Everything Q1 and Q2 concluded about one assignment."""

    allowed: bool
    enforced: bool                 # is Q2 policy in force for this evaluation?
    unclassified: bool             # no task_class on the card -> class fit unasserted
    refusal: Optional[RoutingRefusal] = None


def _decision_for(entry: Mapping[str, Any], adoption: Optional[str], profile: str,
                  field_name: str) -> str:
    """The decision that forbids this assignment: the profile's own, else the adoption, else the
    rule clause itself (most profiles have no recorded routing decision — that is the finding)."""
    decided_by = entry.get("decided_by")
    if decided_by:
        return str(decided_by)
    if adoption:
        return str(adoption)
    return f"fleet-routing-rule.json#profiles.{profile}.{field_name}"


def _review_competence(entry: Mapping[str, Any], task_class: str, lane: str) -> bool:
    """Does this profile's own record let it review *task_class* on the review lane?

    The review lane judges a card by the WORK it hands over, so a class in ``may_serve`` is
    *competence* — but competence is not duty. The record must also grant review of that class in
    ``review_competence``, with ``review_competence_decided_by`` naming the decision; otherwise a
    code predicate quietly widens the canon and a profile starts reviewing work its own note
    forbids (t_d90101d2 review: hottr-bot "never a general review lane", researcher "reviews
    evidence claims only", ux-ui-audit-agent "review class stays design-review"). The grant can
    only repeat a class already in ``may_serve``: a profile cannot review work it could not do.
    ``must_not_serve`` and ``review_allowed`` are checked before this runs and
    ``excluded_from_general_review`` here, so they never need repeating.
    """
    if lane != "review" or entry.get("excluded_from_general_review"):
        return False
    if not entry.get("review_competence_decided_by"):
        return False
    granted = [str(item) for item in (entry.get("review_competence") or [])]
    if task_class not in granted:
        return False
    may_serve = [str(item) for item in (entry.get("may_serve") or [])]
    return task_class in may_serve


def capable_profiles(
    rule: Mapping[str, Any], task_class: Optional[str], *, lane: str,
    exclude: Sequence[str] = (),
) -> tuple[str, ...]:
    """Profiles *rule* would accept for ``(task_class, lane)``, in declaration order.

    Read-only and best-effort: a refusal that names no alternative leaves the owner of a parked
    card guessing, which is what t_2e0cd7e4's ``review_refused`` events did. Mirrors
    :func:`assessment`'s own order of clauses so a name returned here really would be admitted.
    """
    if task_class is None:
        return ()
    skip = {str(name) for name in exclude}
    found: list[str] = []
    for name, entry in (rule.get("profiles") or {}).items():
        if name in skip or not isinstance(entry, Mapping):
            continue
        must_not = [str(item) for item in (entry.get("must_not_serve") or [])]
        if "*" in must_not or task_class in must_not:
            continue
        if lane == "review":
            if not entry.get("review_allowed"):
                continue
            listed = task_class in [str(item) for item in (entry.get("review_classes") or [])]
            if not (listed or _review_competence(entry, task_class, lane)):
                continue
        elif task_class not in [str(item) for item in (entry.get("may_serve") or [])]:
            continue
        found.append(str(name))
    return tuple(found)


def assessment(
    profile: str,
    task_class: Optional[str],
    *,
    lane: str,
    identifiers: Sequence[str] = (),
    rule: Optional[Mapping[str, Any]] = None,
    path: Optional[str] = None,
    enforce: Optional[bool] = None,
) -> Assessment:
    """Evaluate Q1 then Q2 for one ``(profile, class, lane)`` assignment.

    *identifiers* is the exact preload list the worker would receive
    (:func:`hermes_cli.kanban_db.worker_skill_identifiers`); Q1 is skipped when it is empty.
    *enforce* overrides the rule's own switch (used by checks to prove both outcomes).
    """
    profile = (profile or "").strip()
    declared = normalize_task_class(task_class)

    if rule is None and enforce is not False:
        rule, resolved = load_rule(path)
    else:
        resolved = rule_path(path) if path else None
    rule = rule or {}
    enabled, status, adoption = enforcement_state(rule)
    if enforce is not None:
        enabled = enforce is True
        if enabled and status.upper().startswith("PROPOSED"):
            status = "ENFORCED (override)"
    config_version = rule_config_version(rule) if rule else None
    rule_str = str(resolved) if resolved else None

    # --- Q1: mechanical, rule-independent ------------------------------------------------
    missing: list[str] = []
    if identifiers:
        from hermes_cli.kanban_skill_preflight import unsatisfiable_preload

        missing = list(unsatisfiable_preload(profile, list(identifiers)) or [])
    if missing:
        entry = (rule.get("profiles") or {}).get(profile) or {}
        return Assessment(
            allowed=False, enforced=enabled, unclassified=declared is None,
            refusal=RoutingRefusal(
                question="Q1", profile=profile, task_class=declared, lane=lane,
                reason=(f"profile {profile!r} cannot start a {lane!r} worker turn: "
                        f"identifier(s) {', '.join(repr(m) for m in missing)} did not resolve, "
                        "so no preloaded skill would load"),
                missing=tuple(missing),
                decision_id=_decision_for(entry, adoption, profile, "review_duty"),
                rule_path=rule_str, rule_status=status, config_version=config_version,
            ),
        )

    # --- Q2: policy ----------------------------------------------------------------------
    if not enabled:
        # Evaluated but not in force. Report the class gap without refusing anything.
        return Assessment(allowed=True, enforced=False, unclassified=declared is None)

    entry = (rule.get("profiles") or {}).get(profile)
    if not isinstance(entry, Mapping):
        return Assessment(
            allowed=False, enforced=True, unclassified=declared is None,
            refusal=RoutingRefusal(
                question="Q2", profile=profile, task_class=declared, lane=lane,
                reason=(f"profile {profile!r} is not in the routing rule, so it holds no routing "
                        "permission - deny by default"),
                decision_id=adoption or "fleet-routing-rule.json#profiles",
                rule_path=rule_str, rule_status=status, config_version=config_version,
            ),
        )

    must_not = [str(item) for item in (entry.get("must_not_serve") or [])]

    if "*" in must_not:
        return Assessment(
            allowed=False, enforced=True, unclassified=declared is None,
            refusal=RoutingRefusal(
                question="Q2", profile=profile, task_class=declared, lane=lane,
                reason=f"profile {profile!r} is retired from dispatch (must_not_serve: ['*'])",
                decision_id=_decision_for(entry, adoption, profile, "must_not_serve"),
                rule_path=rule_str, rule_status=status, config_version=config_version,
            ),
        )
    if lane == "review" and not entry.get("review_allowed"):
        return Assessment(
            allowed=False, enforced=True, unclassified=declared is None,
            refusal=RoutingRefusal(
                question="Q2", profile=profile, task_class=declared, lane=lane,
                reason=(f"profile {profile!r} holds no review duty (review_allowed is false), so it "
                        f"must not be routed a {lane!r} card"),
                decision_id=_decision_for(entry, adoption, profile, "review_allowed"),
                rule_path=rule_str, rule_status=status, config_version=config_version,
                capable_profiles=capable_profiles(rule, declared, lane=lane, exclude=(profile,)),
            ),
        )
    if declared is not None:
        if declared in must_not:
            return Assessment(
                allowed=False, enforced=True, unclassified=False,
                refusal=RoutingRefusal(
                    question="Q2", profile=profile, task_class=declared, lane=lane,
                    reason=f"class {declared!r} is explicitly forbidden to profile {profile!r}",
                    decision_id=_decision_for(entry, adoption, profile, "must_not_serve"),
                    rule_path=rule_str, rule_status=status, config_version=config_version,
                    capable_profiles=capable_profiles(
                        rule, declared, lane=lane, exclude=(profile,)),
                ),
            )
        allowed_classes = entry.get("review_classes" if lane == "review" else "may_serve") or []
        if declared not in [str(item) for item in allowed_classes]:
            field_name = "review_classes" if lane == "review" else "may_serve"
            if not _review_competence(entry, declared, lane):
                return Assessment(
                    allowed=False, enforced=True, unclassified=False,
                    refusal=RoutingRefusal(
                        question="Q2", profile=profile, task_class=declared, lane=lane,
                        reason=(f"profile {profile!r} is not routed class {declared!r} on the "
                                f"{lane!r} lane (not listed in {field_name}); deny by default"),
                        decision_id=_decision_for(entry, adoption, profile, field_name),
                        rule_path=rule_str, rule_status=status, config_version=config_version,
                        capable_profiles=capable_profiles(
                            rule, declared, lane=lane, exclude=(profile,)),
                    ),
                )

    return Assessment(allowed=True, enforced=True, unclassified=declared is None)


def assignment_refusal(
    profile: str,
    task_class: Optional[str],
    *,
    lane: str,
    identifiers: Sequence[str] = (),
    rule: Optional[Mapping[str, Any]] = None,
    path: Optional[str] = None,
    enforce: Optional[bool] = None,
) -> Optional[RoutingRefusal]:
    """The refusal for this assignment, or ``None`` when it may be routed."""
    return assessment(
        profile, task_class, lane=lane, identifiers=identifiers,
        rule=rule, path=path, enforce=enforce,
    ).refusal
