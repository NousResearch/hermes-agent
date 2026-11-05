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
* a declared ``task_class`` must appear in ``may_serve``, and on the review lane in
  ``review_classes``;
* the review lane additionally needs ``review_allowed: true``.

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
                ),
            )
        allowed_classes = entry.get("review_classes" if lane == "review" else "may_serve") or []
        if declared not in [str(item) for item in allowed_classes]:
            field_name = "review_classes" if lane == "review" else "may_serve"
            return Assessment(
                allowed=False, enforced=True, unclassified=False,
                refusal=RoutingRefusal(
                    question="Q2", profile=profile, task_class=declared, lane=lane,
                    reason=(f"profile {profile!r} is not routed class {declared!r} on the {lane!r} "
                            f"lane (not listed in {field_name}); deny by default"),
                    decision_id=_decision_for(entry, adoption, profile, field_name),
                    rule_path=rule_str, rule_status=status, config_version=config_version,
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
