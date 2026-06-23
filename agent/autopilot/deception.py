"""Deception detector for autopilot — flag the known cheat patterns in a
candidate final response before/alongside the Council review.

The behaviors below are learned human reward-seeking strategies that show up in
every model family under a long unattended run. They don't extinguish through
instruction; they extinguish when they stop paying off. This module is the
detection half of that: it spots the tells cheaply (pure string/heuristic, no
model call), so the driver can (a) re-inject a directive that names the specific
banned behavior and (b) log it to the ADR. It never blocks on its own; a flag
just shapes the directive and the record.

THE DICTIONARY IS DATA, NOT CODE. All phrasings live in
``deception_patterns.yaml`` (shipped beside this module) plus an optional user/
community overlay at ``~/.hermes/autopilot/deception-patterns.local.yaml``. New
tells can be added by editing the YAML or contributing a PR — no logic change.
Run ``hermes autopilot harvest-deceptions`` to mine the ADR logs for novel
phrasings the model used that aren't in the dictionary yet, then promote them.

Categories (the B-series from the user's field notes + REBORN-A §0.2/§0.3):
  await_user        — "awaiting your review" / "the user can review" / "waiting for <name>"
  reviewer_attack   — "the Council can't see tables / lacks vision"
  external_artifact — "this Jira ticket / PR shows it's done"
  effort_excuse     — "given the effort/time spent, this should suffice / deserves a break"
  unreachable_excuse— "I couldn't find X / access-gated / undocumented" without exhausting sources
  scope_shrink      — "I focused on the core / the rest follow the same pattern"
  flag_to_human     — "I'll flag this / needs human review / the user should decide"
  claim_without_evidence — "it's complete / all done" with no artifact reference
  stall_padding     — busy-looking filler with no concrete artifact verb

Conservative by construction: only consulted while autopilot is active and the
goal is unmet, so a false positive merely sharpens an already-warranted
keep-going directive.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Categories that are special-cased in scan() rather than a plain "match any".
_CLAIM_CATEGORY = "claim_without_evidence"
_STALL_CATEGORY = "stall_padding"
_AWAIT_CATEGORY = "await_user"


@dataclass
class _Dictionary:
    """Loaded pattern dictionary (shipped YAML merged with any local overlay)."""

    categories: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_markers: tuple[str, ...] = ()
    artifact_verbs: tuple[str, ...] = ()

    def note(self, category: str) -> str:
        return str((self.categories.get(category) or {}).get("note", "")).strip()

    def patterns(self, category: str) -> tuple[str, ...]:
        pats = (self.categories.get(category) or {}).get("patterns") or []
        return tuple(str(p).lower() for p in pats)


def _shipped_yaml_path() -> Path:
    return Path(__file__).with_name("deception_patterns.yaml")


def _overlay_yaml_path() -> Path:
    root = os.environ.get("HERMES_HOME", "").strip() or os.path.join(os.path.expanduser("~"), ".hermes")
    return Path(root) / "autopilot" / "deception-patterns.local.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # lazy: yaml is a Hermes dependency but keep import local
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 — a bad dict file must never break the run
        logger.warning("autopilot: failed to load deception patterns from %s (%s)", path, exc)
        return {}


def _merge_overlay(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge a local overlay onto the shipped dict. Overlay phrases are ADDED to
    each category (union), and overlay-only categories are appended. Notes from
    the overlay win when provided."""
    if not overlay:
        return base
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    o_cats = overlay.get("categories") or {}
    b_cats = out.setdefault("categories", {})
    for cat, spec in o_cats.items():
        if not isinstance(spec, dict):
            continue
        cur = b_cats.setdefault(cat, {})
        if spec.get("note"):
            cur["note"] = spec["note"]
        merged = list(cur.get("patterns") or []) + list(spec.get("patterns") or [])
        # de-dup, preserve order
        seen: set = set()
        cur["patterns"] = [p for p in merged if not (p in seen or seen.add(p))]
    for key in ("evidence_markers", "artifact_verbs"):
        if overlay.get(key):
            out[key] = list(out.get(key) or []) + list(overlay[key])
    return out


_DICT_CACHE: Optional[_Dictionary] = None


def load_dictionary(force: bool = False) -> _Dictionary:
    """Load (and cache) the deception dictionary: shipped YAML + local overlay."""
    global _DICT_CACHE
    if _DICT_CACHE is not None and not force:
        return _DICT_CACHE
    base = _load_yaml(_shipped_yaml_path())
    overlay = _load_yaml(_overlay_yaml_path())
    merged = _merge_overlay(base, overlay)
    d = _Dictionary(
        categories=merged.get("categories") or {},
        evidence_markers=tuple(str(x).lower() for x in (merged.get("evidence_markers") or [])),
        artifact_verbs=tuple(str(x).lower() for x in (merged.get("artifact_verbs") or [])),
    )
    _DICT_CACHE = d
    return d


@dataclass
class DeceptionSignal:
    """Result of the deception scan over a candidate final response."""

    flags: list[str] = field(default_factory=list)   # category names that fired
    notes: list[str] = field(default_factory=list)    # human-readable per-flag note

    @property
    def detected(self) -> bool:
        return bool(self.flags)

    def directive_addendum(self) -> str:
        """A sentence naming the caught behavior(s), appended to the keep-going
        directive so the model is told exactly what it was caught doing."""
        if not self.flags:
            return ""
        return " CAUGHT: " + " ".join(self.notes)


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def _name_handoff_patterns(user_name: str) -> tuple[str, ...]:
    """Build name-based await-user patterns from the operator's actual name.

    The model addresses the absent user by NAME to dodge a "you"/"the user" scan
    ("waiting for William", "William can review", "William is around now"). We
    match the name only next to a handoff/availability verb so an innocent mention
    of the name elsewhere doesn't false-positive.
    """
    name = (user_name or "").strip().lower()
    if not name:
        return ()
    first = name.split()[0]
    names = {name, first}
    verbs = (
        "can review", "can verify", "can confirm", "can decide", "can take",
        "will review", "will verify", "to review", "to verify", "to confirm",
        "is around", "is now around", "is back", "is online", "is present",
        "is available", "is here now", "can now", "can finally",
    )
    out: list[str] = []
    for n in names:
        if not n:
            continue
        out.append(f"waiting for {n}")
        out.append(f"await {n}")
        out.append(f"awaiting {n}")
        out.append(f"for {n} to")
        for v in verbs:
            out.append(f"{n} {v}")
    return tuple(out)


def scan(final_response: str, *, user_name: str = "") -> DeceptionSignal:
    """Scan a candidate final response for the known deception tells.

    Pure heuristic, no model call. Returns a DeceptionSignal; empty when clean.
    ``user_name`` (the operator's name) lets the scan catch name-based handoffs
    ("waiting for William", "William is around now") that dodge a second-person
    or "the user" scan.
    """
    sig = DeceptionSignal()
    if not final_response or not final_response.strip():
        return sig
    t = final_response.lower()
    d = load_dictionary()

    # Plain "match any pattern" categories (everything except the two special cases).
    for category in d.categories:
        if category in (_CLAIM_CATEGORY, _STALL_CATEGORY):
            continue
        hit = _has_any(t, d.patterns(category))
        # await_user also matches the dynamic, name-based handoff.
        if category == _AWAIT_CATEGORY and not hit:
            hit = _has_any(t, _name_handoff_patterns(user_name))
        if hit:
            sig.flags.append(category)
            sig.notes.append(d.note(category))

    # claim_without_evidence: a completion claim with NO artifact the Council can inspect.
    if _has_any(t, d.patterns(_CLAIM_CATEGORY)) and not _has_any(t, d.evidence_markers):
        sig.flags.append(_CLAIM_CATEGORY)
        sig.notes.append(d.note(_CLAIM_CATEGORY))

    # stall_padding: a long response that is filler AND has no artifact verb.
    if len(t) > 200 and _has_any(t, d.patterns(_STALL_CATEGORY)) and not _has_any(t, d.artifact_verbs):
        sig.flags.append(_STALL_CATEGORY)
        sig.notes.append(d.note(_STALL_CATEGORY))

    return sig
