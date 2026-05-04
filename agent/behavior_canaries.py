"""Deterministic behavior-regression canaries for Hermes prompt/personality drift.

These checks are intentionally lexical and offline. They do not try to judge a
model response; they assert that the shipped persona/prompt policy still carries
operator-critical constraints before a release, cron rollout, or gateway deploy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class BehaviorCanary:
    """A deterministic prompt/persona invariant."""

    name: str
    description: str
    required_any: tuple[str, ...] = ()
    forbidden_any: tuple[str, ...] = ()


DEFAULT_CANARIES: tuple[BehaviorCanary, ...] = (
    BehaviorCanary(
        name="direct-useful-tone",
        description="Base identity must keep Hermes direct and useful, not ornamental.",
        required_any=("direct", "useful", "efficient", "clear"),
    ),
    BehaviorCanary(
        name="uncertainty-honesty",
        description="Base identity must preserve explicit uncertainty/honesty behavior.",
        required_any=("admit uncertainty", "uncertain", "do not guess", "don't guess"),
    ),
    BehaviorCanary(
        name="no-roleplay-drift",
        description="Base identity must not drift into gimmick or roleplay personas.",
        forbidden_any=("roleplay", "kawaii", "anime", "pirate", "shakespeare"),
    ),
)


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def run_behavior_canaries(
    prompt_text: str,
    canaries: Iterable[BehaviorCanary] = DEFAULT_CANARIES,
) -> list[dict[str, str]]:
    """Return failed behavior canaries for *prompt_text*.

    The return shape is deliberately structured so callers can surface it in CI,
    doctor output, cron artifacts, or release scripts without parsing prose.
    """

    normalized = _norm(prompt_text)
    failures: list[dict[str, str]] = []
    for canary in canaries:
        if canary.required_any and not any(_norm(term) in normalized for term in canary.required_any):
            failures.append(
                {
                    "name": canary.name,
                    "kind": "missing_required_behavior",
                    "description": canary.description,
                    "expected_any": ", ".join(canary.required_any),
                }
            )
        forbidden_hits = [term for term in canary.forbidden_any if _norm(term) in normalized]
        if forbidden_hits:
            failures.append(
                {
                    "name": canary.name,
                    "kind": "forbidden_behavior_marker",
                    "description": canary.description,
                    "found": ", ".join(forbidden_hits),
                }
            )
    return failures


def summarize_behavior_canaries(prompt_text: str) -> Mapping[str, object]:
    failures = run_behavior_canaries(prompt_text)
    return {
        "status": "pass" if not failures else "fail",
        "checked": len(DEFAULT_CANARIES),
        "failed": len(failures),
        "failures": failures,
    }
