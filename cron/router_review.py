"""Router-first routing outcome review for Hermes supervisor.

Reads routing decision records from OpenClaw and applies policy checks,
producing [REVIEW_REQUEST] prompt text for the Hermes supervisor agent to
evaluate as [ACK] or [ESCALATION_NOTICE].

Endpoint assumption
-------------------
``blockrun/auto`` routes through ``openclaw-cli-proxy`` on this machine at::

    http://127.0.0.1:11435/v1

This is the canonical deployed endpoint per the router-first plan
(docs/plans/router-first-autonomous-cron-orchestration-2026-05-15.md §Item 6).
Use 11435 (proxy) unless a direct-8402 probe explicitly validates that path.

Primary source — routing-decisions.json
---------------------------------------
OpenClaw persists gateway-level routing decisions (written by
``src/agents/routing-decisions.ts``) to::

    ~/.openclaw/routing-decisions.json

The envelope shape is::

    {
        "decisions": [
            {
                "timestamp": "<ISO-8601>",
                "agentId": "<agent>" (optional),
                "complexityTier": "<tier>",
                "selectedModel": "<model>" (optional),
                "resolvedModel": "<upstream>" (optional),
                ...
            }
        ],
        "stats": {...},
        "updatedAt": "<ISO-8601>"
    }

This file exists as soon as the gateway starts making routing decisions.
``load_routing_decisions`` reads it and returns ``(outcomes, file_was_present)``
so callers can distinguish "file absent" from "file present with no recent
activity".

Secondary source — routing-outcomes.jsonl (future)
--------------------------------------------------
When OpenClaw Items 1-4 land, cron runs will write per-job JSONL records to::

    ~/.openclaw/routing-outcomes.jsonl

Each record contains at minimum:
  - ``job_name`` (str)
  - ``job_id`` (str)
  - ``selected_model`` (str | null) — e.g. "blockrun/auto"
  - ``resolved_model`` (str | null) — e.g. "gn100/qwen3.6:35b-a3b"
  - ``job_type`` (str | null) — "simple", "coding", "tool-heavy", "reasoning"
  - ``router_pin`` (str | null) — [router-pin: <reason>] text if present
  - ``timestamp`` (ISO-8601 str)
  - ``error`` (str | null)
  - ``consecutive_local_failures`` (int, default 0)

``load_routing_outcomes`` handles this JSONL format and is retained for
forward-compatibility once Items 1-4 land.

Usage (from a cron job)
-----------------------
The Hermes scheduler cannot execute arbitrary Python in a job prompt, so the
review job uses the static ``REVIEW_JOB_STATIC_PROMPT`` as its ``prompt``
field.  That prompt instructs the Hermes agent (which has file tools available)
to load the decisions file and apply the policy checks.

For programmatic use (integration tests, operator scripts)::

    from cron.router_review import load_routing_decisions, review_outcomes, format_review_prompt
    outcomes, present = load_routing_decisions()
    violations = review_outcomes(outcomes)
    prompt = format_review_prompt(outcomes, violations, source_present=present)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical blockrun/auto endpoint — openclaw-cli-proxy on this machine.
#: Plan assumption: use 11435 unless direct-8402 probe explicitly validates.
BLOCKRUN_AUTO_ENDPOINT = "http://127.0.0.1:11435/v1"

#: Primary source: OpenClaw gateway routing decisions (JSON envelope, written by
#: src/agents/routing-decisions.ts). Exists as soon as the gateway makes decisions.
ROUTING_DECISIONS_PATH = Path.home() / ".openclaw" / "routing-decisions.json"

#: Secondary/future source: per-cron-job routing outcomes (JSONL, written by
#: OpenClaw Items 1-4, not yet landed). Kept for forward-compatibility.
ROUTING_OUTCOMES_PATH = Path.home() / ".openclaw" / "routing-outcomes.jsonl"

#: Model prefixes that indicate cloud/strong inference (escalation path).
_CLOUD_MODEL_PREFIXES = ("claude", "anthropic", "openai", "gemini", "codex", "gpt")

#: Model prefixes that indicate local inference (GN100 / Sparky path).
_LOCAL_MODEL_PREFIXES = ("gn100", "local/", "ollama", "qwen", "llama", "mistral")

#: Job types where escalation to a cloud model is expected.
_ESCALATION_EXPECTED_TYPES = ("coding", "tool-heavy", "reasoning")

#: Threshold for "repeated local failures" policy check.
_CONSECUTIVE_LOCAL_FAILURE_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RoutingOutcome:
    """A single routing outcome record from an OpenClaw cron run."""

    job_name: str
    job_id: str
    selected_model: Optional[str]
    resolved_model: Optional[str]
    job_type: Optional[str]
    router_pin: Optional[str]
    timestamp: str
    error: Optional[str] = None
    consecutive_local_failures: int = 0


@dataclass
class PolicyViolation:
    """A routing policy violation detected by ``review_outcomes``."""

    severity: str  # "warning" | "critical"
    message: str
    job_name: str


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_routing_outcomes(
    path: Optional[Path] = None,
    max_age_minutes: int = 60,
) -> List[RoutingOutcome]:
    """Load recent routing outcome records from the OpenClaw JSONL file.

    This is the future per-cron-job format written by OpenClaw Items 1-4.
    The file does not exist until those items land; this function returns an
    empty list when the file is absent — use ``load_routing_decisions`` for
    the currently-available source.

    Args:
        path: Override for ``ROUTING_OUTCOMES_PATH`` (useful in tests).
        max_age_minutes: Discard records older than this many minutes.

    Returns:
        List of ``RoutingOutcome`` objects, newest-first, filtered to the
        requested time window.  Returns an empty list when the file is absent
        or unreadable.
    """
    outcomes_path = path if path is not None else ROUTING_OUTCOMES_PATH
    if not outcomes_path.exists():
        logger.debug("Routing outcomes file not found at %s", outcomes_path)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    outcomes: List[RoutingOutcome] = []

    try:
        with outcomes_path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed routing outcome on line %d: %s", lineno, exc
                    )
                    continue

                ts_str = record.get("timestamp", "")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts < cutoff:
                            continue
                    except ValueError:
                        logger.warning("Unrecognised timestamp %r on line %d", ts_str, lineno)
                        continue

                outcomes.append(
                    RoutingOutcome(
                        job_name=str(record.get("job_name") or "unknown"),
                        job_id=str(record.get("job_id") or ""),
                        selected_model=record.get("selected_model") or None,
                        resolved_model=record.get("resolved_model") or None,
                        job_type=record.get("job_type") or None,
                        router_pin=record.get("router_pin") or None,
                        timestamp=ts_str,
                        error=record.get("error") or None,
                        consecutive_local_failures=int(
                            record.get("consecutive_local_failures") or 0
                        ),
                    )
                )
    except OSError as exc:
        logger.warning("Could not read routing outcomes from %s: %s", outcomes_path, exc)

    # Return newest-first so callers see the most recent events first.
    outcomes.sort(key=lambda o: o.timestamp, reverse=True)
    return outcomes


def load_routing_decisions(
    path: Optional[Path] = None,
    max_age_minutes: int = 60,
) -> tuple[List[RoutingOutcome], bool]:
    """Load routing decisions from OpenClaw's routing-decisions.json.

    OpenClaw persists gateway-level routing decisions (not per-cron-job
    outcomes) to this file via ``src/agents/routing-decisions.ts``. The
    envelope shape is ``{"decisions": [...], "stats": {...}, "updatedAt": "..."}``,
    where each decision carries at minimum ``timestamp``, ``selectedModel``,
    ``resolvedModel``, ``agentId``, and ``complexityTier``.

    Field mapping to ``RoutingOutcome``:

    - ``agentId`` → ``job_name`` (``"gateway/<agentId>"`` or
      ``"gateway-decision"`` when absent)
    - ``complexityTier`` → ``job_type`` (passed through; OpenClaw tiers such as
      ``"research:reasoning"`` differ from cron job types — policy checks that
      require exact values like ``"simple"`` or ``"coding"`` will not fire on
      unrecognised tier values)
    - ``selectedModel`` → ``selected_model``
    - ``resolvedModel`` → ``resolved_model``
    - ``router_pin``, ``consecutive_local_failures``, ``error`` → defaults
      (None / 0) — not recorded at the gateway layer

    Args:
        path: Override for ``ROUTING_DECISIONS_PATH`` (useful in tests).
        max_age_minutes: Discard decisions older than this many minutes.

    Returns:
        ``(outcomes, file_was_present)`` — ``file_was_present`` is ``False``
        only when the file does not exist on disk; it is ``True`` even when
        the file is present but contains no decisions within the time window.
        This lets callers distinguish "file absent" from "file present with no
        recent activity" when constructing review prompts.
    """
    decisions_path = path if path is not None else ROUTING_DECISIONS_PATH
    if not decisions_path.exists():
        logger.debug("Routing decisions file not found at %s", decisions_path)
        return [], False

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    outcomes: List[RoutingOutcome] = []

    try:
        with decisions_path.open(encoding="utf-8") as fh:
            envelope = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Could not read routing decisions from %s: %s", decisions_path, exc
        )
        # File exists but is unreadable; report as present so the prompt can
        # surface the read error rather than silently claiming "file absent".
        return [], True

    raw_decisions = envelope.get("decisions") or []
    for record in raw_decisions:
        if not isinstance(record, dict):
            continue

        ts_str = record.get("timestamp", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    continue
            except ValueError:
                logger.warning(
                    "Unrecognised timestamp %r in routing decisions file", ts_str
                )
                continue

        agent_id = record.get("agentId") or None
        outcomes.append(
            RoutingOutcome(
                job_name=(f"gateway/{agent_id}" if agent_id else "gateway-decision"),
                job_id="",
                selected_model=record.get("selectedModel") or None,
                resolved_model=record.get("resolvedModel") or None,
                job_type=record.get("complexityTier") or None,
                router_pin=None,
                timestamp=ts_str,
                error=None,
                consecutive_local_failures=0,
            )
        )

    outcomes.sort(key=lambda o: o.timestamp, reverse=True)
    return outcomes, True


# ---------------------------------------------------------------------------
# Policy checks
# ---------------------------------------------------------------------------


def _is_local_model(model: Optional[str]) -> bool:
    """Return True if *model* indicates local/GN100 inference."""
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) for p in _LOCAL_MODEL_PREFIXES)


def _is_cloud_model(model: Optional[str]) -> bool:
    """Return True if *model* indicates cloud/strong inference."""
    if not model:
        return False
    m = model.lower()
    return any(m.startswith(p) for p in _CLOUD_MODEL_PREFIXES)


def review_outcomes(outcomes: List[RoutingOutcome]) -> List[PolicyViolation]:
    """Apply routing policy checks to a list of recent outcomes.

    The five checks (from plan Item 4 / Item 6 done-when criteria):

    1. **Local not used for simple work** — ``job_type=="simple"`` resolved to
       a cloud model without a ``router_pin`` marker.
    2. **No escalation for complex work** — ``job_type`` in coding/tool-heavy/
       reasoning resolved to a local model instead of cloud.
    3. **Missing resolved_model metadata** — router-first job ran but
       ``resolved_model`` is ``None`` (OpenClaw Items 1-4 not yet wired).
    4. **Repeated local failures** — ``consecutive_local_failures`` at or
       above the threshold; GN100 / local inference may be down.
    5. **Unexplained opt-out pin** — explicit non-router model with no
       ``router_pin`` marker (for non-coding job types where a pin is
       unexpected and should be documented).

    Returns a list of ``PolicyViolation`` objects (may be empty).
    """
    violations: List[PolicyViolation] = []

    for o in outcomes:
        # ------------------------------------------------------------------
        # Check 1 — simple work routed to cloud without justification
        # ------------------------------------------------------------------
        if (
            o.job_type == "simple"
            and o.selected_model == "blockrun/auto"
            and _is_cloud_model(o.resolved_model)
            and not o.router_pin
        ):
            violations.append(
                PolicyViolation(
                    severity="warning",
                    message=(
                        f"Simple job resolved to cloud model {o.resolved_model!r} "
                        f"(selected: {o.selected_model!r}). "
                        "Local GN100 expected for simple work; check ClawRouter classification."
                    ),
                    job_name=o.job_name,
                )
            )

        # ------------------------------------------------------------------
        # Check 2 — coding/reasoning work not escalating
        # ------------------------------------------------------------------
        if (
            o.job_type in _ESCALATION_EXPECTED_TYPES
            and o.selected_model == "blockrun/auto"
            and _is_local_model(o.resolved_model)
        ):
            violations.append(
                PolicyViolation(
                    severity="warning",
                    message=(
                        f"{o.job_type.capitalize()} job did not escalate: "
                        f"resolved to local {o.resolved_model!r}. "
                        "Expected cloud escalation for this job type."
                    ),
                    job_name=o.job_name,
                )
            )

        # ------------------------------------------------------------------
        # Check 3 — missing resolved_model metadata
        # ------------------------------------------------------------------
        if o.selected_model == "blockrun/auto" and o.resolved_model is None:
            violations.append(
                PolicyViolation(
                    severity="warning",
                    message=(
                        "Router-first job has no resolved_model recorded. "
                        "OpenClaw Items 1-4 integration may not yet be active, "
                        "or ClawRouter is not exposing the upstream model."
                    ),
                    job_name=o.job_name,
                )
            )

        # ------------------------------------------------------------------
        # Check 4 — repeated local model failures
        # ------------------------------------------------------------------
        if o.consecutive_local_failures >= _CONSECUTIVE_LOCAL_FAILURE_THRESHOLD:
            violations.append(
                PolicyViolation(
                    severity="critical",
                    message=(
                        f"Job has {o.consecutive_local_failures} consecutive local model failures. "
                        "GN100 / local inference may be unavailable. "
                        "Verify ai.openclaw.claude-cli-proxy at 11435 and GN100 tunnel at 11436."
                    ),
                    job_name=o.job_name,
                )
            )

        # ------------------------------------------------------------------
        # Check 5 — unexplained explicit pin (non-router model, no marker)
        # ------------------------------------------------------------------
        if (
            o.selected_model
            and o.selected_model != "blockrun/auto"
            and not o.router_pin
            # Coding pins are expected in phase one per the plan — only flag
            # non-coding job types where an unexplained pin is a surprise.
            and o.job_type not in _ESCALATION_EXPECTED_TYPES
        ):
            violations.append(
                PolicyViolation(
                    severity="warning",
                    message=(
                        f"Explicit model pin {o.selected_model!r} has no "
                        "[router-pin: <reason>] marker in the job description. "
                        "Add a marker or migrate to blockrun/auto."
                    ),
                    job_name=o.job_name,
                )
            )

    return violations


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def format_review_prompt(
    outcomes: List[RoutingOutcome],
    violations: List[PolicyViolation],
    *,
    source_present: bool = True,
) -> str:
    """Build a ``[REVIEW_REQUEST]`` prompt for the Hermes supervisor agent.

    The returned string contains inline routing decision data so the agent can
    decide ``[ACK]`` (all-clear) or ``[ESCALATION_NOTICE]`` (action needed)
    without needing to call file-read tools.

    Args:
        outcomes: Recent routing outcomes/decisions to review.
        violations: Policy violations detected by ``review_outcomes``.
        source_present: Whether the source file was found on disk.  Pass the
            second element of ``load_routing_decisions()``'s return value.  When
            ``False`` and ``outcomes`` is empty, the prompt explicitly says the
            file is absent; when ``True`` and ``outcomes`` is empty, it says the
            file is present but no fresh decisions were found in the review window.
    """
    lines: List[str] = [
        "[REVIEW_REQUEST] Router-first routing outcome review.",
        f"Endpoint assumption: {BLOCKRUN_AUTO_ENDPOINT} (openclaw-cli-proxy).",
        f"Outcomes reviewed: {len(outcomes)}.",
    ]

    if not outcomes:
        if not source_present:
            lines.append(
                f"Source file not found at {ROUTING_DECISIONS_PATH}. "
                "OpenClaw gateway has not yet written any routing decisions."
            )
        else:
            lines.append(
                "Source file is present but contains no router-first decisions "
                "within the review window. No policy violations detectable."
            )
    else:
        router_first = [o for o in outcomes if o.selected_model == "blockrun/auto"]
        pinned = [o for o in outcomes if o.selected_model and o.selected_model != "blockrun/auto"]
        missing_resolved = [
            o for o in outcomes if o.selected_model == "blockrun/auto" and o.resolved_model is None
        ]
        lines.append(
            f"Router-first jobs: {len(router_first)}.  "
            f"Pinned (explicit model): {len(pinned)}.  "
            f"Missing resolved_model: {len(missing_resolved)}."
        )

    if not violations:
        lines.append("Policy checks: all passed.")
    else:
        critical = [v for v in violations if v.severity == "critical"]
        warnings = [v for v in violations if v.severity == "warning"]
        lines.append(
            f"Policy violations: {len(critical)} critical, {len(warnings)} warning."
        )
        for v in violations:
            lines.append(f"  [{v.severity.upper()}] {v.job_name}: {v.message}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Static job prompt template
# ---------------------------------------------------------------------------

#: Static prompt for the ``router-first-review`` cron job entry.
#:
#: The Hermes cron scheduler sends this verbatim to the Hermes agent.  With
#: file tools enabled (``enabled_toolsets: ["file"]``), the agent reads
#: ``ROUTING_DECISIONS_PATH`` directly and applies the policy checks.  The
#: future per-job JSONL source (``ROUTING_OUTCOMES_PATH``) is checked as a
#: supplementary fallback once Items 1-4 land in clawdbot.
#:
#: For programmatic use prefer:
#:   outcomes, present = load_routing_decisions()
#:   prompt = format_review_prompt(outcomes, review_outcomes(outcomes), source_present=present)
REVIEW_JOB_STATIC_PROMPT = f"""\
[REVIEW_REQUEST] Router-first routing outcome review.
Endpoint assumption: {BLOCKRUN_AUTO_ENDPOINT} (openclaw-cli-proxy).

Read the routing decisions file at {ROUTING_DECISIONS_PATH} if it exists.
The file is a JSON envelope: {{"decisions": [...], "stats": {{...}}, "updatedAt": "..."}}.
Each decision has at minimum: timestamp, selectedModel, resolvedModel, agentId, complexityTier.

Apply the following policy checks to decisions from the last 60 minutes:

1. Simple jobs (job_type=simple, selected_model=blockrun/auto) resolved to \
cloud model without router_pin → [warning].
2. Coding/tool-heavy/reasoning jobs (selected_model=blockrun/auto) resolved \
to local model instead of escalating → [warning].
3. Router-first job (selected_model=blockrun/auto) with no resolved_model \
recorded → [warning] (ClawRouter may not be exposing the upstream model).
4. Job with consecutive_local_failures >= {_CONSECUTIVE_LOCAL_FAILURE_THRESHOLD} \
→ [critical] (GN100 / local inference down).
5. Explicit model pin (selected_model != blockrun/auto) with no \
[router-pin: <reason>] marker for a non-coding job → [warning].

If {ROUTING_DECISIONS_PATH} does not exist, also check {ROUTING_OUTCOMES_PATH} \
(JSONL, one record per cron job — written by OpenClaw Items 1-4, may not yet exist).

If neither file exists, output:
[ACK] No routing decisions yet. OpenClaw gateway has not written any decisions.

If the file exists but contains no decisions in the last 60 minutes, output:
[ACK] Routing decisions file present. No decisions in the review window.

If all checks pass, output:
[ACK] Routing decisions reviewed. All policy checks passed.

If any violation is found, output:
[ESCALATION_NOTICE] <brief summary of highest-severity violation and agent/job name>.
"""
