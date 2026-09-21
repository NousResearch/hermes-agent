#!/usr/bin/env python3
"""Self-assessment submission contract for Denji review cycles.

Validates a submission payload (profile, cycle, answers, ts) and appends a
validated ``profile.self_eval.submit`` event to the profile activity ledger.

This is a code-only contract: it validates input shape, checks cycle and
profile constraints, then records the event.  No live governance mutation,
no cron registration, no enforcement mode change — it is a pure function
that callers (tests, review-cycle scripts, or a future cron wrapper) invoke
with an explicit payload.

Usage as a library::

    from denji_self_eval_submit import submit_self_eval
    result = submit_self_eval(payload, hermes_home=Path("/tmp/fake"))

Usage as a CLI::

    HERMES_HOME=/tmp/fake python3 denji-self-eval-submit.py \\
        --profile octacon --cycle weekly --answers '{"quality": 4}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as a standalone script (scripts/ is on sys.path when imported
# by tests that insert the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_cli.profile_activity_ledger import append_event, query_events

VALID_CYCLES = ("weekly", "monthly", "quarterly")
EVENT_TYPE = "profile.self_eval.submit"


class SubmissionError(ValueError):
    """Raised when a self-assessment submission fails validation."""


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a self-assessment submission payload.

    Required fields:
        - profile: non-empty string (the profile submitting the eval)
        - cycle: one of VALID_CYCLES
        - answers: dict mapping question keys to answers (may be empty but
          must be a dict)
        - ts: integer epoch timestamp (optional; defaults to now)

    Optional fields:
        - actor_profile: who triggered the submission (defaults to profile)

    Returns the normalised payload with defaults filled in.

    Raises SubmissionError on validation failure.
    """
    if not isinstance(payload, dict):
        raise SubmissionError("payload must be a dict")

    profile = payload.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise SubmissionError("profile must be a non-empty string")
    profile = profile.strip()

    cycle = payload.get("cycle")
    if cycle not in VALID_CYCLES:
        raise SubmissionError(
            f"cycle must be one of {VALID_CYCLES}, got {cycle!r}"
        )

    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise SubmissionError("answers must be a dict")

    ts = payload.get("ts")
    if ts is None:
        ts = int(time.time())
    if not isinstance(ts, int) or ts < 0:
        raise SubmissionError("ts must be a non-negative integer epoch")

    actor = payload.get("actor_profile") or profile

    return {
        "profile": profile,
        "cycle": cycle,
        "answers": answers,
        "ts": ts,
        "actor_profile": actor,
    }


def submit_self_eval(
    payload: dict[str, Any],
    *,
    hermes_home: Path | None = None,
) -> dict[str, Any]:
    """Validate and record a self-assessment submission.

    Sets ``HERMES_HOME`` (if *hermes_home* is provided) so the ledger writes
    to the fixture tree, validates the payload, appends a
    ``profile.self_eval.submit`` event, then queries the ledger to confirm
    the event round-tripped.

    Returns a dict with:
        - event_id: the ledger event id
        - verified: True if the event was confirmed via query_events
        - payload: the normalised payload that was recorded

    No live mutation: callers must pass a fixture ``hermes_home``.  When
    ``hermes_home`` is None, the current ``HERMES_HOME`` env var is used.
    """
    if hermes_home is not None:
        os.environ["HERMES_HOME"] = str(hermes_home)

    normalised = validate_payload(payload)

    event_id = append_event(
        source="denji-self-eval-submit",
        event_type=EVENT_TYPE,
        actor_profile=normalised["actor_profile"],
        target_profile=normalised["profile"],
        object_type="profile.self_eval",
        object_id=normalised["cycle"],
        summary=f"Self-eval submission for {normalised['profile']} ({normalised['cycle']})",
        payload={
            "cycle": normalised["cycle"],
            "answers": normalised["answers"],
        },
        occurred_at=normalised["ts"],
    )

    # Verify the event round-tripped through the ledger schema.
    events = query_events(
        event_types=[EVENT_TYPE],
        target_profile=normalised["profile"],
        limit=10,
    )
    verified = any(e.get("event_id") == event_id for e in events)

    return {
        "event_id": event_id,
        "verified": verified,
        "payload": normalised,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Submit a self-assessment event to the profile activity ledger"
    )
    ap.add_argument("--profile", required=True, help="Profile submitting the eval")
    ap.add_argument(
        "--cycle", required=True, choices=VALID_CYCLES, help="Review cycle"
    )
    ap.add_argument(
        "--answers", default="{}", help="JSON dict of answers"
    )
    ap.add_argument(
        "--ts", type=int, default=None, help="Epoch timestamp (default: now)"
    )
    args = ap.parse_args()

    try:
        answers = json.loads(args.answers)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --answers is not valid JSON: {exc}", file=sys.stderr)
        return 2

    payload = {
        "profile": args.profile,
        "cycle": args.cycle,
        "answers": answers,
        "ts": args.ts,
    }

    try:
        result = submit_self_eval(payload)
    except SubmissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
