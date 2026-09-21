#!/usr/bin/env python3
"""Logboard open→acknowledge→close linkage for governance artifacts.

Models the lifecycle of a governance alert/issue stored as JSON artifacts in
a fixture ``HERMES_HOME/governance/logboard`` tree.  Each artifact has a
``state`` field (open / ack / closed) and optional ``closure_link`` pointing
to a review packet.  Transitions are validated against a strict state
machine:

    open → ack → closed

Direct open→closed is rejected (an acknowledge step is mandatory so that
issues are never silently dropped).  A closure-link to a review packet is
required for the ack→closed transition.

This is a code-only contract: it reads/writes JSON files in a fixture tree.
No live governance mutation, no cron registration, no enforcement mode
change.

Usage as a library::

    from logboard_closure_link import (
        load_artifact, transition, close_with_link,
    )
    artifact = load_artifact(path)
    transition(artifact, "ack")
    close_with_link(artifact, review_packet_id="rp_2026Q3")

Usage as a CLI::

    python3 logboard_closure_link.py --logboard /tmp/fake/governance/logboard \\
        --artifact alert-001 --transition ack
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

VALID_STATES = ("open", "ack", "closed")
TRANSITIONS = {
    "open": {"ack"},
    "ack": {"closed"},
    "closed": set(),  # terminal
}


class TransitionError(ValueError):
    """Raised when a state transition is invalid."""


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_artifact(
    *,
    artifact_id: str | None = None,
    title: str = "",
    severity: str = "medium",
    board: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new logboard artifact in the ``open`` state."""
    artifact = {
        "artifact_id": artifact_id or f"alert-{uuid.uuid4().hex[:8]}",
        "state": "open",
        "title": title,
        "severity": severity,
        "board": board,
        "created_at": _now_iso(),
        "transitions": [
            {"to": "open", "ts": _now_iso()},
        ],
        "closure_link": None,
    }
    if extra:
        artifact.update(extra)
    return artifact


def load_artifact(path: Path) -> dict[str, Any]:
    """Load a logboard artifact from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "state" not in data:
        raise TransitionError(f"artifact at {path} has no 'state' field")
    if data["state"] not in VALID_STATES:
        raise TransitionError(
            f"artifact at {path} has invalid state {data['state']!r}"
        )
    return data


def save_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Save a logboard artifact to a JSON file (pretty-printed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _validate_transition(artifact: dict[str, Any], target: str) -> None:
    if target not in VALID_STATES:
        raise TransitionError(f"invalid target state {target!r}")
    current = artifact["state"]
    if target not in TRANSITIONS.get(current, set()):
        raise TransitionError(
            f"invalid transition {current!r} → {target!r}: "
            f"allowed from {current!r}: {sorted(TRANSITIONS.get(current, set()))}"
        )


def transition(
    artifact: dict[str, Any], target: str, *, actor: str | None = None
) -> dict[str, Any]:
    """Transition an artifact to *target* state in-memory.

    Returns the mutated artifact (caller is responsible for saving).
    """
    _validate_transition(artifact, target)
    artifact["state"] = target
    artifact.setdefault("transitions", []).append(
        {"to": target, "ts": _now_iso(), "actor": actor}
    )
    return artifact


def close_with_link(
    artifact: dict[str, Any],
    *,
    review_packet_id: str,
    actor: str | None = None,
) -> dict[str, Any]:
    """Transition an artifact from ``ack`` to ``closed`` with a closure link.

    The closure link ties the closed issue to a review packet id so that
    auditors can trace which review cycle disposed of the alert.
    """
    if not review_packet_id or not isinstance(review_packet_id, str):
        raise TransitionError("review_packet_id must be a non-empty string")
    _validate_transition(artifact, "closed")
    artifact["state"] = "closed"
    artifact["closure_link"] = review_packet_id
    artifact.setdefault("transitions", []).append(
        {
            "to": "closed",
            "ts": _now_iso(),
            "actor": actor,
            "closure_link": review_packet_id,
        }
    )
    return artifact


def validate_lifecycle(artifact: dict[str, Any]) -> bool:
    """Validate the full transition history of a closed artifact.

    Returns True iff:
        - The transition sequence starts at ``open``.
        - It passes through ``ack`` before ``closed``.
        - The final state is ``closed`` and has a ``closure_link``.
    """
    transitions = artifact.get("transitions", [])
    if not transitions:
        return False
    states = [t.get("to") for t in transitions]
    if not states or states[0] != "open":
        return False
    if "ack" not in states:
        return False
    ack_idx = states.index("ack")
    closed_idx = states.index("closed") if "closed" in states else -1
    if closed_idx == -1 or closed_idx < ack_idx:
        return False
    if artifact.get("state") != "closed":
        return False
    if not artifact.get("closure_link"):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Logboard open→ack→close lifecycle manager"
    )
    ap.add_argument("--logboard", required=True, help="Path to logboard dir")
    ap.add_argument("--artifact", required=True, help="Artifact id (filename stem)")
    ap.add_argument(
        "--transition",
        required=True,
        choices=["ack", "closed"],
        help="Target state",
    )
    ap.add_argument("--review-packet", default=None, help="Review packet id (for closed)")
    ap.add_argument("--actor", default=None, help="Actor performing the transition")
    args = ap.parse_args()

    logboard = Path(args.logboard)
    path = logboard / f"{args.artifact}.json"

    if not path.exists():
        print(f"ERROR: artifact not found: {path}", file=sys.stderr)
        return 2

    artifact = load_artifact(path)

    if args.transition == "closed":
        if not args.review_packet:
            print("ERROR: --review-packet required for closed transition", file=sys.stderr)
            return 2
        close_with_link(
            artifact, review_packet_id=args.review_packet, actor=args.actor
        )
    else:
        transition(artifact, args.transition, actor=args.actor)

    save_artifact(path, artifact)
    print(json.dumps(artifact, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
