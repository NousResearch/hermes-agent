#!/usr/bin/env python3
"""Torben Signal/open-loop capture surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torben_open_loops import add_loop, load_loops

ERIC_SIGNAL_SENDER = "+15163843337"
DEFAULT_SCORE_THRESHOLD = 0.7


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_sender(sender: str | None) -> str:
    digits = re.sub(r"\D+", "", str(sender or ""))
    return f"+{digits}" if digits else ""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def capture_key(*, kind: str, text: str, source_id: str | None = None) -> str:
    material = f"{kind}:{source_id or ''}:{_normalize_text(text)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _load_dedupe(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(payload, dict):
        return {}
    return {str(key): int(value) for key, value in payload.items()}


def _write_dedupe(path: Path, payload: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_confirmation(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _task_from_signal(text: str) -> tuple[str | None, str | None]:
    stripped = text.strip()
    match = re.match(r"^(?:task|todo|loop|remember)\s*[:\-]\s*(?P<item>.+)$", stripped, flags=re.I)
    if match:
        return match.group("item").strip(), None
    words = stripped.split()
    if len(words) < 3:
        return None, "What should I track, and what outcome do you want?"
    if not re.search(r"\b(call|email|schedule|book|renew|pay|send|draft|follow up|check|find|finish|submit)\b", stripped, re.I):
        return None, "What is the concrete next action for this?"
    return stripped, None


def _confirmation(loop_id: int, state: str, item: str) -> str:
    return f"Captured loop #{loop_id} as {state}: {item}"


def capture_signal(
    *,
    text: str,
    sender: str,
    tracker_path: Path,
    dedupe_path: Path,
    confirmations_path: Path,
    domain: str = "admin",
) -> dict[str, Any]:
    normalized_sender = _normalize_sender(sender)
    if normalized_sender != ERIC_SIGNAL_SENDER:
        return {
            "status": "rejected",
            "reason": "sender_not_eric",
            "sender": normalized_sender,
        }
    item, clarify = _task_from_signal(text)
    if clarify:
        return {
            "status": "clarify",
            "question": clarify,
            "wakeAgent": True,
        }
    assert item is not None
    key = capture_key(kind="signal", text=item)
    dedupe = _load_dedupe(dedupe_path)
    if key in dedupe:
        loop_id = dedupe[key]
        return {
            "status": "duplicate",
            "loop_id": loop_id,
            "confirmation": f"Already tracking loop #{loop_id}: {item}",
            "wakeAgent": True,
        }
    loop = add_loop(path=tracker_path, item=item, state="next-action", owner="eric", domain=domain, note=f"capture_key={key}")
    dedupe[key] = loop.id
    _write_dedupe(dedupe_path, dedupe)
    confirmation = {
        "schema": "torben.capture-confirmation.v1",
        "created_at": _iso(),
        "channel": "signal",
        "recipient": normalized_sender,
        "loop_id": loop.id,
        "text": _confirmation(loop.id, loop.state, loop.item),
    }
    _append_confirmation(confirmations_path, confirmation)
    return {
        "status": "captured",
        "loop": loop.to_dict(),
        "confirmation": confirmation["text"],
        "wakeAgent": True,
    }


def capture_auto_candidate(
    *,
    text: str,
    source: str,
    source_id: str,
    score: float,
    tracker_path: Path,
    dedupe_path: Path,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    domain: str = "admin",
) -> dict[str, Any]:
    key = capture_key(kind=source, text=text, source_id=source_id)
    if score < threshold:
        return {
            "status": "withheld",
            "reason": "below_validation_threshold",
            "score": score,
            "threshold": threshold,
        }
    dedupe = _load_dedupe(dedupe_path)
    if key in dedupe:
        return {"status": "duplicate", "loop_id": dedupe[key], "score": score}
    loop = add_loop(
        path=tracker_path,
        item=text,
        state="next-action",
        owner="eric",
        domain=domain,
        note=f"candidate_source={source};source_id={source_id};score={score};capture_key={key}",
    )
    dedupe[key] = loop.id
    _write_dedupe(dedupe_path, dedupe)
    return {"status": "candidate", "loop": loop.to_dict(), "score": score, "threshold": threshold}


def _default_state_path(name: str) -> Path:
    home = os.getenv("HERMES_HOME")
    root = Path(home).expanduser() if home else Path(".")
    return root / "state" / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", default=None)
    parser.add_argument("--dedupe", default=None)
    parser.add_argument("--confirmations", default=None)
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    signal = subparsers.add_parser("signal")
    signal.add_argument("--sender", required=True)
    signal.add_argument("--domain", default="admin")
    signal.add_argument("text", nargs="+")

    auto = subparsers.add_parser("auto")
    auto.add_argument("--source", required=True)
    auto.add_argument("--source-id", required=True)
    auto.add_argument("--score", type=float, required=True)
    auto.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    auto.add_argument("--domain", default="admin")
    auto.add_argument("text", nargs="+")
    args = parser.parse_args(argv)

    tracker = Path(args.tracker) if args.tracker else _default_state_path("torben-open-loops.csv")
    dedupe = Path(args.dedupe) if args.dedupe else _default_state_path("torben-capture-dedupe.json")
    confirmations = Path(args.confirmations) if args.confirmations else _default_state_path("torben-capture-confirmations.jsonl")
    if args.command == "signal":
        payload = capture_signal(
            text=" ".join(args.text),
            sender=args.sender,
            tracker_path=tracker,
            dedupe_path=dedupe,
            confirmations_path=confirmations,
            domain=args.domain,
        )
    elif args.command == "auto":
        payload = capture_auto_candidate(
            text=" ".join(args.text),
            source=args.source,
            source_id=args.source_id,
            score=args.score,
            threshold=args.threshold,
            tracker_path=tracker,
            dedupe_path=dedupe,
            domain=args.domain,
        )
    else:
        raise ValueError(f"Unhandled command: {args.command}")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
