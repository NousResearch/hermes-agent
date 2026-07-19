#!/usr/bin/env python3
"""
Retroactive session labeling for the finetune pipeline.

Stateless command-driven tool for labeling historical sessions and individual
assistant turns. Designed to run inside Hermes via /finetune retro <subcommand>
where each invocation does one thing and returns. State lives in feedback.jsonl.

Usage:
    python retro.py list [--limit N]              Show top N priority sessions
    python retro.py show <session_id>              Show full conversation
    python retro.py good <session_id> [turns]      Label good (all or specific turns)
    python retro.py bad <session_id> [turns]       Label bad (all or specific turns)
    python retro.py skip <session_id>              Skip — drop priority on this session
    python retro.py stats                          Show labeling progress

Turn syntax:
    "1,3,5"     individual turns
    "1-5"       inclusive range
    "1,3-5,7"   mixed
    omit        all assistant turns in the session
"""

import argparse
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import common
from common import (
    FEEDBACK_PATH, SCORED_DIR,
    load_config, read_jsonl, append_jsonl, content_to_text, logger,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ── Priority ranking ────────────────────────────────────────────────────

WEIGHTS = {
    "uncertainty": 0.50,   # MVP weights — boundary_proximity dropped (Phase 2)
    "tool_density": 0.25,
    "turn_count":   0.10,
    "recency":      0.15,
}

RECENCY_HALFLIFE_DAYS = 14.0


def _uncertainty_score(composite: float) -> float:
    """Peaks at 0.5, drops to 0 at 0.0 and 1.0."""
    return 1.0 - abs(2.0 * composite - 1.0)


def _tool_density(tool_call_count: int) -> float:
    """Saturates at 10 tool calls."""
    return min(1.0, tool_call_count / 10.0)


def _turn_count_score(assistant_turns: int) -> float:
    """Saturates at 15 assistant turns."""
    return min(1.0, assistant_turns / 15.0)


def _recency_decay(started_at_iso: str) -> float:
    """Exponential decay with 14-day half-life.

    New extractions serialize timezone-aware UTC timestamps; legacy records
    are naive *local* time (old extract.py used datetime.fromtimestamp).
    Interpret naive values as local so legacy data isn't skewed by the UTC
    offset.
    """
    if not started_at_iso or not isinstance(started_at_iso, str):
        return 0.0
    try:
        ts = datetime.fromisoformat(started_at_iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0.0
    if ts.tzinfo is None:
        ts = ts.astimezone()  # naive == legacy local time
    now = datetime.now(timezone.utc)
    days_old = (now - ts).total_seconds() / 86400.0
    return math.exp(-0.693 * days_old / RECENCY_HALFLIFE_DAYS)


def compute_priority(session: Dict) -> float:
    """Compute priority score for a session (higher = label this first)."""
    scoring = session.get("scoring", {})
    composite = scoring.get("composite_score", 0.5)
    metadata = session.get("metadata", {})
    tool_calls = metadata.get("tool_call_count", 0)

    turns = session.get("turns", [])
    assistant_turns = sum(1 for t in turns if t.get("role") == "assistant")

    started_at = session.get("started_at", "")

    return (
        WEIGHTS["uncertainty"]  * _uncertainty_score(composite)
        + WEIGHTS["tool_density"] * _tool_density(tool_calls)
        + WEIGHTS["turn_count"]   * _turn_count_score(assistant_turns)
        + WEIGHTS["recency"]      * _recency_decay(started_at)
    )


# ── Feedback I/O ─────────────────────────────────────────────────────────

def load_feedback() -> List[Dict]:
    """Load all feedback records."""
    return read_jsonl(common.FEEDBACK_PATH)


def labeled_session_ids(feedback: List[Dict]) -> Set[str]:
    """Set of session IDs that have at least one retro session-level label.

    Excludes skip records — those are tracked by skipped_session_ids and
    should not count as 'labeled' for queue-filtering purposes.
    """
    return {
        r.get("session_id")
        for r in feedback
        if r.get("source") == "retro"
        and r.get("turn_index") is None
        and r.get("signal") != "skip"
        and r.get("session_id")
    }


def skipped_session_ids(feedback: List[Dict]) -> Set[str]:
    """Set of session IDs explicitly skipped via retro."""
    return {
        r.get("session_id")
        for r in feedback
        if r.get("source") == "retro"
        and r.get("signal") == "skip"
        and r.get("session_id")
    }


def has_turn_label(feedback: List[Dict], session_id: str) -> bool:
    """Whether any turn in this session has a retro label."""
    return any(
        r.get("source") == "retro"
        and r.get("session_id") == session_id
        and r.get("turn_index") is not None
        for r in feedback
    )


def write_label(
    session_id: str,
    signal: str,
    score: float,
    turn_index: Optional[int] = None,
    note: Optional[str] = None,
) -> None:
    """Append a label record to feedback.jsonl."""
    record = {
        "session_id": session_id,
        "signal": signal,
        "score": score,
        "source": "retro",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if turn_index is not None:
        record["turn_index"] = turn_index
    if note:
        record["note"] = note
    append_jsonl(common.FEEDBACK_PATH, [record])


# ── Session loading ──────────────────────────────────────────────────────

def load_all_scored() -> List[Dict]:
    """Load every scored session from disk."""
    sessions = []
    for path in sorted(common.SCORED_DIR.glob("scored_*.jsonl")):
        sessions.extend(read_jsonl(path))
    # De-dup by session_id, keeping the most recently scored copy
    by_id: Dict[str, Dict] = {}
    for s in sessions:
        sid = s.get("session_id")
        if sid:
            by_id[sid] = s
    return list(by_id.values())


def find_session(session_id: str) -> Optional[Dict]:
    """Look up a session by ID, with prefix matching."""
    sessions = load_all_scored()
    # Exact match first
    for s in sessions:
        if s.get("session_id") == session_id:
            return s
    # Prefix match
    matches = [s for s in sessions if s.get("session_id", "").startswith(session_id)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous session ID '{session_id}' — {len(matches)} matches:")
        for m in matches[:10]:
            print(f"  {m.get('session_id')}")
        if len(matches) > 10:
            print(f"  ... and {len(matches) - 10} more")
        # Compute the minimum unique prefix length so the user knows what to type
        min_unique = _min_unique_prefix_length(session_id, [m.get("session_id", "") for m in matches])
        if min_unique:
            sample = matches[0].get("session_id", "")[:min_unique]
            print(f"\n  Try a longer prefix — e.g. '{sample}' for the first match.")
        return None
    return None


def _min_unique_prefix_length(prefix: str, ids: List[str]) -> int:
    """Return the smallest length at which the first ID in `ids` becomes unique."""
    if not ids:
        return 0
    target = ids[0]
    for length in range(len(prefix) + 1, len(target) + 1):
        target_prefix = target[:length]
        if sum(1 for i in ids if i.startswith(target_prefix)) == 1:
            return length
    return len(target)


def parse_turn_spec(spec: str, max_turn: int) -> List[int]:
    """Parse a turn specifier like '1,3-5,7' into a sorted list of turn numbers."""
    if not spec:
        return list(range(1, max_turn + 1))

    result: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                lo, hi = part.split("-", 1)
                lo_i, hi_i = int(lo), int(hi)
                result.update(range(min(lo_i, hi_i), max(lo_i, hi_i) + 1))
            except ValueError:
                print(f"Bad turn spec: '{part}' (expected 'N-M')")
                return []
        else:
            try:
                result.add(int(part))
            except ValueError:
                print(f"Bad turn spec: '{part}' (expected integer)")
                return []

    valid = sorted(t for t in result if 1 <= t <= max_turn)
    invalid = sorted(t for t in result if t < 1 or t > max_turn)
    if invalid:
        print(f"Ignoring out-of-range turns: {invalid} (session has {max_turn} assistant turns)")
    return valid


# ── Subcommands ──────────────────────────────────────────────────────────

def cmd_list(args):
    """Show the priority queue of unlabeled sessions."""
    sessions = load_all_scored()
    if not sessions:
        print("No scored sessions found. Run /finetune extract && /finetune score first.")
        return

    feedback = load_feedback()
    labeled = labeled_session_ids(feedback) | skipped_session_ids(feedback)
    # Sessions with turn-level labels leave the queue too — labeling turns
    # 2 and 4 of a session is a completed review, not a pending one.
    unlabeled = [
        s for s in sessions
        if s.get("session_id") not in labeled
        and not has_turn_label(feedback, s.get("session_id"))
    ]

    if not unlabeled:
        print(f"All {len(sessions)} sessions have been labeled. /finetune retro stats")
        return

    ranked = sorted(
        ((compute_priority(s), s) for s in unlabeled),
        key=lambda x: x[0],
        reverse=True,
    )

    print(f"[Retro queue] {len(unlabeled)} unlabeled / {len(sessions)} total scored")
    print()

    for i, (priority, s) in enumerate(ranked[:args.limit], 1):
        sid = s.get("session_id", "")
        scoring = s.get("scoring", {})
        composite = scoring.get("composite_score", 0.0)
        bucket = scoring.get("bucket", "?")
        meta = s.get("metadata", {})
        n_turns = sum(1 for t in s.get("turns", []) if t.get("role") == "assistant")
        n_tools = meta.get("tool_call_count", 0)
        started = s.get("started_at", "")[:10]

        # First user turn as preview
        preview = ""
        for t in s.get("turns", []):
            if t.get("role") == "user":
                preview = content_to_text(t.get("content")).strip().replace("\n", " ")[:60]
                break

        print(f"  {i:>2}. {sid[:20]:<20}  pri={priority:.2f}  "
              f"score={composite:.2f} [{bucket:<7}]  "
              f"{n_turns}t/{n_tools}T  {started}")
        if preview:
            print(f"      {preview}{'…' if len(preview) >= 60 else ''}")

    print()
    print("Next: /finetune retro show <id_prefix>")


def cmd_show(args):
    """Show the full conversation of a session with turn numbers."""
    session = find_session(args.session_id)
    if session is None:
        print(f"Session not found: {args.session_id}")
        sys.exit(1)

    sid = session.get("session_id", "")
    started = session.get("started_at", "")
    meta = session.get("metadata", {})
    scoring = session.get("scoring", {})
    composite = scoring.get("composite_score", 0.0)
    turn_scores = dict(scoring.get("turn_scores", []))

    print(f"Session: {sid}")
    print(f"  Started:  {started}")
    print(f"  Source:   {meta.get('source', '?')}  Model: {meta.get('model', '?')}")
    print(f"  Score:    {composite:.2f} [{scoring.get('bucket', '?')}]")
    print(f"  Turns:    {len(session.get('turns', []))}  "
          f"(tool calls: {meta.get('tool_call_count', 0)})")
    print()

    # Check existing retro labels
    feedback = load_feedback()
    existing = [
        r for r in feedback
        if r.get("session_id") == sid and r.get("source") == "retro"
    ]
    if existing:
        print(f"  [Already has {len(existing)} retro label(s)]")
        print()

    assistant_counter = 0
    for i, turn in enumerate(session.get("turns", [])):
        role = turn.get("role", "?")
        content = content_to_text(turn.get("content")).strip()

        if role == "system":
            print(f"  [system] {content[:200]}{'…' if len(content) > 200 else ''}")
        elif role == "user":
            print(f"  [user] {content[:600]}{'…' if len(content) > 600 else ''}")
        elif role == "assistant":
            assistant_counter += 1
            t_score = turn_scores.get(i, turn_scores.get(str(i), 0.5))
            t_bucket = "good" if t_score >= 0.7 else ("bad" if t_score < 0.4 else "neutral")
            print(f"  [assistant t{assistant_counter}, score {t_score:.2f} {t_bucket}]")
            print(f"    {content[:600]}{'…' if len(content) > 600 else ''}")
            if turn.get("tool_calls"):
                tc_names = [
                    tc.get("function", {}).get("name", "?")
                    for tc in turn["tool_calls"]
                    if isinstance(tc, dict)
                ]
                print(f"    [tool_calls: {', '.join(tc_names)}]")
        elif role == "tool":
            tname = turn.get("tool_name", "")
            print(f"  [tool {tname}] {content[:200]}{'…' if len(content) > 200 else ''}")
        print()

    n_assistant = assistant_counter
    print(f"  ({n_assistant} assistant turns total)")
    print()
    print("Label with:")
    print(f"  /finetune retro good {sid}        — all turns good")
    print(f"  /finetune retro good {sid} 2,4    — turns 2 and 4 good")
    print(f"  /finetune retro bad {sid} 3       — turn 3 bad")
    print(f"  /finetune retro skip {sid}        — skip this session")


def _apply_label(session_id: str, signal: str, turn_spec: Optional[str]) -> None:
    """Common label-write logic."""
    session = find_session(session_id)
    if session is None:
        print(f"Session not found: {session_id}")
        sys.exit(1)

    sid = session.get("session_id", "")
    score = 1.0 if signal == "good" else 0.0

    n_assistant = sum(1 for t in session.get("turns", []) if t.get("role") == "assistant")

    if not turn_spec:
        # Session-level label — write ONE session-scope record (no
        # turn_index). Expanding it into per-turn records would clobber
        # earlier turn-level labels; keeping it session-scoped lets
        # turn-specific records always win regardless of write order
        # (score.py applies session-level first, then per-turn overrides).
        write_label(sid, signal, score)
        print(f"Labeled session {sid[:12]} as {signal} ({n_assistant} turns; "
              f"existing per-turn labels still take precedence)")
    else:
        turns = parse_turn_spec(turn_spec, n_assistant)
        if not turns:
            print("No valid turns to label.")
            sys.exit(1)
        for turn_num in turns:
            write_label(sid, signal, score, turn_index=turn_num)
        print(f"Labeled turns {turns} of session {sid[:12]} as {signal}")


def cmd_good(args):
    _apply_label(args.session_id, "good", args.turns)


def cmd_bad(args):
    _apply_label(args.session_id, "bad", args.turns)


def cmd_skip(args):
    """Mark a session skipped — drops it out of the queue without labeling."""
    session = find_session(args.session_id)
    if session is None:
        print(f"Session not found: {args.session_id}")
        sys.exit(1)

    sid = session.get("session_id", "")
    write_label(sid, "skip", 0.5)  # 0.5 = neutral, doesn't affect scoring
    print(f"Skipped session {sid[:12]}")


def cmd_stats(args):
    """Show retro labeling progress."""
    sessions = load_all_scored()
    feedback = load_feedback()

    retro_records = [r for r in feedback if r.get("source") == "retro"]
    session_level = [r for r in retro_records if r.get("turn_index") is None and r.get("signal") != "skip"]
    turn_level = [r for r in retro_records if r.get("turn_index") is not None]
    skipped = [r for r in retro_records if r.get("signal") == "skip"]

    good_session = sum(1 for r in session_level if r.get("signal") == "good")
    bad_session = sum(1 for r in session_level if r.get("signal") == "bad")
    good_turns = sum(1 for r in turn_level if r.get("signal") == "good")
    bad_turns = sum(1 for r in turn_level if r.get("signal") == "bad")

    labeled_sids = labeled_session_ids(feedback) | skipped_session_ids(feedback)
    unlabeled_count = sum(
        1 for s in sessions
        if s.get("session_id") not in labeled_sids
        and not has_turn_label(feedback, s.get("session_id"))
    )

    print("[Retro statistics]")
    print(f"  Total scored sessions:      {len(sessions)}")
    print(f"  Sessions in queue:          {unlabeled_count}")
    print(f"  Session-level labels:       {good_session} good, {bad_session} bad, {len(skipped)} skipped")
    print(f"  Turn-level labels:          {good_turns} good, {bad_turns} bad")
    print(f"  Total feedback records:     {len(feedback)}")


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Retroactive labeling for finetune",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="Show priority queue")
    p_list.add_argument("--limit", type=int, default=10, help="Number of sessions to show")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show a session's full conversation")
    p_show.add_argument("session_id", help="Session ID (prefix match supported)")
    p_show.set_defaults(func=cmd_show)

    p_good = sub.add_parser("good", help="Label session/turns as good")
    p_good.add_argument("session_id")
    p_good.add_argument("turns", nargs="?", default=None,
                        help="Turn spec like '1,3-5' (default: all assistant turns)")
    p_good.set_defaults(func=cmd_good)

    p_bad = sub.add_parser("bad", help="Label session/turns as bad")
    p_bad.add_argument("session_id")
    p_bad.add_argument("turns", nargs="?", default=None,
                       help="Turn spec like '1,3-5' (default: all assistant turns)")
    p_bad.set_defaults(func=cmd_bad)

    p_skip = sub.add_parser("skip", help="Skip a session — removes from queue")
    p_skip.add_argument("session_id")
    p_skip.set_defaults(func=cmd_skip)

    p_stats = sub.add_parser("stats", help="Show labeling progress")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
