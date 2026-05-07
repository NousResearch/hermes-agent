"""CLI confirm UI for Phase 2 auto-memory proposals.

Called from cli.py's exit handler (before shutdown_memory_provider).
Shows the user a list of proposed memory entries, asks which to accept
edit / reject, and commits accepted ones via the conflict-resolution
pipeline.

Design:
  * BLOCKS the exit by ~1-2 LLM calls + user input. That's intentional —
    the user is exiting; they have a moment to review.
  * Single Q-press to accept all, single d-press to discard all, batch
    mode for power users.
  * Edit support: pick an entry by letter, get prompt-toolkit input
    pre-populated with the proposal, edit and re-submit.
  * Each entry shows the conflict verdict (NEW / DUPLICATE / REFINEMENT
    / CONTRADICTION) before the user decides. Contradictions surface
    BOTH the new and existing fact text.

The UI is plain-print + input(). prompt_toolkit niceties are nice but
this runs on session exit when the prompt_toolkit session may already
be torn down.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _shorten(text: str, width: int = 90) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _wrap_indented(text: str, indent: str = "      ", width: int = 100) -> str:
    """Render full text wrapped to ``width`` and prefixed by ``indent`` per line.

    Used when we have only a handful of proposals and want the user to see
    the entire content rather than a truncated head. Newlines in the source
    are normalized to spaces first so the rendered block is one logical
    paragraph wrapped to terminal width.
    """
    flat = text.strip().replace("\n", " ")
    if len(flat) <= width:
        return f"{indent}{flat}"
    out: List[str] = []
    line: str = ""
    for word in flat.split():
        if not line:
            line = word
            continue
        if len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}"
    if line:
        out.append(line)
    return "\n".join(f"{indent}{ln}" for ln in out)


def _print_separator() -> None:
    print("─" * 78, flush=True)


def _countdown_for_review(seconds: int = 3) -> bool:
    """Block briefly with a 'press any key to review' countdown.

    Returns True when the user pressed a key (caller should fall through
    to the interactive prompt) or False when the timer ran out (caller
    should auto-accept all).

    On a non-tty (CI, gateway, redirected stdin), returns False
    immediately — there's no human watching to press anything, and we
    don't want to gate session exit on a 3-second wait.

    Falls back to the regular interactive prompt (returning True) on any
    error: the goal is to never accidentally drop proposals because
    raw-mode tty manipulation hit a corner case.
    """
    import sys
    import time
    try:
        # Stdin must be a real tty — gateway/cron/CI all run with
        # redirected stdin and select would block forever or report
        # spurious readiness.
        if not sys.stdin.isatty():
            return False
    except Exception:
        return False

    try:
        import select
        import termios
        import tty
    except Exception:
        # Windows or other platforms without termios — skip the
        # countdown, hand control straight to the interactive prompt.
        return True

    try:
        fd = sys.stdin.fileno()
    except (ValueError, OSError, Exception):
        # Pseudo-files (pytest capture, some IDE consoles) raise on
        # fileno(). Treat as non-tty and auto-accept.
        return False
    try:
        original = termios.tcgetattr(fd)
    except termios.error:
        return True  # Not a real terminal — bail to interactive path.

    interrupted = False
    try:
        tty.setcbreak(fd)
        end = time.monotonic() + seconds
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            # Refresh the inline countdown each second.
            ticks = int(remaining) + 1
            sys.stdout.write(
                f"\rAuto-accepting all in {ticks}s — press any key to review... "
            )
            sys.stdout.flush()
            ready, _, _ = select.select([sys.stdin], [], [], min(1.0, remaining))
            if ready:
                # Drain the keystroke so it doesn't bleed into the next
                # prompt's input buffer.
                try:
                    sys.stdin.read(1)
                except Exception:
                    pass
                interrupted = True
                break
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
        except termios.error:
            pass
        # Clear the countdown line so the next print starts on a clean row.
        sys.stdout.write("\r" + " " * 78 + "\r")
        sys.stdout.flush()
    return interrupted


def _classify_proposals(proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run conflict classification on each proposal. Returns annotated list.

    Each annotated entry has the original fields plus:
        verdict: ConflictVerdict
        outcome: dict (the would-be apply_verdict result, NOT applied)
    """
    from tools.memory_extraction import conflict
    annotated: List[Dict[str, Any]] = []
    for p in proposals:
        try:
            v = conflict.classify(p["content"])
        except Exception as e:
            logger.warning("memory confirm: classify failed: %s", e)
            from tools.memory_extraction.conflict import ConflictVerdict
            v = ConflictVerdict(verdict="NEW", rationale=f"classify failed: {e}")
        annotated.append({**p, "verdict": v})
    return annotated


def _commit_proposal(p: Dict[str, Any]) -> Dict[str, Any]:
    from tools.memory_extraction import conflict
    return conflict.apply_verdict(
        p["verdict"], p, auto_commit=True,
    )


def confirm_and_commit(
    session_id: str,
    final_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the confirm UI. Returns a summary dict matching on_session_end.

    Safe to call when there are no pending proposals — just returns a
    summary with all-zero counts.
    """
    summary: Dict[str, Any] = {
        "session_id": session_id,
        "buffered": 0,
        "final_proposed": 0,
        "committed": 0,
        "skipped": 0,
        "actions": [],
    }
    if not session_id:
        return summary

    # Step 1: get the current buffer + run final extraction pass to
    # reconcile. We piggyback on the existing on_session_end logic but
    # pass our own confirm_callback.
    try:
        from tools.memory_extraction import extractor, buffer as _buf
    except Exception as e:
        logger.warning("memory confirm: extractor import failed: %s", e)
        return summary

    if not extractor.is_enabled():
        return summary

    buffered = _buf.get_session_entries(session_id)
    if not buffered and not final_messages:
        return summary

    print()
    _print_separator()
    print("Memory: reviewing proposals from this session...")
    _print_separator()

    def _confirm_callback(proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _interactive_review(proposals)

    summary = extractor.on_session_end(
        session_id, final_messages or [],
        interactive=True,
        confirm_callback=_confirm_callback,
    )

    print()
    _print_separator()
    proposed_total = summary["final_proposed"]
    proposed_noun = "entry" if proposed_total == 1 else "entries"
    print(
        f"Memory: committed {summary['committed']} of "
        f"{proposed_total} proposed {proposed_noun}."
    )
    if summary["committed"]:
        for action in summary["actions"]:
            tag = {
                "stored": "+",
                "refined": "~",
                "deduplicated": "=",
                "superseded": "!",
            }.get(action.get("outcome", ""), "?")
            print(f"  {tag} {_shorten(action.get('content') or '')}")
    _print_separator()
    return summary


def _interactive_review(
    proposals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Interactive triage. Returns the user-approved subset.

    Each proposal is enriched with a conflict verdict before the user
    sees it so they can make informed decisions on REFINEMENT /
    CONTRADICTION cases.

    Display rules:
      - When N <= 3, content is shown in full (wrapped to terminal width)
        rather than truncated, since the user can easily read all of it.
      - When N >= 4, content is shortened to fit one line; the user can
        run ``show <letter>`` to expand a single entry.
      - Each entry shows its tier+target (warm-tier categories or hot
        tier targets like ``hot:user`` / ``hot:memory``) so accept-all
        doesn't quietly bloat the always-loaded prompt budget.
      - Existing entries with semantic overlap (REFINEMENT / DUPLICATE /
        CONTRADICTION) display the matched fact text inline so the user
        can spot near-duplicate accretion before approving.

    Auto-accept countdown:
      - After rendering the proposals, a 3-second "press any key to
        review" countdown runs. If the user presses anything, control
        falls through to the interactive prompt as before. If the timer
        expires, all proposals are auto-accepted. This is the fast path
        for the common case where the user is just exiting and the
        proposals look fine; the explicit prompt remains available for
        edits / rejects / partial accepts.
      - Non-tty stdin (gateway, cron, CI) skips the countdown and
        auto-accepts immediately — no human watching means no point
        gating exit on a wall-clock wait.
    """
    if not proposals:
        return []

    annotated = _classify_proposals(proposals)
    n = len(annotated)
    show_full = n <= 3

    # Grammar: "1 entry" vs "N entries"
    noun = "entry" if n == 1 else "entries"

    print()
    print(f"{n} proposed memory {noun} from this session:")
    print()

    for i, p in enumerate(annotated):
        _render_proposal(i, p, show_full=show_full)

    print()

    # Auto-accept countdown. Returns True when the user wants to review
    # interactively, False when the timer expired and we should accept
    # everything as-shown.
    if not _countdown_for_review(seconds=3):
        return annotated

    print("Choices:")
    if show_full:
        print("  letters (e.g. 'a c') — accept those entries")
    else:
        print("  letters (e.g. 'a c d') — accept those entries")
        print("  'show <letter>' — print one entry's full content")
    print("  'all' — accept everything")
    print("  'none' — reject everything (proposals dropped)")
    print("  'reject <letter>' — drop a single entry, then re-prompt")
    print("  'edit <letter>' — edit one entry's content before deciding")
    print("  'skip' — leave proposals in the buffer for next session")
    print()

    # Prompt default. For N <= 3, "all" is the safe default (the user has
    # seen every entry in full). For larger batches, force an explicit
    # selection — pressing Enter with no input is a no-op.
    default_label = "all" if show_full else "no default — pick letters"
    prompt_str = f"Accept which? [{default_label}]: "

    while True:
        try:
            raw = input(prompt_str).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n(skipping — proposals remain buffered)")
            return []

        if not raw:
            if show_full:
                return annotated
            print("  pick letters, or type 'all' / 'none' / 'skip'")
            continue

        if raw == "all":
            return annotated

        if raw == "none":
            return []

        if raw == "skip":
            # Re-stash will happen automatically when we return [] AND
            # auto_commit_session_end is False.
            return []

        if raw.startswith("show "):
            idx = _resolve_letter(raw.split(" ", 1)[1].strip(), len(annotated))
            if idx is None:
                continue
            print()
            _render_proposal(idx, annotated[idx], show_full=True)
            print()
            continue

        if raw.startswith("reject "):
            idx = _resolve_letter(raw.split(" ", 1)[1].strip(), len(annotated))
            if idx is None:
                continue
            dropped = annotated.pop(idx)
            print(f"  dropped: {_shorten(dropped['content'])}")
            if not annotated:
                print("  (no entries left)")
                return []
            # Re-render the remaining list with fresh letters and re-prompt.
            print()
            print(f"{len(annotated)} entries remaining:")
            print()
            new_show_full = len(annotated) <= 3
            for j, q in enumerate(annotated):
                _render_proposal(j, q, show_full=new_show_full)
            print()
            continue

        if raw.startswith("edit "):
            idx = _resolve_letter(raw.split(" ", 1)[1].strip(), len(annotated))
            if idx is None:
                continue
            current = annotated[idx]["content"]
            print(f"\nCurrent: {current}")
            try:
                new_text = input("New text (blank = keep existing): ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if new_text:
                annotated[idx]["content"] = new_text
                # Re-run classification with the edited content
                from tools.memory_extraction import conflict
                annotated[idx]["verdict"] = conflict.classify(new_text)
                print(f"  edited; new verdict: {annotated[idx]['verdict'].verdict}")
            continue

        # Letter list
        chosen: List[Dict[str, Any]] = []
        invalid = False
        for tok in raw.replace(",", " ").split():
            if len(tok) != 1:
                print(f"  invalid token {tok!r}")
                invalid = True
                break
            idx = ord(tok) - ord("a")
            if not (0 <= idx < len(annotated)):
                print(f"  out of range: {tok!r}")
                invalid = True
                break
            chosen.append(annotated[idx])
        if not invalid:
            return chosen


def _resolve_letter(letter: str, count: int) -> Optional[int]:
    """Validate a single-letter selector, print an error and return None on failure."""
    if not letter or len(letter) != 1:
        print(f"  invalid letter {letter!r}")
        return None
    idx = ord(letter) - ord("a")
    if not (0 <= idx < count):
        print(f"  out of range: {letter!r}")
        return None
    return idx


def _render_proposal(i: int, p: Dict[str, Any], *, show_full: bool) -> None:
    """Print one annotated proposal with tier indicator + dedup hint.

    ``show_full=True`` renders the content wrapped to terminal width;
    ``False`` truncates to one line (used in dense lists).
    """
    letter = chr(ord("a") + i)
    v = p["verdict"]
    verdict_tag = {
        "NEW": "+ NEW",
        "DUPLICATE": "= DUPE",
        "REFINEMENT": "~ REFINE",
        "CONTRADICTION": "! CONFLICT",
    }.get(v.verdict, v.verdict)

    # Tier indicator. All Phase 2 auto-extracted proposals currently land
    # in the warm tier (extractor.on_session_end → conflict.apply_verdict
    # → warm_store.add). If a proposal carries an explicit ``tier``/``target``
    # field (e.g. from a future hot-tier extractor), surface it here
    # instead so the user can tell warm:preferences from hot:user at a
    # glance.
    tier = (p.get("tier") or "warm").lower()
    if tier == "hot":
        tier_label = f"hot:{p.get('target') or 'memory'}"
    else:
        tier_label = f"warm:{p.get('category') or 'general'}"

    if show_full:
        # Header line with metadata, then the full content wrapped below.
        print(f"  [{letter}] [{verdict_tag}] [{tier_label}]")
        print(_wrap_indented(p["content"]))
    else:
        print(f"  [{letter}] [{verdict_tag}] [{tier_label}] {_shorten(p['content'])}")

    if v.verdict == "REFINEMENT" and v.matched_content:
        print(f"      existing: {_shorten(v.matched_content, 80)}")
        if v.merged_content:
            print(f"      merged:   {_shorten(v.merged_content, 80)}")

    # Dedup hint for non-REFINEMENT/DUPLICATE/CONTRADICTION cases. When
    # the conflict classifier returned NEW but FTS5 surfaced candidates
    # with token overlap, flag the closest one so the user can manually
    # spot near-duplicate accretion the LLM missed.
    if v.verdict == "NEW" and v.candidates:
        top = v.candidates[0]
        existing_text = top.get("content") or ""
        if existing_text:
            print(f"      similar to existing: {_shorten(existing_text, 80)}")

    if v.verdict == "DUPLICATE" and v.matched_content:
        print(f"      duplicate of: {_shorten(v.matched_content, 80)}")

    if v.verdict == "CONTRADICTION" and v.matched_content:
        print(f"      conflicts with: {_shorten(v.matched_content, 80)}")

    if p.get("rationale"):
        print(f"      reason: {p['rationale']}")
