"""Boundary-aware partial compression — "summarize up to here".

Inspired by Claude Code's Rewind menu "Summarize up to here" action
(v2.1.139–v2.1.142, Week 20, May 2026):
https://code.claude.com/docs/en/whats-new/2026-w20

IX Agency already has ``/compress`` (full-history compaction) and an
automatic token-budget tail-protection heuristic inside
``ContextCompressor``. What was missing is *user-chosen* boundary
control: "fold everything before this point into a summary, but keep
my most recent N exchanges exactly as they are." That is the value of
the Claude Code feature — the user decides the compression boundary
instead of leaving it to the token-budget heuristic.

This module owns the pure, side-effect-free split logic so both the
CLI (``cli.py::_manual_compress``) and the gateway
(``gateway/run.py::_handle_compress_command``) share one
implementation. The slash-command surfaces handle compression of the
*head* via the existing ``_compress_context`` pipeline (preserving all
the session-rotation / lock / memory-notify machinery) and then
re-append the verbatim *tail* returned here.

Design notes / invariants honored:

* **Role alternation.** The compressed head ends with summary/handoff
  content (assistant- or user-role, possibly a trailing todo snapshot).
  The verbatim tail must begin with a ``user`` message so the rejoined
  history keeps the user↔assistant alternation that providers validate.
  :func:`split_history_for_partial_compress` snaps the tail boundary
  backwards to the nearest ``user`` turn so the rejoin is always legal.

* **No silent context mutation.** This is a manual, user-invoked
  action. It rotates the session exactly like ``/compress`` does (via
  the caller), so the prompt-cache reset is explicit and expected, not
  silent.

* **Conservative defaults.** ``keep_last`` counts *exchanges* (a user
  turn plus its following assistant/tool turns), defaulting to 2. The
  split never compresses if doing so would leave nothing in the head.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Default number of recent exchanges to preserve verbatim when the user
#: runs ``/compress here`` without an explicit count.
DEFAULT_KEEP_LAST = 2

#: Hard ceiling so a fat-fingered ``/compress here 9999`` doesn't turn
#: into a no-op surprise — clamp instead.
MAX_KEEP_LAST = 100


def parse_partial_compress_args(
    raw_args: str,
) -> Tuple[bool, int, Optional[str]]:
    """Parse the argument string after ``/compress``.

    Recognizes the boundary-aware forms:

    * ``here``            → partial compress, keep ``DEFAULT_KEEP_LAST``
    * ``here 4``          → partial compress, keep 4 exchanges
    * ``--keep 4``        → partial compress, keep 4 exchanges
    * ``up to here``      → alias for ``here`` (matches Claude Code's
                            menu label "Summarize up to here")

    Anything else is treated as a focus topic for the existing full
    ``/compress <focus>`` behavior.

    Returns ``(partial, keep_last, focus_topic)``:

    * ``partial`` — True when a boundary-aware form was requested.
    * ``keep_last`` — exchanges to preserve verbatim (only meaningful
      when ``partial`` is True).
    * ``focus_topic`` — focus string for full compression, or None.
      Always None when ``partial`` is True (the two modes are exclusive;
      a focused partial compress is not a documented Claude Code
      behavior and would muddy the UX).
    """
    text = (raw_args or "").strip()
    if not text:
        return False, DEFAULT_KEEP_LAST, None

    lowered = text.lower()

    # Normalize the "up to here" alias to "here".
    if lowered.startswith("up to here"):
        lowered = lowered[len("up to ") :]
        text = text[len("up to ") :]

    tokens = lowered.split()

    # Form: here [N]
    if tokens and tokens[0] == "here":
        keep = DEFAULT_KEEP_LAST
        if len(tokens) >= 2:
            keep = _coerce_keep(tokens[1])
        return True, keep, None

    # Form: --keep N  (or --keep=N)
    if tokens and tokens[0] in ("--keep", "-k") and len(tokens) >= 2:
        return True, _coerce_keep(tokens[1]), None
    if tokens and tokens[0].startswith("--keep="):
        return True, _coerce_keep(tokens[0].split("=", 1)[1]), None

    # Otherwise: full compression with this as the focus topic.
    return False, DEFAULT_KEEP_LAST, text or None


def extract_compress_flags(raw_args: str) -> Tuple[str, bool, bool]:
    """Strip ``--preview``/``--dry-run``/``--aggressive`` flags from the
    argument string after ``/compress`` (or its ``/compact`` alias).

    Flags may appear anywhere and coexist with the positional forms
    (``here [N]``, ``--keep N``, or a focus topic); the returned
    remainder is what :func:`parse_partial_compress_args` should see.

    Returns ``(remaining_args, preview, aggressive_requested)``:

    * ``preview`` — True when ``--preview`` or ``--dry-run`` was given.
      The caller must report what WOULD be compressed (message counts,
      token estimate, boundary) and make **no changes**.
    * ``aggressive_requested`` — True when ``--aggressive`` was given.
      The current surfaces do not implement an LLM-free hard-truncate
      path (it would need its own transcript-persistence branch outside
      the guarded ``_compress_context`` rotation machinery), so callers
      surface a "not supported" note instead of silently treating the
      flag as a focus topic.
    """
    preview = False
    aggressive = False
    kept: List[str] = []
    for tok in (raw_args or "").split():
        low = tok.lower()
        if low in ("--preview", "--dry-run", "--dryrun"):
            preview = True
        elif low == "--aggressive":
            aggressive = True
        else:
            kept.append(tok)
    return " ".join(kept), preview, aggressive


def summarize_compress_preview(
    history: List[Dict[str, Any]],
    partial: bool,
    keep_last: int,
    focus_topic: Optional[str],
    approx_tokens: int,
) -> Dict[str, Any]:
    """Build the ``/compress --preview`` report — pure, no side effects.

    Shared by the CLI (``cli.py::_manual_compress``) and the gateway
    (``gateway/slash_commands.py::_handle_compress_command``) so both
    surfaces report the same numbers the real run would use.

    Returns a dict with ``head_count``/``tail_count``/``lines`` where
    ``lines`` is a ready-to-print list of report strings.
    """
    total = len(history)
    head = list(history)
    tail: List[Dict[str, Any]] = []
    effective_partial = partial
    if partial:
        head, tail = split_history_for_partial_compress(history, keep_last)
        if not tail:
            # Same degenerate-split fallback the real run applies.
            effective_partial = False
            head, tail = list(history), []

    lines = [
        "Preview — no changes made.",
        f"Would compress {len(head)} of {total} message(s) "
        f"(~{approx_tokens:,} tokens currently in context).",
    ]
    if effective_partial:
        lines.append(
            f"Boundary: keeping the last {keep_last} exchange(s) "
            f"({len(tail)} message(s)) verbatim."
        )
    elif partial:
        lines.append(
            "Boundary: 'here' split would keep everything — "
            "falling back to full compression."
        )
    if focus_topic:
        lines.append(f'Focus topic: "{focus_topic}"')
    lines.append("Run the command again without --preview to apply.")

    return {
        "head_count": len(head),
        "tail_count": len(tail),
        "total": total,
        "partial": effective_partial,
        "lines": lines,
    }


def _coerce_keep(value: str) -> int:
    """Parse a keep-count token, clamping to [1, MAX_KEEP_LAST]."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_KEEP_LAST
    if n < 1:
        return 1
    if n > MAX_KEEP_LAST:
        return MAX_KEEP_LAST
    return n


def split_history_for_partial_compress(
    history: List[Dict[str, Any]],
    keep_last: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split ``history`` into ``(head, tail)`` for partial compression.

    ``head`` is the earlier portion that will be summarized; ``tail`` is
    the most recent ``keep_last`` exchanges, preserved verbatim.

    An *exchange* is counted by ``user``-role messages: keeping N
    exchanges means keeping everything from the Nth-most-recent ``user``
    message onward. This guarantees the tail starts on a ``user`` turn,
    so when the caller rejoins ``compressed_head + tail`` the
    user↔assistant alternation stays valid (the compressed head's
    trailing content is followed by a fresh user turn).

    Returns ``(head, tail)``. If the split would leave the head empty
    (not enough history to compress meaningfully), returns
    ``(history, [])`` — signaling the caller to fall back to full
    compression or report "nothing to do".
    """
    if keep_last < 1:
        keep_last = 1

    n = len(history)
    if n == 0:
        return [], []

    # Walk backwards collecting the indices of the most recent `keep_last`
    # user-message starts. The tail begins at the earliest such index.
    user_starts: List[int] = []
    for idx in range(n - 1, -1, -1):
        if history[idx].get("role") == "user":
            user_starts.append(idx)
            if len(user_starts) >= keep_last:
                break

    if not user_starts:
        # No user turns at all (degenerate) — nothing sensible to keep
        # as a "recent exchange"; treat as full compression.
        return list(history), []

    boundary = user_starts[-1]  # earliest of the kept user starts

    head = history[:boundary]
    tail = history[boundary:]

    # If everything is in the tail (nothing left to compress), signal the
    # caller to fall back to full compression rather than producing a
    # no-op that rotates the session for no benefit.
    if not head:
        return list(history), []

    return head, tail


def rejoin_compressed_head_and_tail(
    compressed_head: List[Dict[str, Any]],
    tail: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Concatenate a compressed head with the verbatim tail, defending
    the seam against an illegal user→user / assistant→assistant adjacency.

    In normal operation the compressed head ends with the head's own
    protected verbatim tail (the ``ContextCompressor`` always preserves a
    recent window), which terminates on an ``assistant``/``tool`` turn —
    so ``assistant → user`` at the seam is already valid. But the head
    compressor's exact output shape is not contractually guaranteed (a
    plugin context engine could return something that ends on a ``user``
    turn, or a degenerate single-summary message). Rather than trust the
    seam, this helper inspects the boundary and, if the last head message
    and the first tail message share a ``user``/``assistant`` role, folds
    the tail's first message content onto the head's last message so the
    rejoined list never violates provider role-alternation rules.

    ``tool`` messages are left alone — consecutive ``tool`` entries are
    the one legal repetition (parallel tool results).
    """
    if not tail:
        return list(compressed_head)
    if not compressed_head:
        return list(tail)

    head = list(compressed_head)
    rest = list(tail)

    last = head[-1]
    first = rest[0]
    last_role = last.get("role")
    first_role = first.get("role")

    if last_role == first_role and last_role in ("user", "assistant"):
        # Illegal adjacency. Merge the tail's first message text into the
        # head's last message so alternation is preserved. Only string
        # contents are merged inline; structured/multimodal contents fall
        # back to dropping the redundant standalone (the content is
        # preserved by concatenation when both are strings).
        last_content = last.get("content")
        first_content = first.get("content")
        if isinstance(last_content, str) and isinstance(first_content, str):
            merged = dict(last)
            merged["content"] = f"{last_content}\n\n{first_content}"
            head[-1] = merged
            rest = rest[1:]
        else:
            # Can't safely string-merge multimodal content. Insert a
            # minimal bridging turn so the seam alternates rather than
            # losing data.
            bridge_role = "assistant" if first_role == "user" else "user"
            head.append({"role": bridge_role, "content": ""})

    return head + rest
