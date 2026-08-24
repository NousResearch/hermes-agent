"""Boundary-aware partial compression — "summarize up to here".

* **Role alternation.** The compressed head ends with summary/handoff content (assistant- or user-
role, possibly a trailing todo snapshot). The verbatim tail must begin with a ``user`` message so
the rejoined history keeps the user↔assistant alternation that providers validate.
* **No silent context mutation.** Manual, user-invoked; rotates the session exactly like
``/compress`` (via the caller), so the prompt-cache reset is explicit and expected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Recent exchanges preserved verbatim when ``/compress here`` has no explicit count.
DEFAULT_KEEP_LAST = 2

#: Hard ceiling so a fat-fingered ``/compress here 9999`` clamps instead of no-op'ing.
MAX_KEEP_LAST = 100


def parse_partial_compress_args(raw_args: str) -> Tuple[bool, int, Optional[str]]:
    """Parse the argument string after ``/compress`` into ``(partial, keep_last, focus_topic)``.

    ``here [N]`` / ``up to here [N]`` / ``--keep N`` / ``-k N`` / ``--keep=N`` select the
    boundary-aware form; anything else is a focus topic for full ``/compress <focus>``.
    """
    text = (raw_args or "").strip()
    if not text:
        return False, DEFAULT_KEEP_LAST, None

    lowered = text.lower()
    if lowered.startswith("up to here"):
        lowered = lowered[len("up to ") :]
        text = text[len("up to ") :]

    tokens = lowered.split()
    head = tokens[0] if tokens else ""

    if head == "here":
        keep = _coerce_keep(tokens[1]) if len(tokens) >= 2 else DEFAULT_KEEP_LAST
        return True, keep, None
    if head in ("--keep", "-k") and len(tokens) >= 2:
        return True, _coerce_keep(tokens[1]), None
    if head.startswith("--keep="):
        return True, _coerce_keep(head.split("=", 1)[1]), None

    return False, DEFAULT_KEEP_LAST, text or None


def extract_compress_flags(raw_args: str) -> Tuple[str, bool, bool]:
    """Strip ``--preview``/``--dry-run``/``--aggressive`` (anywhere in the string) and return
    ``(remaining_args, preview, aggressive_requested)``.

    ``preview`` means report what WOULD be compressed and change nothing. No surface implements an
    LLM-free hard-truncate path, so callers surface "not supported" for ``--aggressive`` instead of
    treating it as a focus topic.
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

    Shared by the CLI and the gateway slash handler so both report the numbers the real run would
    use. Returns ``head_count``/``tail_count``/``total``/``partial``/``lines`` (ready to print).
    """
    total = len(history)
    head, tail = list(history), []
    effective_partial = False
    if partial:
        head, tail = split_history_for_partial_compress(history, keep_last)
        # Same degenerate-split fallback the real run applies.
        effective_partial = bool(tail)

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
        lines.append("Boundary: 'here' split would keep everything — falling back to full compression.")
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
    return max(1, min(n, MAX_KEEP_LAST))


def split_history_for_partial_compress(
    history: List[Dict[str, Any]],
    keep_last: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split ``history`` into ``(head, tail)``: head is summarized, tail is the last ``keep_last``
    exchanges kept verbatim.

    Exchanges are counted by ``user`` messages so the tail always starts on a user turn and
    ``compressed_head + tail`` keeps alternation valid. Returns ``(history, [])`` when the head
    would be empty (or there are no user turns), signaling the caller to fall back to full
    compression rather than rotating the session for a no-op.
    """
    keep_last = max(keep_last, 1)
    if not history:
        return [], []

    # Walk backwards to the earliest of the most recent `keep_last` user-message starts.
    boundary = None
    seen = 0
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") == "user":
            boundary = idx
            seen += 1
            if seen >= keep_last:
                break

    if not boundary:  # no user turns, or everything is in the tail
        return list(history), []
    return history[:boundary], history[boundary:]


def rejoin_compressed_head_and_tail(
    compressed_head: List[Dict[str, Any]],
    tail: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Concatenate a compressed head with the verbatim tail, defending the seam's role alternation.

    The head compressor's output shape isn't contractually guaranteed (a plugin engine could end on
    a user turn). If the last head message and first tail message share a user/assistant role, the
    tail's first string content is folded onto the head's last message; non-string (multimodal)
    content gets a minimal bridging turn instead so nothing is lost. ``tool`` messages are left
    alone — consecutive tool entries are the one legal repetition (parallel results).
    """
    if not tail:
        return list(compressed_head)
    if not compressed_head:
        return list(tail)

    head = list(compressed_head)
    rest = list(tail)
    last, first = head[-1], rest[0]
    last_role, first_role = last.get("role"), first.get("role")

    if last_role == first_role and last_role in ("user", "assistant"):
        last_content = last.get("content")
        first_content = first.get("content")
        if isinstance(last_content, str) and isinstance(first_content, str):
            head[-1] = {**last, "content": f"{last_content}\n\n{first_content}"}
            rest = rest[1:]
        else:
            head.append({"role": "assistant" if first_role == "user" else "user", "content": ""})

    return head + rest
