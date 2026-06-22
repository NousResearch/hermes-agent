"""``CompactionStats`` — the single, typed, self-checking source of truth for a
context-compaction's before/after composition, shared by every announce path
(session-hygiene, in-turn LCM, in-turn built-in, overflow, manual /compress).

Design (from the approved spec, 7 Opus review passes):

- ONE typed object carries the whole breakdown; the formatter renders from it.
- ``validate()`` returns ``(ok, reason)`` and **never raises** — and is NOT
  called from ``__post_init__``. Live paths build a stats object inside
  try/except and degrade to the two-line announce on ``not ok``; a reconcile
  failure can never reach the user's reply.
- ``assert_reconciles()`` raises — for tests/CI ONLY.
- The MESSAGE axis is EXACT; the TOKEN axis allows a small estimator tolerance.
- Every ``*_tokens`` field is an ``estimate_messages_tokens_rough`` output over
  its row subset — NEVER the model's live ``prompt_tokens`` (same-estimator
  contract), so the additive identities are real cross-checks, not noise.

Bucket model::

    pre_messages  = cleared + folded + kept           (every removed/folded/kept row)
    eligible      = kept + folded                       (the hygiene-filter survivors / engine pop)
    cleared       = pre - eligible                      (filtered-out: tool + system + contentless-asst)
    post_messages = kept + summary_messages + anchor    (what the model sees next)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Token axis tolerance: the rough estimator is additive over disjoint row sets,
# so a clean partition reconciles within a couple tokens. Keep small — a real
# bucketing bug moves far more than this.
_TOKEN_TOL = 8


@dataclass
class CompactionStats:
    # ── message axis (EXACT) ──
    pre_messages: int
    post_messages: int
    eligible_count: int
    kept_messages: int
    summary_messages: int
    anchor_messages: int
    cleared_count: int
    folded_count: int
    # ── token axis (±estimator tolerance; all from estimate_messages_tokens_rough) ──
    pre_tokens: int
    post_tokens: int
    kept_tokens: int
    summary_tokens: int
    anchor_tokens: int
    cleared_tokens: int
    folded_tokens: int
    # ── optional sub-split of `cleared` (only when Phase-0 attribution is clean) ──
    cleared_tool_count: Optional[int] = None
    cleared_tool_tokens: Optional[int] = None
    cleared_other_count: Optional[int] = None
    cleared_other_tokens: Optional[int] = None

    # NOTE: deliberately NO validation in __post_init__ (keeps any raise off the
    # hot path; callers invoke validate()/assert_reconciles() explicitly).

    @property
    def freed_tokens(self) -> int:
        return self.pre_tokens - self.post_tokens

    @property
    def freed_pct(self) -> Optional[int]:
        if self.pre_tokens <= 0:
            return None
        return max(0, min(100, round(self.freed_tokens / self.pre_tokens * 100)))

    def validate(self) -> Tuple[bool, str]:
        """Return ``(ok, reason)``. Never raises. ``reason`` empty when ok."""
        # ── message axis: EXACT ──
        if self.cleared_count + self.folded_count + self.kept_messages != self.pre_messages:
            return False, (
                f"msg axis: cleared {self.cleared_count} + folded {self.folded_count} "
                f"+ kept {self.kept_messages} != pre {self.pre_messages}"
            )
        if self.cleared_count != self.pre_messages - self.eligible_count:
            return False, (
                f"eligible: cleared {self.cleared_count} != pre {self.pre_messages} "
                f"- eligible {self.eligible_count}"
            )
        if self.kept_messages + self.folded_count != self.eligible_count:
            return False, (
                f"eligible: kept {self.kept_messages} + folded {self.folded_count} "
                f"!= eligible {self.eligible_count}"
            )
        if self.post_messages != self.kept_messages + self.summary_messages + self.anchor_messages:
            return False, (
                f"post msg: kept {self.kept_messages} + summary {self.summary_messages} "
                f"+ anchor {self.anchor_messages} != post {self.post_messages}"
            )
        # ── zero-fold first-class (the literal 222→222 shape) ──
        if self.folded_count == 0:
            if self.summary_messages != 0 or self.summary_tokens != 0:
                return False, (
                    f"zero-fold: folded==0 requires summary_messages==0 and "
                    f"summary_tokens==0 (got {self.summary_messages}/{self.summary_tokens})"
                )
            if self.kept_messages != self.eligible_count:
                return False, (
                    f"zero-fold: folded==0 requires kept==eligible "
                    f"({self.kept_messages} != {self.eligible_count})"
                )
        # ── token axis: ±tolerance ──
        if self.pre_tokens <= 0:
            return False, f"pre_tokens must be > 0 (got {self.pre_tokens})"
        if abs((self.cleared_tokens + self.folded_tokens + self.kept_tokens) - self.pre_tokens) > _TOKEN_TOL:
            return False, (
                f"token pre: cleared {self.cleared_tokens} + folded {self.folded_tokens} "
                f"+ kept {self.kept_tokens} != pre {self.pre_tokens} (tol {_TOKEN_TOL})"
            )
        if abs((self.kept_tokens + self.summary_tokens + self.anchor_tokens) - self.post_tokens) > _TOKEN_TOL:
            return False, (
                f"token post: kept {self.kept_tokens} + summary {self.summary_tokens} "
                f"+ anchor {self.anchor_tokens} != post {self.post_tokens} (tol {_TOKEN_TOL})"
            )
        # freed identity with the anchor term (Pass-2 blocker fix):
        # cleared + folded - summary - anchor == freed
        # NOTE: this identity is the algebraic difference of the two axis checks
        # above (pre = cleared+folded+kept ; post = kept+summary+anchor →
        # pre-post = cleared+folded-summary-anchor). Each axis tolerates ±_TOKEN_TOL
        # independently, so the compounded error here is bounded by 2×_TOKEN_TOL
        # (worst case ε_pre=+tol, ε_post=-tol). Using a single _TOKEN_TOL here would
        # spuriously fail — and silently degrade to the two-line form — on data that
        # passed both axis checks. The estimator is exactly additive over disjoint
        # subsets today (ε≈0), but widen the bound so a future rounding estimator
        # can't trip this latent trap.
        _FREED_TOL = 2 * _TOKEN_TOL
        freed_check = (
            self.cleared_tokens + self.folded_tokens
            - self.summary_tokens - self.anchor_tokens
        )
        if abs(freed_check - self.freed_tokens) > _FREED_TOL:
            return False, (
                f"freed: cleared {self.cleared_tokens} + folded {self.folded_tokens} "
                f"- summary {self.summary_tokens} - anchor {self.anchor_tokens} "
                f"= {freed_check} != freed {self.freed_tokens} (tol {_FREED_TOL})"
            )
        # ── optional sub-split must sum to cleared ──
        if self.cleared_tool_count is not None or self.cleared_other_count is not None:
            t = self.cleared_tool_count or 0
            o = self.cleared_other_count or 0
            if t + o != self.cleared_count:
                return False, (
                    f"sub-split: tool {t} + other {o} != cleared {self.cleared_count}"
                )
        return True, ""

    def assert_reconciles(self) -> None:
        """Raise ``ValueError`` if not reconciling. TESTS/CI ONLY — never on the hot path."""
        ok, reason = self.validate()
        if not ok:
            raise ValueError(f"CompactionStats does not reconcile: {reason}")


# Identifies an LCM summary message in a compressed/active context.
_LCM_SUMMARY_RE = None  # lazy-compiled below


def _is_summary_message(content: str) -> bool:
    global _LCM_SUMMARY_RE
    if _LCM_SUMMARY_RE is None:
        import re
        _LCM_SUMMARY_RE = re.compile(
            r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]"
        )
    return bool(content) and bool(_LCM_SUMMARY_RE.search(content))


def build_hygiene_stats(
    *,
    raw_history: List[dict],
    eligible_msgs: List[dict],
    compressed: List[dict],
    estimator,
) -> "CompactionStats":
    """Build a reconciling ``CompactionStats`` from the session-hygiene path's real data.

    All counts/tokens are MEASURED independently over disjoint row subsets of the
    SAME population (no back-derivation), so ``validate()`` is a real cross-check:

    - ``pre`` = the full raw transcript (`raw_history`).
    - ``eligible`` = the role-filtered subset fed to the throwaway compressor
      (`eligible_msgs` = user/assistant-with-content). ``cleared = pre - eligible``
      (tool + system + contentless-assistant rows the filter removed).
    - ``compressed`` = the LCM output written back. Within it: summary message(s)
      (LCM markers), the system anchor (role == "system"), and the kept tail
      (everything else). ``kept`` rows are a subset of ``eligible``; ``folded =
      eligible - kept``.

    Token sums all use the SAME ``estimator`` over each subset (same-estimator
    contract). Returns a stats object; the caller validates + degrades on failure.
    """
    pre_msgs = list(raw_history or [])
    elig = list(eligible_msgs or [])
    comp = list(compressed or [])

    summary_rows = [m for m in comp if _is_summary_message(m.get("content") or "")]
    anchor_rows = [m for m in comp if m.get("role") == "system"]
    kept_rows = [
        m for m in comp
        if m.get("role") != "system" and not _is_summary_message(m.get("content") or "")
    ]

    pre_messages = len(pre_msgs)
    eligible_count = len(elig)
    cleared_count = pre_messages - eligible_count
    kept_messages = len(kept_rows)
    folded_count = eligible_count - kept_messages
    summary_messages = len(summary_rows)
    anchor_messages = len(anchor_rows)
    post_messages = kept_messages + summary_messages + anchor_messages

    cleared_rows = _disjoint_remainder(pre_msgs, elig)

    return CompactionStats(
        pre_messages=pre_messages,
        post_messages=post_messages,
        eligible_count=eligible_count,
        kept_messages=kept_messages,
        summary_messages=summary_messages,
        anchor_messages=anchor_messages,
        cleared_count=cleared_count,
        folded_count=folded_count,
        pre_tokens=int(estimator(pre_msgs)),
        post_tokens=int(estimator(comp)),
        kept_tokens=int(estimator(kept_rows)),
        summary_tokens=int(estimator(summary_rows)) if summary_rows else 0,
        anchor_tokens=int(estimator(anchor_rows)) if anchor_rows else 0,
        cleared_tokens=int(estimator(cleared_rows)) if cleared_rows else 0,
        folded_tokens=int(estimator(_fold_rows(elig, kept_rows))),
    )


def _row_signature(m: dict) -> Tuple[Optional[str], str]:
    """Collision-resistant ``(role, content-hash)`` key for fallback subtraction.

    Used only when ``id()``-identity matching fails (the population was copied).
    Hashes the FULL content rather than a truncated prefix: templated tool
    results or assistant turns frequently share a long identical prefix, so a
    ``content[:200]`` key collided and the Counter subtraction deducted the wrong
    row — making the granular announce silently degrade to the two-line form on
    exactly those structural sessions. A full-content hash avoids that. (Hash
    bounds key size vs. carrying multi-KB strings as dict keys; SHA-1 is for
    de-dup, not security, so it's fine here.)
    """
    import hashlib

    content = m.get("content")
    if not isinstance(content, str):
        # tool-call-only / structured content → stringify deterministically
        content = repr(content)
    digest = hashlib.sha1(content.encode("utf-8", "surrogatepass")).hexdigest()
    return (m.get("role"), digest)


def _disjoint_remainder(whole: List[dict], subset: List[dict]) -> List[dict]:
    """Rows in ``whole`` not present (by identity) in ``subset`` — for cleared_tok.

    Uses id()-identity when the same objects flow through, else falls back to a
    role+content signature so token attribution is over the right rows.
    """
    sub_ids = {id(m) for m in subset}
    rem = [m for m in whole if id(m) not in sub_ids]
    if len(rem) == len(whole) - len(subset):
        return rem
    # identity didn't line up (copies); fall back to signature subtraction
    from collections import Counter
    want = Counter(_row_signature(m) for m in subset)
    out = []
    for m in whole:
        s = _row_signature(m)
        if want.get(s, 0) > 0:
            want[s] -= 1
        else:
            out.append(m)
    return out


def _fold_rows(eligible: List[dict], kept: List[dict]) -> List[dict]:
    """Eligible rows NOT in the kept tail — the folded population (token source)."""
    kept_ids = {id(m) for m in kept}
    rem = [m for m in eligible if id(m) not in kept_ids]
    if len(rem) == len(eligible) - len(kept):
        return rem
    from collections import Counter
    want = Counter(_row_signature(m) for m in kept)
    out = []
    for m in eligible:
        s = _row_signature(m)
        if want.get(s, 0) > 0:
            want[s] -= 1
        else:
            out.append(m)
    return out


def build_inturn_stats(
    *,
    messages: List[dict],
    compressed: List[dict],
    estimator,
) -> "CompactionStats":
    """Build a reconciling ``CompactionStats`` for the in-turn (LCM) done-site.

    The in-turn path does NOT role-filter — the whole input ``messages`` list is
    the population, so ``cleared == 0`` and ``eligible == pre``. The compressed
    output splits into summary message(s) (LCM markers), the leading anchor
    (role == "system" / preserved-objective), and the kept tail (remainder);
    ``folded = eligible - kept``. All token sums use the SAME ``estimator`` over
    each subset (message-level — consistent, never the request-level estimate).

    Caller validates + degrades on failure (e.g. an exotic compressed shape).
    """
    pre_msgs = list(messages or [])
    comp = list(compressed or [])

    summary_rows = [m for m in comp if _is_summary_message(m.get("content") or "")]
    anchor_rows = [m for m in comp if m.get("role") == "system"]
    kept_rows = [
        m for m in comp
        if m.get("role") != "system" and not _is_summary_message(m.get("content") or "")
    ]

    pre_messages = len(pre_msgs)
    eligible_count = pre_messages  # no role filter on the in-turn path
    kept_messages = len(kept_rows)
    folded_count = eligible_count - kept_messages
    summary_messages = len(summary_rows)
    anchor_messages = len(anchor_rows)
    post_messages = kept_messages + summary_messages + anchor_messages

    return CompactionStats(
        pre_messages=pre_messages,
        post_messages=post_messages,
        eligible_count=eligible_count,
        kept_messages=kept_messages,
        summary_messages=summary_messages,
        anchor_messages=anchor_messages,
        cleared_count=0,
        folded_count=folded_count,
        pre_tokens=int(estimator(pre_msgs)),
        post_tokens=int(estimator(comp)),
        kept_tokens=int(estimator(kept_rows)),
        summary_tokens=int(estimator(summary_rows)) if summary_rows else 0,
        anchor_tokens=int(estimator(anchor_rows)) if anchor_rows else 0,
        cleared_tokens=0,
        folded_tokens=int(estimator(_fold_rows(pre_msgs, kept_rows))),
    )
