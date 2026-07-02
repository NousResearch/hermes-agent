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

import hashlib
from collections import Counter
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
    # ── optional sub-split of `cleared` (only when attribution is clean) ──
    cleared_tool_count: Optional[int] = None
    cleared_tool_tokens: Optional[int] = None
    cleared_other_count: Optional[int] = None
    cleared_other_tokens: Optional[int] = None
    # ── optional sub-split of `folded` (in-turn path; only when attribution is clean) ──
    folded_tool_count: Optional[int] = None
    folded_tool_tokens: Optional[int] = None
    folded_other_count: Optional[int] = None
    folded_other_tokens: Optional[int] = None
    # ── PRE-side kept tokens (hygiene path) — distinct from comp-side `kept_tokens` ──
    # The hygiene path has TWO different "kept" populations that must NOT be conflated:
    #   * `kept_tokens`     = estimator(comp-side kept tail)  → the POST identity
    #     (kept + summary + anchor == post_tokens == estimator(comp)).
    #   * `kept_pre_tokens` = estimator(pre-side kept rows)   → the PRE identity
    #     (cleared + folded + kept_pre == pre_tokens).
    # When LCM sanitizes the kept tail (cleans assistant content / strips tool
    # scaffolding), a comp kept row no longer signature-matches its raw original,
    # so the two populations diverge in TOKENS (live 2026-06-22 reconcile failures).
    # Each is an INDEPENDENT estimator() call over its own disjoint rows (no
    # back-derivation — the dead-guard trap). DEFAULT None → falls back to
    # `kept_tokens` for the in-turn path + legacy callers, where the kept rows ARE
    # comp-side so the two are equal by construction.
    kept_pre_tokens: Optional[int] = None
    # ── PRE-side kept MESSAGE COUNT (hygiene path) — distinct from comp-side `kept_messages` ──
    # The SAME two-population split as `kept_pre_tokens`, on the message axis:
    #   * `kept_messages`     = COUNT of the comp-side kept tail (what's actually in
    #     live context now) → the POST identity + the user-facing "kept N recent chat".
    #   * `kept_pre_messages` = COUNT of pre-side kept rows (raw rows that survived) →
    #     the PRE / eligible identities (cleared + folded + kept_pre == pre).
    # When LCM sanitizes the kept tail, a comp kept row no longer signature-matches its
    # raw original, so the pre-side partition finds ZERO matches while the comp tail is
    # full — the message-axis form of the 2026-06-22 divergence. Measuring `kept_messages`
    # pre-side made the granular announce render "kept 0 recent chat" (post_messages was
    # also computed tautologically as kept_pre + summary + anchor, so it under-counted in
    # lockstep and never tripped validate()). DEFAULT None → falls back to `kept_messages`
    # for the in-turn path + legacy/direct-construction callers (kept is comp-side there).
    kept_pre_messages: Optional[int] = None

    # ── A-floor approximate-attribution flag (in-turn) ──
    # True when the kept_pre/folded split came from the exhaustive single-walk
    # fallback (``_partition_pre_by_comp_kept``) rather than exact whole-tail
    # alignment or an authoritative engine record. TOTALS still reconcile exactly;
    # only the kept/folded *split* is signature-approximate (bounded by the kept-tail
    # fraction). The caller labels the render + emits COMPACTION_STATS_APPROX_ATTRIBUTION
    # so a heavy session silently running the floor is observable, never silent.
    approx_attribution: bool = False

    # ── RAW kept-tail token UPPER BOUND (in-turn A-floor guard) ──
    # estimator(messages[-fresh_tail_count:]) — the raw (pre-sanitize) size of the
    # tail region the engine keeps. This is the TRUE magnitude of the A-floor's
    # possible gross misattribution, and unlike kept_tokens (sanitized comp-side,
    # can be stripped small) or kept_pre_tokens (0 when signature match fails), it is
    # match- AND sanitize-INDEPENDENT — computed straight from the raw input suffix.
    # The render-vs-degrade guard keys off this so a heavily-sanitized large tail
    # can't slip under the threshold (Greptile P1 ×2, PR #109). None on the aligned
    # / legacy paths (guard only applies to the approx_attribution floor).
    raw_tail_tokens: Optional[int] = None

    # NOTE: deliberately NO validation in __post_init__ (keeps any raise off the
    # hot path; callers invoke validate()/assert_reconciles() explicitly).

    @property
    def _kept_pre_messages(self) -> int:
        """PRE-identity kept message count; defaults to comp-side ``kept_messages``
        when a caller (in-turn path / legacy) didn't supply a distinct pre-side value."""
        return self.kept_messages if self.kept_pre_messages is None else self.kept_pre_messages

    @property
    def _kept_pre_tokens(self) -> int:
        """PRE-identity kept tokens; defaults to comp-side ``kept_tokens`` when a
        caller (in-turn path / legacy) didn't supply a distinct pre-side value."""
        return self.kept_tokens if self.kept_pre_tokens is None else self.kept_pre_tokens

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
        # PRE / eligible identities use the PRE-side kept count (raw rows that
        # survived); the POST identity uses the comp-side kept count (the actual
        # kept tail in `compressed`). The two diverge on the hygiene path when LCM
        # sanitizes the kept tail (a comp row no longer signature-matches its raw
        # original) — the message-axis twin of the kept_pre_tokens split. On the
        # in-turn/legacy path the two are equal by construction (kept is comp-side).
        if self.cleared_count + self.folded_count + self._kept_pre_messages != self.pre_messages:
            return False, (
                f"msg axis: cleared {self.cleared_count} + folded {self.folded_count} "
                f"+ kept {self._kept_pre_messages} != pre {self.pre_messages}"
            )
        if self.cleared_count != self.pre_messages - self.eligible_count:
            return False, (
                f"eligible: cleared {self.cleared_count} != pre {self.pre_messages} "
                f"- eligible {self.eligible_count}"
            )
        if self._kept_pre_messages + self.folded_count != self.eligible_count:
            return False, (
                f"eligible: kept {self._kept_pre_messages} + folded {self.folded_count} "
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
            if self._kept_pre_messages != self.eligible_count:
                return False, (
                    f"zero-fold: folded==0 requires kept==eligible "
                    f"({self._kept_pre_messages} != {self.eligible_count})"
                )
        # ── token axis: ±tolerance ──
        if self.pre_tokens <= 0:
            return False, f"pre_tokens must be > 0 (got {self.pre_tokens})"
        if abs((self.cleared_tokens + self.folded_tokens + self._kept_pre_tokens) - self.pre_tokens) > _TOKEN_TOL:
            return False, (
                f"token pre: cleared {self.cleared_tokens} + folded {self.folded_tokens} "
                f"+ kept {self._kept_pre_tokens} != pre {self.pre_tokens} (tol {_TOKEN_TOL})"
            )
        if abs((self.kept_tokens + self.summary_tokens + self.anchor_tokens) - self.post_tokens) > _TOKEN_TOL:
            return False, (
                f"token post: kept {self.kept_tokens} + summary {self.summary_tokens} "
                f"+ anchor {self.anchor_tokens} != post {self.post_tokens} (tol {_TOKEN_TOL})"
            )
        # freed identity with the anchor term (Pass-2 blocker fix):
        # freed = pre - post. With the two distinct kept populations (hygiene path),
        #   pre  = cleared + folded + kept_pre        (pre-side kept)
        #   post = kept_comp + summary + anchor       (comp-side kept)
        # so freed = pre - post
        #          = cleared + folded + kept_pre - kept_comp - summary - anchor.
        # The (kept_pre - kept_comp) term is ZERO on the in-turn/legacy path (kept is
        # comp-side there) but NON-ZERO on hygiene when LCM sanitized the kept tail —
        # which is exactly the 2026-06-22 live bug. Include it so the freed identity is
        # the true algebraic difference of the two axis checks, both measured
        # independently (no back-derivation). Each axis tolerates ±_TOKEN_TOL, plus the
        # kept-difference is two more independent estimator calls, so widen the bound.
        _FREED_TOL = 3 * _TOKEN_TOL
        freed_check = (
            self.cleared_tokens + self.folded_tokens + self._kept_pre_tokens
            - self.kept_tokens - self.summary_tokens - self.anchor_tokens
        )
        if abs(freed_check - self.freed_tokens) > _FREED_TOL:
            return False, (
                f"freed: cleared {self.cleared_tokens} + folded {self.folded_tokens} "
                f"+ kept_pre {self._kept_pre_tokens} - kept {self.kept_tokens} "
                f"- summary {self.summary_tokens} - anchor {self.anchor_tokens} "
                f"= {freed_check} != freed {self.freed_tokens} (tol {_FREED_TOL})"
            )
        # ── optional sub-split of `cleared` must sum to cleared (count + tokens) ──
        # tokens are EXACT: other_tokens is derived as (cleared_tokens - tool_tokens)
        # by the producer (D-7), so any drift here is a real producer bug, not rounding.
        if self.cleared_tool_count is not None or self.cleared_other_count is not None:
            t = self.cleared_tool_count or 0
            o = self.cleared_other_count or 0
            if t + o != self.cleared_count:
                return False, (
                    f"cleared sub-split: tool {t} + other {o} != cleared {self.cleared_count}"
                )
            tt = self.cleared_tool_tokens or 0
            ot = self.cleared_other_tokens or 0
            if tt + ot != self.cleared_tokens:
                return False, (
                    f"cleared sub-split tokens: {tt}+{ot} != cleared {self.cleared_tokens}"
                )
        # ── optional sub-split of `folded` must sum to folded (count + tokens, EXACT) ──
        if self.folded_tool_count is not None or self.folded_other_count is not None:
            t = self.folded_tool_count or 0
            o = self.folded_other_count or 0
            if t + o != self.folded_count:
                return False, (
                    f"folded sub-split: tool {t} + other {o} != folded {self.folded_count}"
                )
            tt = self.folded_tool_tokens or 0
            ot = self.folded_other_tokens or 0
            if tt + ot != self.folded_tokens:
                return False, (
                    f"folded sub-split tokens: {tt}+{ot} != folded {self.folded_tokens}"
                )
        return True, ""

    def assert_reconciles(self) -> None:
        """Raise ``ValueError`` if not reconciling. TESTS/CI ONLY — never on the hot path."""
        ok, reason = self.validate()
        if not ok:
            raise ValueError(f"CompactionStats does not reconcile: {reason}")


# Identifies an LCM summary message in a compressed/active context.
_LCM_SUMMARY_RE = None  # lazy-compiled below


def hygiene_eligible_msgs(history: List[dict]) -> List[dict]:
    """The session-hygiene eligible filter: user/assistant rows WITH content.

    This is the SINGLE source of truth for which transcript rows the hygiene
    compressor operates on (tool / system / contentless-assistant rows are
    removed). The gateway hygiene block and the reconciliation replay probe both
    call this so the filter can never drift between production and verification.
    Returns shallow `{role, content}` dicts, matching the gateway's snapshot.
    """
    return [
        {"role": m.get("role"), "content": m.get("content")}
        for m in (history or [])
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]


_TEXT_PART_TYPES = ("text", "input_text", "output_text")


def _extract_part_text(value):
    """Mirror of the LCM plugin's ``_extract_text_part_value`` (returns ``str | None``).

    str → itself; dict → a nested ``value`` then ``content`` string; else ``None``.
    Bounded to one level of nesting (matches canonical; a marker never lives deeper).
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = value.get("value")
        if isinstance(nested, str):
            return nested
        nested = value.get("content")
        if isinstance(nested, str):
            return nested
    return None


def _part_text(part) -> str:
    """Text contributed by a single content part — type-gated, never raises."""
    if isinstance(part, str):
        return part  # bare-string passthrough (canonical)
    if isinstance(part, dict) and part.get("type") in _TEXT_PART_TYPES:
        # `is None` fall-through (NOT falsy) — faithful to canonical: a present-but-
        # non-extractable `text` still falls through to `content`.
        text = _extract_part_text(part.get("text"))
        if text is None:
            text = _extract_part_text(part.get("content"))
        if text:
            return text
    return ""


def _content_to_text(content) -> str:
    """Coerce a message ``content`` to a searchable string for marker detection.

    Byte-faithful mirror of the canonical LCM extractor
    (``plugins/context_engine/lcm/message_content.py::text_content_for_pattern_matching``
    + ``_extract_text_part_value``), kept in core because core must not import a
    plugin. In-turn compaction feeds API-shaped messages whose ``content`` is a
    LIST of content blocks (``{"type": "text", "text": …}``, ``tool_use``,
    ``tool_result``) rather than a flat string (the hygiene path's shape). Only
    text-typed parts contribute, so a marker-shaped string under a structural key
    (``tool_use``/``tool_result``/``id``/``name``) is never searched, and parts are
    joined with ``"\\n"`` so the single-line summary-marker regex cannot be
    synthesized across a part boundary. Robust to None/dict/list/str so the caller
    can never raise on an exotic content shape.

    ONE deliberate, documented narrowing deviation from canonical: where canonical
    falls through to ``normalize_content_value`` (``json.dumps``) for non-text-typed
    content, this returns ``""``. The sole consumer is a text-only marker regex
    that never wants dumped structure; ``""`` is strictly narrower (can only ever
    match fewer, never a match canonical wouldn't), which is the safe direction.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(t for t in (_part_text(p) for p in content) if t)
    if isinstance(content, dict):  # bare dict — defensive (no live producer)
        return _part_text(content)
    return ""


def _is_summary_message(content) -> bool:
    global _LCM_SUMMARY_RE
    if _LCM_SUMMARY_RE is None:
        import re
        _LCM_SUMMARY_RE = re.compile(
            r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]"
        )
    text = _content_to_text(content)
    return bool(text) and bool(_LCM_SUMMARY_RE.search(text))


# Internal scaffolding marker the LCM engine sets on the summary message it
# assembles (see plugins/context_engine/lcm/engine.py). ``_``-prefixed so the
# transport sanitizer strips it before any provider request (and the Anthropic/
# Gemini/Bedrock adapters rebuild messages by allowlist and never copy it) — so
# it never reaches the wire and never perturbs the prompt cache.
_LCM_SUMMARY_TAG = "_lcm_summary"


def _is_summary_row(msg, *, engine_is_lcm: bool = False, on_tag_missing=None) -> bool:
    """True if a message is an LCM summary row.

    Fast path: the LCM engine tags the summary message it assembles with the
    internal ``_lcm_summary`` marker (structural — exact, independent of whether
    the marker text survives content flattening). Fallback: the PR #99-hardened
    ``_content_to_text`` regex for built-in-engine / pre-tag / non-LCM rows.

    ``is True`` (not truthy) so a deserialized "yes"/1/None can't flip detection.
    Restores the call sites' ``or ""`` None-guard. When ``engine_is_lcm`` and the
    regex fallback DOES match a summary whose tag is absent, ``on_tag_missing`` is
    called once so the silent fast-path/fallback is observable (the
    COMPACTION_STATS_TAG_MISSING tripwire) rather than a dark revert to the regex.
    """
    if isinstance(msg, dict) and msg.get(_LCM_SUMMARY_TAG) is True:
        return True
    content = msg.get("content") if isinstance(msg, dict) else msg
    hit = _is_summary_message(content or "")
    if hit and engine_is_lcm and on_tag_missing is not None:
        on_tag_missing()  # LCM session: regex matched a summary, but the tag was absent
    return hit


def _classify_summary_ids(comp, *, engine_is_lcm: bool = False, on_tag_missing=None):
    """Return ``{id(m) for m in comp that are summary rows}``, firing the
    tag-missing tripwire AT MOST ONCE per call.

    Sharing one classification across the summary/kept partitions (a) avoids
    re-running ``_is_summary_message`` twice per row and (b) ensures
    ``on_tag_missing`` can't fire twice for the same compaction.
    """
    fired = {"n": 0}

    def _tw():
        if fired["n"] == 0 and on_tag_missing is not None:
            fired["n"] = 1
            on_tag_missing()

    out = set()
    for m in comp:
        if _is_summary_row(m, engine_is_lcm=engine_is_lcm, on_tag_missing=_tw):
            out.add(id(m))
    return out


def build_hygiene_stats(
    *,
    raw_history: List[dict],
    eligible_msgs: List[dict],
    compressed: List[dict],
    estimator,
    engine_is_lcm: bool = False,
    on_tag_missing=None,
) -> "CompactionStats":
    """Build a reconciling ``CompactionStats`` from the session-hygiene path's real data.

    All counts/tokens are MEASURED independently over disjoint row subsets of the
    SAME population (no back-derivation), so ``validate()`` is a real cross-check:

    - ``pre`` = the full raw transcript (`raw_history`).
    - ``eligible`` = the role-filtered subset fed to the throwaway compressor
      (`eligible_msgs` = user/assistant-with-content).
    - ``compressed`` = the LCM output written back. Within it: summary message(s)
      (LCM markers), the system anchor (role == "system"), and the **kept tail**
      (everything else).

    Partition (the 2026-06-22 fix — identity-aware, robust to a kept tail that is
    NOT a clean subset of ``eligible``):

      Every ``pre`` row lands in exactly one of three disjoint buckets, classified
      by whether it survived verbatim into the kept tail and whether the filter
      kept it as eligible:

        * ``kept``    — pre rows that appear verbatim in the compressed kept tail
                        (matched by identity, then by full-content signature for
                        copied populations). Any role: if LCM ever keeps a raw
                        tool/system row in the fresh tail, it counts here, NOT
                        double-counted in ``cleared``.
        * ``folded``  — eligible rows that were NOT kept (folded into the summary).
        * ``cleared`` — non-eligible rows (tool/system/contentless-assistant the
                        filter removed) that were NOT kept.

      ``cleared + folded + kept == pre`` holds BY CONSTRUCTION over identity-
      partitioned rows, and each bucket's tokens are an INDEPENDENT estimator call
      over its own disjoint row set (never ``pre - others`` — that would make
      ``validate()`` a tautology and kill the guard).

    Token sums all use the SAME ``estimator`` over each subset (same-estimator
    contract). Returns a stats object; the caller validates + degrades on failure.
    """
    pre_msgs = list(raw_history or [])
    elig = list(eligible_msgs or [])
    comp = list(compressed or [])

    # Classify each compressed row once (id-keyed) so summary detection is shared
    # between the summary/kept partitions and the tag-missing tripwire fires at
    # most once per build (not once per partition).
    _summary_ids = _classify_summary_ids(
        comp, engine_is_lcm=engine_is_lcm, on_tag_missing=on_tag_missing
    )
    summary_rows = [m for m in comp if id(m) in _summary_ids]
    anchor_rows = [m for m in comp if m.get("role") == "system"]
    kept_compressed_rows = [
        m for m in comp
        if m.get("role") != "system" and id(m) not in _summary_ids
    ]

    # ── Identity-aware three-way partition of `pre` ──
    # The gateway hands us `raw_history` as COPIES and `eligible_msgs` as the
    # filtered ORIGINALS, and `compressed`'s kept tail as yet more copies — so
    # id() identity does NOT line up across the three lists. We therefore
    # partition `pre` by walking it once and classifying each row by signature
    # membership (consume-once multisets) against the kept tail and the eligible
    # set, in that priority order:
    #   kept   = pre rows whose signature matches a kept-tail row  (survived verbatim)
    #   folded = remaining pre rows whose signature matches an eligible row
    #   cleared= everything else (tool/system/contentless the filter removed,
    #            not kept)
    # Priority kept > folded ensures a row LCM kept verbatim is counted ONCE in
    # `kept`, never double-counted in `cleared`/`folded`. Each bucket's tokens
    # are then an INDEPENDENT estimator call over its own rows.
    _kept_want = Counter(_row_signature(m) for m in kept_compressed_rows)
    _elig_want = Counter(_row_signature(m) for m in elig)
    kept_rows: List[dict] = []
    folded_rows: List[dict] = []
    cleared_rows: List[dict] = []
    for m in pre_msgs:
        sig = _row_signature(m)
        if _kept_want.get(sig, 0) > 0:
            _kept_want[sig] -= 1
            kept_rows.append(m)
            # consume an eligible slot if this kept row was eligible (keeps the
            # eligible multiset honest so a folded row can't reuse the slot)
            if _elig_want.get(sig, 0) > 0:
                _elig_want[sig] -= 1
        elif _elig_want.get(sig, 0) > 0:
            _elig_want[sig] -= 1
            folded_rows.append(m)
        else:
            cleared_rows.append(m)

    pre_messages = len(pre_msgs)
    # Two kept populations on the message axis (twin of the kept_pre_tokens split):
    #   kept_messages     = COUNT of the comp-side kept tail (truth in live context).
    #   kept_pre_messages = COUNT of pre-side kept rows (raw rows that survived).
    # They diverge when LCM sanitizes the kept tail (pre-side finds no signature
    # match). post_messages is MEASURED as len(comp) — NOT kept+summary+anchor, which
    # would be tautological with validate()'s POST identity and hide a partition bug.
    kept_pre_messages = len(kept_rows)
    kept_messages = len(kept_compressed_rows)
    folded_count = len(folded_rows)
    cleared_count = len(cleared_rows)
    summary_messages = len(summary_rows)
    anchor_messages = len(anchor_rows)
    post_messages = len(comp)

    cleared_tokens = int(estimator(cleared_rows)) if cleared_rows else 0
    # Optional tool/other sub-split of the cleared population (derive-by-subtraction,
    # parent `cleared_tokens` untouched). Degrade to None on any failure (D-6).
    _ctc = _ctt = _coc = _cot = None
    try:
        _ctc, _ctt, _coc, _cot = _tool_other_split(cleared_rows, cleared_tokens, estimator)
    except Exception:
        _ctc = _ctt = _coc = _cot = None

    return CompactionStats(
        pre_messages=pre_messages,
        post_messages=post_messages,
        # eligible_count must satisfy validate()'s two coupled PRE-axis identities:
        #   cleared == pre - eligible   AND   kept_pre + folded == eligible.
        # Under the identity partition `cleared + folded + kept_pre == pre`, BOTH hold
        # iff `eligible == kept_pre + folded` (every retained row — folded into the
        # summary OR kept verbatim — is on the "eligible/retained" side; `cleared`
        # is exactly what was removed). Uses the PRE-side kept count (raw rows that
        # survived): the eligible axis is about the raw transcript partition, not the
        # post-sanitization comp tail. (kept_messages, the comp-side count, only
        # carries the POST identity + the user-facing display.)
        eligible_count=kept_pre_messages + folded_count,
        kept_messages=kept_messages,
        kept_pre_messages=kept_pre_messages,
        summary_messages=summary_messages,
        anchor_messages=anchor_messages,
        cleared_count=cleared_count,
        folded_count=folded_count,
        pre_tokens=int(estimator(pre_msgs)),
        post_tokens=int(estimator(comp)),
        # POST identity: comp-side kept tail (the actual fresh tail in `compressed`).
        kept_tokens=int(estimator(kept_compressed_rows)) if kept_compressed_rows else 0,
        # PRE identity: pre-side kept rows (raw_history rows that survived). Distinct
        # population — diverges from comp-side when LCM sanitized the kept tail.
        kept_pre_tokens=int(estimator(kept_rows)) if kept_rows else 0,
        summary_tokens=int(estimator(summary_rows)) if summary_rows else 0,
        anchor_tokens=int(estimator(anchor_rows)) if anchor_rows else 0,
        cleared_tokens=cleared_tokens,
        folded_tokens=int(estimator(folded_rows)) if folded_rows else 0,
        cleared_tool_count=_ctc,
        cleared_tool_tokens=_ctt,
        cleared_other_count=_coc,
        cleared_other_tokens=_cot,
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
    want = Counter(_row_signature(m) for m in kept)
    out = []
    for m in eligible:
        s = _row_signature(m)
        if want.get(s, 0) > 0:
            want[s] -= 1
        else:
            out.append(m)
    return out


def _partition_pre_by_comp_kept(
    eligible: List[dict], kept: List[dict]
) -> Tuple[List[dict], List[dict]]:
    """Single consume-once partition of ``eligible`` into (kept_pre, folded).

    The A-floor (fallback) partition for the in-turn path. Walks ``eligible``
    ONCE, classifying each row as *matched* (a pre-side row whose signature is in
    the comp-kept multiset → ``kept_pre``) or *unmatched* (→ ``folded``). The two
    buckets are disjoint and together exhaust ``eligible`` BY CONSTRUCTION
    (``len(kept_pre) + len(folded) == len(eligible)``) — so a separate
    ``estimator()`` over each bucket reconciles to ``estimator(eligible)`` within
    estimator rounding, regardless of WHY whole-tail alignment failed.

    Prefers ``id()``-identity when the same row objects flow through; falls back to
    a collision-resistant ``(role, full-content-hash)`` **multiset** (consume-once,
    decrement on match) so a duplicated signature (repeated tool scaffolds, short
    identical turns on heavy sessions) cannot be matched twice or steal another
    row's slot. This is the can-never-RECONCILE-fail floor for any path that lacks
    Option B's authoritative engine record (built-in engine, overflow, a future
    engine, or an LCM generation mismatch).

    NOTE: the kept_pre/folded *attribution* here is signature-approximate (a
    sanitized-then-unmatchable kept row lands in ``folded``); the TOTALS reconcile
    exactly. The caller labels A-floor renders as approximate and measures the
    gross bound (degrade-to-two-line above threshold).
    """

    kept_ids = {id(m) for m in kept}
    matched = [m for m in eligible if id(m) in kept_ids]
    if len(matched) == len(kept):
        # identity lined up cleanly — complement is exact
        folded = [m for m in eligible if id(m) not in kept_ids]
        return matched, folded
    # copies / sanitized tail: consume-once signature multiset (collision-safe)
    want = Counter(_row_signature(m) for m in kept)
    kept_pre: List[dict] = []
    folded: List[dict] = []
    for m in eligible:
        s = _row_signature(m)
        if want.get(s, 0) > 0:
            want[s] -= 1
            kept_pre.append(m)
        else:
            folded.append(m)
    return kept_pre, folded


_PROVENANCE_KEY = "_src_idx"


def harvest_provenance_partition(messages, kept_rows):
    """Option B: read the EXACT pre-side kept partition off provenance-stamped rows.

    The LCM engine stamps ``_src_idx`` (index into the original ``messages``) onto
    each tail row it hands to ``_assemble_context`` on a **single-pass** compaction.
    The pipeline's shallow-copies (``dict(msg)`` / ``msg.copy()``) carry the key
    through every drop / strip / content-rewrite stage by construction; synthetic
    tool stubs are freshly built and therefore lack it. So the kept region of the
    returned ``compressed`` splits cleanly into:

      * ``kept_pre_indices``  — origins of the surviving real kept rows (the EXACT
        pre-side kept set; no inference, no replay).
      * ``stub_count``        — kept rows WITHOUT ``_src_idx`` (engine-synthesized
        placeholders; sourced independently from the rows themselves, NOT derived
        as ``comp_count − kept_pre`` — a genuine independent count).

    Returns ``(kept_pre_indices, stub_count)`` when provenance is present + valid
    (every index in range, no duplicates), else ``None`` → caller falls to the
    A-floor. Validity is checked here so a malformed stamp can never produce a
    confidently-wrong split. Caller MUST strip ``_src_idx`` from ``compressed``
    after harvest (it must not reach the wire/cache).
    """
    n = len(messages)
    kept_pre_indices = []
    stub_count = 0
    seen = set()
    any_stamped = False
    for m in kept_rows:
        if not isinstance(m, dict) or _PROVENANCE_KEY not in m:
            stub_count += 1
            continue
        any_stamped = True
        idx = m.get(_PROVENANCE_KEY)
        if not isinstance(idx, int) or idx < 0 or idx >= n or idx in seen:
            return None  # malformed/duplicate provenance → unsafe, fall to A-floor
        seen.add(idx)
        kept_pre_indices.append(idx)
    if not any_stamped:
        return None  # no provenance present at all (not a stamped single-pass LCM
        #              compaction, or a non-B caller) → fall to replay / A-floor
    return kept_pre_indices, stub_count


def strip_provenance(messages) -> int:
    """Remove ``_src_idx`` from every row in-place; return how many were stripped.

    Load-bearing (the only thing keeping the internal provenance key off the
    wire/cache). Idempotent. Callers run this on ``compressed`` immediately after
    harvest, before the list flows onward.
    """
    stripped = 0
    for m in messages:
        if isinstance(m, dict) and _PROVENANCE_KEY in m:
            del m[_PROVENANCE_KEY]
            stripped += 1
    return stripped


def _is_tool_message(m) -> bool:
    """True when a message is a tool-result / tool-call carrier.

    Covers BOTH shapes the two compaction paths see:
      * hygiene path — flat ``role == "tool"`` rows.
      * in-turn path — API content-block rows where a tool result rides on a
        ``role == "user"`` message with a ``tool_result`` block, and a tool call
        rides on a ``role == "assistant"`` message with a ``tool_use`` block.
    So the "tool-result messages" breakout works on the live in-turn shape, not
    only the flat hygiene shape.
    """
    if not isinstance(m, dict):
        return False
    if m.get("role") == "tool":
        return True
    content = m.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in {
                "tool_result", "tool_use", "tool_call",
            }:
                return True
    return False


def _tool_other_split(rows, parent_tokens, estimator):
    """Split a row-list into (tool_count, tool_tokens, other_count, other_tokens).

    ``other_tokens`` is DERIVED by subtraction (``parent_tokens - tool_tokens``),
    NOT estimated — so ``tool+other == parent_tokens`` EXACTLY despite the
    estimator's ceil-of-sum non-additivity, and the parent total is left untouched
    (D-7; keeps the freed/pre/post axes provably unperturbed). The caller MUST pass
    the SAME list whose length is the parent bucket's count and whose estimate is
    ``parent_tokens`` (the post-fold list), so counts and tokens both partition by
    construction regardless of which dedup branch produced it. ``estimator`` returns
    an int, so no ``int()`` wrap is needed. Tool classification is shape-aware
    (flat ``role==tool`` AND API ``tool_result``/``tool_use`` content blocks).
    """
    tool = [m for m in rows if _is_tool_message(m)]
    other = [m for m in rows if not _is_tool_message(m)]
    tool_tokens = estimator(tool) if tool else 0
    other_tokens = parent_tokens - tool_tokens
    return (len(tool), tool_tokens, len(other), other_tokens)


def _inturn_content_struct(content):
    """Structural (NON-``str()``-flattened) content key for alignment matching.

    ``str(content)`` can collide two structurally-different list/dict rows (the
    in-turn path carries list/multipart content — the #95 lineage), so the cut
    search compares content by shape: a scalar passes through; a list compares
    element-wise by ``(type, text)``. Used ONLY to compare a replayed sanitized
    slice against the comp-side kept tail — never rendered.
    """
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                out.append(("part", b.get("type"), _part_text(b)))
            else:
                out.append(("part", None, b if isinstance(b, str) else repr(b)))
        return tuple(out)
    if isinstance(content, str):
        return ("scalar", content)
    return ("scalar", repr(content))


def _inturn_tool_calls_struct(tool_calls):
    """Structural key for an assistant row's ``tool_calls``.

    The sanitizer PRESERVES assistant tool_calls (and uses them to decide which
    stub tool-results to insert), so two assistant rows with identical sanitized
    visible content but DIFFERENT tool calls are distinct rows — omitting
    tool_calls from the match key would let a wrong cut compare equal and produce
    a reconciled-but-wrong folded/kept split (Greptile P1, PR #106). Normalize by
    the call ``id`` + function name/arguments when present (dict-order-independent),
    falling back to ``repr`` for an unexpected shape.
    """
    if not tool_calls:
        return ()
    out = []
    for call in tool_calls:
        if isinstance(call, dict):
            fn = call.get("function")
            if isinstance(fn, dict):
                out.append((call.get("id"), fn.get("name"), fn.get("arguments")))
            else:
                out.append((call.get("id"), call.get("name"), call.get("arguments"), repr(fn)))
        else:
            out.append(repr(call))
    return tuple(out)


def _inturn_norm_row(m):
    return (
        m.get("role"),
        _inturn_content_struct(m.get("content")),
        m.get("tool_call_id"),
        _inturn_tool_calls_struct(m.get("tool_calls")),
    )


def _inturn_norm(msgs):
    return tuple(_inturn_norm_row(m) for m in msgs)


def find_inturn_kept_cut(messages, comp_kept, sanitize, fresh_tail_count, *, slack=8):
    """Whole-tail-replay alignment: the index ``cut`` such that the REAL LCM
    sanitizer applied to ``messages[cut:]`` reproduces the comp-side kept tail.

    The LCM active context is ``[anchor] + [summaries] + sanitize(fresh tail)``,
    where the fresh tail is a contiguous suffix of ``messages`` (possibly shortened
    by an assembly budget). The sanitizer's drop + stub-insert are TAIL-GLOBAL
    (``_sanitize_tool_pairs`` operates on the whole tail as a set), so a per-row
    predicate cannot reproduce them — only replaying the batch sanitizer over a
    candidate slice can. We therefore search ``cut`` and accept on EXACT equality
    of ``sanitize(messages[cut:])`` to ``comp_kept`` (never a token sum, so a wrong
    cut cannot pass). The search starts at the expected boundary
    (``len(messages) - fresh_tail_count``) and expands outward, so the common case
    is one replay and the budget-trimmed shorter tail (true cut *interior*) is still
    covered. Returns the cut index, or ``None`` when no slice in the window
    reproduces ``comp_kept`` (→ caller fails ``validate()`` → honest two-line
    degrade). ``sanitize`` must be the engine's real
    ``_sanitize_active_context_messages`` bound method.
    """
    if sanitize is None or not fresh_tail_count:
        return None
    target = _inturn_norm(comp_kept)
    n = len(messages)
    expected = max(0, n - int(fresh_tail_count))
    lo = max(0, n - int(fresh_tail_count) - int(slack))
    candidates = sorted(range(lo, n + 1), key=lambda c: (abs(c - expected), c))
    for cut in candidates:
        try:
            if _inturn_norm(sanitize(messages[cut:])) == target:
                return cut
        except Exception:
            continue
    return None


def build_inturn_stats(
    *,
    messages: List[dict],
    compressed: List[dict],
    estimator,
    engine_is_lcm: bool = False,
    on_tag_missing=None,
    sanitize=None,
    fresh_tail_count=None,
) -> "CompactionStats":
    """Build a reconciling ``CompactionStats`` for the in-turn (LCM) done-site.

    The in-turn path does NOT role-filter — the whole input ``messages`` list is
    the population, so ``cleared == 0`` and ``eligible == pre``. The compressed
    output is ``[anchor] + [summaries] + sanitize(fresh tail)``.

    There are TWO distinct "kept" populations that must NOT be conflated (the
    PR #101 lesson, here on the in-turn path):

    * **comp-side kept** (``kept_rows``) = the actual fresh tail in ``compressed``,
      AFTER the LCM sanitizer stripped assistant content / dropped empty rows /
      inserted synthetic tool stubs. Feeds the POST identity + the user-facing
      "kept N recent chat" display.
    * **pre-side kept** (``kept_pre_rows``) = the RAW ``messages`` rows that became
      that tail. Feeds the PRE identity (``cleared + folded + kept_pre == pre``).

    When the engine's real ``sanitize`` + ``fresh_tail_count`` are supplied, the
    cut between folded | kept-tail is found by whole-tail replay
    (:func:`find_inturn_kept_cut`) so ``folded`` AND ``kept_pre`` are both measured
    pre-side over a true partition of ``messages`` → PRE reconciles even when the
    sanitizer made the comp tail diverge from its raw originals. Without them
    (built-in / overflow / manual paths, or a legacy caller), it falls back to the
    pre-side complement of the comp-kept rows (``_fold_rows``), where ``kept_pre``
    is left unset and the property defaults to comp-side (equal by construction on
    a non-sanitizing engine). If alignment fails (no slice reproduces the comp
    tail), ``folded``/``kept_pre`` are set so ``validate()`` fails → the caller
    degrades to the honest two-line announce.

    Caller validates + degrades on failure.
    """
    pre_msgs = list(messages or [])
    comp = list(compressed or [])

    _summary_ids = _classify_summary_ids(
        comp, engine_is_lcm=engine_is_lcm, on_tag_missing=on_tag_missing
    )
    summary_rows = [m for m in comp if id(m) in _summary_ids]
    anchor_rows = [m for m in comp if m.get("role") == "system"]
    kept_rows = [
        m for m in comp
        if m.get("role") != "system" and id(m) not in _summary_ids
    ]

    pre_messages = len(pre_msgs)
    eligible_count = pre_messages  # no role filter on the in-turn path
    kept_messages = len(kept_rows)          # comp-side (POST identity + display)
    summary_messages = len(summary_rows)
    anchor_messages = len(anchor_rows)
    post_messages = len(comp)               # MEASURED, not derived (no tautology)

    # ── Pre-side partition: folded | kept_pre (the PRE identity) ──
    # Precedence: (B) provenance harvest off _src_idx-stamped kept rows — EXACT,
    # the engine told us the origins; → (replay) whole-tail alignment — exact when
    # it aligns; → (A-floor) exhaustive single-walk partition — reconciles-always,
    # signature-approximate. ``kept_pre_*`` stay None only on legacy/non-sanitizing
    # paths where the _kept_pre_* property defaults to comp-side.
    kept_pre_messages = None
    kept_pre_tokens = None
    approx_attribution = False
    raw_tail_tokens = None
    b_engaged = False
    cut = None

    # (B) Option B — provenance partition (exact, no inference). The kept rows carry
    # ``_src_idx`` (origin index into ``pre_msgs``) when the engine stamped a
    # single-pass compaction; synthetic stubs lack it. kept_pre = the stamped
    # origins; fold = pre rows whose index is NOT a kept origin. Exact attribution.
    _b = harvest_provenance_partition(pre_msgs, kept_rows)
    if _b is not None:
        _kept_idx, _stub_count = _b
        _kept_idx_set = set(_kept_idx)
        kept_pre_rows = [pre_msgs[i] for i in _kept_idx]
        fold_rows = [m for i, m in enumerate(pre_msgs) if i not in _kept_idx_set]
        kept_pre_messages = len(kept_pre_rows)
        kept_pre_tokens = int(estimator(kept_pre_rows)) if kept_pre_rows else 0
        folded_count = len(fold_rows)
        b_engaged = True

    if not b_engaged:
        if sanitize is not None and fresh_tail_count:
            cut = find_inturn_kept_cut(pre_msgs, kept_rows, sanitize, fresh_tail_count)
        if cut is not None:
            kept_pre_rows = pre_msgs[cut:]
            fold_rows = pre_msgs[:cut]
            kept_pre_messages = len(kept_pre_rows)
            kept_pre_tokens = int(estimator(kept_pre_rows)) if kept_pre_rows else 0
            folded_count = len(fold_rows)
        else:
            # A-floor: exhaustive single-walk pre-side partition. When whole-tail
            # alignment can't reproduce the in-context-sanitized comp tail (the real
            # heavy-session failure mode), partition ``pre`` ONCE into (kept_pre, folded)
            # against the comp-kept multiset. Both buckets are measured pre-side over a
            # disjoint exhaustive partition, so ``folded_tokens + kept_pre_tokens ==
            # pre_tokens`` BY CONSTRUCTION (within estimator rounding) — validate()
            # reconciles instead of re-exhibiting the old mixed pre/comp-side gap. The
            # kept/folded *attribution* is signature-approximate (bounded by the kept-tail
            # fraction, which is a small contiguous suffix — the folded bulk is a prefix
            # and always classifies correctly), so totals are exact and only the split is
            # approximate. ``approx_attribution`` flags the caller to label + watch it.
            kept_pre_rows, fold_rows = _partition_pre_by_comp_kept(pre_msgs, kept_rows)
            kept_pre_messages = len(kept_pre_rows)
            kept_pre_tokens = int(estimator(kept_pre_rows)) if kept_pre_rows else 0
            folded_count = len(fold_rows)
            approx_attribution = True
        # RAW kept-tail upper bound for the caller's gross-error guard: the raw
        # (pre-sanitize) size of the tail region the engine keeps. Match- AND
        # sanitize-independent (computed from the raw suffix), so a heavily-stripped
        # tail can't make the guard under-report (Greptile P1 ×2). Use the engine's
        # fresh_tail_count when known, else the comp-side kept count as a proxy.
        _tail_n = int(fresh_tail_count) if fresh_tail_count else kept_messages
        if _tail_n > 0:
            raw_tail_tokens = int(estimator(pre_msgs[-_tail_n:])) if pre_msgs else 0

    folded_tokens = int(estimator(fold_rows)) if fold_rows else 0
    # Optional tool/other sub-split of the folded population (derive-by-subtraction,
    # parent `folded_tokens` untouched). Degrade to None on any failure (D-6).
    _ftc = _ftt = _foc = _fot = None
    try:
        _ftc, _ftt, _foc, _fot = _tool_other_split(fold_rows, folded_tokens, estimator)
    except Exception:
        _ftc = _ftt = _foc = _fot = None

    return CompactionStats(
        pre_messages=pre_messages,
        post_messages=post_messages,
        eligible_count=eligible_count,
        kept_messages=kept_messages,
        kept_pre_messages=kept_pre_messages,
        summary_messages=summary_messages,
        anchor_messages=anchor_messages,
        cleared_count=0,
        folded_count=folded_count,
        pre_tokens=int(estimator(pre_msgs)),
        post_tokens=int(estimator(comp)),
        kept_tokens=int(estimator(kept_rows)) if kept_rows else 0,
        kept_pre_tokens=kept_pre_tokens,
        summary_tokens=int(estimator(summary_rows)) if summary_rows else 0,
        anchor_tokens=int(estimator(anchor_rows)) if anchor_rows else 0,
        cleared_tokens=0,
        folded_tokens=folded_tokens,
        folded_tool_count=_ftc,
        folded_tool_tokens=_ftt,
        folded_other_count=_foc,
        folded_other_tokens=_fot,
        approx_attribution=approx_attribution,
        raw_tail_tokens=raw_tail_tokens,
    )
