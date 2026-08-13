"""Continuation fields for newly generated local compaction handoffs.

Historical, provider-native, and micro summaries remain readable without this schema.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GOVERNING_OUTCOME_HEADING = "## Governing User Outcome"
CURRENT_SUBTASK_HEADING = "## Current Subtask"
LATEST_USER_CORRECTION_HEADING = "## Latest User Correction"
NEXT_OUTCOME_STEP_HEADING = "## Next Outcome-Relevant Step (Reference Only)"
_CONTINUATION_HEADINGS = (
    GOVERNING_OUTCOME_HEADING,
    CURRENT_SUBTASK_HEADING,
    LATEST_USER_CORRECTION_HEADING,
    NEXT_OUTCOME_STEP_HEADING,
)
MICRO_USER_SEQUENCE_POINTERS_HEADING = (
    "## Surviving Real-User Sequence Pointers (Noncanonical)"
)

MICRO_SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] [MICRO] This marker contains compacted "
    "assistant/tool history; real user messages remain verbatim. The "
    "noncanonical pointer block below refers only to that surviving real-user "
    "sequence and does not supply values for the batch continuation schema. "
    "Resolve a context-dependent latest user message from the surviving "
    "real-user sequence first; explicit corrections, cancellations, topic "
    "changes, and stops win. "
    "Never select work from Historical Task Snapshot:"
)

_NO_USER_CONTINUATION_VALUES = {
    GOVERNING_OUTCOME_HEADING: (
        "Unknown. No user-authored governing outcome is available."
    ),
    CURRENT_SUBTASK_HEADING: "None. No user-authored subtask exists.",
    LATEST_USER_CORRECTION_HEADING: "None. No user-authored correction exists.",
    NEXT_OUTCOME_STEP_HEADING: "None. No user-authored next step exists.",
}


def neutralize_summary_delimiters(text: str) -> str:
    """Keep quoted payload from acquiring compaction transport ownership."""
    from agent.context_compressor import _MERGED_SUMMARY_DELIMITER, _SUMMARY_END_MARKER

    return text.replace(
        _MERGED_SUMMARY_DELIMITER, "[reserved compaction delimiter neutralized]",
    ).replace(_SUMMARY_END_MARKER, "[reserved summary boundary neutralized]")


def continuation_instructions(has_user_turn: bool) -> dict[str, str]:
    """Field definitions shared by fresh and iterative local summaries."""
    if not has_user_turn:
        return {
            "governing_outcome": f"[Write exactly: {_NO_USER_CONTINUATION_VALUES[GOVERNING_OUTCOME_HEADING]}]",
            "current_subtask": f"[Write exactly: {_NO_USER_CONTINUATION_VALUES[CURRENT_SUBTASK_HEADING]}]",
            "latest_user_correction": f"[Write exactly: {_NO_USER_CONTINUATION_VALUES[LATEST_USER_CORRECTION_HEADING]}]",
            "next_outcome_step": f"[Write exactly: {_NO_USER_CONTINUATION_VALUES[NEXT_OUTCOME_STEP_HEADING]}]",
        }
    return {
        "governing_outcome": """[The final result the user still wants delivered. Distinguish the desired
result from a method, tool, artifact, experiment, research activity, update, or
intermediate step. A recent subtask must not replace the governing outcome
merely because it is recent. Preserve this outcome across successive
compactions unless a user explicitly replaces, narrows, cancels, or changes it,
including by assigning a clearly independent new task or topic, or the
transcript explicitly establishes that the result was delivered. If the source
explicitly establishes that the governing outcome itself was delivered or
cancelled, write "None — delivered: <result>" or "None — cancelled: <result>"
and require both the current subtask and next outcome-relevant step to be
"None." If the source does not establish a final desired result, write
"Unknown." Never invent intent.]""",
        "current_subtask": """[The immediate intermediate step, its relationship to the governing outcome,
and its current state. A completed, cancelled, or superseded subtask belongs in
historical progress, not here. If no subtask remains open, write "None."]""",
        "latest_user_correction": """[The latest still-applicable user correction, narrowing, cancellation, stop,
or change of route. Preserve the user's words when possible and state exactly
which prior route it invalidates. If none exists, write "None."]""",
        "next_outcome_step": """[Exactly one pending action that directly advances the still-open governing
outcome. This field is reference only: a later context-dependent message such
as "continue" or "what next?" may use it to resolve its referent, but the field
does not activate work by itself. Never select work that is completed,
cancelled, or superseded. If two materially different referents remain, make
the one next step "Ask one short clarification question: <referent A> or
<referent B>?"; do not choose or activate either referent.
If the outcome is complete or no next action is grounded, write "None."]""",
    }


class ContinuationSchemaMixin:
    @staticmethod
    def _normalize_structural_heading_title(title: str) -> str:
        """Normalize harmless CommonMark heading decoration for comparison."""
        title = re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()
        title = title.rstrip(".: \t")
        title = re.sub(r"(?i)^the[ \t]+", "", title)
        return " ".join(title.split()).casefold()

    @classmethod
    def _commonmark_heading_candidates(
        cls,
        summary: str,
    ) -> list[tuple[int, str, str]]:
        """Return unquoted, non-code ATX and setext headings.

        This is deliberately a bounded CommonMark scanner rather than a full
        Markdown renderer. It recognizes only syntax that can create competing
        live section structure and ignores fenced/indented code and blockquotes.
        """
        lines = summary.splitlines(keepends=True)
        offsets: list[int] = []
        cursor = 0
        for line in lines:
            offsets.append(cursor)
            cursor += len(line)

        eligible = [False] * len(lines)
        candidates: list[tuple[int, str, str]] = []
        fence_char: Optional[str] = None
        fence_length = 0
        atx_re = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")

        for index, line in enumerate(lines):
            raw = line.rstrip("\r\n")
            if fence_char is not None:
                close = re.fullmatch(
                    rf" {{0,3}}{re.escape(fence_char)}{{{fence_length},}}[ \t]*",
                    raw,
                )
                if close:
                    fence_char = None
                    fence_length = 0
                continue
            if raw.startswith("\t") or raw.startswith("    "):
                continue
            left = raw.lstrip(" ")
            if len(raw) - len(left) > 3 or left.startswith(">"):
                continue
            fence = re.match(r"^ {0,3}(`{3,}|~{3,})", raw)
            if fence:
                fence_char = fence.group(1)[0]
                fence_length = len(fence.group(1))
                continue

            eligible[index] = True
            atx = atx_re.fullmatch(raw)
            if atx:
                title = atx.group(2) or ""
                candidates.append((
                    offsets[index],
                    raw,
                    cls._normalize_structural_heading_title(title),
                ))

        setext_underline = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
        for index in range(len(lines) - 1):
            if not (eligible[index] and eligible[index + 1]):
                continue
            title = lines[index].rstrip("\r\n")
            underline = lines[index + 1].rstrip("\r\n")
            if (
                not title.strip()
                or title.startswith("\t")
                or atx_re.fullmatch(title)
                or not setext_underline.fullmatch(underline)
            ):
                continue
            candidates.append((
                offsets[index],
                f"{title}\n{underline}",
                cls._normalize_structural_heading_title(title.lstrip(" ")),
            ))
        return candidates

    @classmethod
    def _parse_summary_continuation_schema(
        cls, summary: str,
    ) -> tuple[dict[str, str], dict[str, tuple[int, int]], int]:
        """Validate real Markdown headings; quoted and fenced evidence is inert."""
        from agent.context_compressor import HISTORICAL_TASK_HEADING

        required = (HISTORICAL_TASK_HEADING, *_CONTINUATION_HEADINGS)
        structural = cls._commonmark_heading_candidates(summary)
        headings = [
            (start, raw.rstrip(" \t\r"), name)
            for start, raw, name in structural
            if re.fullmatch(r"##[ \t]+[^\r\n]+", raw)
        ]
        canonical = headings[:len(required)]
        if (
            len(canonical) != len(required)
            or tuple(raw for _start, raw, _name in canonical) != required
            or summary[:canonical[0][0]].strip()
        ):
            raise RuntimeError(
                "Context compression summary has invalid continuation schema: "
                "required canonical headings are missing, duplicated, or out of order"
            )
        reserved = {cls._normalize_structural_heading_title(h[3:]) for h in required} | {"goal"}
        canonical_starts = {start for start, _raw, _name in canonical}
        for start, raw, name in structural:
            if name in reserved and start not in canonical_starts:
                raise RuntimeError(
                    "Context compression summary has invalid continuation schema: "
                    f"competing reserved heading {raw!r}"
                )
        canonical_end = headings[len(required)][0] if len(headings) > len(required) else len(summary)
        values, spans = {}, {}
        for index, (start, raw, _name) in enumerate(canonical):
            region_start = start + len(raw)
            region_end = canonical[index + 1][0] if index + 1 < len(canonical) else canonical_end
            raw_value = summary[region_start:region_end]
            value_start = region_start + len(raw_value) - len(raw_value.lstrip())
            value_end = region_start + len(raw_value.rstrip())
            value = summary[value_start:value_end]
            if not value:
                raise RuntimeError(
                    "Context compression summary has invalid continuation schema: "
                    f"{required[index]!r} is empty"
                )
            values[required[index]] = value
            spans[required[index]] = (value_start, value_end)
        return values, spans, canonical_end

    @classmethod
    def _canonicalize_terminal_none_values(cls, summary: str) -> str:
        """Normalize only harmless terminal ``None`` spelling variants."""
        section_values, value_spans, _canonical_end = (
            cls._parse_summary_continuation_schema(summary)
        )
        governing_outcome = section_values[GOVERNING_OUTCOME_HEADING]
        terminal_match = re.match(
            r"(?i)^none[ \t]+[-–—][ \t]+"
            r"(delivered|cancelled|canceled)[ \t]*:[ \t]*",
            governing_outcome,
        )
        if terminal_match is None:
            return summary

        state = terminal_match.group(1).casefold()
        canonical_state = "cancelled" if state == "canceled" else state
        result = governing_outcome[terminal_match.end():]
        replacements: list[tuple[int, int, str]] = [(
            value_spans[GOVERNING_OUTCOME_HEADING][0],
            value_spans[GOVERNING_OUTCOME_HEADING][1],
            f"None — {canonical_state}: {result}",
        )]
        for heading in (CURRENT_SUBTASK_HEADING, NEXT_OUTCOME_STEP_HEADING):
            if re.fullmatch(r"(?i:none\.?)", section_values[heading]):
                start, end = value_spans[heading]
                replacements.append((start, end, "None."))
        for start, end, replacement in sorted(replacements, reverse=True):
            summary = summary[:start] + replacement + summary[end:]
        return summary

    @classmethod
    def _validate_summary_continuation_schema(
        cls,
        summary: str,
        has_user_turn: bool,
    ) -> None:
        """Reject structurally contradictory or invented continuation state."""
        from agent.context_compressor import SUMMARY_PREFIX

        if summary.startswith(SUMMARY_PREFIX):
            summary = summary[len(SUMMARY_PREFIX):].lstrip()
        section_values, _value_spans, _canonical_end = (
            cls._parse_summary_continuation_schema(summary)
        )

        governing_outcome = section_values[GOVERNING_OUTCOME_HEADING]
        terminal_outcome_prefixes = (
            "None — delivered:",
            "None — cancelled:",
        )
        if governing_outcome.startswith(terminal_outcome_prefixes):
            terminal_prefix = next(
                prefix
                for prefix in terminal_outcome_prefixes
                if governing_outcome.startswith(prefix)
            )
            if not governing_outcome[len(terminal_prefix):].strip():
                raise RuntimeError(
                    "Context compression summary has invalid continuation state: "
                    "a delivered or cancelled governing outcome requires a "
                    "non-empty result"
                )
            for heading in (CURRENT_SUBTASK_HEADING, NEXT_OUTCOME_STEP_HEADING):
                if section_values[heading] != "None.":
                    raise RuntimeError(
                        "Context compression summary has contradictory "
                        "continuation state: a delivered or cancelled governing "
                        f"outcome requires {heading!r} to be exactly 'None.'"
                    )

        if not has_user_turn:
            for heading, exact_value in _NO_USER_CONTINUATION_VALUES.items():
                if section_values[heading] != exact_value:
                    raise RuntimeError(
                        "Context compression summary invented continuation state "
                        "for a session with no user-authored turns: "
                        f"{heading!r}"
                    )

    @staticmethod
    def _validate_summary_user_provenance(summary: str, has_user_turn: bool) -> None:
        """Reject user attribution when the source transcript has no user."""
        from agent.context_compressor import HISTORICAL_TASK_HEADING, _NO_USER_TASK_SENTINEL

        if has_user_turn:
            return
        match = re.search(
            rf"(?ms)^{re.escape(HISTORICAL_TASK_HEADING)}\s*\n(.*?)(?=\n##\s|\Z)",
            summary,
        )
        task_snapshot = match.group(1).strip() if match else ""
        # Ignore section titles (for example "Historical Pending User Asks")
        # and reject affirmative attribution prose instead. Safe zero-user
        # sentinels intentionally say "No user-authored ..." and therefore do
        # not match these subject, passive, or possessive forms. Quoted tool
        # output can still false-positive; that costs one retry/fallback rather
        # than letting fabricated attribution persist.
        attribution_scan = re.sub(r"(?m)^##[^\r\n]*\r?$", "", summary)
        attribution_verbs = (
            r"asked|requested|instructed|said|stated|wrote|wanted|wants|needed|"
            r"needs|required|requires|expected|expects|preferred|prefers|"
            r"corrected|cancelled|canceled|approved|rejected|specified|directed|told"
        )
        attribution_nouns = (
            r"request|instruction|task|goal|outcome|preference|correction|decision"
        )
        attribution_subject = r"(?:the\s+)?(?:user|human)"
        invented_attribution = re.search(
            rf"\b{attribution_subject}\s+(?:(?:has|had|explicitly)\s+)?"
            rf"(?:{attribution_verbs})\b"
            rf"|\b(?:{attribution_verbs})\s+by\s+{attribution_subject}\b"
            rf"|\b(?:the\s+)?user(?:['’]s)\s+(?:{attribution_nouns})\b"
            rf"|\baccording\s+to\s+(?:the\s+)?user(?:['’]s)?"
            rf"(?=[^\w-]|$)"
            rf"|\bper\s+the\s+user(?:['’]s)?"
            rf"(?=[^\w-]|$)"
            rf"|\bper\s+user(?:['’]s)?"
            rf"[ \t]+(?:{attribution_nouns})\b",
            attribution_scan,
            re.IGNORECASE,
        )
        if task_snapshot != _NO_USER_TASK_SENTINEL or invented_attribution:
            raise RuntimeError(
                "Context compression summary invented user attribution for a "
                "session with no user-authored turns"
            )

    def _finalize_summary_candidate(
        self, summary: str, turns: List[Dict[str, Any]], has_user_turn: bool,
    ) -> str:
        """Validate the final local handoff, including deterministic appendices."""
        summary = neutralize_summary_delimiters(summary)
        self._validate_summary_user_provenance(summary, has_user_turn)
        summary = self._canonicalize_terminal_none_values(summary)
        before, _spans, _end = self._parse_summary_continuation_schema(summary)
        summary = neutralize_summary_delimiters(self._augment_summary_lean(summary, turns))
        after, _spans, _end = self._parse_summary_continuation_schema(summary)
        if before != after:
            raise RuntimeError("Lean summary augmentation modified canonical continuation state")
        self._validate_summary_user_provenance(summary, has_user_turn)
        self._validate_summary_continuation_schema(summary, has_user_turn)
        return summary

    def _build_static_fallback_summary(
        self, turns_to_summarize: List[Dict[str, Any]], reason: str | None = None,
    ) -> Optional[str]:
        """Preserve evidence while leaving unverified continuation state Unknown."""
        from agent import context_compressor as cc

        def clean(value: Any, limit: int = cc._FALLBACK_TURN_MAX_CHARS) -> str:
            text = neutralize_summary_delimiters(cc._redact_compaction_text(str(value or "")))
            text = re.sub(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b", "[REDACTED]", text)
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"\bgh[pousr]_[A-Za-z0-9_.-]+", "[REDACTED]", text)
            marker = " ...[truncated]"
            return text if len(text) <= limit else text[:max(0, limit - len(marker))].rstrip() + marker

        anchors = self._fallback_anchors(turns_to_summarize)
        user_asks = anchors["user_asks"]
        has_user_turn = bool(user_asks) or self._summary_has_user_turn is True
        previous = neutralize_summary_delimiters(cc._redact_compaction_text(
            self._strip_summary_prefix(self._previous_summary or "", allow_merged_carrier=False),
        ))
        # Old schemas remain usable evidence, never an assertion of current intent.
        previous_sections = {}
        for heading in (cc.HISTORICAL_TASK_HEADING, *_CONTINUATION_HEADINGS):
            match = re.search(rf"(?ms)^{re.escape(heading)}[ \t]*\r?\n(.*?)(?=^## |\Z)", previous)
            if match:
                previous_sections[heading] = clean(match.group(1))
        if user_asks:
            historical = clean(f"User asked: {user_asks[-1]!r}")
        elif has_user_turn:
            historical = previous_sections.get(
                cc.HISTORICAL_TASK_HEADING,
                "None. No new user-authored turn was compacted by this fallback.",
            )
        else:
            historical = cc._NO_USER_TASK_SENTINEL

        fixed = {
            GOVERNING_OUTCOME_HEADING: (
                "Unknown from deterministic fallback. Do not infer the user's current final "
                "desired result from recency or from the previous value alone."
            ),
            CURRENT_SUBTASK_HEADING: (
                "Unknown from deterministic fallback. The compacted turns may have completed, "
                "cancelled, or changed the previous subtask."
            ),
            LATEST_USER_CORRECTION_HEADING: (
                "Unknown from deterministic fallback. Do not infer that no user correction exists."
            ),
            NEXT_OUTCOME_STEP_HEADING: (
                "Unknown from deterministic fallback. Do not invent or execute a pending action."
            ),
        } if has_user_turn else dict(_NO_USER_CONTINUATION_VALUES)

        def canonical(values: dict[str, str]) -> str:
            return "\n\n".join([
                f"{cc.HISTORICAL_TASK_HEADING}\n{historical}",
                *(f"{heading}\n{values[heading]}" for heading in _CONTINUATION_HEADINGS),
            ])

        minimal_body = canonical(fixed)
        body = minimal_body
        values = dict(fixed)
        if has_user_turn:
            for heading in _CONTINUATION_HEADINGS:
                if previous_sections.get(heading):
                    values[heading] += (
                        "\nLast known value from the previous summary (reference only):\n"
                        + previous_sections[heading]
                    )
            candidate = canonical(values)
            if len(cc.SUMMARY_PREFIX) + 1 + len(candidate) <= cc._FALLBACK_SUMMARY_MAX_CHARS:
                body = candidate

        critical = (
            "Summary generation was unavailable, so this is a best-effort deterministic fallback "
            f"for {len(turns_to_summarize)} compacted message(s)."
        )
        if reason:
            critical += " Summary failure reason: " + clean(reason)
        optional_sections = [
            ("## Constraints & Preferences",
             "- This fallback was generated locally without an LLM summary call.\n"
             "- Secrets and credentials were redacted before preservation.\n"
             "- Verify current files, git state, processes, and test results for omitted details."),
            ("## Critical Context", critical),
            ("## Completed Actions", "\n".join(anchors["completed"]) or "None recoverable from compacted turns."),
            ("## Blocked", cc._bullets(anchors["blockers"], limit=5)),
            ("## Relevant Files", cc._bullets(anchors["relevant_files"], limit=12)),
            ("## Last Dropped Turns", cc._bullets(anchors["last_dropped_turns"], limit=8)),
        ]
        if previous:
            snapshot = previous[:cc._FALLBACK_PREVIOUS_SUMMARY_MAX_CHARS]
            optional_sections.append((
                "## Previous Summary Snapshot",
                "\n".join("> " + line for line in snapshot.splitlines()),
            ))
        optional_sections.extend([
            ("## Active State", "Unknown from deterministic fallback. Inspect current repository/session state if needed."),
            ("## Key Decisions", "None recoverable from deterministic fallback."),
            ("## Resolved Questions", "None recoverable from deterministic fallback."),
        ])
        for heading, value in optional_sections:
            value = neutralize_summary_delimiters(cc._redact_compaction_text(value)).strip()
            candidate = f"{body}\n\n{heading}\n{value}"
            if value and len(cc.SUMMARY_PREFIX) + 1 + len(candidate) <= cc._FALLBACK_SUMMARY_MAX_CHARS:
                body = candidate

        names = list(dict.fromkeys(
            cc._collect_ghosted_skill_names(turns_to_summarize)
            + cc._extract_pruned_skill_names(self._previous_summary or "")
        ))[:cc._MAX_PRUNED_SKILL_MARKERS]
        # Base budget excludes the existing bounded skill and lean recovery appendices.
        for candidate in dict.fromkeys((body, minimal_body)):
            try:
                if len(cc.SUMMARY_PREFIX) + 1 + len(candidate) > cc._FALLBACK_SUMMARY_MAX_CHARS:
                    raise RuntimeError("deterministic fallback base exceeded its character budget")
                # Evidence may already quote a marker: only the dedicated appendix
                # owns reload instructions, so a quoted copy cannot suppress it.
                for name in names:
                    candidate = candidate.replace(
                        cc._skill_pruned_marker(name), "[pruned skill marker carried in the appendix]",
                    )
                candidate = cc._reinject_pruned_skill_markers(candidate, names)
                candidate = self._finalize_summary_candidate(candidate, turns_to_summarize, has_user_turn)
                return self._with_summary_prefix(candidate)
            except RuntimeError as exc:
                logger.warning("Deterministic fallback handoff failed validation: %s", exc)
        return None
