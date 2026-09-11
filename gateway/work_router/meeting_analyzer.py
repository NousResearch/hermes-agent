from __future__ import annotations

import json
from typing import Mapping

from .meeting_strategy import RoundAnalysis


class RoundAnalysisParseError(ValueError):
    pass


_REQUIRED_FIELDS = (
    "key_conflicts",
    "uncertainties",
    "evidence_gaps",
    "challenged_assumptions",
    "promising_alternatives",
    "participants_needed_next",
    "next_round_focus",
    "convergence_recommended",
)


class MeetingRoundAnalyzer:
    def __init__(self, *, participants: tuple[str, ...]) -> None:
        if not participants:
            raise ValueError("meeting participants are required")
        if len(set(participants)) != len(participants):
            raise ValueError("meeting participants must be unique")
        self.participants = participants

    def build_prompt(
        self,
        *,
        objective: str,
        round_id: int,
        contributions: Mapping[str, str],
    ) -> str:
        if round_id < 1:
            raise ValueError("round_id must be >= 1")

        lines = [
            "You are Demian, the PM and meeting facilitator.",
            f"Meeting objective: {objective}",
            f"Round: {round_id}",
            "",
            "Analyze the discussion process, not a predetermined answer.",
            "Do not predetermine the conclusion.",
            "Preserve meaningful minority opinions.",
            "Identify conflicts, uncertainties, unsupported evidence, "
            "challenged assumptions, and promising alternatives.",
            "Allow participants to reframe the problem when justified.",
            "Select only the participants actually needed for the next round.",
            "Recommend convergence only when material conflicts, "
            "uncertainties, and evidence gaps are resolved.",
            "",
            "Contributions:",
        ]

        for profile in self.participants:
            statement = str(contributions.get(profile, "")).strip()
            if statement:
                lines.append(f"- {profile}: {statement}")

        lines.extend(
            [
                "",
                "Return JSON only with exactly these fields:",
                "{",
                "  \"key_conflicts\": [],",
                "  \"uncertainties\": [],",
                "  \"evidence_gaps\": [],",
                "  \"challenged_assumptions\": [],",
                "  \"promising_alternatives\": [],",
                "  \"participants_needed_next\": [],",
                "  \"next_round_focus\": \"\",",
                "  \"convergence_recommended\": false",
                "}",
            ]
        )

        return "\n".join(lines)

    def parse(self, response_text: str) -> RoundAnalysis:
        try:
            raw = json.loads(response_text.strip())
        except (json.JSONDecodeError, TypeError) as exc:
            raise RoundAnalysisParseError(
                "meeting analysis must be valid JSON"
            ) from exc

        if not isinstance(raw, dict):
            raise RoundAnalysisParseError(
                "meeting analysis must be a JSON object"
            )

        missing = tuple(
            field for field in _REQUIRED_FIELDS if field not in raw
        )
        if missing:
            raise RoundAnalysisParseError(
                "missing meeting analysis field(s): " + ", ".join(missing)
            )

        key_conflicts = self._string_tuple(
            raw["key_conflicts"], "key_conflicts"
        )
        uncertainties = self._string_tuple(
            raw["uncertainties"], "uncertainties"
        )
        evidence_gaps = self._string_tuple(
            raw["evidence_gaps"], "evidence_gaps"
        )
        challenged_assumptions = self._string_tuple(
            raw["challenged_assumptions"],
            "challenged_assumptions",
        )
        promising_alternatives = self._string_tuple(
            raw["promising_alternatives"],
            "promising_alternatives",
        )
        participants_needed_next = self._string_tuple(
            raw["participants_needed_next"],
            "participants_needed_next",
        )

        unknown = tuple(
            profile
            for profile in participants_needed_next
            if profile not in self.participants
        )
        if unknown:
            raise RoundAnalysisParseError(
                "unknown meeting participant(s): " + ", ".join(unknown)
            )

        focus = raw["next_round_focus"]
        if not isinstance(focus, str):
            raise RoundAnalysisParseError(
                "next_round_focus must be text"
            )
        focus = " ".join(focus.split())

        convergence = raw["convergence_recommended"]
        if not isinstance(convergence, bool):
            raise RoundAnalysisParseError(
                "convergence_recommended must be boolean"
            )

        if key_conflicts or uncertainties or evidence_gaps:
            convergence = False

        return RoundAnalysis(
            key_conflicts=key_conflicts,
            uncertainties=uncertainties,
            evidence_gaps=evidence_gaps,
            challenged_assumptions=challenged_assumptions,
            promising_alternatives=promising_alternatives,
            participants_needed_next=participants_needed_next,
            next_round_focus=focus,
            convergence_recommended=convergence,
        )

    @staticmethod
    def _string_tuple(
        value: object,
        field: str,
    ) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise RoundAnalysisParseError(
                f"{field} must be a list"
            )

        items = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise RoundAnalysisParseError(
                    f"{field} must contain non-empty text"
                )
            items.append(" ".join(item.split()))

        if len(set(items)) != len(items):
            raise RoundAnalysisParseError(
                f"{field} must not contain duplicates"
            )

        return tuple(items)
