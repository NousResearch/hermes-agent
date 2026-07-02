"""Compact phase handoff records for budget-sensitive agent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable


_DATA_IMAGE_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=_-]+")
_MAX_FIELD_CHARS = 1500


def _clean_text(value: Any, *, max_chars: int = _MAX_FIELD_CHARS) -> str:
    text = str(value or "")
    text = _DATA_IMAGE_RE.sub("[image data omitted; use artifact path]", text)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [checkpoint field truncated]"
    return text


def _clean_list(values: Iterable[Any], *, max_item_chars: int = _MAX_FIELD_CHARS) -> list[str]:
    return [_clean_text(value, max_chars=max_item_chars) for value in values or []]


@dataclass(frozen=True)
class PhaseCheckpoint:
    """Small, explicit handoff between repair, verification, and UI phases.

    It intentionally stores conclusions and artifact paths, not raw logs,
    screenshots, or tool dumps. The JSON output is safe to feed into a fresh
    short verification phase without dragging earlier context along.
    """

    objective: str
    changes: list[str] = field(default_factory=list)
    verification_passed: list[str] = field(default_factory=list)
    next_phase_inputs: list[str] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)
    stop_reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "objective": _clean_text(self.objective),
            "changes": _clean_list(self.changes),
            "verification_passed": _clean_list(self.verification_passed),
            "next_phase_inputs": _clean_list(self.next_phase_inputs),
            "artifact_paths": _clean_list(self.artifact_paths, max_item_chars=500),
            "stop_reason": _clean_text(self.stop_reason, max_chars=500),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, indent=2)


def build_phase_checkpoint(**kwargs: Any) -> PhaseCheckpoint:
    return PhaseCheckpoint(**kwargs)

