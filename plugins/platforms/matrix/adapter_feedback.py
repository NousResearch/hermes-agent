"""Policy for automatic Matrix read receipts and processing reactions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gateway.config import PlatformConfig
from gateway.platforms._shared import extra_or_secret
from gateway.platforms.event import ProcessingOutcome


class ReadReceiptMode(StrEnum):
    IMMEDIATE = "immediate"
    AFTER_PROCESSING = "after_processing"
    DISABLED = "disabled"

    @classmethod
    def from_value(cls, value: object) -> ReadReceiptMode:
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.IMMEDIATE

    def should_send_on_completion(self, outcome: ProcessingOutcome) -> bool:
        return (
            self is ReadReceiptMode.AFTER_PROCESSING
            and outcome != ProcessingOutcome.CANCELLED
        )


@dataclass(frozen=True)
class MatrixFeedbackPolicy:
    read_receipts: ReadReceiptMode
    reactions: bool

    @classmethod
    def from_config(cls, config: PlatformConfig) -> MatrixFeedbackPolicy:
        reactions = extra_or_secret(config.extra, "reactions", "MATRIX_REACTIONS", True)
        return cls(
            read_receipts=ReadReceiptMode.from_value(config.extra.get("read_receipts")),
            reactions=str(reactions).strip().lower() not in {"false", "0", "no", "off"},
        )
