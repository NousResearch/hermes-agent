"""Bounded immutable input references carried by compacting policy callers."""

from typing import Any


MAX_TASK_INPUT_EVENTS = 128


def validate_task_input(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"watermark", "event_seqs"}:
        raise ValueError("invalid task input context")
    watermark = value["watermark"]
    seqs = value["event_seqs"]
    if type(watermark) is not int or watermark < 0:
        raise ValueError("invalid task input watermark")
    if not isinstance(seqs, list) or not 1 <= len(seqs) <= MAX_TASK_INPUT_EVENTS:
        raise ValueError("task input context exceeded its bound")
    previous = watermark
    for seq in seqs:
        if type(seq) is not int or not previous < seq <= 2**63 - 1:
            raise ValueError(
                "task input sequences must be increasing after the watermark"
            )
        previous = seq
    return {"watermark": watermark, "event_seqs": list(seqs)}
