from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


LIST_FIELDS = {"allowed_actions", "approval_gates"}


@dataclass(frozen=True)
class StandingOrder:
    name: str
    scope: str = ""
    trigger: str = ""
    allowed_actions: list[str] = field(default_factory=list)
    approval_gates: list[str] = field(default_factory=list)
    escalation_rules: str = ""
    output_policy: str = ""


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _coerce_order(name: str, fields: dict[str, str]) -> StandingOrder:
    return StandingOrder(
        name=name.strip(),
        scope=fields.get("scope", ""),
        trigger=fields.get("trigger", ""),
        allowed_actions=_split_list(fields.get("allowed_actions", "")),
        approval_gates=_split_list(fields.get("approval_gates", "")),
        escalation_rules=fields.get("escalation_rules", ""),
        output_policy=fields.get("output_policy", ""),
    )


def parse_standing_orders(markdown: str) -> list[StandingOrder]:
    """Parse simple markdown standing-order sections.

    MVP format:
      ## Order name
      - scope: ...
      - trigger: ...
    """
    orders: list[StandingOrder] = []
    current_name: str | None = None
    current_fields: dict[str, str] = {}

    def flush() -> None:
        nonlocal current_name, current_fields
        if current_name:
            orders.append(_coerce_order(current_name, current_fields))
        current_name = None
        current_fields = {}

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            flush()
            current_name = line[3:].strip()
            continue
        if not current_name or not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        current_fields[key.strip()] = value.strip()
    flush()
    return orders


def load_standing_orders(path: str | Path | None, fallbacks: Iterable[str | Path] = ()) -> list[StandingOrder]:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.extend(Path(p) for p in fallbacks)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return parse_standing_orders(candidate.read_text(encoding="utf-8"))
    return []

