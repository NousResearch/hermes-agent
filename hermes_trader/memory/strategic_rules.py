"""Load distilled strategic rules separate from raw chat history."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml

from hermes_trader.config import TRADER_HOME_SUBDIR

BUNDLED_RULES_PATH = Path(__file__).resolve().parent / "strategic_rules.yaml"


@dataclass
class StrategicRules:
    version: int = 1
    updated_at: Optional[str] = None
    regime_name: str = "neutral"
    regime_notes: str = ""
    positive_heuristics: List[dict[str, str]] = field(default_factory=list)
    negative_constraints: List[dict[str, str]] = field(default_factory=list)

    def to_context_snippets(self) -> List[dict[str, Any]]:
        snippets: list[dict[str, Any]] = []
        snippets.append(
            {
                "kind": "strategic_regime",
                "trust": "advisory",
                "regime": self.regime_name,
                "notes": self.regime_notes,
            }
        )
        for item in self.positive_heuristics:
            snippets.append(
                {
                    "kind": "positive_heuristic",
                    "trust": "advisory",
                    "id": item.get("id", ""),
                    "rule": item.get("rule", ""),
                }
            )
        for item in self.negative_constraints:
            snippets.append(
                {
                    "kind": "negative_constraint",
                    "trust": "advisory",
                    "id": item.get("id", ""),
                    "rule": item.get("rule", ""),
                }
            )
        return snippets


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def default_strategic_rules_path() -> Path:
    env_path = os.environ.get("HERMES_TRADER_STRATEGIC_RULES", "").strip()
    if env_path:
        return Path(env_path)
    return _hermes_home() / TRADER_HOME_SUBDIR / "strategic_rules.yaml"


def save_strategic_rules(rules: StrategicRules, path: Optional[Path | str] = None) -> Path:
    target = Path(path) if path is not None else default_strategic_rules_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": rules.version,
        "updated_at": rules.updated_at,
        "regime": {
            "name": rules.regime_name,
            "notes": rules.regime_notes,
        },
        "positive_heuristics": rules.positive_heuristics,
        "negative_constraints": rules.negative_constraints,
    }
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return target


def load_strategic_rules(path: Optional[Path | str] = None) -> StrategicRules:
    target = Path(path) if path is not None else default_strategic_rules_path()
    if not target.is_file():
        target = BUNDLED_RULES_PATH
    with open(target, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{target}: strategic rules root must be a mapping")
    regime = data.get("regime") or {}
    return StrategicRules(
        version=int(data.get("version", 1)),
        updated_at=data.get("updated_at"),
        regime_name=str(regime.get("name", "neutral")),
        regime_notes=str(regime.get("notes", "")),
        positive_heuristics=list(data.get("positive_heuristics") or []),
        negative_constraints=list(data.get("negative_constraints") or []),
    )