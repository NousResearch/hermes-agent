"""
SOUL.md Parser — Borge Extension

Reads the Borge frontmatter block from SOUL.md and constructs a
ValueSystem + EmotionalDefaults from it.

Compatible with standard Hermes SOUL.md (no frontmatter → uses defaults).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

from .value_system import EmotionalDefaults, Value, ValueSystem

log = logging.getLogger(__name__)


def parse_soul_frontmatter(soul_path: str) -> ValueSystem:
    path = Path(soul_path)
    if not path.exists():
        log.debug(f"[SoulParser] {soul_path} not found — using defaults")
        return ValueSystem()

    content = path.read_text(encoding="utf-8")
    fm = _extract_frontmatter(content)
    if not fm:
        log.debug("[SoulParser] No frontmatter found — using defaults")
        return ValueSystem()

    return _build_value_system(fm)


def _extract_frontmatter(content: str) -> Optional[dict]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as e:
        log.warning(f"[SoulParser] YAML parse error: {e}")
        return None


def _build_value_system(fm: dict) -> ValueSystem:
    ed_cfg = fm.get("emotional_defaults", {})
    emotional_defaults = EmotionalDefaults(
        valence_baseline=float(ed_cfg.get("valence_baseline", 0.0)),
        arousal_baseline=float(ed_cfg.get("arousal_baseline", 0.5)),
        tau_valence=float(ed_cfg.get("tau_valence", 5.0)),
        tau_arousal=float(ed_cfg.get("tau_arousal", 2.0)),
        frustrated_threshold=float(ed_cfg.get("frustrated_threshold", -0.4)),
        excited_threshold=float(ed_cfg.get("excited_threshold", 0.7)),
    )

    values_cfg = fm.get("values", {})
    primary_raw = values_cfg.get("primary", [])
    primary_values = [
        Value(
            id=v.get("id", f"value_{i}"),
            description=v.get("description", ""),
            weight=float(v.get("weight", 0.7)),
        )
        for i, v in enumerate(primary_raw)
        if isinstance(v, dict)
    ]
    if not primary_values:
        primary_values = ValueSystem._defaults()

    hard_constraints = values_cfg.get("constraints", [])
    aesthetic = values_cfg.get("aesthetic", {})

    return ValueSystem(
        primary_values=primary_values,
        hard_constraints=hard_constraints,
        aesthetic_preferences=aesthetic,
        emotional_defaults=emotional_defaults,
    )
