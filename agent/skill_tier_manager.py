"""
Skill Tier Manager — usage-based skill lifecycle management.

Automatically classifies skills into three tiers to optimize context token usage:

    BUILTIN  — core skills, always injected into system prompt
    FREQUENT — high-usage skills (§10), auto-injected by relevance
    ARCHIVED — low-usage skills, zero token cost, loaded on demand

Promotion rules:
    archived → frequent:  ≥3 uses in the last 7 days
    frequent → archived:  7 consecutive days unused
    cold archive:         30 days unused (advisory)

Metadata stored in ~/.hermes/skill_tiers.json.

Tier limits, builtin skill names, and the metadata path are all configurable
so downstream forks (e.g. hermes-agent-cn) can set their own defaults.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# -- Default constants --------------------------------------------------------

DEFAULT_META_PATH = Path.home() / ".hermes" / "skill_tiers.json"

# Promotion/Demotion thresholds
PROMOTION_THRESHOLD = 3          # uses in the rolling 7-day window to promote
DEMOTION_DAYS = 7                # consecutive idle days before demotion
COLD_ARCHIVE_DAYS = 30           # consecutive idle days before cold-archive hint
MAX_FREQUENT = 10                # max skills in the FREQUENT tier
MAX_BUILTIN = 8                  # max skills in the BUILTIN tier


# -- Data model ---------------------------------------------------------------

class SkillTier(Enum):
    BUILTIN = "builtin"
    FREQUENT = "frequent"
    ARCHIVED = "archived"


@dataclass
class SkillMeta:
    """Per-skill metadata persisted to disk."""

    name: str
    tier: str = SkillTier.ARCHIVED.value
    usage_count: int = 0
    last_used: str = ""                         # ISO-8601
    weekly_usage: List[int] = field(default_factory=lambda: [0, 0, 0, 0])  # last 4 weeks
    promoted_at: str = ""
    archived_at: str = ""
    pinned: bool = False
    description: str = ""

    # -- Internal helpers -----------------------------------------------------

    def _pad_weekly(self):
        while len(self.weekly_usage) < 4:
            self.weekly_usage.insert(0, 0)
        self.weekly_usage = self.weekly_usage[-4:]

    # -- Public API -----------------------------------------------------------

    def record_use(self):
        """Record one usage event."""
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()
        self._pad_weekly()
        self.weekly_usage[-1] += 1

    def should_promote(self) -> bool:
        """Check promotion condition: ≥3 uses in the current weekly window."""
        if self.tier != SkillTier.ARCHIVED.value:
            return False
        self._pad_weekly()
        return self.weekly_usage[-1] >= PROMOTION_THRESHOLD

    def should_demote(self) -> bool:
        """Check demotion condition: 7 idle days as FREQUENT."""
        if self.tier != SkillTier.FREQUENT.value or self.pinned:
            return False
        if not self.last_used:
            return False
        try:
            last = datetime.fromisoformat(self.last_used)
            return (datetime.now() - last).days >= DEMOTION_DAYS
        except (ValueError, TypeError):
            return False

    def should_cold_archive(self) -> bool:
        """Check cold-archive condition: 30 idle days."""
        if not self.last_used:
            return False
        try:
            last = datetime.fromisoformat(self.last_used)
            return (datetime.now() - last).days >= COLD_ARCHIVE_DAYS
        except (ValueError, TypeError):
            return False


@dataclass
class SkillsMetaStore:
    """Global container for all skill metadata."""

    skills: Dict[str, SkillMeta] = field(default_factory=dict)
    tier_limits: Dict[str, int] = field(default_factory=lambda: {
        SkillTier.BUILTIN.value: MAX_BUILTIN,
        SkillTier.FREQUENT.value: MAX_FREQUENT,
    })
    last_evaluation: str = ""


# -- Manager ------------------------------------------------------------------

class SkillTierManager:
    """Manages skill tier classification, usage tracking, and lifecycle.

    Args:
        meta_path: Path to the JSON metadata file.
        builtin_skills: Names of skills considered always-active (builtin).
        tier_limits: Optional per-tier capacity overrides.
    """

    def __init__(
        self,
        meta_path: Optional[Path] = None,
        builtin_skills: Optional[List[str]] = None,
        tier_limits: Optional[Dict[str, int]] = None,
    ):
        self.meta_path = meta_path or DEFAULT_META_PATH
        self._builtin_skills = builtin_skills or []
        self._tier_limits_override = tier_limits or {}
        self._store: SkillsMetaStore = SkillsMetaStore()
        self._loaded = False

    # -- Persistence ----------------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return
        self.load()
        self._loaded = True

    def load(self) -> SkillsMetaStore:
        """Load metadata from disk (or initialise defaults)."""
        try:
            if self.meta_path.exists():
                raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
                skills = {}
                for name, data in raw.get("skills", {}).items():
                    skills[name] = SkillMeta(
                        name=name,
                        tier=data.get("tier", SkillTier.ARCHIVED.value),
                        usage_count=data.get("usage_count", 0),
                        last_used=data.get("last_used", ""),
                        weekly_usage=data.get("weekly_usage", [0, 0, 0, 0]),
                        promoted_at=data.get("promoted_at", ""),
                        archived_at=data.get("archived_at", ""),
                        pinned=data.get("pinned", False),
                        description=data.get("description", ""),
                    )
                self._store = SkillsMetaStore(
                    skills=skills,
                    tier_limits=raw.get("tier_limits", {
                        SkillTier.BUILTIN.value: MAX_BUILTIN,
                        SkillTier.FREQUENT.value: MAX_FREQUENT,
                    }),
                    last_evaluation=raw.get("last_evaluation", ""),
                )
            else:
                self._init_defaults()
        except Exception as e:
            logger.warning("Failed to load %s: %s — using defaults", self.meta_path, e)
            self._init_defaults()

        self._loaded = True

        # Override tier limits if provided via constructor
        if self._tier_limits_override:
            self._store.tier_limits.update(self._tier_limits_override)

        return self._store

    def _init_defaults(self):
        """Seed metadata for configured builtin skills."""
        self._store = SkillsMetaStore()
        now = datetime.now().isoformat()
        for name in self._builtin_skills:
            self._store.skills[name] = SkillMeta(
                name=name,
                tier=SkillTier.BUILTIN.value,
                usage_count=0,
                last_used="",
                weekly_usage=[0, 0, 0, 0],
                description="",
            )

    def save(self):
        """Persist metadata to disk."""
        self._ensure_loaded()
        try:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "skills": {
                    name: asdict(meta)
                    for name, meta in self._store.skills.items()
                },
                "tier_limits": self._store.tier_limits,
                "last_evaluation": self._store.last_evaluation,
            }
            self.meta_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to save %s: %s", self.meta_path, e)

    # -- Queries --------------------------------------------------------------

    def get_skill(self, name: str) -> Optional[SkillMeta]:
        self._ensure_loaded()
        return self._store.skills.get(name)

    def get_tier(self, name: str) -> SkillTier:
        meta = self.get_skill(name)
        return SkillTier(meta.tier) if meta else SkillTier.ARCHIVED

    def get_skills_by_tier(self, tier: SkillTier) -> List[str]:
        """Return skill names in the given tier, sorted by usage (desc)."""
        self._ensure_loaded()
        pairs = [
            (n, m.usage_count)
            for n, m in self._store.skills.items()
            if m.tier == tier.value
        ]
        pairs.sort(key=lambda x: -x[1])
        return [n for n, _ in pairs]

    def get_active_skills(self) -> List[str]:
        """Return skills that should be injected into the system prompt.

        Builtin skills are always included; up to MAX_FREQUENT Frequent skills
        (by usage) are appended.
        """
        builtin = self.get_skills_by_tier(SkillTier.BUILTIN)
        frequent = self.get_skills_by_tier(SkillTier.FREQUENT)
        limit = self._store.tier_limits.get(SkillTier.FREQUENT.value, MAX_FREQUENT)
        return builtin + frequent[:limit]

    def get_archived_skills(self) -> List[str]:
        """Return all archived skill names (zero token cost)."""
        return self.get_skills_by_tier(SkillTier.ARCHIVED)

    # -- Operations -----------------------------------------------------------

    def record_usage(self, skill_name: str):
        """Record one usage for *skill_name* and check real-time promotion."""
        self._ensure_loaded()
        skill_name = skill_name.strip().lower()

        if skill_name not in self._store.skills:
            self._store.skills[skill_name] = SkillMeta(
                name=skill_name,
                tier=SkillTier.ARCHIVED.value,
            )

        meta = self._store.skills[skill_name]
        meta.record_use()

        if meta.should_promote():
            self._promote(skill_name)

        self.save()

    def _promote(self, skill_name: str):
        """Promote *skill_name* from archived to frequent."""
        meta = self._store.skills.get(skill_name)
        if not meta:
            return

        frequent = self.get_skills_by_tier(SkillTier.FREQUENT)
        limit = self._store.tier_limits.get(SkillTier.FREQUENT.value, MAX_FREQUENT)
        if len(frequent) >= limit:
            # Evict the least-used unpinned frequent skill
            evict_candidates = [
                (n, self._store.skills[n].usage_count)
                for n in frequent
                if not self._store.skills[n].pinned
            ]
            if evict_candidates:
                evict_candidates.sort(key=lambda x: x[1])
                self._demote(evict_candidates[0][0])

        meta.tier = SkillTier.FREQUENT.value
        meta.promoted_at = datetime.now().isoformat()
        logger.info("Skill promoted: %s → frequent", skill_name)

    def _demote(self, skill_name: str):
        """Demote *skill_name* from frequent to archived."""
        meta = self._store.skills.get(skill_name)
        if not meta or meta.pinned:
            return
        meta.tier = SkillTier.ARCHIVED.value
        meta.archived_at = datetime.now().isoformat()
        meta.weekly_usage = [0, 0, 0, 0]
        logger.info("Skill demoted: %s → archived", skill_name)

    def promote(self, skill_name: str):
        """Manually promote *skill_name* to frequent."""
        self._ensure_loaded()
        self._promote(skill_name)
        self.save()

    def demote(self, skill_name: str):
        """Manually demote *skill_name* to archived."""
        self._ensure_loaded()
        self._demote(skill_name)
        self.save()

    def pin_skill(self, skill_name: str):
        """Protect a skill from auto-demotion."""
        meta = self.get_skill(skill_name)
        if meta:
            meta.pinned = True
            self.save()

    def unpin_skill(self, skill_name: str):
        """Remove pin protection."""
        meta = self.get_skill(skill_name)
        if meta:
            meta.pinned = False
            self.save()

    def set_builtin(self, skill_name: str):
        """Manually promote *skill_name* to builtin tier."""
        if skill_name not in self._store.skills:
            return
        builtin_count = len(self.get_skills_by_tier(SkillTier.BUILTIN))
        limit = self._store.tier_limits.get(SkillTier.BUILTIN.value, MAX_BUILTIN)
        if builtin_count >= limit:
            logger.warning("Builtin tier full (%d), cannot add %s", limit, skill_name)
            return
        self._store.skills[skill_name].tier = SkillTier.BUILTIN.value
        self.save()

    def set_builtin_skills(self, skills: List[str]):
        """Set the list of builtin skills (used by prompt builder on init)."""
        self._ensure_loaded()
        # Ensure each listed skill exists in metadata as builtin
        for name in skills:
            if name not in self._store.skills:
                self._store.skills[name] = SkillMeta(
                    name=name,
                    tier=SkillTier.BUILTIN.value,
                )
            else:
                self._store.skills[name].tier = SkillTier.BUILTIN.value
        self.save()

    # -- Batch evaluation -----------------------------------------------------

    def evaluate_promotions(self) -> Dict[str, List[str]]:
        """Run batch promotion/demotion evaluation (designed for daily cron).

        Returns a dict with keys ``promoted``, ``demoted``, ``cold_archive``.
        """
        self._ensure_loaded()

        promoted = []
        demoted = []
        cold = []

        for name, meta in list(self._store.skills.items()):
            if meta.should_promote():
                self._promote(name)
                promoted.append(name)

            if meta.should_demote():
                self._demote(name)
                demoted.append(name)

            if meta.should_cold_archive():
                cold.append(name)

        if promoted or demoted:
            self._store.last_evaluation = datetime.now().isoformat()
            self.save()

        logger.info(
            "Batch evaluation: %d promoted, %d demoted, %d cold-archive suggested",
            len(promoted), len(demoted), len(cold),
        )
        return {"promoted": promoted, "demoted": demoted, "cold_archive": cold}

    # -- Statistics -----------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics."""
        self._ensure_loaded()
        tiers = {"builtin": 0, "frequent": 0, "archived": 0}
        total_usage = 0
        for meta in self._store.skills.values():
            tiers[meta.tier] = tiers.get(meta.tier, 0) + 1
            total_usage += meta.usage_count

        active_count = (
            tiers["builtin"]
            + min(tiers["frequent"], self._store.tier_limits.get("frequent", MAX_FREQUENT))
        )
        total_count = sum(tiers.values())
        token_saved_pct = round((1 - active_count / max(total_count, 1)) * 100)

        return {
            "total_skills": total_count,
            "by_tier": tiers,
            "total_usage": total_usage,
            "active_skills": active_count,
            "token_saved_pct": token_saved_pct,
            "last_evaluation": self._store.last_evaluation,
        }

    def print_stats(self) -> str:
        """Return a human-readable English stats report."""
        stats = self.get_stats()
        lines = [
            "=== Skill Tier Statistics ===",
            f"Total: {stats['total_skills']}",
            f"  Builtin:    {stats['by_tier']['builtin']} (always loaded)",
            f"  Frequent:   {stats['by_tier']['frequent']} (auto-matched)",
            f"  Archived:   {stats['by_tier']['archived']} (on-demand)",
            f"Active: {stats['active_skills']}/{stats['total_skills']}",
            f"Total uses: {stats['total_usage']}",
            f"Token saved: ~{stats['token_saved_pct']}%",
        ]
        if stats["last_evaluation"]:
            lines.append(f"Last evaluation: {stats['last_evaluation']}")
        return "\n".join(lines)


# -- Global convenience accessor ---------------------------------------------

_mgr_instance: Optional[SkillTierManager] = None


def get_skill_manager() -> SkillTierManager:
    """Return the global SkillTierManager singleton."""
    global _mgr_instance
    if _mgr_instance is None:
        _mgr_instance = SkillTierManager()
    return _mgr_instance


def record_skill_usage(skill_name: str):
    """Convenience: record one usage via the global manager."""
    get_skill_manager().record_usage(skill_name)
