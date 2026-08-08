"""
Self Evolution Plugin — Evolution Executor
============================================

Executes approved evolution proposals with rollback support.

Design reference: Claude Code plugins/ralph-wiggum/
  - Self-referential feedback loop: execute → verify → rollback if needed
  - Each change has a "completion promise" (verification criteria)
  - Iteration > Perfection
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from self_evolution import db
from self_evolution.models import Proposal, ImprovementUnit

logger = logging.getLogger(__name__)

from self_evolution.paths import DATA_DIR as STRATEGIES_DIR, STRATEGIES_FILE, ARCHIVE_DIR
from self_evolution.paths import SKILLS_DIR, MEMORIES_DIR


class EvolutionExecutor:
    """Execute approved evolution proposals.

    Supported proposal types:
      - skill: create a new skill via skill_manager_tool
      - strategy: update strategy rules
      - memory: update PERFORMANCE.md via memory_tool
      - tool_preference: update tool preference config
    """

    def execute(self, proposal: Proposal):
        """Execute an approved proposal."""
        logger.info("Executing proposal: %s (%s)", proposal.id, proposal.proposal_type)

        try:
            match proposal.proposal_type:
                case "skill":
                    self._create_skill(proposal)
                case "strategy":
                    self._update_strategy(proposal)
                case "memory":
                    self._update_memory(proposal)
                case "tool_preference":
                    self._update_tool_preference(proposal)
                case "code_improvement":
                    self._save_optimization_request(proposal)

            # Mark as executed
            db.update(
                "evolution_proposals",
                {"status": "executed", "resolved_at": time.time()},
                where="id = ?",
                where_params=(proposal.id,),
            )

            # Create improvement tracking unit
            self._create_tracking_unit(proposal)

            logger.info("Proposal %s executed successfully", proposal.id)

        except Exception as exc:
            logger.exception("Failed to execute proposal %s: %s", proposal.id, exc)
            db.update(
                "evolution_proposals",
                {"status": "execution_failed", "resolved_at": time.time()},
                where="id = ?",
                where_params=(proposal.id,),
            )

    def check_and_rollback(self):
        """Check active improvement units and rollback if needed.

        Called during dream consolidation to verify previous changes.
        """
        units = db.fetch_all("improvement_units", where="status = 'active'")

        for unit_data in units:
            unit = ImprovementUnit(
                id=unit_data["id"],
                proposal_id=unit_data["proposal_id"],
                change_type=unit_data["change_type"],
                version=unit_data.get("version", 0),
                baseline_score=unit_data.get("baseline_score", 0),
                current_score=unit_data.get("current_score", 0),
                sessions_sampled=unit_data.get("sessions_sampled", 0),
                min_sessions=unit_data.get("min_sessions", 10),
                min_improvement=unit_data.get("min_improvement", 0.05),
                max_regression=unit_data.get("max_regression", 0.10),
            )

            # Update current score from recent sessions
            self._update_unit_score(unit)

            if unit.should_revert:
                self._revert(unit)
                logger.warning("Rolled back improvement unit %s", unit.id)
            elif unit.should_promote:
                self._promote(unit)
                logger.info("Promoted improvement unit %s", unit.id)

    # ── Proposal Type Handlers ────────────────────────────────────────────

    def _create_skill(self, proposal: Proposal):
        """Create a new skill via the skill_manager_tool."""
        from self_evolution.strategy_store import StrategyStore

        store = StrategyStore()
        skill_dir = SKILLS_DIR / proposal.id
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_content = (
            f"---\n"
            f"name: {proposal.id}\n"
            f"description: {proposal.title}\n"
            f"---\n\n"
            f"{proposal.description}\n"
        )
        (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")
        logger.info("Created learned skill: %s", skill_dir)

    def _update_strategy(self, proposal: Proposal):
        """Update strategy rules file with version tracking."""
        from self_evolution.strategy_store import StrategyStore

        store = StrategyStore()
        current = store.load()

        # Check for duplicate strategies by title similarity
        rules = current.get("rules", [])
        existing_titles = {r.get("name", "").strip().lower() for r in rules}
        if proposal.title.strip().lower() in existing_titles:
            logger.warning("Skipping duplicate strategy: %s", proposal.title)
            return

        # Archive current version
        version = current.get("version", 0) + 1
        store.archive(version - 1)

        # Parse new strategy from proposal description
        new_strategy = {
            "id": proposal.id,
            "name": proposal.title,
            "type": "learned",
            "description": proposal.description,
            "hint_text": proposal.description,
            "conditions": [],
            "severity": "medium",
            "created_at": time.time(),
        }

        # Add to strategies
        rules.append(new_strategy)
        current["rules"] = rules
        current["version"] = version

        store.save(current)
        logger.info("Updated strategies to version %d", version)

        # Invalidate injector cache so new strategy takes effect immediately
        from self_evolution.strategy_injector import invalidate_cache
        invalidate_cache()

    def _update_memory(self, proposal: Proposal):
        """Update PERFORMANCE.md via the memory system."""
        perf_path = MEMORIES_DIR / "PERFORMANCE.md"
        perf_path.parent.mkdir(parents=True, exist_ok=True)

        existing = ""
        if perf_path.exists():
            existing = perf_path.read_text(encoding="utf-8")

        # Append new entry
        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        entry = f"\n## [{timestamp}] 自动学习\n{proposal.description}\n"

        # Keep file under reasonable size (last 50 entries)
        entries = (existing + entry).split("\n## ")
        if len(entries) > 50:
            entries = entries[-50:]

        perf_path.write_text("\n## ".join(entries), encoding="utf-8")
        logger.info("Updated PERFORMANCE.md")

    def _update_tool_preference(self, proposal: Proposal):
        """Update tool preference config."""
        prefs_path = STRATEGIES_DIR / "tool_preferences.json"
        prefs = {}
        if prefs_path.exists():
            prefs = json.loads(prefs_path.read_text(encoding="utf-8"))

        prefs[proposal.id] = {
            "description": proposal.description,
            "expected_impact": proposal.expected_impact,
            "created_at": time.time(),
        }

        prefs_path.write_text(
            json.dumps(prefs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Updated tool preferences: %s", proposal.id)

    # ── Tracking & Verification ───────────────────────────────────────────

    def _create_tracking_unit(self, proposal: Proposal):
        """Create an improvement tracking unit after execution.

        Inspired by Ralph Wiggum's completion_promise pattern.
        """
        # Get baseline score from recent sessions
        recent = db.fetch_all(
            "session_scores",
            order_by="created_at DESC",
            limit=10,
        )
        baseline = (
            sum(s.get("composite_score", 0) for s in recent) / len(recent)
            if recent else 0
        )

        unit = ImprovementUnit(
            id=f"unit-{uuid.uuid4().hex[:8]}",
            proposal_id=proposal.id,
            change_type=proposal.proposal_type,
            baseline_score=baseline,
            min_sessions=10,
            min_improvement=0.05,
            max_regression=0.10,
        )

        db.insert("improvement_units", unit.to_db_row())
        logger.info("Created tracking unit: %s (baseline=%.3f)", unit.id, baseline)

    def _update_unit_score(self, unit: ImprovementUnit):
        """Update the current score for an improvement unit."""
        # Count sessions since this unit was created
        unit_data = db.fetch_one("improvement_units", where="id = ?", params=(unit.id,))
        if not unit_data:
            return

        created_at = unit_data.get("created_at", 0)
        recent = db.fetch_all(
            "session_scores",
            where="created_at >= ?",
            params=(created_at,),
            order_by="created_at DESC",
        )

        if recent:
            current_score = sum(s.get("composite_score", 0) for s in recent) / len(recent)
            sessions_sampled = len(recent)

            db.update(
                "improvement_units",
                {
                    "current_score": current_score,
                    "sessions_sampled": sessions_sampled,
                },
                where="id = ?",
                where_params=(unit.id,),
            )
            unit.current_score = current_score
            unit.sessions_sampled = sessions_sampled

    def _revert(self, unit: ImprovementUnit):
        """Revert a change by restoring the previous version."""
        from self_evolution.strategy_store import StrategyStore

        store = StrategyStore()
        if unit.version > 0:
            old = store.load_archive(unit.version - 1)
            if old:
                store.save(old)

        db.update(
            "improvement_units",
            {"status": "reverted", "resolved_at": time.time()},
            where="id = ?",
            where_params=(unit.id,),
        )

    def _promote(self, unit: ImprovementUnit):
        """Promote an improvement unit from active to permanent."""
        db.update(
            "improvement_units",
            {"status": "promoted", "resolved_at": time.time()},
            where="id = ?",
            where_params=(unit.id,),
        )

    # ── Code Improvement (save request document) ────────────────────────────

    def _save_optimization_request(self, proposal: Proposal):
        """Save a code improvement request as a document.

        Does NOT auto-modify code. The user reviews the request and decides
        whether to implement changes manually or via Claude Code.
        """
        req_dir = DATA_DIR / "optimization_requests"
        req_dir.mkdir(parents=True, exist_ok=True)
        doc_path = req_dir / f"{proposal.id}.md"

        doc_content = (
            f"# 程序优化需求\n\n"
            f"**标题**: {proposal.title}\n"
            f"**预期影响**: {proposal.expected_impact}\n"
            f"**风险评估**: {proposal.risk_assessment}\n"
            f"**回滚方案**: {proposal.rollback_plan}\n"
            f"**创建时间**: {time.strftime('%Y-%m-%d %H:%M', time.localtime())}\n\n"
            f"---\n\n"
            f"{proposal.description}\n"
        )

        doc_path.write_text(doc_content, encoding="utf-8")
        logger.info("Saved optimization request: %s", doc_path)
