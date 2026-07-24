#!/usr/bin/env python3
"""
Phase 2: Decay Automation

Runs hourly to:
  1. Recalculate temporal decay weights for all facts
  2. Update freshness tiers
  3. Archive facts >90 days old
  4. Log changes to temporal_decay_log
"""

import json
import sqlite3
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass, asdict

DB_PATH = Path.home() / ".hermes/memory-engine/db/memory.db"

# Decay parameters
DECAY_LAMBDA = 0.05  # e^(-0.05 * days)
RECENT_DAYS = 7
MEDIUM_DAYS = 30
OLD_DAYS = 90

FRESHNESS_TIERS = {
    "recent": {"max_days": RECENT_DAYS, "boost": 2.0},
    "medium": {"max_days": MEDIUM_DAYS, "boost": 1.0},
    "old": {"max_days": OLD_DAYS, "boost": 0.5},
    "archive": {"max_days": float("inf"), "boost": 0.1},
}


@dataclass
class DecayUpdate:
    """Change to a fact's decay weight."""
    fact_id: str
    old_weight: float
    new_weight: float
    old_tier: str
    new_tier: str
    days_old: int
    archived: bool = False


class DecayScheduler:
    """Manages temporal decay recalculation and archiving."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()

    @staticmethod
    def calculate_decay_weight(days_old: int) -> float:
        """Calculate exponential decay weight."""
        return math.exp(-DECAY_LAMBDA * days_old)

    @staticmethod
    def get_freshness_tier(days_old: int) -> str:
        """Get freshness tier based on days old."""
        if days_old <= RECENT_DAYS:
            return "recent"
        elif days_old <= MEDIUM_DAYS:
            return "medium"
        elif days_old <= OLD_DAYS:
            return "old"
        else:
            return "archive"

    @staticmethod
    def get_freshness_boost(tier: str) -> float:
        """Get boost multiplier for freshness tier."""
        return FRESHNESS_TIERS.get(tier, {}).get("boost", 0.1)

    def run_decay_recalculation(self) -> Dict:
        """
        Recalculate decay weights for all active facts.
        
        Returns:
            Stats dict with changes
        """
        now = datetime.now()
        updates = []
        archived_count = 0

        rows = self.conn.execute("""
            SELECT id, created_at, decay_weight, freshness_tier, status
            FROM quantum_facts
            WHERE status NOT IN ('abandoned')
        """).fetchall()

        for row in rows:
            fact_id = row["id"]
            created_at = datetime.fromisoformat(row["created_at"])
            old_weight = row["decay_weight"] or 1.0
            old_tier = row["freshness_tier"] or "recent"

            # Calculate new metrics
            days_old = (now - created_at).days
            new_weight = self.calculate_decay_weight(days_old)
            new_tier = self.get_freshness_tier(days_old)
            boost = self.get_freshness_boost(new_tier)
            final_weight = min(1.0, new_weight * boost)

            # Check if should archive
            should_archive = days_old >= OLD_DAYS and old_tier != "archived"

            # Update fact
            new_status = "archived" if should_archive else row["status"]
            self.conn.execute(
                """
                UPDATE quantum_facts
                SET decay_weight = ?,
                    freshness_tier = ?,
                    status = ?
                WHERE id = ?
                """,
                (final_weight, new_tier, new_status, fact_id),
            )

            # Log change
            if old_weight != final_weight or old_tier != new_tier:
                update = DecayUpdate(
                    fact_id=fact_id,
                    old_weight=old_weight,
                    new_weight=final_weight,
                    old_tier=old_tier,
                    new_tier=new_tier,
                    days_old=days_old,
                    archived=should_archive,
                )
                updates.append(update)

                # Log to temporal_decay_log
                self.conn.execute(
                    """
                    INSERT INTO temporal_decay_log
                      (fact_id, old_weight, new_weight, days_since_created, decay_reason)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        fact_id,
                        old_weight,
                        final_weight,
                        days_old,
                        f"scheduled_recalc|tier:{old_tier}→{new_tier}",
                    ),
                )

                if should_archive:
                    archived_count += 1

        self.conn.commit()

        return {
            "total_facts": len(rows),
            "updated": len(updates),
            "archived": archived_count,
            "timestamp": now.isoformat(),
            "updates": [asdict(u) for u in updates],
        }

    def get_archival_candidates(self, days_threshold: int = OLD_DAYS) -> List[Dict]:
        """
        Get facts that are candidates for archiving.
        
        Args:
            days_threshold: Age in days to consider for archival
        
        Returns:
            List of fact dicts
        """
        cutoff_date = (
            datetime.now() - timedelta(days=days_threshold)
        ).isoformat()

        rows = self.conn.execute(
            """
            SELECT id, summary, created_at, decay_weight, activation_count
            FROM quantum_facts
            WHERE status NOT IN ('abandoned', 'archived')
            AND created_at < ?
            ORDER BY created_at ASC
            LIMIT 100
            """,
            (cutoff_date,),
        ).fetchall()

        return [dict(r) for r in rows]

    def archive_facts(self, fact_ids: List[str]) -> Dict:
        """
        Archive a list of facts.
        
        Args:
            fact_ids: List of fact IDs to archive
        
        Returns:
            Stats dict
        """
        archived = 0

        for fact_id in fact_ids:
            self.conn.execute(
                """
                UPDATE quantum_facts
                SET status = 'archived'
                WHERE id = ?
                """,
                (fact_id,),
            )

            self.conn.execute(
                """
                INSERT INTO temporal_decay_log
                  (fact_id, decay_reason)
                VALUES (?, 'manual_archival')
                """,
                (fact_id,),
            )

            archived += 1

        self.conn.commit()

        return {
            "archived": archived,
            "timestamp": datetime.now().isoformat(),
        }

    def get_decay_status(self) -> Dict:
        """Get current decay and archival statistics."""
        freshness = self.conn.execute(
            """
            SELECT freshness_tier, COUNT(*) as cnt
            FROM quantum_facts
            WHERE status NOT IN ('abandoned')
            GROUP BY freshness_tier
            """
        ).fetchall()

        status_dist = self.conn.execute(
            """
            SELECT status, COUNT(*) as cnt
            FROM quantum_facts
            GROUP BY status
            """
        ).fetchall()

        weight_dist = self.conn.execute(
            """
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN decay_weight > 0.7 THEN 1 ELSE 0 END) as high,
              SUM(CASE WHEN decay_weight BETWEEN 0.3 AND 0.7 THEN 1 ELSE 0 END) as medium,
              SUM(CASE WHEN decay_weight < 0.3 THEN 1 ELSE 0 END) as low
            FROM quantum_facts
            WHERE status NOT IN ('abandoned')
            """
        ).fetchone()

        last_run = self.conn.execute(
            "SELECT MAX(calculated_at) as last_run FROM temporal_decay_log"
        ).fetchone()

        return {
            "freshness_distribution": dict(freshness) or {},
            "status_distribution": dict(status_dist) or {},
            "weight_distribution": {
                "total": weight_dist["total"] or 0,
                "high": weight_dist["high"] or 0,
                "medium": weight_dist["medium"] or 0,
                "low": weight_dist["low"] or 0,
            },
            "last_recalculation": last_run["last_run"] or "never",
        }

    def setup_cron(self) -> str:
        """Return cron job line for hourly decay recalculation."""
        return "0 * * * * python3 ~/.hermes/memory-engine/scripts/decay_scheduler.py run"


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Decay Scheduler — Phase 2")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Run decay recalculation (hourly)")
    sub.add_parser("status", help="Decay status")
    sub.add_parser("candidates", help="Show archival candidates")
    
    arch = sub.add_parser("archive", help="Archive specific facts")
    arch.add_argument("fact_ids", nargs="+")
    
    sub.add_parser("cron", help="Show cron setup command")

    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scheduler = DecayScheduler()
    scheduler.connect()

    try:
        if args.command == "run":
            result = scheduler.run_decay_recalculation()
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"\n  DECAY RECALCULATION")
                print(f"  {'═' * 50}\n")
                print(f"  Total facts:  {result['total_facts']}")
                print(f"  Updated:      {result['updated']}")
                print(f"  Archived:     {result['archived']}\n")

        elif args.command == "status":
            status = scheduler.get_decay_status()
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"\n  DECAY STATUS")
                print(f"  {'═' * 50}\n")
                print(f"  Freshness Distribution:")
                for tier, cnt in status["freshness_distribution"].items():
                    print(f"    {tier:12s} {cnt}")
                print(f"\n  Status Distribution:")
                for s, cnt in status["status_distribution"].items():
                    print(f"    {s:12s} {cnt}")
                print(f"\n  Weight Distribution:")
                wd = status["weight_distribution"]
                print(f"    High (>0.7)   {wd['high']}")
                print(f"    Medium        {wd['medium']}")
                print(f"    Low (<0.3)    {wd['low']}")
                print(f"\n  Last run: {status['last_recalculation']}\n")

        elif args.command == "candidates":
            candidates = scheduler.get_archival_candidates()
            if args.json:
                print(json.dumps(candidates, indent=2, default=str))
            else:
                print(f"\n  ARCHIVAL CANDIDATES (>90 days)")
                print(f"  {'═' * 50}\n")
                for c in candidates[:20]:
                    age = (datetime.now() - datetime.fromisoformat(c["created_at"])).days
                    print(
                        f"  {c['id'][:12]:12s} {age:3d}d  {(c['summary'] or '')[:40]}"
                    )
                print(f"\n  {len(candidates)} candidates\n")

        elif args.command == "archive":
            result = scheduler.archive_facts(args.fact_ids)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\n  ARCHIVED {result['archived']} facts\n")

        elif args.command == "cron":
            print(f"\n  Add this to crontab (crontab -e):\n")
            print(f"  {scheduler.setup_cron()}\n")

        else:
            parser.print_help()

    finally:
        scheduler.close()


if __name__ == "__main__":
    main()
