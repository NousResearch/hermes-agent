"""
Skill Evolution Engine

Applies evolutionary dynamics to the skill library:
  - Fitness scoring: F = success_rate × usage_weight × recency × f_reduction
  - Selection: surface low-fitness candidates for pruning
  - Variation: suggest high-fitness skills for generalisation
  - Heredity: successful patterns propagate via skill creation

Fitness is tracked in SQLite (skill_fitness table).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS skill_fitness (
    skill_name        TEXT PRIMARY KEY,
    invocation_count  INTEGER DEFAULT 0,
    success_count     INTEGER DEFAULT 0,
    avg_f_reduction   REAL DEFAULT 0.0,
    last_used         TEXT,
    fitness           REAL DEFAULT 0.5
);
"""


@dataclass
class SkillFitnessRecord:
    skill_name: str
    invocation_count: int = 0
    success_count: int = 0
    avg_f_reduction: float = 0.0
    last_used: Optional[str] = None
    fitness: float = 0.5

    @property
    def success_rate(self) -> float:
        if self.invocation_count == 0:
            return 0.5
        return self.success_count / self.invocation_count

    def compute_fitness(self) -> float:
        if self.last_used is None:
            recency = 0.3
        else:
            try:
                days = (datetime.now() - datetime.fromisoformat(self.last_used)).days
            except ValueError:
                days = 30
            recency = math.exp(-0.05 * days)
        usage_weight = math.log(1.0 + self.invocation_count)
        f_bonus = 1.0 + max(0.0, self.avg_f_reduction)
        self.fitness = round(self.success_rate * usage_weight * recency * f_bonus, 4)
        return self.fitness


class SkillEvolutionEngine:
    PRUNE_THRESHOLD      = 0.15
    GENERALIZE_THRESHOLD = 0.75
    MIN_INVOCATIONS      = 3

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    def record_invocation(
        self,
        skill_name: str,
        success: bool,
        f_reduction: float = 0.0,
    ) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM skill_fitness WHERE skill_name = ?",
                (skill_name,),
            ).fetchone()
            if row is None:
                rec = SkillFitnessRecord(skill_name=skill_name)
            else:
                rec = SkillFitnessRecord(**dict(row))
            rec.invocation_count += 1
            if success:
                rec.success_count += 1
            rec.last_used = datetime.now().isoformat()
            rec.avg_f_reduction = 0.9 * rec.avg_f_reduction + 0.1 * f_reduction
            rec.compute_fitness()
            conn.execute(
                """INSERT INTO skill_fitness
                   (skill_name, invocation_count, success_count, avg_f_reduction,
                    last_used, fitness)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(skill_name) DO UPDATE SET
                     invocation_count=excluded.invocation_count,
                     success_count=excluded.success_count,
                     avg_f_reduction=excluded.avg_f_reduction,
                     last_used=excluded.last_used,
                     fitness=excluded.fitness""",
                (
                    rec.skill_name, rec.invocation_count, rec.success_count,
                    rec.avg_f_reduction, rec.last_used, rec.fitness,
                ),
            )

    def prune_candidates(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT skill_name FROM skill_fitness
                   WHERE fitness < ? AND invocation_count >= ?
                   ORDER BY fitness ASC""",
                (self.PRUNE_THRESHOLD, self.MIN_INVOCATIONS),
            ).fetchall()
        return [r["skill_name"] for r in rows]

    def generalise_candidates(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT skill_name FROM skill_fitness
                   WHERE fitness > ? ORDER BY fitness DESC""",
                (self.GENERALIZE_THRESHOLD,),
            ).fetchall()
        return [r["skill_name"] for r in rows]

    def all_fitness(self) -> list[SkillFitnessRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM skill_fitness ORDER BY fitness DESC"
            ).fetchall()
        return [SkillFitnessRecord(**dict(r)) for r in rows]

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
