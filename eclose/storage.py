"""
Eclose Storage - SQLite-based persistent storage for proposals and evolution history.
"""

import sqlite3
import json
import os
from pathlib import Path
from dataclasses import asdict
from typing import Optional

from eclose.evolution.proposal import EvolutionProposal
from eclose.evolution.execution import ExecutionResult


class EcloseStorage:
    """SQLite-based persistent storage for Eclose data."""

    def __init__(self, db_path: str = None):
        """Initialize storage with database path."""
        if db_path is None:
            # Default to ~/.hermes/eclose.db
            home = Path.home()
            hermes_dir = home / ".hermes"
            hermes_dir.mkdir(exist_ok=True)
            db_path = str(hermes_dir / "eclose.db")

        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Proposals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                background TEXT,
                solution TEXT,
                expected_impact TEXT,
                risks TEXT,
                metadata TEXT,
                gap_type TEXT,
                gap_severity TEXT,
                gap_description TEXT,
                gap_evidence TEXT,
                requires_approval INTEGER,
                status TEXT DEFAULT 'pending',
                created_at REAL,
                updated_at REAL
            )
        """)

        # Evolution history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id TEXT,
                status TEXT,
                results TEXT,
                verification TEXT,
                executed_at REAL,
                FOREIGN KEY (proposal_id) REFERENCES proposals(id)
            )
        """)

        conn.commit()
        conn.close()

    def save_proposal(self, proposal: EvolutionProposal) -> bool:
        """Save a proposal to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO proposals (
                    id, title, background, solution, expected_impact,
                    risks, metadata, gap_type, gap_severity,
                    gap_description, gap_evidence, requires_approval,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                proposal.id,
                proposal.title,
                json.dumps(proposal.background),
                json.dumps(proposal.solution),
                json.dumps(proposal.expected_impact),
                json.dumps(proposal.risks),
                json.dumps(proposal.metadata),
                proposal.gap.gap_type.value if proposal.gap else None,
                proposal.gap.severity.value if proposal.gap else None,
                proposal.gap.description if proposal.gap else None,
                json.dumps(proposal.gap.evidence) if proposal.gap else None,
                int(proposal.requires_approval),
                'pending',
                proposal.gap.timestamp if proposal.gap else None,
                None,
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving proposal: {e}")
            return False
        finally:
            conn.close()

    def get_proposal(self, proposal_id: str) -> Optional[EvolutionProposal]:
        """Get a proposal by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # Reconstruct proposal (simplified)
        # In a full implementation, you'd reconstruct the full object
        return row

    def get_pending_proposals(self) -> list:
        """Get all pending proposals."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM proposals WHERE status = 'pending' ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_proposal_status(self, proposal_id: str, status: str) -> bool:
        """Update proposal status (approved, rejected, etc.)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "UPDATE proposals SET status = ?, updated_at = ? WHERE id = ?",
                (status, None, proposal_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"Error updating proposal: {e}")
            return False
        finally:
            conn.close()

    def save_execution_result(self, result: ExecutionResult) -> bool:
        """Save an execution result to history."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO evolution_history (
                    proposal_id, status, results, verification, executed_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                result.proposal_id,
                result.status,
                json.dumps(result.results),
                json.dumps(result.verification),
                None,
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving execution result: {e}")
            return False
        finally:
            conn.close()

    def get_evolution_history(self, limit: int = 100) -> list:
        """Get evolution history."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM evolution_history ORDER BY executed_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_statistics(self) -> dict:
        """Get storage statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM proposals WHERE status = 'pending'")
        pending_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM proposals WHERE status = 'approved'")
        approved_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM proposals WHERE status = 'rejected'")
        rejected_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM evolution_history")
        history_count = cursor.fetchone()["count"]

        conn.close()

        return {
            "pending_proposals": pending_count,
            "approved_proposals": approved_count,
            "rejected_proposals": rejected_count,
            "total_executions": history_count,
        }
