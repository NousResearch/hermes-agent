"""SQLite database operations for Mission Control integration."""

import json
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class MissionControlDatabase:
    """Manages SQLite database for Mission Control tasks and webhook deliveries."""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._ensure_db_exists()
        
    def _ensure_db_exists(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Tasks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mc_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mc_task_id INTEGER NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    priority TEXT,
                    assigned_to TEXT,
                    created_by TEXT,
                    project_id INTEGER,
                    workspace_id INTEGER,
                    metadata TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    accepted_at INTEGER,
                    completed_at INTEGER,
                    hermes_session_id TEXT
                )
            """)
            
            # Agents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mc_agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mc_agent_id INTEGER NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    role TEXT,
                    last_seen_at INTEGER,
                    updated_at INTEGER NOT NULL
                )
            """)
            
            # Webhook delivery log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mc_webhook_deliveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    received_at INTEGER NOT NULL,
                    processed_at INTEGER,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
            """)
            
            # Indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_tasks_status ON mc_tasks(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_tasks_assigned ON mc_tasks(assigned_to)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_webhook_event_id ON mc_webhook_deliveries(event_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mc_webhook_status ON mc_webhook_deliveries(status)")
            
            conn.commit()
            
        logger.info("[mc] Database initialized at %s", self.db_path)
        
    def create_task(self, task_data: Dict[str, Any]) -> bool:
        """Create a new task from MC webhook data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = int(datetime.now().timestamp())
                conn.execute(
                    """
                    INSERT OR REPLACE INTO mc_tasks (
                        mc_task_id, title, description, status, priority,
                        assigned_to, created_by, project_id, workspace_id,
                        metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_data.get("id"),
                        task_data.get("title", "Untitled"),
                        task_data.get("description", ""),
                        task_data.get("status", "inbox"),
                        task_data.get("priority", "medium"),
                        task_data.get("assigned_to"),
                        task_data.get("created_by"),
                        task_data.get("project_id"),
                        task_data.get("workspace_id"),
                        json.dumps(task_data.get("metadata", {})),
                        task_data.get("created_at", now),
                        now
                    )
                )
                conn.commit()
                logger.info("[mc] Task %d created: %s", task_data.get("id"), task_data.get("title"))
                return True
        except Exception as e:
            logger.error("[mc] Failed to create task: %s", e)
            return False
            
    # Whitelist of allowed columns for update (prevents SQL injection)
    ALLOWED_COLUMNS = {
        "title", "description", "status", "priority", "assigned_to",
        "created_by", "project_id", "workspace_id", "metadata"
    }
    
    def update_task(self, task_id: int, updates: Dict[str, Any]) -> bool:
        """Update existing task fields."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = int(datetime.now().timestamp())
                
                # Build dynamic update with column whitelist validation
                fields = []
                values = []
                for key, value in updates.items():
                    # Validate column name against whitelist (prevents SQL injection)
                    if key not in self.ALLOWED_COLUMNS:
                        logger.warning("[mc] Ignoring invalid column in update: %s", key)
                        continue
                        
                    if key == "metadata":
                        value = json.dumps(value)
                    fields.append(f"{key} = ?")
                    values.append(value)
                    
                if not fields:
                    logger.warning("[mc] No valid fields to update for task %d", task_id)
                    return False
                    
                fields.append("updated_at = ?")
                values.append(now)
                values.append(task_id)
                
                query = f"UPDATE mc_tasks SET {', '.join(fields)} WHERE mc_task_id = ?"
                conn.execute(query, values)
                conn.commit()
                logger.info("[mc] Task %d updated", task_id)
                return True
        except Exception as e:
            logger.error("[mc] Failed to update task %d: %s", task_id, e)
            return False
            
    def accept_task(self, task_id: int) -> bool:
        """Mark task as accepted by Hermes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = int(datetime.now().timestamp())
                conn.execute(
                    "UPDATE mc_tasks SET accepted_at = ?, updated_at = ? WHERE mc_task_id = ?",
                    (now, now, task_id)
                )
                conn.commit()
                logger.info("[mc] Task %d accepted", task_id)
                return True
        except Exception as e:
            logger.error("[mc] Failed to accept task %d: %s", task_id, e)
            return False
            
    def complete_task(self, task_id: int) -> bool:
        """Mark task as completed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = int(datetime.now().timestamp())
                conn.execute(
                    "UPDATE mc_tasks SET completed_at = ?, updated_at = ? WHERE mc_task_id = ?",
                    (now, now, task_id)
                )
                conn.commit()
                logger.info("[mc] Task %d completed", task_id)
                return True
        except Exception as e:
            logger.error("[mc] Failed to complete task %d: %s", task_id, e)
            return False
            
    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get task by MC task ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM mc_tasks WHERE mc_task_id = ?",
                    (task_id,)
                ).fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error("[mc] Failed to get task %d: %s", task_id, e)
            return None
            
    def log_webhook_delivery(self, event_id: str, event_type: str, 
                             payload_hash: str, status: str = "received",
                             error_message: Optional[str] = None) -> bool:
        """Log webhook delivery for idempotency and audit."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = int(datetime.now().timestamp())
                conn.execute(
                    """
                    INSERT OR REPLACE INTO mc_webhook_deliveries
                    (event_id, event_type, payload_hash, received_at, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, event_type, payload_hash, now, status, error_message)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error("[mc] Failed to log webhook delivery: %s", e)
            return False
            
    def is_duplicate_event(self, event_id: str) -> bool:
        """Check if event was already processed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT 1 FROM mc_webhook_deliveries WHERE event_id = ?",
                    (event_id,)
                ).fetchone()
                return row is not None
        except Exception as e:
            logger.error("[mc] Failed to check duplicate: %s", e)
            return False
            
    def mark_event_processed(self, event_id: str, error: Optional[str] = None):
        """Mark event as processed (or failed)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = int(datetime.now().timestamp())
                status = "failed" if error else "processed"
                conn.execute(
                    """
                    UPDATE mc_webhook_deliveries 
                    SET processed_at = ?, status = ?, error_message = ?
                    WHERE event_id = ?
                    """,
                    (now, status, error, event_id)
                )
                conn.commit()
        except Exception as e:
            logger.error("[mc] Failed to mark event processed: %s", e)