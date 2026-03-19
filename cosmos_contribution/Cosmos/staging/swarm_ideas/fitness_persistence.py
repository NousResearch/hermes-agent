"""
Fitness storage module for persisting fitness scores.

This module provides an async interface for saving and retrieving fitness scores using SQLite.
"""

import asyncio
import json
from typing import Dict, List, Optional
from loguru import logger
import sqlite3
from datetime import datetime

from Cosmos.core.config import get_config

class FitnessStorage:
    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize with default path: data/fitness_scores.db"""
        self.db_path = db_path or "data/fitness_scores.db"
        self._inited = False

    async def initialize(self) -> None:
        """Create SQLite schema with indices. Idempotent."""
        if self._inited:
            return
            
        await self._execute_sql(
            """
            CREATE TABLE IF NOT EXISTS fitness_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                score REAL NOT NULL,
                metric_type TEXT NOT NULL,
                task_id TEXT,
                context TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._inited = True
        logger.info("Fitness storage initialized")

    async def save_score(
        self,
        agent_id: str,
        score: float,
        metric_type: str = "general_fitness",
        task_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> int:
        """Persist score.

        Args:
            agent_id: The unique identifier for the agent.
            score: The fitness score to persist.
            metric_type: The type of metric (default: "general_fitness")
            task_id: Optional task identifier for which the score was generated.
            context: Optional context dictionary (default: None)
            timestamp: Optional timestamp (default: current time)

        Returns:
            The integer ID of the saved record.

        Raises:
            ValueError: If score is not within [0, 1]
            sqlite3.Error: If database operation fails
        """
        if score < 0 or score > 1:
            raise ValueError("Score must be between 0 and 1")
            
        if timestamp is None:
            timestamp = datetime.now()
            
        context_str = None
        if context is not None:
            context_str = json.dumps(context)
            
        params = (
            agent_id,
            score,
            metric_type,
            task_id,
            context_str,
            timestamp.isoformat() if timestamp else None
        )
        
        try:
            result = await self._execute_sql(
                """
                INSERT INTO fitness_scores 
                (agent_id, score, metric_type, task_id, context, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                params,
                fetchone=True
            )
            return result
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise

    async def _execute_sql(self, sql: str, params: tuple = (), fetchone: bool = False, fetchall: bool = False) -> Any:
        """Execute a SQL command and return the result.

        This method runs the SQL in a separate thread to avoid blocking the event loop.

        Args:
            sql: The SQL statement to execute.
            params: Parameters to pass to the SQL statement.
            fetchone: Whether to fetch one result (default: False).
            fetchall: Whether to fetch all results (default: False).

        Returns:
            The fetched results, or the rowcount for an INSERT/UPDATE/DELETE.

        Raises:
            Exception: If the SQL execution fails.
        """
        return await asyncio.to_thread(self._blocking_execute, sql, params, fetchone, fetchall)
        
    def _blocking_execute(self, sql: str, params: tuple = (), fetchone: bool = False, fetchall: bool = False) -> Any:
        """Blocking implementation of execute_sql to be run in a thread."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            
            if fetchone:
                result = cursor.fetchone()
            elif fetchall:
                result = cursor.fetchall()
            else:
                result = cursor.rowcount
                
            conn.commit()
            conn.close()
            return result
        except sqlite3.Error as e:
            logger.error(f"SQL execution failed: {e}")
            conn.rollback() if conn else None
            conn.close() if conn else None
            raise

    async def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, '_conn'):
            self._conn.close()
        logger.info("Fitness storage closed")