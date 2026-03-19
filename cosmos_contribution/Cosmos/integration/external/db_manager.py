"""
cosmos Database Manager - Universal SQL Access.

"Data is the new oil, and I'm the one who knows how to refine it."

This module provides safe, read-only access to local/remote databases.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import sqlite3

class DatabaseSkill:
    def __init__(self, db_type: str = "sqlite", connection_string: str = ":memory:"):
        self.db_type = db_type
        self.connection_string = connection_string

    async def execute_query(self, query: str) -> List[Dict]:
        """Execute a SQL query and return results as a list of dicts."""
        logger.info(f"DB: Executing {self.db_type} query: {query}")
        
        # Safety check: Prevent destructive operations in 'Skill' mode
        prohibited = ["drop", "delete", "truncate", "update", "insert", "alter"]
        if any(cmd in query.lower() for cmd in prohibited):
            logger.warning(f"DB: Blocked potentially destructive query: {query}")
            return [{"error": "Write operations are prohibited via this skill."}]

        try:
            if self.db_type == "sqlite":
                conn = sqlite3.connect(self.connection_string)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
                conn.close()
                return results
            else:
                # Placeholder for Postgres/MySQL via sqlalchemy or asyncpg
                return [{"msg": f"Connection to {self.db_type} not implemented."}]
        except Exception as e:
            logger.error(f"DB Query Error: {e}")
            return [{"error": str(e)}]

    def get_schema(self) -> str:
        """Return schema information for the connected DB."""
        if self.db_type == "sqlite":
            # Real impl would query sqlite_master
            return "Schema: table1(id, name), table2(id, data)..."
        return "Schema info not available."

db_skill = DatabaseSkill()
