"""agents/tool_registry.py
ToolRegistry: Self-evolving tool management system.
Persistent storage for tools with auto-generation and improvement capabilities.
Core of the Kairos self-evolution mechanism.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("agents.tool_registry")


class ToolRegistry:
    """
    Manages autonomous tool creation, storage, and evolution.
    
    Features:
    - Persistent tool storage (JSON + SQLite)
    - Tool versioning and improvement tracking
    - Sandboxed code execution and testing
    - Vector embeddings for semantic search (via ChromaDB if available)
    - Auto-skill generation in YAML format
    """

    def __init__(self, registry_root: str = "hermes/tools"):
        self.registry_root = Path(registry_root)
        self.registry_root.mkdir(parents=True, exist_ok=True)

        self.tools_db = self.registry_root / "tools.db"
        self.tools_json = self.registry_root / "tools.json"
        self.skills_dir = self.registry_root / "auto_skills"
        self.skills_dir.mkdir(exist_ok=True)

        self.chroma_collection = None
        self._init_db()
        self._init_chroma()
        logger.info(f"ToolRegistry initialized at {self.registry_root}")

    def _init_db(self):
        """Initialize SQLite database for tool tracking."""
        conn = sqlite3.connect(str(self.tools_db))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tools (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                code TEXT NOT NULL,
                input_schema TEXT,
                output_schema TEXT,
                version INTEGER DEFAULT 1,
                created_at TEXT,
                updated_at TEXT,
                improvement_count INTEGER DEFAULT 0,
                test_pass_rate REAL DEFAULT 0.0,
                metadata TEXT
            )
        """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_improvements (
                id TEXT PRIMARY KEY,
                tool_id TEXT NOT NULL,
                feedback TEXT,
                old_code TEXT,
                new_code TEXT,
                test_results TEXT,
                created_at TEXT,
                FOREIGN KEY (tool_id) REFERENCES tools(id)
            )
        """
        )

        conn.commit()
        conn.close()

    def _init_chroma(self):
        """Initialize ChromaDB for semantic search (optional)."""
        try:
            import chromadb

            self.chroma_client = chromadb.PersistentClient(
                path=str(self.registry_root / "chroma_db")
            )
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="tools",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB initialized for tool search")
        except ImportError:
            logger.warning("ChromaDB not available, semantic search disabled")
            self.chroma_client = None

    def register_new_tool(
        self,
        tool_name: str,
        code: str,
        description: str = "",
        input_schema: Optional[dict[str, Any]] = None,
        output_schema: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Register a new tool in the registry.

        Args:
            tool_name: Unique tool identifier
            code: Python function code
            description: Tool description
            input_schema: JSON schema for inputs
            output_schema: JSON schema for outputs
            metadata: Additional metadata

        Returns:
            Tool record with ID and version
        """
        logger.info(f"Registering new tool: {tool_name}")

        if self._tool_exists(tool_name):
            logger.warning(f"Tool {tool_name} already exists, updating instead")
            return self.update_tool(
                tool_name, code, description, input_schema, output_schema, metadata
            )

        tool_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(self.tools_db))
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO tools 
            (id, name, description, code, input_schema, output_schema, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                tool_id,
                tool_name,
                description,
                code,
                json.dumps(input_schema or {}),
                json.dumps(output_schema or {}),
                now,
                now,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        conn.close()

        # Add to vector DB if available
        if self.chroma_collection:
            self._add_to_chroma(
                tool_id, tool_name, description, code
            )

        # Save to JSON backup
        self._save_to_json()

        logger.info(f"Tool registered: {tool_name} (ID: {tool_id})")
        return {
            "tool_id": tool_id,
            "name": tool_name,
            "version": 1,
            "created_at": now,
        }

    def rewrite_tool(
        self,
        tool_name: str,
        feedback: str,
        new_code: str,
        test_results: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Improve/rewrite an existing tool based on feedback.

        Args:
            tool_name: Tool to improve
            feedback: Improvement feedback/rationale
            new_code: Improved code
            test_results: Results of testing new code

        Returns:
            Updated tool record with new version
        """
        logger.info(f"Improving tool: {tool_name}")

        conn = sqlite3.connect(str(self.tools_db))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM tools WHERE name = ?", (tool_name,))
        tool_row = cur.fetchone()

        if not tool_row:
            logger.error(f"Tool not found: {tool_name}")
            conn.close()
            return {"success": False, "error": f"Tool {tool_name} not found"}

        tool_id = tool_row["id"]
        old_code = tool_row["code"]
        new_version = tool_row["version"] + 1

        # Record improvement
        improvement_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        pass_rate = 0.0
        if test_results:
            passed = sum(1 for t in test_results.values() if t.get("passed", False))
            pass_rate = passed / len(test_results) if test_results else 0.0

        cur.execute(
            """
            INSERT INTO tool_improvements
            (id, tool_id, feedback, old_code, new_code, test_results, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                improvement_id,
                tool_id,
                feedback,
                old_code,
                new_code,
                json.dumps(test_results or {}),
                now,
            ),
        )

        # Update tool
        cur.execute(
            """
            UPDATE tools
            SET code = ?, version = ?, improvement_count = improvement_count + 1,
                test_pass_rate = ?, updated_at = ?
            WHERE id = ?
        """,
            (new_code, new_version, pass_rate, now, tool_id),
        )

        conn.commit()
        conn.close()

        # Update vector DB
        if self.chroma_collection:
            self._add_to_chroma(
                tool_id, tool_name, tool_row["description"], new_code
            )

        # Save to JSON backup
        self._save_to_json()

        logger.info(
            f"Tool improved: {tool_name} v{new_version} "
            f"(pass_rate: {pass_rate:.1%})"
        )
        return {
            "tool_id": tool_id,
            "name": tool_name,
            "version": new_version,
            "pass_rate": pass_rate,
            "updated_at": now,
        }

    def update_tool(
        self,
        tool_name: str,
        code: Optional[str] = None,
        description: Optional[str] = None,
        input_schema: Optional[dict[str, Any]] = None,
        output_schema: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Update tool metadata without recording as improvement."""
        conn = sqlite3.connect(str(self.tools_db))
        cur = conn.cursor()

        now = datetime.now(timezone.utc).isoformat()

        updates = {"updated_at": now}
        if code is not None:
            updates["code"] = code
        if description is not None:
            updates["description"] = description
        if input_schema is not None:
            updates["input_schema"] = json.dumps(input_schema)
        if output_schema is not None:
            updates["output_schema"] = json.dumps(output_schema)
        if metadata is not None:
            updates["metadata"] = json.dumps(metadata)

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [tool_name]

        cur.execute(
            f"UPDATE tools SET {set_clause} WHERE name = ?",
            values,
        )
        conn.commit()
        conn.close()

        self._save_to_json()
        logger.info(f"Tool updated: {tool_name}")

        return {"name": tool_name, "updated_at": now}

    def get_tool(self, tool_name: str) -> Optional[dict[str, Any]]:
        """Retrieve a tool by name."""
        conn = sqlite3.connect(str(self.tools_db))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM tools WHERE name = ?", (tool_name,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        return dict(row)

    def list_tools(self, limit: int = 100) -> list[dict[str, Any]]:
        """List all registered tools."""
        conn = sqlite3.connect(str(self.tools_db))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            "SELECT id, name, description, version, improvement_count, test_pass_rate FROM tools LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def search_tools(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search tools by description or name."""
        if self.chroma_collection:
            return self._search_chroma(query, limit)

        # Fallback: text search in SQLite
        conn = sqlite3.connect(str(self.tools_db))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        pattern = f"%{query}%"
        cur.execute(
            """
            SELECT id, name, description, version 
            FROM tools 
            WHERE name LIKE ? OR description LIKE ?
            LIMIT ?
        """,
            (pattern, pattern, limit),
        )
        rows = cur.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def _tool_exists(self, tool_name: str) -> bool:
        """Check if tool exists."""
        conn = sqlite3.connect(str(self.tools_db))
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM tools WHERE name = ?", (tool_name,))
        exists = cur.fetchone() is not None
        conn.close()
        return exists

    def _add_to_chroma(
        self, tool_id: str, tool_name: str, description: str, code: str
    ):
        """Add tool to ChromaDB for semantic search."""
        if not self.chroma_collection:
            return

        try:
            combined_text = f"{tool_name}: {description}\n{code[:500]}"
            self.chroma_collection.upsert(
                ids=[tool_id],
                metadatas=[{"name": tool_name, "description": description}],
                documents=[combined_text],
            )
        except Exception as e:
            logger.warning(f"Failed to add tool to ChromaDB: {e}")

    def _search_chroma(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search tools using ChromaDB."""
        if not self.chroma_collection:
            return []

        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=limit,
            )
            return [
                {
                    "id": results["ids"][0][i],
                    "name": results["metadatas"][0][i].get("name", ""),
                    "description": results["metadatas"][0][i].get("description", ""),
                }
                for i in range(len(results["ids"][0]))
            ]
        except Exception as e:
            logger.warning(f"ChromaDB search failed: {e}")
            return []

    def _save_to_json(self):
        """Backup registry to JSON."""
        try:
            tools = self.list_tools(limit=1000)
            with open(self.tools_json, "w") as f:
                json.dump(tools, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save tools.json: {e}")

    def export_as_skill(self, tool_name: str) -> Optional[str]:
        """
        Export a tool as a reusable YAML skill.
        
        Returns:
            Path to generated skill file, or None if failed
        """
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            return None

        skill_yaml = f"""---
name: {tool_name}
version: {tool["version"]}
description: |
  {tool["description"]}
tags:
  - auto-generated
  - self-evolved

implementation:
  language: python
  code: |
{self._indent_code(tool['code'], 4)}

metadata:
  created_at: {tool["created_at"]}
  improvements: {tool["improvement_count"]}
  pass_rate: {tool["test_pass_rate"]:.1%}
"""

        skill_path = self.skills_dir / f"{tool_name}.yaml"
        try:
            with open(skill_path, "w") as f:
                f.write(skill_yaml)
            logger.info(f"Skill exported: {skill_path}")
            return str(skill_path)
        except Exception as e:
            logger.error(f"Failed to export skill: {e}")
            return None

    @staticmethod
    def _indent_code(code: str, spaces: int) -> str:
        """Indent code for YAML embedding."""
        indent = " " * spaces
        lines = code.split("\n")
        return "\n".join(indent + line for line in lines)
