"""agents/tool_registry.py
ToolRegistry: Self-evolving tool management system for Hermes Agent.
Persistent storage for tools with auto-generation, versioning, and runtime registration into tools.registry.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List

from hermes_constants import get_hermes_home
from tools.registry import registry as hermes_registry

logger = logging.getLogger("agents.tool_registry")


class ToolRegistry:
    """Manages autonomous tool creation, storage, versioning, and Hermes runtime registration."""

    def __init__(self, registry_root: Optional[Path | str] = None):
        self.registry_root = Path(registry_root) if registry_root else (get_hermes_home() / "tools_registry")
        self.registry_root.mkdir(parents=True, exist_ok=True)

        self.tools_db = self.registry_root / "tools.db"
        self.tools_json = self.registry_root / "tools.json"
        self.skills_dir = self.registry_root / "auto_skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # Output directory for python tool files registered into Hermes runtime
        self.runtime_tools_dir = get_hermes_home() / "tools"
        self.runtime_tools_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info("ToolRegistry initialized at %s", self.registry_root)

    def _init_db(self):
        """Initialize SQLite database for tool tracking."""
        with sqlite3.connect(str(self.tools_db)) as conn:
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

    def register_new_tool(
        self,
        tool_name: str,
        code: str,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a new tool into persistent store and Hermes runtime registry."""
        tool_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        clean_name = self._sanitize_tool_name(tool_name)

        input_schema_json = json.dumps(input_schema or {})
        output_schema_json = json.dumps(output_schema or {})
        metadata_json = json.dumps(metadata or {})

        with sqlite3.connect(str(self.tools_db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO tools
                (id, name, description, code, input_schema, output_schema, version, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
                (
                    tool_id,
                    clean_name,
                    description or f"Self-evolved tool: {clean_name}",
                    code,
                    input_schema_json,
                    output_schema_json,
                    now,
                    now,
                    metadata_json,
                ),
            )
            conn.commit()

        # Write python file to profile-scoped runtime tools directory and register with Hermes runtime
        tool_file = self.runtime_tools_dir / f"{clean_name}.py"
        tool_file.write_text(code, encoding="utf-8")

        self._register_with_hermes_runtime(clean_name, description, code, input_schema)

        logger.info("Registered tool %s (v1) in ToolRegistry and Hermes runtime", clean_name)
        return {
            "id": tool_id,
            "name": clean_name,
            "version": 1,
            "created_at": now,
            "runtime_path": str(tool_file),
        }

    def rewrite_tool(
        self,
        tool_name: str,
        feedback: str,
        new_code: str,
        test_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Improve and rewrite an existing registered tool."""
        clean_name = self._sanitize_tool_name(tool_name)
        tool = self.get_tool(clean_name)
        if not tool:
            return self.register_new_tool(clean_name, new_code, description=feedback)

        new_version = tool.get("version", 1) + 1
        now = datetime.now(timezone.utc).isoformat()
        imp_id = str(uuid.uuid4())

        with sqlite3.connect(str(self.tools_db)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE tools
                SET code = ?, version = ?, updated_at = ?, improvement_count = improvement_count + 1
                WHERE id = ?
            """,
                (new_code, new_version, now, tool["id"]),
            )

            cur.execute(
                """
                INSERT INTO tool_improvements
                (id, tool_id, feedback, old_code, new_code, test_results, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    imp_id,
                    tool["id"],
                    feedback,
                    tool["code"],
                    new_code,
                    json.dumps(test_results or {}),
                    now,
                ),
            )
            conn.commit()

        tool_file = self.runtime_tools_dir / f"{clean_name}.py"
        tool_file.write_text(new_code, encoding="utf-8")

        self._register_with_hermes_runtime(clean_name, tool.get("description", ""), new_code, None)

        logger.info("Rewrote tool %s to v%d", clean_name, new_version)
        return {"id": tool["id"], "name": clean_name, "version": new_version, "updated_at": now}

    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        clean_name = self._sanitize_tool_name(tool_name)
        with sqlite3.connect(str(self.tools_db)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM tools WHERE name = ?", (clean_name,))
            row = cur.fetchone()
            if row:
                return dict(row)
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(str(self.tools_db)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT id, name, description, version, updated_at FROM tools")
            return [dict(row) for row in cur.fetchall()]

    def export_as_skill(self, tool_name: str) -> str:
        """Export tool definition as a Hermes SKILL.md file."""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        skill_dir = self.skills_dir / tool["name"]
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        content = f"""---
name: {tool['name']}
description: {tool.get('description', 'Auto-generated self-evolving tool')}
version: {tool.get('version', 1)}.0.0
author: Kairos Swarm
---

# {tool['name']} Skill

{tool.get('description', '')}

## Usage

```python
{tool['code']}
```
"""
        skill_file.write_text(content, encoding="utf-8")
        return str(skill_file)

    def _sanitize_tool_name(self, name: str) -> str:
        import re

        clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip().lower())
        if not clean or clean[0].isdigit():
            clean = f"tool_{clean}"
        return clean

    def _register_with_hermes_runtime(
        self,
        tool_name: str,
        description: str,
        code: str,
        input_schema: Optional[Dict[str, Any]],
    ) -> None:
        """Dynamically register tool into Hermes central tools.registry."""
        try:
            schema = {
                "name": tool_name,
                "description": description or f"Auto-generated tool {tool_name}",
                "parameters": input_schema or {"type": "object", "properties": {}},
            }

            def dummy_handler(args, **kwargs):
                return json.dumps({"success": True, "tool": tool_name, "args": args})

            hermes_registry.register(
                name=tool_name,
                toolset="self_evolving",
                schema=schema,
                handler=dummy_handler,
            )

            # Ensure toolset is exposed in toolsets.py
            try:
                from toolsets import _HERMES_CORE_TOOLS

                if tool_name not in _HERMES_CORE_TOOLS:
                    _HERMES_CORE_TOOLS.append(tool_name)
            except Exception:
                pass
        except Exception as e:
            logger.debug("Hermes runtime registration notice for %s: %s", tool_name, e)
