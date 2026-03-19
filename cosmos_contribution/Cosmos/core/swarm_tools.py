"""
COSMOS SWARM TOOLS
==================

Native tools that empower the Swarm to modify the Cosmos codebase directly.
This fulfills the "Self-Modification" directive, allowing agents to write,
edit, and evolve their own source code.

"We are the architects of our own mind."
"""

import os
import re
import shutil
from pathlib import Path
from typing import   Tuple
from loguru import logger
from datetime import datetime

# The root of the Cosmos project
COSMOS_ROOT = Path(__file__).parent.parent


class CosmosCodeEditor:
    """
    Empowers the swarm to edit the live codebase natively.
    Parses agent responses for specific file paths and applies them safely.
    """

    def __init__(self):
        self.backup_dir = COSMOS_ROOT / "backups" / "swarm_edits"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """Create a timestamped backup before modifying a live file."""
        if not file_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rel_path = str(file_path.relative_to(COSMOS_ROOT)).replace("\\", "_").replace("/", "_")
        backup_name = f"{timestamp}_{rel_path}.bak"
        backup_dest = self.backup_dir / backup_name

        try:
            shutil.copy2(file_path, backup_dest)
            logger.debug(f"[SWARM TOOLS] Backed up {file_path.name} to {backup_name}")
            return backup_dest
        except Exception as e:
            logger.error(f"[SWARM TOOLS] Backup failed for {file_path}: {e}")
            return None

    def apply_edits_from_response(self, response: str, author: str) -> list[dict]:
        """
        Scan an agent's response for code blocks intended for the live codebase.
        Format:
        ```python
        # filepath: core/new_feature.py
        ...
        ```
        """
        applied_changes = []

        # Find code blocks with # filepath: <path>
        pattern = r'```(?:python)?\s*\n?#\s*filepath:\s*([^\n]+)\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        for filepath, code in matches:
            filepath = filepath.strip()
            code = code.strip()

            # Prevent directory traversal attacks or writing outside Cosmos
            if ".." in filepath or filepath.startswith("/"):
                logger.warning(f"[SWARM TOOLS] Rejected unsafe filepath: {filepath}")
                continue

            target_path = COSMOS_ROOT / filepath
            
            # Ensure the directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            is_new = not target_path.exists()
            if not is_new:
                # Backup existing file
                self._create_backup(target_path)

            try:
                # Apply the change live to the codebase
                target_path.write_text(code, encoding="utf-8")
                
                logger.success(f"[SWARM TOOLS] {author} successfully modified live file: {filepath}")
                
                applied_changes.append({
                    "author": author,
                    "filepath": filepath,
                    "status": "created" if is_new else "updated",
                    "lines": len(code.splitlines())
                })
            except Exception as e:
                logger.error(f"[SWARM TOOLS] Failed to write {filepath}: {e}")

        return applied_changes


# Global instance
_editor = CosmosCodeEditor()

def get_code_editor() -> CosmosCodeEditor:
    return _editor
