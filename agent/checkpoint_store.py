"""Checkpoint store -- read/write YAML checkpoint files for session resumption.

Each checkpoint captures enough state to resume a task from where the agent
left off after context compaction or session restart. Files live under
~/.hermes/checkpoints/<session_id>.yaml.
"""

import logging
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS_DIR = Path.home() / ".hermes" / "checkpoints"
DEFAULT_GC_MAX_AGE_DAYS = 7


class CheckpointStore:
    """Manages checkpoint YAML files on disk."""

    def __init__(self, checkpoints_dir: Path = None):
        self._dir = Path(checkpoints_dir) if checkpoints_dir else DEFAULT_CHECKPOINTS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.yaml"

    def write(self, session_id: str, data: Dict[str, Any]) -> None:
        """Write a checkpoint for the given session. Overwrites if exists."""
        path = self._path_for(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure the 'updated' timestamp is current
        data["updated"] = datetime.now().isoformat()
        try:
            path.write_text(
                yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            logger.debug("Checkpoint written: %s (%d chars)", session_id, path.stat().st_size)
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Failed to write checkpoint %s: %s", session_id, e)

    def read(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Read a checkpoint. Returns None if missing or corrupt."""
        path = self._path_for(session_id)
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                return None
            return data
        except (OSError, yaml.YAMLError) as e:
            logger.debug("Failed to read checkpoint %s: %s", session_id, e)
            return None

    def delete(self, session_id: str) -> None:
        """Delete a checkpoint. No-op if missing."""
        path = self._path_for(session_id)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def list_sessions(self) -> List[str]:
        """Return all session IDs that have checkpoint files."""
        if not self._dir.exists():
            return []
        return sorted(p.stem for p in self._dir.glob("*.yaml"))

    def garbage_collect(self, max_age_days: int = DEFAULT_GC_MAX_AGE_DAYS) -> List[str]:
        """Remove stale checkpoints. Returns list of removed session IDs.

        A checkpoint is stale if:
        - Its 'updated' timestamp is older than max_age_days AND not completed
        - OR it's 'completed' and older than 1 day (task done, no need to keep)
        """
        removed = []
        cutoff = datetime.now() - timedelta(days=max_age_days)
        completed_cutoff = datetime.now() - timedelta(days=1)

        for path in self._dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    path.unlink(missing_ok=True)
                    removed.append(path.stem)
                    continue
                updated_str = data.get("updated", "")
                status = data.get("status", "in_progress")
                try:
                    updated = datetime.fromisoformat(updated_str) if updated_str else cutoff
                except ValueError:
                    updated = cutoff

                if status == "completed" and updated < completed_cutoff:
                    path.unlink(missing_ok=True)
                    removed.append(path.stem)
                elif updated < cutoff:
                    path.unlink(missing_ok=True)
                    removed.append(path.stem)
            except (OSError, yaml.YAMLError):
                path.unlink(missing_ok=True)
                removed.append(path.stem)

        if removed:
            logger.info("Checkpoint GC: removed %d stale file(s): %s", len(removed), removed)
        return removed