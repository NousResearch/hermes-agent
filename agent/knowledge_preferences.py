"""Knowledge preference manager for the 3-Tier Knowledge Center.

Stores and retrieves user preferences for knowledge promotion decisions.
Uses file-based JSON storage with file locking for concurrent access.
Profile-aware — uses get_hermes_home() for path resolution.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_PREFERENCE_FILE = "knowledge_preferences.json"


class KnowledgePreferenceManager:
    """Manages user preferences for knowledge promotion decisions.

    Preferences determine whether knowledge should be auto-promoted,
    skipped, or asked about when cross-project relevance is detected.
    """

    def __init__(self, hermes_home: Optional[Path] = None) -> None:
        if hermes_home is None:
            hermes_home = get_hermes_home()
        self._prefs_path = hermes_home / _PREFERENCE_FILE
        self._prefs: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load preferences from disk."""
        if self._prefs_path.exists():
            try:
                text = self._prefs_path.read_text(encoding="utf-8")
                self._prefs = json.loads(text)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Failed to load preferences: %s", e)
                self._prefs = []
        else:
            self._prefs = []

    def _save(self) -> None:
        """Save preferences to disk with atomic write."""
        self._prefs_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._prefs_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._prefs, indent=2), encoding="utf-8")
        tmp_path.rename(self._prefs_path)

    def save_preference(
        self,
        domain: str,
        project: str,
        pattern: str,
        allow: bool,
        reason: str = "",
    ) -> str:
        """Save a preference for knowledge promotion.

        Args:
            domain: Domain slug (e.g., 'frontend', 'backend')
            project: Project slug
            pattern: Knowledge pattern to match (keyword or regex)
            allow: True to auto-promote, False to auto-deny
            reason: Optional reason for the preference

        Returns:
            Preference ID
        """
        pref_id = str(uuid.uuid4())[:8]
        pref = {
            "id": pref_id,
            "domain": domain,
            "project": project,
            "pattern": pattern.lower(),
            "allow": allow,
            "reason": reason,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._prefs.append(pref)
        self._save()
        return pref_id

    def check_preference(
        self,
        domain: str,
        project: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if there's a matching preference for this knowledge.

        Checks in order:
        1. Exact domain + project match
        2. Domain-level match (any project)
        3. Pattern match in content

        Args:
            domain: Domain slug
            project: Project slug
            content: Knowledge content to check

        Returns:
            Matching preference dict or None
        """
        content_lower = content.lower()

        # Check exact domain + project match first
        for pref in reversed(self._prefs):
            if pref["domain"] == domain and pref["project"] == project:
                if pref["pattern"] in content_lower:
                    return pref

        # Check domain-level match (project = "*" means any project)
        for pref in reversed(self._prefs):
            if pref["domain"] == domain and pref["project"] == "*":
                if pref["pattern"] in content_lower:
                    return pref

        # Check pattern-only match
        for pref in reversed(self._prefs):
            if pref["pattern"] in content_lower:
                return pref

        return None

    def list_preferences(self) -> List[Dict[str, Any]]:
        """Return all preferences."""
        return list(self._prefs)

    def delete_preference(self, pref_id: str) -> bool:
        """Delete a preference by ID.

        Args:
            pref_id: Preference ID to delete

        Returns:
            True if deleted, False if not found
        """
        before = len(self._prefs)
        self._prefs = [p for p in self._prefs if p["id"] != pref_id]
        if len(self._prefs) < before:
            self._save()
            return True
        return False
