"""
COSMOS KNOWLEDGE LOADER
Loads API references and knowledge on startup.
ALWAYS APPENDS - never overwrites existing knowledge.

This module integrates with the memory system to ensure
persistent knowledge across restarts.
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path("/workspace/Cosmos/cosmos/knowledge")
MEMORY_FILE = Path("/workspace/Cosmos/data/knowledge_memory.json")


class KnowledgeLoader:
    """
    Loads and manages persistent knowledge.
    Integrates with Cosmos's memory system.
    """

    def __init__(self):
        self.knowledge: Dict[str, Any] = {}
        self.load_history: list = []
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure knowledge and data directories exist"""
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _load_existing_memory(self) -> Dict:
        """Load existing memory - NEVER overwrite"""
        if MEMORY_FILE.exists():
            try:
                with open(MEMORY_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load existing memory: {e}")
        return {"entries": [], "loaded_files": [], "last_updated": None}

    def _save_memory(self, memory: Dict):
        """Save memory with append-only semantics"""
        memory["last_updated"] = datetime.now().isoformat()
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(memory, f, indent=2)
            logger.info(f"Saved knowledge memory: {len(memory.get('entries', []))} entries")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def load_knowledge_file(self, filepath: Path) -> Optional[Dict]:
        """Load a single knowledge file"""
        try:
            with open(filepath) as f:
                data = json.load(f)
            logger.info(f"Loaded knowledge: {filepath.name}")
            return data
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None

    def load_all(self) -> Dict[str, Any]:
        """
        Load all knowledge files.
        APPENDS to existing memory - never overwrites.
        """
        memory = self._load_existing_memory()
        loaded_files = set(memory.get("loaded_files", []))
        new_entries = []

        # Find all JSON files in knowledge directory
        if KNOWLEDGE_DIR.exists():
            for filepath in KNOWLEDGE_DIR.glob("*.json"):
                file_id = f"{filepath.name}:{filepath.stat().st_mtime}"

                # Skip if already loaded this version
                if file_id in loaded_files:
                    logger.debug(f"Already loaded: {filepath.name}")
                    continue

                data = self.load_knowledge_file(filepath)
                if data:
                    self.knowledge[filepath.stem] = data
                    loaded_files.add(file_id)
                    new_entries.append({
                        "file": filepath.name,
                        "loaded_at": datetime.now().isoformat(),
                        "keys": list(data.keys()) if isinstance(data, dict) else "array"
                    })
                    self.load_history.append(filepath.name)

        # APPEND new entries to existing ones
        if new_entries:
            memory["entries"] = memory.get("entries", []) + new_entries
            memory["loaded_files"] = list(loaded_files)
            self._save_memory(memory)
            logger.info(f"Added {len(new_entries)} new knowledge entries")

        return self.knowledge

    def get(self, key: str, default: Any = None) -> Any:
        """Get knowledge by key"""
        return self.knowledge.get(key, default)

    def get_api_reference(self, api_name: str) -> Optional[Dict]:
        """Get specific API reference"""
        # Check x_api_reference
        x_ref = self.knowledge.get("x_api_reference", {})
        if api_name.lower() in ["x", "twitter"]:
            return x_ref

        # Check image_api_reference
        img_ref = self.knowledge.get("image_api_reference", {})
        if api_name.lower() in ["grok", "xai"]:
            return img_ref.get("grok_xai")
        if api_name.lower() in ["gemini", "google"]:
            return img_ref.get("gemini_google")

        return None

    def get_status(self) -> Dict:
        """Get loader status"""
        return {
            "knowledge_dir": str(KNOWLEDGE_DIR),
            "loaded_files": self.load_history,
            "total_keys": len(self.knowledge),
            "memory_file": str(MEMORY_FILE)
        }


# Global instance
_knowledge_loader = None

def get_knowledge_loader() -> KnowledgeLoader:
    global _knowledge_loader
    if _knowledge_loader is None:
        _knowledge_loader = KnowledgeLoader()
    return _knowledge_loader

def load_all_knowledge() -> Dict[str, Any]:
    """Load all knowledge on startup"""
    loader = get_knowledge_loader()
    return loader.load_all()

def get_api_reference(api_name: str) -> Optional[Dict]:
    """Get API reference by name"""
    loader = get_knowledge_loader()
    return loader.get_api_reference(api_name)


# Auto-load on import
if __name__ != "__main__":
    try:
        load_all_knowledge()
    except Exception as e:
        logger.warning(f"Auto-load knowledge failed: {e}")
