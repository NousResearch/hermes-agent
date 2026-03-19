"""
Cosmos Knowledge Base

This module provides access to Cosmos's knowledge files for the memory system.
Knowledge is loaded into the retrieval database so bots know what capabilities exist.
ALWAYS APPENDS to memory - never overwrites.
"""

import os
import json
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent


def get_knowledge_files() -> list:
    """Get all knowledge files in the knowledge directory."""
    files = []
    for f in KNOWLEDGE_DIR.glob("*.md"):
        files.append(f)
    for f in KNOWLEDGE_DIR.glob("*.json"):
        files.append(f)
    return files


def load_api_knowledge() -> str:
    """Load the API capabilities knowledge base."""
    api_file = KNOWLEDGE_DIR / "api_capabilities.md"
    if api_file.exists():
        return api_file.read_text(encoding="utf-8")
    return ""


def get_all_knowledge() -> dict:
    """Load all knowledge files into a dictionary."""
    knowledge = {}
    for f in get_knowledge_files():
        if f.suffix == ".md":
            knowledge[f.stem] = f.read_text(encoding="utf-8")
        elif f.suffix == ".json":
            try:
                knowledge[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass
    return knowledge


# Import loader functions
try:
    from .loader import (
        get_knowledge_loader,
        load_all_knowledge,
        get_api_reference,
        KnowledgeLoader
    )
except ImportError:
    # Loader not yet created
    pass
