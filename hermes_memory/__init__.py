"""Hermes curated memory subsystem (file-backed MEMORY.md/USER.md + MemoryManager).

Bundled optional backends ship under ``hermes_memory/plugins/memory/``.
User-installed providers still go in ``HERMES_HOME/plugins/<name>/``.
This package holds the core implementation used by ``run_agent`` and the ``memory`` tool."""

from hermes_memory.builtin_memory_tool import (
    MEMORY_SCHEMA,
    MemoryStore,
    check_memory_requirements,
    get_memory_dir,
    memory_tool,
)
from hermes_memory.memory_manager import (
    MemoryManager,
    build_memory_context_block,
    sanitize_context,
)
from hermes_memory.memory_provider import MemoryProvider

__all__ = [
    "MEMORY_SCHEMA",
    "MemoryProvider",
    "MemoryManager",
    "MemoryStore",
    "build_memory_context_block",
    "check_memory_requirements",
    "get_memory_dir",
    "memory_tool",
    "sanitize_context",
]
