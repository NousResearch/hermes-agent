"""Per-thread scoping of Hermes's built-in memory (``MEMORY.md`` / ``USER.md``).

Upstream Hermes keeps one memory per profile: ``tools.memory_tool.get_memory_dir()`` is
``$HERMES_HOME/memories``, and every agent built in the profile reads and writes it. On a matter
host the whole case team shares one profile, so a note written in one lawyer's private thread
would reach every later turn for every lawyer.

The matter host scopes it by thread instead:

* a ``dm`` turn reads and writes ``<matter home>/users/<userId>/memories/``;
* a ``channel`` turn reads and writes ``<matter home>/shared/memories/``.

How, with no change to Hermes core: ``MemoryStore`` resolves its files through the overridable
``_path_for`` on every load and write, and every consumer (the ``memory`` tool, the system-prompt
snapshot, compaction reloads, the background memory review, which shares the parent agent's store
object) reaches the files through ``agent._memory_store``. The runner therefore swaps that store,
after the agent is built and before its first prompt is assembled, for a :class:`ScopedMemoryStore`
bound to the thread's folder. The binding lives on the store object, not in a context variable,
so a review thread that outlives the turn still writes to the right folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from litco.homes import safe_segment

MEMORIES = "memories"


def memory_dir(home: Path, kind: str, user_id: Optional[str]) -> Path:
    """``users/<userId>/memories`` for a dm thread, ``shared/memories`` for a channel thread."""
    if kind == "dm":
        if not user_id:
            raise ValueError("a dm thread needs a userId")
        return Path(home) / "users" / safe_segment(user_id) / MEMORIES
    return Path(home) / "shared" / MEMORIES


_STORE_CLASS: Optional[type] = None


def _store_class():
    """The ``ScopedMemoryStore`` class, built once (on first use, so Hermes imports lazily)."""
    global _STORE_CLASS
    if _STORE_CLASS is not None:
        return _STORE_CLASS
    from tools.memory_tool import MemoryStore

    class ScopedMemoryStore(MemoryStore):
        """A ``MemoryStore`` whose ``MEMORY.md`` / ``USER.md`` live in one fixed folder."""

        def __init__(self, directory: Path, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.directory = Path(directory)

        def _path_for(self, target: str) -> Path:  # type: ignore[override]
            return self.directory / ("USER.md" if target == "user" else "MEMORY.md")

    _STORE_CLASS = ScopedMemoryStore
    return _STORE_CLASS


def scoped_store(directory: Path, *, like: Any = None):
    """A loaded store bound to ``directory``, with ``like``'s limits and enable flags."""
    cls = _store_class()
    kwargs = {}
    if like is not None:
        kwargs = {"memory_char_limit": like.memory_char_limit, "user_char_limit": like.user_char_limit,
                  "memory_enabled": like.memory_enabled, "user_profile_enabled": like.user_profile_enabled}
    Path(directory).mkdir(parents=True, exist_ok=True)
    store = cls(directory, **kwargs)
    store.load_from_disk()
    return store


def scope_agent_memory(agent: Any, directory: Path):
    """Rebind ``agent``'s built-in memory to ``directory``. A no-op when memory is off for the agent."""
    current = getattr(agent, "_memory_store", None)
    if current is None:
        return None
    store = scoped_store(directory, like=current)
    agent._memory_store = store
    return store
