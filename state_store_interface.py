"""Backend-neutral protocol for the session/state store surface plugins may rely on.

``StateStoreInterface`` is a ``typing.Protocol`` (structural, ``@runtime_checkable``):
a store satisfies it by *having* the methods, not by inheriting anything. The
SQLite ``SessionDB`` (``hermes_state.SessionDB``) is the reference
implementation. Other backends can implement the same shape independently, so
plugins do not need to depend on SQLite internals.

Scope and intent:

* The face below is what real plugins (session-titler, hermes-evolve) actually
  consume today. It is deliberately narrow — every member was extracted from a
  live ``getattr`` probe, not invented.
* The protocol is a **plugin-compatibility check tool**, not a runtime gate:
  use ``isinstance(store, StateStoreInterface)`` in tests and dev-time
  validation to prove a store exposes the contracted face. Nothing in the
  runtime should refuse to call a store merely because it fails this check.
* Private underscore members (``_execute_write``, ``_read_one``,
  ``_is_compression_ancestor``, ...) are **not part of the contract** even
  though some plugins reach for them defensively; they are implementation
  details of the SQLite store and may change or differ per backend.
* New capabilities enter the protocol first, then the implementations; the
  protocol is the versioned contract that lets backends drift independently
  without breaking plugin consumers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = ["StateStoreInterface"]


@runtime_checkable
class StateStoreInterface(Protocol):
    """The state-store surface plugins may rely on, across all backends.

    Structural: implementations do NOT inherit from this protocol. The
    reference SQLite implementation is ``hermes_state.SessionDB``; the
    PostgreSQL backend implements the same members on its facade.
    """

    # ── Title provenance and writes ────────────────────────────────────────
    def get_session_title(self, session_id: str) -> Optional[str]:
        """Title for a session, or None when untitled."""
        ...

    def get_session_title_source(self, session_id: str) -> Optional[str]:
        """Provenance of a session's title (``user``/``llm``/``derived``), or None."""
        ...

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set a title with ``user`` authority; False when the row is untouched."""
        ...

    def set_auto_title(self, session_id: str, title: str, *, source: str) -> bool:
        """Set an automatic title; False when a higher-authority title holds the row."""
        ...

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        """Normalize a title (strip control chars, collapse whitespace); None when empty."""
        ...

    # ``"llm"`` — the automatic-title source a titler writes with.
    TITLE_SOURCE_LLM: str

    # ── Session reads ──────────────────────────────────────────────────────
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Full session row (prompt resolved through the prompt store when present)."""
        ...

    def get_messages_as_conversation(self, session_id: str, **kwargs) -> List[Dict[str, Any]]:
        """Messages for a session in OpenAI conversation format."""
        ...

    def search_messages(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """FTS-backed message search across sessions."""
        ...

    # ── System-prompt invalidation ─────────────────────────────────────────
    def clear_stored_system_prompts(self) -> Dict[str, Any]:
        """Invalidate every stored system-prompt snapshot.

        Sessions keep their rows; the resolved prompt becomes unset so the
        next run/resume rebuilds it from the live configuration. Idempotent:
        with nothing stored (or already cleared) it reports ``cleared == 0``.

        Returns ``{"cleared": <int>, "storage_mode": <str>}`` where
        ``storage_mode`` is ``"out-of-line"`` (prompt table + hash
        references), ``"inline"`` (prompt column on sessions), or
        ``"unknown"`` (neither layout detected).
        """
        ...
