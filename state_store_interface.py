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
* Error channels are part of the contract. Members divide into two classes:
  **boolean-valued writes** (``set_session_title``, ``set_auto_title``) return
  False for every outcome a caller can sensibly retry or skip (missing row,
  lower authority, protected canonical row) and raise ValueError only for
  outcomes the caller MUST handle before retrying (invalid source, title over
  the length cap, title owned by another live session). **Optional-valued
  helpers** (``sanitize_title``) return None only for empty input; every other
  rejection raises, and ``truncate=True`` converts the length rejection into a
  guaranteed-value path. Read members never raise for missing data (None /
  ``[]``) and raise only for genuine storage failures. Backends must keep
  these channels: a caller written against the contract must not meet an
  exception the contract did not declare.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, runtime_checkable

__all__ = ["StateStoreInterface"]

_T = TypeVar("_T")


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
        """Set a title with ``user`` authority; False when the row is missing.

        Raises ValueError for a sanitized title over the reference store's maximum length,
        a title held by another live owner (including a hidden canonical
        Bot Chat), or a rename of that hidden canonical Bot Chat itself.
        """
        ...

    def set_auto_title(self, session_id: str, title: str, *, source: str) -> bool:
        """Set an automatic title from ``derived`` or ``llm`` provenance.

        Raises ValueError for any other source, a sanitized title over
        the reference store's maximum length, or a title held by another live owner
        (including a hidden canonical Bot Chat). Returns False without changing
        a missing row, a title of equal or higher authority, or a hidden
        canonical Bot Chat's title.
        """
        ...

    @staticmethod
    def sanitize_title(title: Optional[str], *, truncate: bool = False) -> Optional[str]:
        """Normalize a title (strip control chars, collapse whitespace); None when empty.

        Length policy (``MAX_TITLE_LENGTH``): by default a normalized title over
        the cap raises ValueError (a caller that cannot handle the title must
        know, not silently keep a different one). ``truncate=True`` asks for a
        value that always exists: the normalized title is cut to the cap and
        re-sanitized, so the result is a valid title to store (this is the
        protocol-programming escape hatch — a store-facing ``Optional[str]``
        caller passes ``truncate=True`` and gets None only for empty input).
        """
        ...

    # ``"llm"`` — the automatic-title source a titler writes with.
    TITLE_SOURCE_LLM: str

    # ── Session reads ──────────────────────────────────────────────────────
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Full session row (prompt resolved through the prompt store when present), or None if missing."""
        ...

    def get_messages_as_conversation(self, session_id: str, **kwargs) -> List[Dict[str, Any]]:
        """Messages for a session in OpenAI conversation format."""
        ...

    def search_messages(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search messages across sessions using the available route.

        The reference SQLite store uses FTS when healthy and a canonical-row
        LIKE fallback when FTS is unavailable or an FTS MATCH read raises
        ``sqlite3.OperationalError``. The fallback supports only its documented
        query subset; ranking, proximity, and other FTS semantics may differ.
        ``[]`` means no match from the route that ran, not a suppressed
        canonical-read failure: errors reading canonical rows propagate.

        Availability is fail-open and NOT distinguishable from the return
        value: index down, index corrupt, and a genuine no-match all return
        ``[]``. A caller that must know probes first — ``is_search_available()``
        for a boolean gate, ``fts_health()`` for detail.
        """
        ...

    def is_search_available(self) -> bool:
        """True when the indexed search route is live for the NEXT ``search_messages`` call.

        False means the next call is still legal but will serve the fallback
        route (canonical-row scan) or ``[]`` — degraded recall, fewer query
        semantics, never an exception for unavailability itself. Fail-open
        discovery: an internal probe error also reports False rather than
        raising, so the pre-call gate itself can never take search down.
        Advisory, point-in-time: with no ``strict`` argument the store may
        fail open (or recover) between this check and the search call.

        Backends without an index report True when their only route is
        canonical and healthy, or False to flag degraded reads.
        """
        ...

    def fts_health(self) -> dict[str, Any]:
        """Diagnostic detail behind ``is_search_available``; read-only, never raises.

        Keys (reference SQLite values): ``available`` (bool — same verdict as
        ``is_search_available``), ``stale`` (bool — fail-open latched or
        observed in shared state), ``routes`` (per-route liveness: the base
        FTS5 index plus the trigram/CJK sub-indexes), ``rebuild`` (deferred
        backfill progress dict or None), ``reason`` (short machine-readable
        unavailability cause or None). Backends may add keys; callers must
        treat unknown keys and missing optional keys tolerantly.
        """
        ...

    # ── System-prompt invalidation ─────────────────────────────────────────
    def clear_stored_system_prompts(self) -> Dict[str, Any]:
        """Invalidate every stored system-prompt snapshot.

        Sessions keep their rows; the resolved prompt becomes unset — the
        protocol's canonical unset value is ``None`` on EVERY layout: the
        reference out-of-line branch NULLs both columns, and a true legacy
        inline row is normalized to NULL as well, so ``get_session(...)[
        "system_prompt"] is None`` means unset regardless of which storage
        layout a database happens to carry. (An inline row that was already
        ``''`` before clear is untouched history, not a cleared value.)
        so the next run/resume rebuilds it from the live configuration.
        The reference SQLite operation is atomic per call, not across this
        method and other store methods. Idempotent:
        with nothing stored (or already cleared) it reports ``cleared == 0``.

        Returns ``{"cleared": <int>, "storage_mode": <str>}`` where
        ``storage_mode`` is ``"out-of-line"`` (prompt table + hash
        references), ``"inline"`` (prompt column on sessions), or
        ``"unknown"`` (neither layout detected).
        """
        ...

    # ── Explicit write atomicity ───────────────────────────────────────────
    def apply_atomic(self, fn: Callable[[Any], _T]) -> _T:
        """Run *fn(store_conn)* as ONE store transaction (unit of work).

        The nine read/write members above are each atomic per call; composing
        them is not. ``apply_atomic`` is the public expression of that
        composition: the backend must land every write *fn* issues in a single
        transaction — all committed together or not at all — across all
        layouts, including mid-``fn`` failures (rollback, partial work never
        visible) and backends where a per-method transaction is the norm.

        ``fn`` receives a connection-like handle. What the handle exposes is
        backend-specific (the reference passes its private writer connection);
        protocol code that stays portable should call ONLY the store's own
        methods inside ``fn`` and treat the handle as an opaque last resort —
        SQL written against it binds the plugin to one backend. Reads inside
        the unit of work see the unit's own uncommitted writes.

        *fn* MUST be idempotent under replay: lock contention can make the
        backend re-run the whole callback after a rollback, so ``fn`` must
        produce the same effect when executed twice against the same starting
        state. Relative mutations (``+=``, ``count = count + 1``) are unsafe
        unless expressed as absolute writes; prefer the store's own
        compare-and-swap style methods. A raised exception propagates after
        rollback: retrying the whole ``apply_atomic`` is then the caller's
        choice.
        """
        ...
