"""Valut — local secret reference system for in-chat credential safety.

When a user wraps sensitive text with ``?/`` ... ``/?`` in a chat message,
the enclosed value is extracted, stored in a local encrypted-feeling JSON
file (~/.hermes/valut.json, 0600), and replaced with an opaque reference
like ``[VLT:a1b2c3d4]``.  The LLM never sees the plaintext.

On output, any ``[VLT:<id>]`` reference is substituted back to the original
value before the user sees it, so the agent can "use" a secret by
referencing its ID without ever knowing the actual value.

Usage
-----
  User:   Hey, my API key is ?/sk-secret-123/? — can you test it?
  Agent sees: Hey, my API key is [VLT:a1b2c3d4] — can you test it?
  Agent says: I'll call the API with [VLT:a1b2c3d4]
  User sees:  I'll call the API with sk-secret-123
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ── trigger regex ──────────────────────────────────────────────────────────
# Matches ``?/`` opener, any content (non-greedy), ``/?`` closer.
# The content is captured as group 1.
_TRIGGER_RE = re.compile(r"\?\/(.+?)\/\?")

# ── output reference regex ─────────────────────────────────────────────────
# Matches ``[VLT:<id>]`` where <id> is ``vlt_`` + 8 hex chars.
_REF_RE = re.compile(r"\[VLT:(vlt_[0-9a-fA-F]{8})\]")


def _valut_path() -> Path:
    """Return the path to the valut store file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "valut.json"


class ValutStore:
    """Thread-safe persistent store for secret reference mappings."""

    def __init__(self, path: Path | None = None):
        self._path = path or _valut_path()
        self._lock = threading.Lock()
        self._entries: dict[str, str] = {}
        self._loaded = False

    # ── persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load entries from the valut JSON file (best-effort)."""
        if self._loaded:
            return
        self._loaded = True
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._entries = {
                        k: v for k, v in data.items()
                        if isinstance(k, str) and isinstance(v, str)
                    }
        except Exception:
            logger.debug("valut load failed — starting empty", exc_info=True)
            self._entries = {}

    def _save(self) -> None:
        """Write entries to disk atomically with 0600 permissions."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")
            # Set restrictive permissions before rename so the window of
            # exposure is as small as possible.
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            tmp.replace(self._path)
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                pass
        except Exception:
            logger.debug("valut save failed", exc_info=True)

    # ── API ────────────────────────────────────────────────────────────

    def store(self, value: str) -> str:
        """Store a secret and return its reference ID (``vlt_XXXXXXXX``).

        If *value* is empty, returns it unchanged.
        """
        if not value:
            return value
        with self._lock:
            self._load()
            # Check if we already have this value stored — reuse the ID.
            for existing_id, existing_val in self._entries.items():
                if existing_val == value:
                    return existing_id
            # Generate a new unique ID.
            while True:
                ref_id = f"vlt_{secrets.token_hex(4)}"
                if ref_id not in self._entries:
                    break
            self._entries[ref_id] = value
            self._save()
            logger.info("valut: stored secret as %s", ref_id)
            return ref_id

    def resolve(self, ref_id: str) -> str | None:
        """Look up a reference ID. Returns ``None`` if not found."""
        if not ref_id:
            return None
        with self._lock:
            self._load()
            return self._entries.get(ref_id)

    def list_ids(self) -> list[str]:
        """Return all stored reference IDs (for user introspection)."""
        with self._lock:
            self._load()
            return sorted(self._entries.keys())

    def remove(self, ref_id: str) -> bool:
        """Remove a stored secret. Returns True if it existed."""
        with self._lock:
            self._load()
            existed = ref_id in self._entries
            if existed:
                del self._entries[ref_id]
                self._save()
            return existed

    def clear(self) -> None:
        """Remove all stored secrets."""
        with self._lock:
            self._entries.clear()
            self._save()


# ── module-level singleton ─────────────────────────────────────────────────

_store: ValutStore | None = None
_store_lock = threading.Lock()


def _get_store() -> ValutStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ValutStore()
    return _store


# ── public helpers ─────────────────────────────────────────────────────────


def sanitize_input(text: str) -> str:
    """Scan *text* for ``?/.../?`` patterns, vault the secrets, return cleaned text.

    Each ``?/secret/?`` becomes ``[VLT:vlt_XXXXXXXX]`` in the returned string.
    If no triggers are found, the original string is returned unchanged.

    SAFETY: The LLM never receives the plaintext — only the opaque reference.
    """
    if not text or "?/" not in text:
        return text

    store = _get_store()

    def _replace(m: re.Match) -> str:
        secret = m.group(1)
        if not secret.strip():
            return m.group(0)  # empty trigger, pass through unchanged
        ref_id = store.store(secret)
        return f"[VLT:{ref_id}]"

    try:
        return _TRIGGER_RE.sub(_replace, text)
    except Exception:
        logger.debug("valut sanitize_input failed", exc_info=True)
        return text


def restore_output(text: str) -> str:
    """Replace ``[VLT:<id>]`` references in *text* with their stored values.

    If a reference can't be resolved, it passes through unchanged so the
    user can see which ID was referenced.
    """
    if not text or "[VLT:" not in text:
        return text

    store = _get_store()

    def _replace(m: re.Match) -> str:
        ref_id = m.group(1)
        resolved = store.resolve(ref_id)
        if resolved is not None:
            return resolved
        return m.group(0)  # unresolved — leave as-is

    try:
        return _REF_RE.sub(_replace, text)
    except Exception:
        logger.debug("valut restore_output failed", exc_info=True)
        return text


def process_message_for_agent(text: str) -> str:
    """Full input pipeline: sanitize before the agent sees it."""
    return sanitize_input(text)


def process_message_for_display(text: str) -> str:
    """Full output pipeline: restore references before the user sees it."""
    return restore_output(text)
