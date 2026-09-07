"""Peer-ID resolution for HonchoSessionManager (config aliases, runtime identities, fallbacks)."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plugins.memory.honcho.session import HonchoSession

logger = logging.getLogger("plugins.memory.honcho.session")

_PEER_ID_HASH_ESCALATION_LENGTHS = (8, 12, 16, 24, 32, 64)


class SessionPeersMixin:
    """Resolve user/assistant/observer peer IDs. Reads ``self._config`` and runtime identities only."""

    def _sanitize_id(self, id_str: str) -> str:
        """Sanitize an ID to match Honcho's pattern: ^[a-zA-Z0-9_-]+"""
        return re.sub(r'[^a-zA-Z0-9_-]', '-', id_str)

    def _cfg(self, name: str, default: Any = None) -> Any:
        return getattr(self._config, name, default) if self._config is not None else default

    def _runtime_user_ids(self) -> list[str]:
        """Runtime identity candidates in lookup order (deduped, blanks dropped)."""
        candidates = [str(v).strip() for v in (self._runtime_user_peer_name, self._runtime_user_peer_name_alt) if v is not None]
        return list(dict.fromkeys(c for c in candidates if c))

    def _peer_aliases(self) -> dict:
        aliases = self._cfg("user_peer_aliases", {})
        return aliases if isinstance(aliases, dict) else {}

    def _explicit_user_peer_ids(self) -> set[str]:
        """Return sanitized user peer IDs that came from explicit config."""
        explicit_ids = {self._sanitize_id(alias.strip()) for alias in self._peer_aliases().values()
                        if isinstance(alias, str) and alias.strip()}
        owner = self._declared_owner_peer_id()
        if owner:
            explicit_ids.add(owner)
        return explicit_ids

    def _generated_runtime_peer_id(self, prefix: str, runtime_id: str) -> str:
        """Stable peer ID for an unknown prefixed runtime user; a hash suffix is added when
        sanitizing changed the ID or it collides with an explicitly configured peer."""
        raw_peer_id = f"{prefix}{runtime_id}"
        sanitized_peer_id = self._sanitize_id(raw_peer_id)
        explicit_ids = self._explicit_user_peer_ids()
        if sanitized_peer_id == raw_peer_id and sanitized_peer_id not in explicit_ids:
            return sanitized_peer_id
        digest = hashlib.sha256(raw_peer_id.encode("utf-8")).hexdigest()
        for hash_len in (*_PEER_ID_HASH_ESCALATION_LENGTHS, len(digest)):
            candidate = f"{sanitized_peer_id}-{digest[:hash_len]}"
            if candidate not in explicit_ids or hash_len == len(digest):
                return candidate

    def _declared_owner_peer_id(self) -> str | None:
        """Sanitized ``peerName`` (the install owner), or None when none is declared."""
        peer_name = str(self._cfg("peer_name") or "").strip()
        return self._sanitize_id(peer_name) if peer_name else None

    def _resolve_user_peer_id(self, key: str) -> str:
        """Honcho user peer ID for this manager/session. Order: pinned peerName -> alias of a
        runtime identity -> (prefixed) runtime identity -> configured peerName -> session-key fallback."""
        peer_name = self._cfg("peer_name")
        if peer_name and self._cfg("pin_peer_name", False) is True:
            return self._sanitize_id(peer_name)

        runtime_ids = self._runtime_user_ids()
        if runtime_ids:
            aliases = self._peer_aliases()
            for runtime_id in runtime_ids:
                alias = aliases.get(runtime_id)
                if isinstance(alias, str) and alias.strip():
                    return self._sanitize_id(alias.strip())
            prefix = self._cfg("runtime_peer_prefix", "")
            prefix = prefix.strip() if isinstance(prefix, str) else ""
            if prefix:
                return self._generated_runtime_peer_id(prefix, runtime_ids[0])
            return self._sanitize_id(runtime_ids[0])

        if peer_name:
            return self._sanitize_id(peer_name)
        channel, sep, chat_id = key.partition(":")
        return self._sanitize_id(f"user-{channel}-{chat_id}" if sep else f"user-default-{key}")

    def _resolve_peer_id(self, session: HonchoSession, peer: str | None) -> str:
        """Resolve a peer alias ('user'/'ai') or explicit peer ID to a concrete, non-empty peer ID."""
        normalized = self._sanitize_id((peer or "user").strip() or "user")
        return {"user": session.user_peer_id, "ai": session.assistant_peer_id}.get(normalized, normalized)

    def _resolve_observer_target(self, session: HonchoSession, peer: str | None) -> tuple[str, str | None]:
        """Resolve (observer, target) peer IDs for context/search/profile queries."""
        target_peer_id = self._resolve_peer_id(session, peer)
        if target_peer_id == session.assistant_peer_id:
            return session.assistant_peer_id, session.assistant_peer_id
        if self._ai_observe_others:
            return session.assistant_peer_id, target_peer_id
        return target_peer_id, None
