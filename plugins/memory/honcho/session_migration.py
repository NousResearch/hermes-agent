"""One-time upload of local memory files (MEMORY.md / USER.md / SOUL.md) into Honcho."""

from __future__ import annotations

import logging
from pathlib import Path

from plugins.memory.honcho.session_auth import HonchoAuthError

logger = logging.getLogger("plugins.memory.honcho.session")

# (filename, upload name, description, peer kind) — peer kind picks user vs assistant peer.
_MEMORY_FILES = (
    ("MEMORY.md", "consolidated_memory.md", "Long-term agent notes and preferences", "user"),
    ("USER.md", "user_profile.md", "User profile and preferences", "user"),
    ("SOUL.md", "agent_soul.md", "Agent persona and identity configuration", "ai"),
)


class SessionMigrationMixin:
    def migrate_memory_files(self, session_key: str, memory_dir: str) -> bool:
        """Upload MEMORY.md / USER.md / SOUL.md to Honcho when it activates on an instance with
        locally consolidated memory; skips missing/empty files. True if at least one uploaded."""
        memory_path = Path(memory_dir)
        if not memory_path.exists():
            return False

        session = self._cache.get(session_key)
        if not session:
            logger.warning("No local session cached for '%s', skipping memory migration", session_key)
            return False
        if session.honcho_session_id not in self._sessions_cache:
            logger.warning("No Honcho session cached for '%s', skipping memory migration", session_key)
            return False

        # Owner-scoped: these files describe the install owner; uploading them under another
        # human's peer would make Honcho attribute the owner's facts to that person. The owner is
        # the CONFIG peerName — never a re-resolution of the session's own peer (that would compare
        # the triggering user to themselves). No declared owner: single-operator only when there is
        # no runtime identity; with one, nobody can be proven to be the owner.
        owner_peer_id = self._declared_owner_peer_id()
        session_is_owner = (session.user_peer_id == owner_peer_id if owner_peer_id is not None
                            else not self._runtime_user_ids())
        if not session_is_owner:
            logger.info("Skipping memory-file migration: session user peer '%s' is not the declared owner (peerName=%s)",
                        session.user_peer_id, owner_peer_id or "unset")
            return False

        uploaded = False
        for filename, upload_name, description, target_kind in _MEMORY_FILES:
            filepath = memory_path / filename
            content = filepath.read_text(encoding="utf-8").strip() if filepath.exists() else ""
            if not content:
                continue
            target_peer_id = session.user_peer_id if target_kind == "user" else session.assistant_peer_id
            wrapped = ("<prior_memory_file>\n<context>\n"
                       "This file was consolidated from local conversations BEFORE Honcho was activated.\n"
                       f"{description}. Treat as foundational context for this user.\n"
                       f"</context>\n\n{content}\n</prior_memory_file>\n")

            def _upload() -> None:
                self._sdk_session(session.honcho_session_id).upload_file(
                    file=(upload_name, wrapped.encode("utf-8"), "text/plain"),
                    peer=self._get_or_create_peer(target_peer_id),
                    metadata={"source": "local_memory", "original_file": filename, "target_peer": target_kind},
                )

            try:
                self._authed_call("memory migration upload", _upload)
                logger.info("Uploaded %s to Honcho for %s (%s peer)", filename, session_key, target_kind)
                uploaded = True
            except HonchoAuthError:
                logger.error("Honcho memory migration stopped after %s: auth failed", filename)
                break
            except Exception as e:
                logger.error("Failed to upload %s to Honcho: %s", filename, e)

        return uploaded
