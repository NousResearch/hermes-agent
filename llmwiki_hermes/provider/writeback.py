"""Session writeback orchestration."""

from __future__ import annotations

import logging
from typing import Any

from llmwiki_hermes.compiler.episodic import create_or_append_episodic_note
from llmwiki_hermes.compiler.source import render_source_note
from llmwiki_hermes.storage.frontmatter import write_note
from llmwiki_hermes.storage.sqlite_index import IndexService
from llmwiki_hermes.storage.vault import VaultService
from llmwiki_hermes.utils.slug import slugify

logger = logging.getLogger(__name__)


class SessionWritebackService:
    """Persist session lifecycle artifacts without blocking the caller."""

    def __init__(self, vault_service: VaultService) -> None:
        self.vault_service = vault_service
        self.index_service = IndexService(vault_service)

    def sync_turn(self, session_id: str, user: str, assistant: str) -> None:
        """Append a single turn to the session log."""

        self.vault_service.append_session_turn(session_id, {"user": user, "assistant": assistant})
        logger.debug("Appended session turn for %s.", session_id)

    def on_pre_compress(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Persist the tail of the conversation before compression."""

        preview = [item for item in messages[-4:]]
        self.vault_service.write_precompress_snapshot(session_id, preview)
        logger.debug("Persisted %s pre-compress message(s) for %s.", len(preview), session_id)

    def on_session_end(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        auto_writeback: bool,
    ) -> None:
        """Persist a session transcript as source and episodic memory."""

        if not auto_writeback:
            logger.debug("Auto writeback disabled for session %s.", session_id)
            return

        transcript = "\n".join(
            f"{item.get('role', 'message')}: {item.get('content', '')}" for item in messages
        ).strip()
        if not transcript:
            logger.debug(
                "Skipping writeback for session %s because transcript is empty.",
                session_id,
            )
            return

        session_name = session_id or "session"
        title = f"Session {session_name}"
        source_frontmatter, source_body, filename = render_source_note(
            content=transcript,
            path=None,
            source_type="chat",
            tags=["session"],
        )
        source_path = self.vault_service.sources_dir / f"src-{slugify(session_name)}-{filename}"
        write_note(source_path, source_frontmatter.model_dump(mode="json"), source_body)
        episodic_path = create_or_append_episodic_note(
            vault_service=self.vault_service,
            title=title,
            content=transcript,
            source_note=source_frontmatter,
            dry_run=False,
        )
        if episodic_path is None:
            logger.warning(
                (
                    "Skipped episodic session writeback for %s because the matched "
                    "note uses an unsupported schema."
                ),
                session_id,
            )
        self.index_service.reindex()
        logger.info("Persisted session writeback artifacts for %s.", session_id)
