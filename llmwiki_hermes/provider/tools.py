"""Read-only tool implementations exposed to Hermes."""

from __future__ import annotations

from typing import Any

from llmwiki_hermes.recall.search import RecallService
from llmwiki_hermes.storage.vault import VaultService


class ProviderTools:
    """Thin wrappers around recall and note retrieval."""

    def __init__(self, recall_service: RecallService, vault_service: VaultService) -> None:
        self.recall_service = recall_service
        self.vault_service = vault_service

    def wiki_recall(
        self,
        query: str,
        memory_type: str = "auto",
        top_k: int = 8,
    ) -> dict[str, Any]:
        return self.recall_service.recall(
            query=query,
            memory_type=memory_type,
            top_k=top_k,
        ).model_dump(mode="json")

    def wiki_get_note(self, id_or_slug: str) -> dict[str, Any]:
        document = self.vault_service.get_note(id_or_slug=id_or_slug)
        return document.model_dump(mode="json")
