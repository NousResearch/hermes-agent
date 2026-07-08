"""Ingest pipeline."""

from __future__ import annotations

from pathlib import Path

from llmwiki_hermes.compiler.service import CompilerService, normalize_text
from llmwiki_hermes.schemas.cli import CommandOutput
from llmwiki_hermes.settings import WikiSettings
from llmwiki_hermes.storage.ingest_inputs import IngestInputLoader, flatten_json_text
from llmwiki_hermes.storage.sqlite_index import IndexService
from llmwiki_hermes.storage.vault import VaultService

__all__ = ["IngestService", "normalize_text", "flatten_json_text"]


class IngestService:
    """Compile raw source material into notes and index entries."""

    def __init__(self, settings: WikiSettings) -> None:
        self.settings = settings
        self.vault_service = VaultService(settings.vault_path)
        self.index_service = IndexService(self.vault_service)
        self.compiler_service = CompilerService(self.vault_service, self.index_service)
        self.input_loader = IngestInputLoader()

    @classmethod
    def from_settings(cls, settings: WikiSettings) -> "IngestService":
        return cls(settings)

    def ingest(
        self,
        path: Path | None,
        stdin: bool,
        recursive: bool,
        tags: list[str],
        source_type: str | None,
        dry_run: bool,
    ) -> CommandOutput:
        """Ingest a file, directory, or STDIN payload."""

        self.vault_service.ensure_initialized()
        inputs = self.input_loader.resolve(path=path, stdin=stdin, recursive=recursive)
        created: list[str] = []
        for loaded_input in inputs.loaded_inputs:
            effective_source_type = source_type or loaded_input.detected_source_type
            created.extend(
                self.compiler_service.ingest_input(
                    input_path=loaded_input.path,
                    raw_content=loaded_input.raw_content,
                    source_type=effective_source_type,
                    tags=tags,
                    dry_run=dry_run,
                )
            )
        if not dry_run and inputs.loaded_inputs:
            self.compiler_service.reindex()
        failed_inputs = [failure.as_dict() for failure in inputs.failed_inputs]
        processed_inputs = inputs.processed_inputs
        successful_inputs = len(inputs.loaded_inputs)
        failed_count = len(failed_inputs)
        if failed_count == 0:
            message = f"Ingested {successful_inputs} input(s)."
        elif successful_inputs > 0:
            message = (
                f"Ingested {successful_inputs} of {processed_inputs} input(s); "
                f"{failed_count} failed."
            )
        else:
            message = f"Failed to ingest {processed_inputs} input(s)."
        return CommandOutput(
            ok=successful_inputs > 0 or processed_inputs == 0,
            message=message,
            data={
                "created_or_updated": created,
                "dry_run": dry_run,
                "processed_inputs": processed_inputs,
                "successful_inputs": successful_inputs,
                "failed_inputs": failed_inputs,
            },
        )
