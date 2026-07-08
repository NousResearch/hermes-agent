"""Pydantic schemas for notes and CLI payloads."""

from llmwiki_hermes.schemas.cli import CommandOutput, RecallHit, RecallResponse
from llmwiki_hermes.schemas.diagnostics import DiagnosticIssue, DiagnosticReport, DiagnosticSeverity
from llmwiki_hermes.schemas.notes import (
    EpisodicNoteFrontmatter,
    NoteDocument,
    SemanticNoteFrontmatter,
    SourceNoteFrontmatter,
    validate_note_frontmatter,
)

__all__ = [
    "CommandOutput",
    "RecallHit",
    "RecallResponse",
    "DiagnosticIssue",
    "DiagnosticReport",
    "DiagnosticSeverity",
    "NoteDocument",
    "SourceNoteFrontmatter",
    "SemanticNoteFrontmatter",
    "EpisodicNoteFrontmatter",
    "validate_note_frontmatter",
]
