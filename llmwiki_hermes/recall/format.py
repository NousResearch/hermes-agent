"""Formatting helpers for recall output."""

from __future__ import annotations

from llmwiki_hermes.schemas.cli import RecallHit
from llmwiki_hermes.types import NoteKind


def format_recall_block(results: list[RecallHit]) -> str:
    """Build the compact provider prefetch block."""

    semantic = [hit for hit in results if hit.kind == NoteKind.SEMANTIC]
    episodic = [hit for hit in results if hit.kind == NoteKind.EPISODIC]
    source = [hit for hit in results if hit.kind == NoteKind.SOURCE]
    source_refs = sorted({ref for hit in results for ref in hit.source_refs})

    lines: list[str] = ["[Semantic Recall]"]
    if semantic:
        for index, hit in enumerate(semantic, start=1):
            lines.append(f"{index}. {hit.id} — {hit.title}")
            lines.append(f"   {hit.snippet}")
    else:
        lines.append("None")

    lines.append("")
    lines.append("[Episodic Recall]")
    if episodic:
        for index, hit in enumerate(episodic, start=1):
            lines.append(f"{index}. {hit.id} — {hit.title}")
            lines.append(f"   {hit.snippet}")
    else:
        lines.append("None")

    lines.append("")
    lines.append("[Source Recall]")
    if source:
        for index, hit in enumerate(source, start=1):
            lines.append(f"{index}. {hit.id} — {hit.title}")
            lines.append(f"   {hit.snippet}")
    else:
        lines.append("None")

    lines.append("")
    lines.append("[Source Refs]")
    if source_refs:
        lines.extend(f"- {item}" for item in source_refs)
    else:
        lines.append("- none")

    return "\n".join(lines).strip()
