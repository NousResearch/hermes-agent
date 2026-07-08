"""Typer app exposed as the Hermes `wiki` subcommand."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from llmwiki_hermes.constants import PROVIDER_NAME
from llmwiki_hermes.errors import LlmWikiError
from llmwiki_hermes.schemas.cli import CommandOutput

if TYPE_CHECKING:
    from llmwiki_hermes.settings import WikiSettings

app = typer.Typer(help="LLM-Wiki provider commands for Hermes.")


def echo_output(output: CommandOutput, as_json: bool = False) -> None:
    """Render CLI output as either human text or JSON."""

    if as_json:
        typer.echo(json.dumps(output.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    typer.echo(render_human_output(output))


def render_human_output(output: CommandOutput) -> str:
    """Render a stable human-readable summary from the JSON payload."""

    lines = [output.message]
    if _is_ingest_output(output.data):
        lines.extend(_render_ingest_details(output.data))
    elif _is_doctor_output(output.data):
        lines.extend(_render_doctor_details(output.data))
    elif _is_reindex_output(output.data):
        lines.extend(_render_reindex_details(output.data))
    elif _is_compact_output(output.data):
        lines.extend(_render_compact_details(output.data))
    return "\n".join(lines)


def _is_doctor_output(data: dict[str, Any]) -> bool:
    return "issues" in data and "stats" in data


def _is_ingest_output(data: dict[str, Any]) -> bool:
    return {
        "created_or_updated",
        "processed_inputs",
        "successful_inputs",
        "failed_inputs",
    } <= data.keys()


def _is_reindex_output(data: dict[str, Any]) -> bool:
    return {"note_files", "indexed_notes", "skipped_notes", "failed_notes"} <= data.keys()


def _is_compact_output(data: dict[str, Any]) -> bool:
    return "summary" in data and "semantic_duplicate_candidates" in data


def _render_ingest_details(data: dict[str, Any]) -> list[str]:
    failed_inputs = data.get("failed_inputs", [])
    lines = [
        "Summary:",
        (
            f"- mode: {'dry-run' if data.get('dry_run') else 'apply'}, "
            f"processed: {data.get('processed_inputs', 0)}, "
            f"successful: {data.get('successful_inputs', 0)}, "
            f"failed: {len(failed_inputs)}"
        ),
        f"- created/updated notes: {len(data.get('created_or_updated', []))}",
    ]
    if failed_inputs:
        lines.append("Failures:")
        for failure in failed_inputs:
            lines.append(f"- [{failure['code']}] {failure['path']}: {failure['message']}")
    return lines


def _render_doctor_details(data: dict[str, Any]) -> list[str]:
    stats = data.get("stats", {})
    issues = data.get("issues", [])
    severity_counts = data.get("severity_counts", {})
    lines = [
        "Summary:",
        (
            f"- note files: {stats.get('note_files', 0)} total, "
            f"{stats.get('valid_notes', 0)} valid, "
            f"{stats.get('invalid_notes', 0)} invalid, "
            f"{stats.get('indexed_notes', 0)} indexed"
        ),
        (
            f"- note health: {stats.get('orphan_semantic_notes', 0)} orphan semantic, "
            f"{stats.get('orphan_episodic_notes', 0)} orphan episodic, "
            f"{stats.get('duplicate_source_hash_groups', 0)} duplicate source hash group(s)"
        ),
        (
            f"- index health: {stats.get('missing_index_entries', 0)} missing entry, "
            f"{stats.get('stale_index_entries', 0)} stale entry"
        ),
        (
            f"- schema health: {stats.get('legacy_schema_notes', 0)} legacy note(s), "
            f"{stats.get('notes_by_schema_version', {})}"
        ),
    ]
    if severity_counts:
        counts = ", ".join(
            f"{count} {severity}" for severity, count in sorted(severity_counts.items())
        )
        lines.append(f"- issue severity: {counts}")
    if issues:
        lines.append("Issues:")
        for issue in issues:
            location = issue["path"]
            lines.append(f"- [{issue['severity']}] {issue['code']} {location}: {issue['message']}")
        recovery_workflow = data.get("recovery_workflow", [])
        if recovery_workflow:
            lines.append(f"Recovery workflow: {' -> '.join(recovery_workflow)}")
            lines.append(
                "If issues remain after reindex, inspect the listed files and repair them manually."
            )
    return lines


def _render_reindex_details(data: dict[str, Any]) -> list[str]:
    lines = [
        "Summary:",
        (
            f"- files scanned: {data.get('note_files', 0)}, "
            f"indexed: {data.get('indexed_notes', 0)}, "
            f"skipped: {data.get('skipped_notes', 0)}"
        ),
        (
            f"- index contents: {data.get('chunk_count', 0)} chunk(s), "
            f"{data.get('link_count', 0)} link(s)"
        ),
        f"- index path: {data.get('index_path', '')}",
    ]
    failed_notes = data.get("failed_notes", [])
    if failed_notes:
        lines.append("Skipped notes:")
        for issue in failed_notes:
            lines.append(f"- [{issue['severity']}] {issue['path']}: {issue['message']}")
        lines.append("Recommended next step: run doctor after reindex to confirm remaining issues.")
    else:
        lines.append("Recommended next step: run doctor after reindex to confirm index health.")
    return lines


def _render_compact_details(data: dict[str, Any]) -> list[str]:
    summary = data.get("summary", {})
    lines = [
        "Summary:",
        f"- mode: {'dry-run' if data.get('dry_run') else 'apply'}",
        (
            f"- candidate groups: {summary.get('total_groups', 0)} total, "
            f"{summary.get('semantic_duplicate_groups', 0)} semantic duplicate, "
            f"{summary.get('semantic_source_conflict_groups', 0)} semantic source conflict, "
            f"{summary.get('episodic_near_duplicate_groups', 0)} episodic near-duplicate"
        ),
    ]
    candidate_lines = _compact_candidate_lines(data)
    if candidate_lines:
        lines.append("Candidates:")
        lines.extend(candidate_lines)
        lines.append("Review candidate notes manually before making any content changes.")
    else:
        lines.append("No candidate groups detected. No manual cleanup is suggested right now.")
    return lines


def _compact_candidate_lines(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for candidate in data.get("semantic_duplicate_candidates", []):
        note_ids = ", ".join(note["id"] for note in candidate.get("notes", []))
        lines.append(
            "- semantic duplicate "
            f"{candidate.get('key', 'group')}: {note_ids} "
            f"(canonical: {candidate.get('canonical_note_id', 'n/a')}, "
            f"reason: {candidate.get('match_reason', 'n/a')})"
        )
    for candidate in data.get("semantic_source_conflict_candidates", []):
        note_ids = ", ".join(note["id"] for note in candidate.get("notes", []))
        lines.append(
            "- semantic source conflict "
            f"{candidate.get('source_ref', 'source')}: {note_ids} "
            f"(canonical: {candidate.get('canonical_note_id', 'n/a')}, "
            f"reason: {candidate.get('match_reason', 'n/a')})"
        )
    for candidate in data.get("episodic_near_duplicate_candidates", []):
        note_ids = ", ".join(note["id"] for note in candidate.get("notes", []))
        lines.append(
            f"- episodic near-duplicate {candidate.get('date', 'date')} / "
            f"{candidate.get('title_key', 'title')}: {note_ids} "
            f"(canonical: {candidate.get('canonical_note_id', 'n/a')}, "
            f"reason: {candidate.get('match_reason', 'n/a')})"
        )
    return lines


def abort_on_error(exc: Exception) -> None:
    """Convert domain exceptions into user-facing CLI failures."""

    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=1) from exc


def load_settings(vault: Path | None) -> "WikiSettings":
    """Load settings from CLI args, then Hermes profile config, then defaults."""

    from llmwiki_hermes.settings import WikiSettings

    config_path = None
    hermes_home = os.getenv("HERMES_HOME")
    if not hermes_home:
        try:
            from hermes_constants import get_hermes_home  # type: ignore[import-untyped]
        except ImportError:
            hermes_home = None
        else:
            hermes_home = str(get_hermes_home())
    if hermes_home:
        candidate = Path(hermes_home) / PROVIDER_NAME / "config.yaml"
        if candidate.exists():
            config_path = candidate
    return WikiSettings.load(vault_path=vault, config_path=config_path)


@app.command()
def init(
    vault: Path = typer.Option(..., help="Base path that will contain the LLM-Wiki vault."),
    force: bool = typer.Option(False, help="Overwrite system files if they already exist."),
    json_output: bool = typer.Option(False, "--json", help="Render output as JSON."),
) -> None:
    """Initialize a new vault."""

    from llmwiki_hermes.storage.vault import VaultService

    try:
        service = VaultService.from_user_path(vault)
        output = service.initialize(force=force)
        echo_output(output, as_json=json_output)
    except (LlmWikiError, FileNotFoundError, OSError) as exc:
        abort_on_error(exc)


@app.command()
def ingest(
    path: Path | None = typer.Argument(None, help="Path to a file or directory to ingest."),
    vault: Path | None = typer.Option(None, help="Existing LLM-Wiki vault path."),
    stdin: bool = typer.Option(False, help="Read content from STDIN."),
    recursive: bool = typer.Option(False, help="Recurse into directories."),
    tags: str = typer.Option("", help="Comma-separated tags."),
    source_type: str | None = typer.Option(None, help="Explicit source type."),
    dry_run: bool = typer.Option(False, help="Preview outputs without writing notes."),
    json_output: bool = typer.Option(False, "--json", help="Render output as JSON."),
) -> None:
    """Ingest source material into the vault."""

    from llmwiki_hermes.compiler.ingest import IngestService

    try:
        settings = load_settings(vault)
        service = IngestService.from_settings(settings)
        output = service.ingest(
            path=path,
            stdin=stdin,
            recursive=recursive,
            tags=[item.strip() for item in tags.split(",") if item.strip()],
            source_type=source_type,
            dry_run=dry_run,
        )
        echo_output(output, as_json=json_output)
        if not output.ok:
            raise typer.Exit(code=1)
    except (LlmWikiError, FileNotFoundError, OSError, ValueError) as exc:
        abort_on_error(exc)


@app.command()
def reindex(
    vault: Path | None = typer.Option(None, help="Existing LLM-Wiki vault path."),
    json_output: bool = typer.Option(False, "--json", help="Render output as JSON."),
) -> None:
    """Rebuild the SQLite sidecar index."""

    from llmwiki_hermes.storage.sqlite_index import IndexService
    from llmwiki_hermes.storage.vault import VaultService

    try:
        settings = load_settings(vault)
        output = IndexService(VaultService(settings.vault_path)).reindex()
        echo_output(output, as_json=json_output)
    except (LlmWikiError, FileNotFoundError, OSError) as exc:
        abort_on_error(exc)


@app.command()
def recall(
    query: str = typer.Option(..., "--query", "-q", help="Query text."),
    vault: Path | None = typer.Option(None, help="Existing LLM-Wiki vault path."),
    memory_type: str = typer.Option("auto", help="auto|semantic|episodic"),
    top_k: int = typer.Option(8, help="Max result count."),
    json_output: bool = typer.Option(False, "--json", help="Render output as JSON."),
) -> None:
    """Explicitly search the knowledge base."""

    from llmwiki_hermes.recall.search import RecallService

    try:
        settings = load_settings(vault)
        service = RecallService.from_settings(settings)
        output = service.recall_cli(query=query, memory_type=memory_type, top_k=top_k)
        echo_output(output, as_json=json_output)
    except (LlmWikiError, FileNotFoundError, OSError, ValueError) as exc:
        abort_on_error(exc)


@app.command()
def doctor(
    vault: Path | None = typer.Option(None, help="Existing LLM-Wiki vault path."),
    json_output: bool = typer.Option(False, "--json", help="Render output as JSON."),
) -> None:
    """Validate the vault and index."""

    from llmwiki_hermes.storage.vault import VaultService

    try:
        settings = load_settings(vault)
        service = VaultService(settings.vault_path)
        service.ensure_initialized()
        output = service.doctor()
        echo_output(output, as_json=json_output)
    except (LlmWikiError, FileNotFoundError, OSError) as exc:
        abort_on_error(exc)


@app.command()
def compact(
    vault: Path | None = typer.Option(None, help="Existing LLM-Wiki vault path."),
    json_output: bool = typer.Option(False, "--json", help="Render output as JSON."),
) -> None:
    """Report potential semantic duplicates and conflicts."""

    from llmwiki_hermes.compiler.semantic import SemanticMaintenanceService

    try:
        settings = load_settings(vault)
        output = SemanticMaintenanceService.from_settings(settings).compact_report()
        echo_output(output, as_json=json_output)
    except (LlmWikiError, FileNotFoundError, OSError) as exc:
        abort_on_error(exc)
