"""Hermes plugin CLI wrapper."""

from __future__ import annotations

from pathlib import Path

from llmwiki_hermes.provider import cli as provider_cli


def wiki_command(args) -> None:
    """Dispatch the active ``hermes wiki`` subcommand."""

    command = getattr(args, "wiki_command", None)
    if command == "init":
        provider_cli.init(
            vault=Path(args.vault),
            force=bool(args.force),
            json_output=bool(args.json_output),
        )
        return
    if command == "ingest":
        provider_cli.ingest(
            path=Path(args.path) if args.path else None,
            vault=Path(args.vault) if args.vault else None,
            stdin=bool(args.stdin),
            recursive=bool(args.recursive),
            tags=str(args.tags or ""),
            source_type=args.source_type,
            dry_run=bool(args.dry_run),
            json_output=bool(args.json_output),
        )
        return
    if command == "reindex":
        provider_cli.reindex(
            vault=Path(args.vault) if args.vault else None,
            json_output=bool(args.json_output),
        )
        return
    if command == "recall":
        provider_cli.recall(
            query=str(args.query),
            vault=Path(args.vault) if args.vault else None,
            memory_type=str(args.memory_type),
            top_k=int(args.top_k),
            json_output=bool(args.json_output),
        )
        return
    if command == "doctor":
        provider_cli.doctor(
            vault=Path(args.vault) if args.vault else None,
            json_output=bool(args.json_output),
        )
        return
    if command == "compact":
        provider_cli.compact(
            vault=Path(args.vault) if args.vault else None,
            json_output=bool(args.json_output),
        )
        return
    raise SystemExit("Unknown wiki command.")


def register_cli(subparser) -> None:
    """Build the ``hermes wiki`` argparse tree."""

    subs = subparser.add_subparsers(dest="wiki_command")

    init_parser = subs.add_parser("init", help="Initialize a new LLM-Wiki vault")
    init_parser.add_argument("--vault", required=True, help="Base path that will contain the vault")
    init_parser.add_argument("--force", action="store_true", help="Overwrite system files")
    init_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Render output as JSON"
    )

    ingest_parser = subs.add_parser("ingest", help="Ingest source material into the vault")
    ingest_parser.add_argument("path", nargs="?", help="Path to a file or directory to ingest")
    ingest_parser.add_argument("--vault", help="Existing LLM-Wiki vault path")
    ingest_parser.add_argument("--stdin", action="store_true", help="Read content from STDIN")
    ingest_parser.add_argument("--recursive", action="store_true", help="Recurse into directories")
    ingest_parser.add_argument("--tags", default="", help="Comma-separated tags")
    ingest_parser.add_argument("--source-type", dest="source_type", help="Explicit source type")
    ingest_parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    ingest_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Render output as JSON"
    )

    reindex_parser = subs.add_parser("reindex", help="Rebuild the SQLite sidecar index")
    reindex_parser.add_argument("--vault", help="Existing LLM-Wiki vault path")
    reindex_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Render output as JSON"
    )

    recall_parser = subs.add_parser("recall", help="Search the knowledge base")
    recall_parser.add_argument("-q", "--query", required=True, help="Query text")
    recall_parser.add_argument("--vault", help="Existing LLM-Wiki vault path")
    recall_parser.add_argument(
        "--memory-type",
        default="auto",
        choices=("auto", "semantic", "episodic"),
        help="Memory type bias",
    )
    recall_parser.add_argument("--top-k", type=int, default=8, help="Max result count")
    recall_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Render output as JSON"
    )

    doctor_parser = subs.add_parser("doctor", help="Validate the vault and index")
    doctor_parser.add_argument("--vault", help="Existing LLM-Wiki vault path")
    doctor_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Render output as JSON"
    )

    compact_parser = subs.add_parser("compact", help="Report maintenance candidates")
    compact_parser.add_argument("--vault", help="Existing LLM-Wiki vault path")
    compact_parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Render output as JSON"
    )

    subparser.set_defaults(func=wiki_command)


__all__ = ["register_cli", "wiki_command"]
