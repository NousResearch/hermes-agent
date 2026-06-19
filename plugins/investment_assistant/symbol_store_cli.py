"""CLI for the investment-assistant symbol data store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .symbol_store import DEFAULT_DATA_ROOT, SymbolDataStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest-root":
        return _ingest_root(args)
    if args.command == "reindex":
        return _reindex(args)
    if args.command == "list":
        return _list_symbols(args)
    if args.command == "show":
        return _show_symbol(args)
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-symbol-store",
        description="Manage the investment-assistant file-backed symbol store.",
    )
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT), help="Long-lived data root.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest-root", help="Merge a batch symbols/ directory into the long-lived store.")
    ingest.add_argument("--source-root", required=True, help="Batch root or symbols directory to ingest.")
    ingest.add_argument("--run-id", default="", help="Run id to stamp on ingested layer entries.")
    ingest.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing layer files.")
    ingest.add_argument("--experiment-id", help="Experiment id to create/update.")
    ingest.add_argument("--experiment-name", default="", help="Experiment display name.")
    ingest.add_argument("--stage-id", default="data_ingest", help="Experiment stage id for the ingest artifact.")
    ingest.add_argument("--json", action="store_true", help="Print JSON summary.")

    reindex = subparsers.add_parser("reindex", help="Rebuild SQLite index from symbols/*/manifest.json.")
    reindex.add_argument("--json", action="store_true", help="Print JSON summary.")

    list_cmd = subparsers.add_parser("list", help="List indexed symbols.")
    list_cmd.add_argument("--layer", help="Filter by layer.")
    list_cmd.add_argument("--status", help="Filter by symbol or layer status.")
    list_cmd.add_argument("--include-deleted", action="store_true", help="Include soft-deleted symbols.")
    list_cmd.add_argument("--json", action="store_true", help="Print JSON rows.")

    show = subparsers.add_parser("show", help="Show a symbol manifest.")
    show.add_argument("symbol")
    return parser


def _ingest_root(args: argparse.Namespace) -> int:
    store = SymbolDataStore(args.root)
    manifests = store.ingest_symbols_root(args.source_root, run_id=args.run_id, overwrite=not args.no_overwrite)
    source_root = Path(args.source_root)
    artifact = _ingest_artifact(source_root, manifests)
    if args.experiment_id:
        store.create_experiment(
            args.experiment_id,
            name=args.experiment_name or args.experiment_id,
            metadata={"source_root": str(source_root)},
        )
        store.put_experiment_stage(
            args.experiment_id,
            args.stage_id,
            artifact,
            metadata={"source_root": str(source_root), "run_id": args.run_id},
        )
    store.rebuild_index([manifest["symbol"] for manifest in manifests])
    if args.json:
        print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"ingested_symbols: {len(manifests)}")
        print(f"root: {Path(args.root)}")
        if args.experiment_id:
            print(f"experiment: {args.experiment_id}")
    return 0


def _reindex(args: argparse.Namespace) -> int:
    store = SymbolDataStore(args.root)
    store.rebuild_index()
    rows = store.list_symbols(include_deleted=True)
    payload = {"symbol_count": len(rows), "root": str(Path(args.root))}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"symbol_count: {len(rows)}")
    return 0


def _list_symbols(args: argparse.Namespace) -> int:
    store = SymbolDataStore(args.root)
    rows = store.list_symbols(include_deleted=args.include_deleted, layer=args.layer, status=args.status)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(f"{row['symbol']}\t{row['status']}\t{row['updated_at']}")
    return 0


def _show_symbol(args: argparse.Namespace) -> int:
    manifest = SymbolDataStore(args.root).get_symbol(args.symbol)
    if not manifest:
        raise SystemExit(f"Unknown symbol: {args.symbol}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _ingest_artifact(source_root: Path, manifests: list[dict]) -> dict:
    batch_audit_path = source_root / "batch_audit.json"
    batch_audit = {}
    if batch_audit_path.exists():
        try:
            batch_audit = json.loads(batch_audit_path.read_text(encoding="utf-8"))
        except Exception:
            batch_audit = {}
    return {
        "artifact_type": "symbol_store_ingest",
        "source_root": str(source_root),
        "symbol_count": len(manifests),
        "symbols": [manifest["symbol"] for manifest in manifests],
        "batch_audit": batch_audit,
    }
