#!/usr/bin/env python3
"""Search the Black Wire Maker's Technical Reference Guide catalog.

Looks up board pinouts, GPIO maps, wiring guides and power profiles from the
Black Wire collection (https://valleytech-black-wire-guide.pages.dev/).
Standard library only; reads a local ``library/catalog.json`` or downloads and
caches the live catalog. Never treats a catalog entry as electrical approval.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_URL = "https://valleytech-black-wire-guide.pages.dev/library/catalog.json"
COLLECTION_RAW_BASE = (
    "https://raw.githubusercontent.com/valleytechsolutions/black-wire-pinouts/main/library/"
)
COLLECTION_LIVE_BASE = "https://valleytech-black-wire-guide.pages.dev/library/"
COLLECTION_REPO_URL = "https://github.com/valleytechsolutions/black-wire-pinouts"
APP_URL = "https://valleytech-black-wire-guide.pages.dev/"

STATUS_LABELS = {
    "pinout-source": "pinout source",
    "functions-only": "pin functions - map needed",
    "reference-only": "pinout needed",
    "missing": "documentation needed",
}

PINOUT_ASSET_TYPES = {
    "pinout image",
    "GPIO reference image",
    "pin function reference",
    "connector reference",
    "GPIO block diagram",
}


def _norm(text: Any) -> str:
    """Uppercase and drop every non-alphanumeric character.

    ``RAK-13002`` and ``RAK13002`` both normalize to ``RAK13002`` while numeric
    suffixes stay significant, so RAK13002 never collapses into RAK13003.
    """
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def _tokens(text: Any) -> list[str]:
    """Normalized alphanumeric tokens, before punctuation is dropped.

    ``RAK13002 WisBlock IO Module`` yields ``RAK13002`` as one token, so an exact
    model code matches a longer marketing name.
    """
    return [_norm(token) for token in re.split(r"[^A-Za-z0-9]+", str(text)) if token]


def load_catalog(
    source: str | None = None, cache_dir: str | None = None, refresh: bool = False
) -> dict:
    """Load the catalog from a path, or download and cache the live catalog."""
    source = source or DEFAULT_CATALOG_URL
    if source.startswith("http://") or source.startswith("https://"):
        cache = Path(cache_dir or tempfile.gettempdir()) / "bwm-catalog.json"
        if cache.exists() and not refresh:
            return json.loads(cache.read_text(encoding="utf-8"))
        with urllib.request.urlopen(source, timeout=120) as response:
            payload = response.read()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
        return json.loads(payload.decode("utf-8"))
    path = Path(source)
    if not path.exists():
        raise SystemExit(f"catalog not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _alias_targets(catalog: dict) -> dict[str, str]:
    targets: dict[str, str] = {}
    for alias, canonical in (catalog.get("boardIdAliases") or {}).items():
        targets[_norm(alias)] = canonical
        targets[_norm(canonical)] = canonical
    return targets


def _score_board(board: dict, query: str, aliases: dict[str, str]) -> int:
    board_id = _norm(board.get("id", ""))
    name = _norm(board.get("name", ""))
    record_aliases = [_norm(a) for a in board.get("aliases", [])]
    search_text = _norm(board.get("searchText", ""))
    brand = _norm(board.get("brand", ""))
    processor = _norm(board.get("processor", ""))

    if board_id == query or name == query or query in record_aliases:
        return 100
    if query in _tokens(board.get("id", "")) or query in _tokens(board.get("name", "")):
        return 96
    if query in aliases and _norm(aliases[query]) == board_id:
        return 95
    if query in search_text:
        return 80
    if name.startswith(query) or board_id.startswith(query):
        return 70
    if query in name or query in board_id:
        return 50
    if brand == query or processor == query:
        return 40
    return 0


def search_boards(
    catalog: dict, query: str, limit: int = 10, exact: bool = False
) -> list[tuple[int, dict]]:
    """Return ``(score, board)`` hits, best first.

    ``exact`` keeps only identifier-level matches (id, name, alias), so a bare
    ``RAK`` query does not flood the result with every RAK module.
    """
    normalized = _norm(query)
    if not normalized:
        return []
    aliases = _alias_targets(catalog)
    hits = []
    for board in catalog.get("boards", []):
        score = _score_board(board, normalized, aliases)
        if score and (not exact or score >= 95):
            hits.append((score, board))
    hits.sort(key=lambda item: (-item[0], item[1].get("name", "")))
    return hits[:limit]


def find_board(catalog: dict, board_id: str) -> dict | None:
    wanted = _norm(board_id)
    for board in catalog.get("boards", []):
        if _norm(board.get("id", "")) == wanted:
            return board
    for board in catalog.get("boards", []):
        if wanted in _norm(board.get("name", "")):
            return board
    return None


def status_label(board: dict) -> str:
    coverage = board.get("pinoutCoverage") or {}
    status = coverage.get("status") or "unknown"
    return STATUS_LABELS.get(status, status)


def _require_board(catalog: dict, board_id: str) -> dict:
    board = find_board(catalog, board_id)
    if board is None:
        raise SystemExit(f"board not found: {board_id}")
    return board


def _load_board_paths(path: str | None) -> dict:
    if not path:
        return {}
    board_paths = Path(path)
    if not board_paths.exists():
        raise SystemExit(f"board paths file not found: {board_paths}")
    return json.loads(board_paths.read_text(encoding="utf-8"))


def cmd_search(args: argparse.Namespace, catalog: dict) -> int:
    hits = search_boards(catalog, args.query, limit=args.limit, exact=args.exact)
    if args.json:
        payload = [
            {
                "id": board.get("id"),
                "name": board.get("name"),
                "brand": board.get("brand"),
                "processor": board.get("processor"),
                "revision": board.get("revision"),
                "status": (board.get("pinoutCoverage") or {}).get("status"),
                "pinouts": board.get("pinouts", 0),
                "assets": len(board.get("assets", [])),
                "score": score,
            }
            for score, board in hits
        ]
        print(json.dumps(payload, indent=2))
        return 0 if payload else 1
    if not hits:
        print(f"no board matches {args.query!r}")
        return 1
    for rank, (score, board) in enumerate(hits, 1):
        print(f"{rank}. {board.get('id')} - {board.get('name')}")
        print(
            f"   {board.get('brand')} · {board.get('family')} · "
            f"{board.get('processor')} · rev {board.get('revision')}"
        )
        print(
            f"   coverage: {status_label(board)} · pinouts {board.get('pinouts', 0)}"
            f" · assets {len(board.get('assets', []))}"
        )
        print(f"   {APP_URL}?board={board.get('id')}")
    return 0


def cmd_show(args: argparse.Namespace, catalog: dict) -> int:
    board = _require_board(catalog, args.board)
    if args.json:
        print(json.dumps(board, indent=2))
        return 0
    coverage = board.get("pinoutCoverage") or {}
    documentation = board.get("documentationCoverage") or {}
    print(f"id: {board.get('id')}")
    print(f"name: {board.get('name')}")
    print(
        f"brand: {board.get('brand')} · family: {board.get('family')} · "
        f"processor: {board.get('processor')}"
    )
    print(f"revision: {board.get('revision')}")
    print(f"coverage: {board.get('coverage')} (status: {status_label(board)})")
    print(f"scope: {board.get('scope')} · partial: {board.get('partial')}")
    for url in board.get("sources", []):
        print(f"source: {url}")
    print(
        "pinout coverage: physical {physicalCount} · function tables {functionTableCount}"
        " · function sheets {functionSheetCount} · purpose rows {purposeRows}".format(
            **{k: coverage.get(k, 0) for k in (
                "physicalCount", "functionTableCount", "functionSheetCount", "purposeRows")}
        )
    )
    print(f"complete device approved: {coverage.get('completeDeviceApproved')}")
    for gap in coverage.get("missing", []):
        print(f"  missing: {gap}")
    print(
        "documentation: datasheets {boardDatasheets} · hardware guides {hardwareGuides}"
        " · schematics {schematics} · component datasheets {componentDatasheets}"
        " · visuals {visualCount}".format(
            **{k: documentation.get(k, 0) for k in (
                "boardDatasheets", "hardwareGuides", "schematics",
                "componentDatasheets", "visualCount")}
        )
    )
    for asset in board.get("assets", []):
        print(
            f"asset [{asset.get('type')}] {asset.get('label')} -> {asset.get('file')}"
        )
        print(f"  review: {asset.get('review')}")
        print(f"  rights: {asset.get('rights')}")
        print(f"  source: {asset.get('url')}")
    connectors = board.get("pinConnectors")
    if connectors:
        print(
            f"pin connectors: method {connectors.get('method')} · "
            f"review {connectors.get('review')}"
        )
        for connector in connectors.get("connectors", []):
            print(
                f"  {connector.get('id')} - {connector.get('name')}: "
                f"{len(connector.get('pins', []))} pins"
            )
    if board.get("powerIds"):
        print(f"power profiles: {', '.join(board['powerIds'])}")
    device = board.get("device")
    if device:
        print(
            f"device: {device.get('category')} · phase {device.get('phase')} · "
            f"tags {', '.join(device.get('tags', []))}"
        )
    return 0


def cmd_pins(args: argparse.Namespace, catalog: dict) -> int:
    board = _require_board(catalog, args.board)
    connectors = board.get("pinConnectors")
    if not connectors:
        print(f"no transcribed pin connectors for {board.get('id')}")
        for asset in board.get("assets", []):
            if asset.get("type") in PINOUT_ASSET_TYPES:
                print(f"  pinout asset: {asset.get('label')} -> {asset.get('file')}")
        return 1
    printed = 0
    for connector in connectors.get("connectors", []):
        haystack = f"{connector.get('id', '')} {connector.get('name', '')}".lower()
        if args.connector and args.connector.lower() not in haystack:
            continue
        pins = connector.get("pins", [])
        if args.pin:
            wanted = args.pin.strip().upper()
            pins = [p for p in pins if wanted == str(p.get("label", "")).upper()]
            if not pins:
                continue
        print(
            f"{connector.get('id')} - {connector.get('name')} "
            f"({len(pins)} pins, rows={connector.get('rows')})"
        )
        for pin in pins:
            functions = ", ".join(str(f) for f in pin.get("functions", []))
            print(f"  {pin.get('position'):>3}  {pin.get('label', ''):<14} {functions}")
        printed += 1
    if not printed:
        print("no connector or pin matched the filter")
        return 1
    print(
        f"transcription review: {connectors.get('review')} - not electrical verification"
    )
    return 0


def cmd_resolve(args: argparse.Namespace, catalog: dict) -> int:
    board = _require_board(catalog, args.board)
    board_paths = _load_board_paths(args.board_paths)
    print(f"deep link: {APP_URL}?board={board.get('id')}")
    page = board_paths.get(board.get("id"))
    if page:
        print(f"board page: {page}")
        print(f"board page raw: {COLLECTION_RAW_BASE.replace('/library/', '/')}{page}")
    assets = board.get("assets", [])
    if args.pinouts_only:
        assets = [a for a in assets if a.get("type") in PINOUT_ASSET_TYPES]
    if not assets:
        print("no matching assets")
        return 1
    for asset in assets:
        file = asset.get("file", "")
        print(f"[{asset.get('type')}] {asset.get('label')}")
        print(f"  local:  library/{file}")
        print(f"  raw:    {COLLECTION_RAW_BASE}{file}")
        print(f"  live:   {COLLECTION_LIVE_BASE}{file}")
        if asset.get("url"):
            print(f"  source: {asset.get('url')}")
    return 0


def cmd_wiring(args: argparse.Namespace, catalog: dict) -> int:
    guides = catalog.get("wiringGuides", [])
    if args.query:
        wanted = _norm(args.query)
        guides = [
            g
            for g in guides
            if wanted
            in _norm(
                " ".join(
                    [g.get("name", ""), g.get("category", ""), g.get("summary", "")]
                    + list(g.get("aliases", []))
                )
            )
        ]
    if not guides:
        print(f"no wiring guide matches {args.query!r}")
        return 1
    for guide in guides:
        print(f"{guide.get('id')} - {guide.get('name')} ({guide.get('category')})")
        print(f"  {guide.get('summary')}")
        print(f"  scope: {guide.get('scope')}")
        columns = guide.get("columns", [])
        for row in guide.get("rows", []):
            print("  | " + " | ".join(str(cell) for cell in row) + " |")
        for step in guide.get("steps", []):
            print(f"  step: {step}")
        for check in guide.get("checks", []):
            print(f"  check: {check}")
        for source in guide.get("sources", []):
            print(f"  source: {source.get('title')} - {source.get('url')}")
        print(f"  review: {guide.get('review')}")
    return 0


def cmd_stats(args: argparse.Namespace, catalog: dict) -> int:
    if args.json:
        print(json.dumps({
            "edition": catalog.get("edition"),
            "generatedAt": catalog.get("generatedAt"),
            "stats": catalog.get("stats"),
        }, indent=2))
        return 0
    print(f"edition: {catalog.get('edition')}")
    print(f"generated: {catalog.get('generatedAt')}")
    for key, value in (catalog.get("stats") or {}).items():
        print(f"  {key}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bwm.py",
        description="Look up pinouts, GPIO maps, wiring guides and power profiles "
        "in the Black Wire catalog.",
    )
    parser.add_argument(
        "--catalog",
        help="path or URL to library/catalog.json (default: live catalog, cached "
        "for the session)",
    )
    parser.add_argument("--cache-dir", help="directory for the downloaded catalog")
    parser.add_argument(
        "--refresh", action="store_true", help="re-download the live catalog"
    )
    parser.add_argument(
        "--board-paths", help="path to catalog/board-paths.json (board page index)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="find boards by model")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument(
        "--exact", action="store_true", help="identifier matches only (id/name/alias)"
    )
    search.add_argument("--json", action="store_true")

    show = subparsers.add_parser("show", help="print one board record")
    show.add_argument("board")
    show.add_argument("--json", action="store_true")

    pins = subparsers.add_parser("pins", help="print transcribed connector pins")
    pins.add_argument("board")
    pins.add_argument("--connector", help="filter by connector id or name")
    pins.add_argument("--pin", help="filter by exact pin label")

    resolve = subparsers.add_parser("resolve", help="print asset URLs and deep link")
    resolve.add_argument("board")
    resolve.add_argument(
        "--pinouts-only", action="store_true", help="only pinout-type assets"
    )

    wiring = subparsers.add_parser("wiring", help="search wiring and protocol guides")
    wiring.add_argument("query", nargs="?")

    stats = subparsers.add_parser("stats", help="print catalog edition and counts")
    stats.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = load_catalog(args.catalog, args.cache_dir, args.refresh)
    handlers = {
        "search": cmd_search,
        "show": cmd_show,
        "pins": cmd_pins,
        "resolve": cmd_resolve,
        "wiring": cmd_wiring,
        "stats": cmd_stats,
    }
    return handlers[args.command](args, catalog)


if __name__ == "__main__":
    sys.exit(main())
