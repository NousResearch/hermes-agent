#!/usr/bin/env python3
"""Canonical Obsidian sync pipeline with bookmark-ID idempotency ledger.

Reads a JSON payload of bookmark records and writes one deterministic note per
bookmark through Obsidian Local REST API. Maintains a ledger keyed by
bookmark_id so repeated runs are idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib import error, parse, request


DEFAULT_API_BASE = "https://127.0.0.1:27124"
DEFAULT_LEDGER = Path.home() / ".hermes" / "state" / "obsidian_bookmark_ledger.json"
DEFAULT_KEY_RELATIVE = Path(".obsidian/plugins/obsidian-local-rest-api/data.json")


def _discover_api_key(vault_path: str | None, explicit_key: str | None, key_file: str | None) -> str:
    if explicit_key:
        return explicit_key

    candidate_files: List[Path] = []
    if key_file:
        candidate_files.append(Path(key_file).expanduser())
    if vault_path:
        candidate_files.append(Path(vault_path).expanduser() / DEFAULT_KEY_RELATIVE)

    for candidate in candidate_files:
        if not candidate.exists():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        for k in ("apiKey", "api_key", "apikey", "token"):
            value = data.get(k)
            if isinstance(value, str) and value.strip():
                return value.strip()

    env_key = os.getenv("OBSIDIAN_API_KEY", "").strip()
    if env_key:
        return env_key

    raise RuntimeError("Obsidian API key not found. Provide --api-key or --vault-path/--key-file.")


def _load_input(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("bookmarks"), list):
        return data["bookmarks"]
    if isinstance(data, list):
        return data
    raise RuntimeError("Input JSON must be a list or an object with a 'bookmarks' list")


def _bookmark_id(item: Dict[str, Any]) -> str:
    for key in ("bookmark_id", "id", "tweet_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    url = str(item.get("url", "")).strip()
    if url:
        return hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    raise RuntimeError("Bookmark item missing bookmark_id/id/tweet_id and url")


def _fingerprint(item: Dict[str, Any]) -> str:
    payload = {
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "author": item.get("author", ""),
        "text": item.get("text", item.get("content", "")),
        "created_at": item.get("created_at", ""),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


def _render_markdown(item: Dict[str, Any], bookmark_id: str) -> str:
    title = str(item.get("title") or f"Bookmark {bookmark_id}")
    url = str(item.get("url", ""))
    author = str(item.get("author", ""))
    created_at = str(item.get("created_at", ""))
    text = str(item.get("text", item.get("content", ""))).strip()

    lines = [
        "---",
        f"bookmark_id: {bookmark_id}",
        f"title: {json.dumps(title, ensure_ascii=False)}",
        f"url: {json.dumps(url, ensure_ascii=False)}",
        f"author: {json.dumps(author, ensure_ascii=False)}",
        f"created_at: {json.dumps(created_at, ensure_ascii=False)}",
        "source: siftly",
        "---",
        "",
        f"# {title}",
        "",
    ]
    if url:
        lines.extend([f"Source: {url}", ""])
    if text:
        lines.extend([text, ""])
    return "\n".join(lines)


def _load_ledger(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def _save_ledger(path: Path, ledger: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def _put_markdown(api_base: str, api_key: str, note_path: str, markdown: str, timeout: int = 20) -> None:
    base = api_base.rstrip("/")
    endpoint = f"{base}/vault/{parse.quote(note_path)}"
    req = request.Request(endpoint, data=markdown.encode("utf-8"), method="PUT")
    req.add_header("Content-Type", "text/markdown")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("X-API-Key", api_key)

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    with request.urlopen(req, timeout=timeout, context=context) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"Obsidian write failed ({resp.status}) for {note_path}")


def sync_bookmarks(
    items: Iterable[Dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    note_root: str,
    ledger_path: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    ledger = _load_ledger(ledger_path)
    copied = 0
    skipped = 0
    failed: List[Tuple[str, str]] = []

    for item in items:
        bid = _bookmark_id(item)
        fp = _fingerprint(item)
        if ledger.get(bid) == fp:
            skipped += 1
            continue

        note_path = f"{note_root.strip('/').strip()}/{bid}.md"
        markdown = _render_markdown(item, bid)
        try:
            if not dry_run:
                _put_markdown(api_base, api_key, note_path, markdown)
            ledger[bid] = fp
            copied += 1
        except (RuntimeError, error.URLError, TimeoutError) as exc:
            failed.append((bid, str(exc)))

    if not dry_run:
        _save_ledger(ledger_path, ledger)

    return {
        "processed": copied + skipped + len(failed),
        "written": copied,
        "skipped": skipped,
        "failed": [{"bookmark_id": bid, "error": msg} for bid, msg in failed],
        "ledger_path": str(ledger_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical Obsidian sync with bookmark-ID ledger")
    parser.add_argument("--input", required=True, help="Input JSON file (list or {'bookmarks': [...]})")
    parser.add_argument("--note-root", default="Siftly/Bookmarks", help="Obsidian note root folder")
    parser.add_argument("--api-base", default=os.getenv("OBSIDIAN_REST_URL", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.getenv("OBSIDIAN_API_KEY"))
    parser.add_argument("--vault-path", default=os.getenv("OBSIDIAN_VAULT"))
    parser.add_argument("--key-file", default=os.getenv("OBSIDIAN_REST_KEY_FILE"))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER), help="Ledger JSON path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        items = _load_input(Path(args.input).expanduser())
        api_key = _discover_api_key(args.vault_path, args.api_key, args.key_file)
        result = sync_bookmarks(
            items,
            api_base=args.api_base,
            api_key=api_key,
            note_root=args.note_root,
            ledger_path=Path(args.ledger).expanduser(),
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
        return 0 if not result["failed"] else 1
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
