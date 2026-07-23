#!/usr/bin/env python3
"""Export mem0/Qdrant memories to human-readable Markdown vault files.

Reads all points from a Qdrant collection that belong to a given user_id
and writes each as a Markdown file to an output directory.

File naming: <qdrant_point_id>.md
Frontmatter: id, agent_id, score, created_at, updated_at
Body: memory text (the 'data' payload field)

Uses the Qdrant HTTP API directly (urllib, no Python SDK required).
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

SCROLL_LIMIT = 200


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_collection_from_mem0_json() -> str | None:
    """Try to read collection_name from ~/.hermes/hermes-agent/mem0.json."""
    mem0_path = Path.home() / ".hermes" / "hermes-agent" / "mem0.json"
    try:
        with mem0_path.open() as f:
            cfg = json.load(f)
        return (
            cfg.get("oss", {})
               .get("vector_store", {})
               .get("config", {})
               .get("collection_name")
        )
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def api_post(qdrant_url: str, path: str, body: dict, timeout: int = 120) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{qdrant_url}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def api_get(qdrant_url: str, path: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(f"{qdrant_url}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Qdrant scrolling
# ---------------------------------------------------------------------------

def scroll_all_points_for_user(qdrant_url: str, collection: str, user_id: str) -> list:
    """Scroll all Qdrant points, returning only those matching user_id."""
    points = []
    offset = None

    while True:
        body: dict = {
            "limit": SCROLL_LIMIT,
            "with_vector": False,
            "with_payload": True,
            "filter": {
                "must": [
                    {
                        "key": "user_id",
                        "match": {"value": user_id},
                    }
                ]
            },
        }
        if offset is not None:
            body["offset"] = offset

        try:
            result = api_post(qdrant_url, f"/collections/{collection}/points/scroll", body)
        except urllib.error.URLError as exc:
            print(f"[ERROR] Qdrant scroll failed: {exc}", file=sys.stderr)
            sys.exit(1)

        batch = result.get("result", {}).get("points", [])
        points.extend(batch)

        next_offset = result.get("result", {}).get("next_page_offset")
        if next_offset is None or not batch:
            break
        offset = next_offset

    return points


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def extract_text(payload: dict) -> str:
    """Pull the memory text from a Qdrant payload (mem0 field ordering)."""
    return (
        payload.get("data")
        or payload.get("text")
        or payload.get("memory")
        or payload.get("content")
        or ""
    )


def render_markdown(point_id: str, payload: dict) -> str:
    """Render a single Qdrant point as a Markdown file with YAML frontmatter."""
    agent_id = payload.get("agent_id", "")
    created_at = payload.get("created_at", "")
    updated_at = payload.get("updated_at", "")
    # Score is not stored in the Qdrant payload — it only exists in search results.
    # We leave it blank; the sync script will not overwrite this field.
    score = payload.get("score", "")

    text = extract_text(payload)

    # Build frontmatter — keep it terse and machine-parseable
    frontmatter_lines = [
        "---",
        f"id: {point_id}",
        f"agent_id: {agent_id}",
        f"score: {score}",
        f"created_at: {created_at}",
        f"updated_at: {updated_at}",
        "---",
    ]
    return "\n".join(frontmatter_lines) + "\n" + text + "\n"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_vault(qdrant_url: str, collection: str, user_id: str, vault_dir: Path) -> int:
    """Export memories to vault. Returns count of files written."""
    vault_dir.mkdir(parents=True, exist_ok=True)

    # Verify Qdrant is up
    try:
        info = api_get(qdrant_url, "/")
        version = info.get("version", "unknown")
        print(f"Qdrant {version} is reachable.", file=sys.stderr)
    except urllib.error.URLError as exc:
        print(f"[ERROR] Cannot reach Qdrant at {qdrant_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Scrolling collection '{collection}' for user_id='{user_id}'...", file=sys.stderr)
    points = scroll_all_points_for_user(qdrant_url, collection, user_id)
    print(f"Found {len(points)} memory point(s).", file=sys.stderr)

    written = 0
    skipped = 0

    for point in points:
        point_id = str(point["id"])
        payload = point.get("payload") or {}

        text = extract_text(payload)
        if not text:
            print(f"[SKIP] {point_id}: no text in payload", file=sys.stderr)
            skipped += 1
            continue

        md_content = render_markdown(point_id, payload)
        dest = vault_dir / f"{point_id}.md"
        dest.write_text(md_content, encoding="utf-8")
        written += 1

    print(f"Vault export complete: {written} written, {skipped} skipped (no text).", file=sys.stderr)
    print(f"Vault location: {vault_dir}", file=sys.stderr)
    return written


def parse_args() -> argparse.Namespace:
    default_collection = load_collection_from_mem0_json() or "hermes_memories"
    default_qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    default_output_dir = Path.home() / ".hermes" / "memories" / "vault"

    parser = argparse.ArgumentParser(
        description="Export mem0/Qdrant memories to human-readable Markdown vault files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--user",
        required=True,
        help="user_id to filter memories by (e.g. 'clark').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to write Markdown files into (created if absent).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=default_qdrant_url,
        help="Base URL for the Qdrant HTTP API. Also reads QDRANT_URL env var.",
    )
    parser.add_argument(
        "--collection",
        default=default_collection,
        help=(
            "Qdrant collection name. Defaults to collection_name from mem0.json "
            "oss.vector_store.config, falling back to 'hermes_memories'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = export_vault(
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        user_id=args.user,
        vault_dir=args.output_dir,
    )
    print(f"Exported {written} memories to {args.output_dir}")


if __name__ == "__main__":
    main()
